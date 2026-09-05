"""vllm-metamodels: prefix caching together with per-request steering.

A steering vector injected at prompt position ``p`` changes the KV cache of every
position ``>= p`` (causal attention leaves positions ``< p`` untouched).  vLLM's
prefix cache keys a block by ``hash(parent_hash, token_ids, extra_keys)``, so two
requests with the same prompt would share ALL blocks -- including the steered ones.
Before 1.1.0.post7 the plugin set ``skip_reading_prefix_cache`` on every hooked
request, which stops *reading* but not *writing*: with ``enable_prefix_caching=True``
(vLLM's default) a steered request still published its steered blocks and a later
plain request with the same prompt silently read them.

This module patches vLLM's ``generate_block_hash_extra_keys`` (called by the request
block hasher for every full block, in the scheduler process) to append a per-request
*salt* to every block whose token range ends after ``salt_from`` (``extra_args["_lens_kv_salt"]
= [salt_from, tag]``):

* blocks entirely before ``salt_from`` keep vLLM's hash -> shared with plain requests and
  with every other steered request of the same prompt template;
* blocks from the one containing token ``salt_from`` onward carry the salt -> only
  requests with the same tag can hit them.

``salt_from = min(steered positions) - 1``: salting from the token BEFORE the marker
guarantees that a steered request recomputes at least two tokens (the marker and its
predecessor) whenever its salted blocks miss, so the recompute can never be scheduled as a
1-token "decode" batch -- under CUDA graphs those replay without hooks and the injection
would be lost silently.

Tags (``extra_args["lens_cache_salt"]`` on the client):

* ``"nonce"`` (default) -> ``n:<unique per request>``: steered blocks are written but never
  reused.  Always exact; the template prefix is still shared.
* ``"payload"`` -> ``p:<sha1 of the steering payload>``: requests with identical prompt AND
  identical vectors share the steered blocks too (GRPO groups).  When the marker is the
  LAST prompt token the patch falls back to a per-request nonce: a full-prompt hit would
  leave a 1-token recompute of the marker itself.
* any other string -> ``u:<string>``: caller-managed sharing.

Early-exit requests (``lens_early_exit``) are salted from token 0: layers above the exit
layer never ran, so their KV is garbage and must never be reusable.  That makes early exit
legal on engines with prefix caching enabled (the worker still requires the salt per request).

``install()`` is idempotent and is called from the plugin's ``register()``, which vLLM runs in
the API process, the engine-core (scheduler) process and every worker.  It also adds
``EngineCore.lens_kv_salt_active`` so the client can verify -- through the engine's own
utility RPC -- that the scheduler process is patched before it relies on the salt.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Sequence

import torch

from vllm_lens._helpers.types import SteeringVector

logger = logging.getLogger(__name__)

KV_SALT_KEY = "_lens_kv_salt"
"""``extra_args`` entry written by the plugin: ``[salt_from_token_idx, tag]``."""

CACHE_SALT_KEY = "lens_cache_salt"
"""User-facing ``extra_args`` key: ``"nonce"`` (default) | ``"payload"`` | explicit string."""

_TRUTHY = ("1", "true", "yes", "on")
_NO_POS = 1 << 62
_PATCH_ATTR = "_vllm_lens_kv_salt_patched"


def _env_truthy(name: str, default: str = "") -> bool:
    return os.environ.get(name, default).strip().lower() in _TRUTHY


# ---------------------------------------------------------------------------
# client side: what to salt
# ---------------------------------------------------------------------------


def min_steered_position(vectors: Sequence[SteeringVector]) -> int | None:
    """Lowest absolute prompt position any vector can touch; 0 for broadcast (2-D)
    vectors; ``None`` when no vector touches any position."""
    lo = _NO_POS
    for sv in vectors:
        act = sv.activations
        if act.dim() == 2:
            return 0
        n_pos = int(act.shape[1])
        pos = list(sv.position_indices[:n_pos]) if sv.position_indices is not None else list(range(n_pos))
        if pos:
            lo = min(lo, min(int(p) for p in pos))
    return None if lo == _NO_POS else lo


def payload_hash(vectors: Sequence[SteeringVector]) -> str:
    """sha1 of everything that determines the injected KV: vector bytes, layers,
    positions, scale, norm_match, mode (order-sensitive)."""
    h = hashlib.sha1()
    for sv in vectors:
        t = sv.activations.detach().to("cpu").contiguous()
        h.update(str((tuple(t.shape), str(t.dtype), list(sv.layer_indices), sv.position_indices, float(sv.scale),
                      bool(sv.norm_match), sv.mode)).encode())
        h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()[:20]


def steering_salt(vectors: Sequence[SteeringVector], mode: str, nonce: str) -> list[Any] | None:
    """``[salt_from, tag]`` for a steered request, or ``None`` if nothing is steered."""
    lo = min_steered_position(vectors)
    if lo is None:
        return None
    if mode == "payload":
        tag = "p:" + payload_hash(vectors)
    elif mode == "nonce":
        tag = "n:" + nonce
    else:
        tag = "u:" + str(mode)
    return [max(0, lo - 1), tag]


def plan_request_kv(
    vectors: Sequence[SteeringVector] | None,
    wants_capture: bool,
    wants_readout: bool,
    early_exit: bool,
    cache_salt: Any,
    nonce: str,
) -> tuple[bool, list[Any] | None]:
    """Decide ``(skip_reading_prefix_cache, kv_salt)`` for one hooked request on an engine
    with prefix caching enabled and the salt patch active.

    * capture / readout -> skip reading (the hooks must see the requested positions);
    * early exit -> salt every block (garbage KV above the exit layer);
    * steering -> salt from the marker's predecessor (nonce / payload / explicit tag);
    * nothing else needs either.
    """
    skip = bool(wants_capture or wants_readout)
    mode = "nonce" if cache_salt is None else str(cache_salt)
    if early_exit:
        return skip, [0, "n:" + nonce]
    if vectors:
        return skip, steering_salt(vectors, mode, nonce)
    return skip, None


# ---------------------------------------------------------------------------
# engine-core side: the block-hash patch
# ---------------------------------------------------------------------------


def _salt_key_for_block(request: Any, end_token_idx: int) -> str | None:
    """The extra hash key a block ``[.., end_token_idx)`` of ``request`` gets, or ``None``."""
    sp = getattr(request, "sampling_params", None)
    extra = getattr(sp, "extra_args", None) if sp is not None else None
    if not extra:
        return None
    salt = extra.get(KV_SALT_KEY)
    if salt is None:
        return None
    salt_from, tag = int(salt[0]), str(salt[1])
    if end_token_idx <= salt_from:
        return None
    if tag.startswith("p:"):
        n_prompt = int(getattr(request, "num_prompt_tokens", 0) or 0)
        if n_prompt and salt_from + 1 >= n_prompt - 1:
            # Marker on the last prompt token: identical requests could hit every full block
            # and recompute the marker alone, possibly as a decode graph -> never share.
            tag = f"n:{getattr(request, 'request_id', id(request))}"
    return f"vllm_lens:{tag}"


def _make_patched(orig):
    def generate_block_hash_extra_keys(request, start_token_idx, end_token_idx, start_mm_idx):
        extra, mm = orig(request, start_token_idx, end_token_idx, start_mm_idx)
        key = _salt_key_for_block(request, end_token_idx)
        if key is None:
            return extra, mm
        return ((*extra, key) if extra else (key,)), mm

    generate_block_hash_extra_keys.__wrapped__ = orig  # type: ignore[attr-defined]
    setattr(generate_block_hash_extra_keys, _PATCH_ATTR, True)
    return generate_block_hash_extra_keys


def install() -> bool:
    """Patch ``vllm.v1.core.kv_cache_utils.generate_block_hash_extra_keys`` in THIS process
    and register the ``lens_kv_salt_active`` utility on ``EngineCore``.  Idempotent.
    Returns True when the patch is active (``VLLM_LENS_KV_SALT=0`` disables it -- then the
    plugin falls back to ``skip_reading_prefix_cache`` for every hooked request)."""
    if not _env_truthy("VLLM_LENS_KV_SALT", "1"):
        return False
    try:
        from vllm.v1.core import kv_cache_utils as ku
    except Exception:  # pragma: no cover - vLLM without the V1 core
        logger.warning("vllm-lens: vllm.v1.core.kv_cache_utils not importable; prefix-cache salting unavailable")
        return False
    fn = getattr(ku, "generate_block_hash_extra_keys", None)
    if fn is None:  # pragma: no cover
        logger.warning("vllm-lens: generate_block_hash_extra_keys not found; prefix-cache salting unavailable")
        return False
    if not getattr(fn, _PATCH_ATTR, False):
        ku.generate_block_hash_extra_keys = _make_patched(fn)
    try:
        from vllm.v1.engine.core import EngineCore

        if not hasattr(EngineCore, "lens_kv_salt_active"):
            EngineCore.lens_kv_salt_active = lambda self: True  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        logger.debug("vllm-lens: could not register EngineCore.lens_kv_salt_active", exc_info=True)
    return True


def is_installed() -> bool:
    try:
        from vllm.v1.core import kv_cache_utils as ku
    except Exception:  # pragma: no cover
        return False
    return bool(getattr(getattr(ku, "generate_block_hash_extra_keys", None), _PATCH_ATTR, False))


def scheduler_active_sync(llm: Any) -> bool:
    """Ask the ENGINE-CORE process (through vLLM's utility RPC) whether it runs the patch."""
    try:
        ec = llm.llm_engine.engine_core
    except Exception:
        return False
    try:
        if hasattr(ec, "call_utility"):
            return bool(ec.call_utility("lens_kv_salt_active"))
        inner = getattr(ec, "engine_core", None)  # in-process client
        if inner is not None:
            return bool(getattr(inner, "lens_kv_salt_active", lambda: False)())
    except Exception:  # noqa: BLE001 - unknown utility -> not patched
        logger.debug("vllm-lens: lens_kv_salt_active utility failed", exc_info=True)
    return False


async def scheduler_active_async(engine: Any) -> bool:
    try:
        ec = engine.engine_core
        if hasattr(ec, "call_utility_async"):
            return bool(await ec.call_utility_async("lens_kv_salt_active"))
        inner = getattr(ec, "engine_core", None)
        if inner is not None:
            return bool(getattr(inner, "lens_kv_salt_active", lambda: False)())
    except Exception:  # noqa: BLE001
        logger.debug("vllm-lens: lens_kv_salt_active utility failed", exc_info=True)
    return False
