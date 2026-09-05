"""
vLLM general plugin that transparently captures residual-stream
activations via worker extension when ``output_residual_stream`` is
passed in ``extra_args``.

Installed automatically via the ``vllm.general_plugins`` entry point
(configured in pyproject.toml). Patches ``EngineArgs.create_engine_config``
to inject the worker extension and eager mode, and patches
``AsyncLLM.generate`` and ``LLM.generate`` to retrieve per-request
activations for both online (async) and offline (sync) usage.

vllm-lens-metamodel environment variables:

``VLLM_LENS_CUDA_GRAPHS=1``
    Do not force ``enforce_eager``.  The plugin sets ``compilation_config``
    to mode ``NONE`` (no torch.compile, so the forward hooks still fire)
    with ``cudagraph_mode=FULL_DECODE_ONLY`` unless you passed a compatible
    one.  Uniform-decode batches then replay CUDA graphs (hooks silent) and
    every batch containing prompt tokens runs eagerly with the hooks live:
    steering and capture apply to **prompt positions only**; 2-D
    (broadcast) steering vectors are rejected.
``VLLM_LENS_COMPILE=1`` (together with ``VLLM_LENS_CUDA_GRAPHS=1``)
    Keep vLLM's torch.compile instead of forcing compilation mode NONE: the layer
    hooks run as an opaque custom op inside the compiled graph (``_compile_op``),
    installed from ``Worker.load_model`` before the model is traced.  Same
    prompt-position-only semantics as CUDA graphs; ``cudagraph_mode`` is forced to
    FULL_DECODE_ONLY.
``VLLM_LENS_BLOCK_RPC=0``
    Disable packing the offline call's single-position vectors into one
    ``set_steering_block`` RPC (falls back to ``set_steering_data_many``).
``VLLM_LENS_DISABLE=1``
    Make the plugin a no-op (plain vLLM), e.g. for baselines.
``VLLM_LENS_FAST_CAPTURE=0``
    Fall back to the 1.1.0 capture path (one blocking ``.cpu()`` slice per
    request per layer-step, one ``get_captured_states`` RPC per request).
    Default on: one gather + one pinned async copy per layer-step, one RPC per
    ``generate()`` call, ``extra_args["capture_positions"]`` honoured.
``VLLM_LENS_EARLY_EXIT=0``
    Never short-circuit forward passes (``extra_args["lens_early_exit"]`` is
    then rejected client-side).
``VLLM_USE_V2_MODEL_RUNNER`` (vLLM's own switch, >= 0.23)
    The plugin defaults it to ``0``: the hooks read the V1 model runner's per-step
    state; with the V2 runner forced on, engine construction fails loudly.
``VLLM_LENS_PREFIX_CACHE=0``
    Hooked requests never READ the prefix cache (the pre-post7 behaviour).  Default on:
    with ``enable_prefix_caching=True`` steering-only requests reuse the cached blocks
    of their prompt template up to the block before the steered position; the blocks
    from there on are salted per request (see ``vllm_lens._kv_salt``).
``VLLM_LENS_SHM=1`` | ``view``
    Same-host zero-copy transport (``_shm``): captured activations come back through one
    POSIX shared-memory segment per ``generate()`` (``1`` = copied out into ordinary
    tensors, ``view`` = zero-copy views that keep the mapping alive via
    ``output.lens_shm``), and the per-call steering / readout vector blocks travel the
    same way.  Off by default; falls back to the pickled RPCs when a segment cannot be
    opened.
``VLLM_LENS_KV_SALT=0``
    Do not patch vLLM's block hasher.  Then hooked requests fall back to skipping the
    cache read, early exit is refused with prefix caching, and -- as in every version
    before post7 -- steered blocks are still WRITTEN to the cache.

Readout (vllm-metamodels): ``extra_args["apply_readout_vectors"] = [ReadoutVector(...)]``
returns ``output.readout`` (per-position cosine / dot products with a per-request
direction, computed in the worker -- no hidden states leave the GPU).
"""

from __future__ import annotations

import copy
import json
import logging
import os
import pickle
import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from typing import TYPE_CHECKING, Any

import torch
import zstandard as zstd

from vllm_lens import _kv_salt, _shm
from vllm_lens._helpers._serialize import serialize_activations, serialize_tensor
from vllm_lens._helpers.types import (
    CAPTURE_POSITIONS_KEY,
    EARLY_EXIT_KEY,
    EMBED_LAYER_INDEX,
    ReadoutVector,
    SteeringVector,
)
from vllm_lens._kv_salt import CACHE_SALT_KEY, KV_SALT_KEY

logger = logging.getLogger(__name__)

_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
_ZSTD_DECOMPRESSOR = zstd.ZstdDecompressor()

if TYPE_CHECKING:
    from vllm import LLM, SamplingParams
    from vllm.v1.engine.async_llm import AsyncLLM

_WORKER_EXT = "vllm_lens._worker_ext.HiddenStatesExtension"
_TRUTHY = ("1", "true", "yes", "on")

# Populated by register() with the original unpatched methods.
_original_create_engine_config: Callable | None = None
_original_generate: Callable | None = None
_original_llm_generate: Callable | None = None
_original_completion_response: Callable | None = None
_original_chat_full_generator: Callable | None = None

# vllm-lens-metamodel: set by _patched_create_engine_config in the process that
# builds the engine config; True when decode batches run as CUDA-graph replays.
_cuda_graphs_enabled: bool = False
_warned_capture_prompt_only: bool = False


def _env_truthy(name: str, default: str = "") -> bool:
    return os.environ.get(name, default).strip().lower() in _TRUTHY


# ---------------------------------------------------------------------------
# vllm-metamodels: prefix caching state (engine setting + scheduler-side salt patch)
# ---------------------------------------------------------------------------

_warned_unsalted: bool = False


def _prefix_caching_enabled(cfg_owner: Any) -> bool:
    try:
        vc = getattr(cfg_owner, "vllm_config", None) or cfg_owner.llm_engine.vllm_config
        return bool(vc.cache_config.enable_prefix_caching)
    except Exception:  # noqa: BLE001
        return False


def _warn_unsalted_once() -> None:
    global _warned_unsalted
    if _warned_unsalted:
        return
    _warned_unsalted = True
    logger.warning(
        "vllm-lens: enable_prefix_caching=True but the scheduler process does not run the "
        "vllm-lens block-hash salt patch (VLLM_LENS_KV_SALT=0, or plugins not loaded there). Hooked "
        "requests skip READING the prefix cache, but their steered / early-exit blocks are still "
        "WRITTEN and a plain request with the same prompt may read steered KV. Disable prefix "
        "caching or enable the patch."
    )


def _prefix_cache_state_sync(llm: Any) -> tuple[bool, bool]:
    """``(prefix caching enabled, salt patch active in the scheduler process)``, cached per engine."""
    st = getattr(llm, "_lens_pc_state", None)
    if st is None:
        enabled = _prefix_caching_enabled(llm)
        active = bool(enabled and _kv_salt.scheduler_active_sync(llm))
        st = (enabled, active)
        llm._lens_pc_state = st
        if enabled and not active:
            _warn_unsalted_once()
    return st


async def _prefix_cache_state_async(engine: Any) -> tuple[bool, bool]:
    st = getattr(engine, "_lens_pc_state", None)
    if st is None:
        enabled = _prefix_caching_enabled(engine)
        active = bool(enabled and await _kv_salt.scheduler_active_async(engine))
        st = (enabled, active)
        engine._lens_pc_state = st
        if enabled and not active:
            _warn_unsalted_once()
    return st


def _effective_caps(caps: dict[str, Any], pc_enabled: bool, salt_active: bool) -> dict[str, Any]:
    """Worker capabilities + the client-side prefix-cache facts.  Early exit with prefix
    caching needs the salt patch (its blocks must never be reusable)."""
    caps = dict(caps or {})
    caps["prefix_caching"] = bool(pc_enabled)
    caps["kv_salt_active"] = bool(salt_active)
    if pc_enabled and caps.get("early_exit") and not salt_active:
        caps["early_exit"] = False
        caps["early_exit_reason"] = (
            "enable_prefix_caching=True and the scheduler process does not run the vllm-lens "
            "block-hash salt patch (skipped layers would leave reusable garbage KV blocks)"
        )
    return caps


def _apply_kv_policy(
    sp: Any,
    vectors: Sequence[SteeringVector] | None,
    wants_capture: bool,
    readouts: Sequence[ReadoutVector] | None,
    early_exit: Any,
    cache_salt: Any,
    nonce: str,
    pc_state: tuple[bool, bool],
    force_skip: bool,
) -> None:
    """Set ``skip_reading_prefix_cache`` / ``extra_args[KV_SALT_KEY]`` on one request's params."""
    hooked = vectors is not None or wants_capture or bool(readouts)
    pc_enabled, salt_active = pc_state
    if not hooked:
        if force_skip:
            sp.skip_reading_prefix_cache = True
        return
    if not (pc_enabled and salt_active):
        # pre-post7 behaviour: hooks need every prompt position computed
        sp.skip_reading_prefix_cache = True
        return
    skip, salt = _kv_salt.plan_request_kv(vectors, wants_capture, bool(readouts), bool(early_exit), cache_salt, nonce)
    if salt is not None:
        if sp.extra_args is None:
            sp.extra_args = {}
        sp.extra_args[KV_SALT_KEY] = salt
    if skip or force_skip or not _env_truthy("VLLM_LENS_PREFIX_CACHE", "1"):
        sp.skip_reading_prefix_cache = True


# ---------------------------------------------------------------------------
# PP merge helper
# ---------------------------------------------------------------------------


def _merge_captured_states(
    states: list[bytes | None] | None,
) -> dict[str, Any] | None:
    """Merge activation captures from multiple PP ranks.

    ``collective_rpc`` returns results in rank order (rank 0, 1, ...).
    With TP, only TP-rank-0 workers capture (others return ``None``).
    Each capturing rank's tensor is sorted by global layer index.
    Because lower PP ranks hold earlier layers, concatenating non-None
    results along dim 0 produces correct global layer ordering.
    """
    if not states:
        return None
    parts: list[dict[str, Any]] = [
        pickle.loads(_ZSTD_DECOMPRESSOR.decompress(s) if s[:4] == _ZSTD_MAGIC else s)
        for s in states
        if s is not None
    ]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]["activations"]
    merged = torch.cat([p["activations"]["residual_stream"] for p in parts], dim=0)
    out: dict[str, Any] = {"residual_stream": merged}
    if "positions" in parts[0]["activations"]:
        out["positions"] = parts[0]["activations"]["positions"]
    return out


def _trim_activations(
    activations: dict[str, Any],
    expected_len: int,
) -> None:
    """Trim residual stream activations and input_ids to the expected length.

    The vLLM v1 scheduler may execute one extra forward pass after the EOS
    stop condition is hit, because ``schedule()`` commits the next step
    before ``update_from_output()`` checks stop conditions.  vLLM itself
    discards the extra output tokens
    (``vllm.v1.core.sched.scheduler.Scheduler.update_from_output`` skips
    already-finished requests), but our activation capture hooks still fire
    during that extra pass.  This trims the surplus positions so the
    residual stream shape is always deterministic.
    """
    rs = activations.get("residual_stream")
    pos = activations.get("positions")
    if pos is not None:
        # fast path: keep the captured positions below expected_len (surplus pass)
        keep = [i for i, p in enumerate(pos) if p < expected_len]
        if len(keep) != len(pos):
            activations["positions"] = [pos[i] for i in keep]
            if rs is not None:
                activations["residual_stream"] = rs[:, keep, :]
        return
    if rs is not None and rs.shape[1] > expected_len:
        activations["residual_stream"] = rs[:, :expected_len, :]
    ids = activations.get("input_ids")
    if ids is not None and len(ids) > expected_len:
        activations["input_ids"] = ids[:expected_len]


def _trim_readout(readout: list[dict[str, Any]], expected_len: int) -> None:
    for r in readout:
        pos = r.get("positions") or []
        keep = [i for i, p in enumerate(pos) if p < expected_len]
        if len(keep) != len(pos):
            r["positions"] = [pos[i] for i in keep]
            r["values"] = r["values"][:, keep]


def _coerce_readouts(readouts: Any) -> list[ReadoutVector] | None:
    if readouts is None:
        return None
    if isinstance(readouts, str):
        readouts = [ReadoutVector.model_validate(d) for d in json.loads(readouts)]
    elif isinstance(readouts, ReadoutVector):
        readouts = [readouts]
    else:
        readouts = [r if isinstance(r, ReadoutVector) else ReadoutVector.model_validate(r) for r in readouts]
    return list(readouts)


def _check_readout_request(
    caps: dict[str, Any] | None,
    readouts: Sequence[ReadoutVector] | None,
    early_exit: Any,
    max_tokens: int | None,
    capture_layers: Any,
) -> None:
    """vllm-metamodels: validate readout / early-exit requests before submission."""
    if readouts:
        if caps and caps.get("readout") is False:
            raise ValueError("vllm-lens: this engine's worker has no readout support")
        if caps and caps.get("multi_stream"):
            bad = sorted({int(l) for rv in readouts for l in rv.layer_indices if l != EMBED_LAYER_INDEX})
            if bad:
                raise ValueError(
                    f"vllm-lens: readout at decoder layer(s) {bad} is unsupported on this hyper-connection "
                    "architecture; use layer_indices=[EMBED_LAYER_INDEX]."
                )
    if early_exit:
        if max_tokens != 1:
            raise ValueError(
                f"vllm-lens: {EARLY_EXIT_KEY} requires max_tokens=1 (the pass stops after the deepest "
                f"requested layer; nothing can be generated), got max_tokens={max_tokens}"
            )
        if capture_layers is True:
            raise ValueError(f"vllm-lens: {EARLY_EXIT_KEY} needs an explicit layer list, not output_residual_stream=True")
        if capture_layers is None and not readouts:
            raise ValueError(f"vllm-lens: {EARLY_EXIT_KEY} without output_residual_stream or apply_readout_vectors")
        if caps is not None and caps and not caps.get("early_exit", False):
            raise ValueError(
                "vllm-lens: early exit is unavailable on this engine: "
                f"{caps.get('early_exit_reason', 'worker predates early exit')}"
            )


def _pack_readouts(
    payloads: dict[str, list[ReadoutVector]],
) -> tuple[dict[str, Any] | None, dict[str, list[ReadoutVector]]]:
    """Single-layer, single-vector keys -> one ``set_readout_block`` payload; the rest
    go through ``set_readout_data_many``."""
    keys: list[str] = []
    vecs: list[torch.Tensor] = []
    layers: list[int] = []
    positions: list[Any] = []
    metrics: list[str] = []
    biases: list[float] = []
    rest: dict[str, list[ReadoutVector]] = {}
    for key, rvs in payloads.items():
        if len(rvs) == 1 and rvs[0].activations.shape[0] == 1:
            rv = rvs[0]
            keys.append(key)
            vecs.append(rv.activations[0].detach().float().cpu())
            layers.append(int(rv.layer_indices[0]))
            positions.append(rv.positions)
            metrics.append(rv.metric)
            biases.append(float(rv.bias))
        else:
            rest[key] = rvs
    if not keys:
        return None, rest
    return {
        "keys": keys,
        "vecs": torch.stack(vecs).contiguous(),
        "layers": layers,
        "positions": positions,
        "metric": metrics,
        "bias": biases,
    }, rest


# ---------------------------------------------------------------------------
# Engine config patch — inject worker extension + eager mode
# ---------------------------------------------------------------------------


def _configure_cuda_graphs(engine_args: Any) -> bool:
    """vllm-lens-metamodel: make ``engine_args`` hook-compatible WITHOUT forcing eager.

    Returns True if decode batches will run as CUDA-graph replays.  The
    forward hooks only fire for eagerly executed layers, so any
    torch.compile mode other than NONE falls back to eager, and
    ``cudagraph_mode`` must be NONE or FULL_DECODE_ONLY (graphs that also
    cover prefill batches would skip the prompt positions).  An explicit
    ``enforce_eager=True`` from the user is respected.
    """
    from vllm.config import CompilationConfig
    from vllm.config.compilation import CompilationMode, CUDAGraphMode

    cc = engine_args.compilation_config
    if isinstance(cc, dict):
        cc = CompilationConfig(**cc)
    elif isinstance(cc, int):
        cc = CompilationConfig(mode=CompilationMode(cc))
    elif cc is None:
        cc = CompilationConfig()

    compile_ok = _env_truthy("VLLM_LENS_COMPILE")
    if compile_ok:
        # vllm-metamodels post7: keep vLLM's torch.compile (its default mode when None); the
        # hooks run as an opaque custom op inside the compiled graph (_compile_op).
        logger.info(
            "vllm-lens: VLLM_LENS_COMPILE=1 -- torch.compile stays on (mode %s); layer hooks run as "
            "the custom op vllm_lens::lens_layer_; decode batches replay full CUDA graphs.",
            getattr(cc.mode, "name", cc.mode),
        )
    elif cc.mode not in (None, CompilationMode.NONE):
        logger.warning(
            "vllm-lens: compilation mode %s would compile the decoder layers and "
            "the forward hooks would not fire; forcing enforce_eager=True. For "
            "CUDA graphs use compilation_config mode=0 (NONE) with "
            "cudagraph_mode=FULL_DECODE_ONLY, or VLLM_LENS_COMPILE=1 to keep "
            "torch.compile with the custom-op hooks.",
            cc.mode,
        )
        engine_args.enforce_eager = True
        return False
    else:
        cc.mode = CompilationMode.NONE
    if cc.cudagraph_mode is None:
        cc.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
    elif cc.cudagraph_mode not in (CUDAGraphMode.NONE, CUDAGraphMode.FULL_DECODE_ONLY):
        logger.warning(
            "vllm-lens: cudagraph_mode %s would run prefill batches inside CUDA "
            "graphs where the hooks do not fire (the compile-mode op records no kernels "
            "during capture); overriding to FULL_DECODE_ONLY.",
            cc.cudagraph_mode.name,
        )
        cc.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
    engine_args.compilation_config = cc

    if engine_args.enforce_eager or cc.cudagraph_mode == CUDAGraphMode.NONE:
        return False
    logger.info(
        "vllm-lens: CUDA graphs enabled (compilation mode %s, cudagraph_mode "
        "%s). Hooks run only in forward passes that contain prompt tokens: "
        "steering and activation capture are prompt-position only.",
        getattr(cc.mode, "name", cc.mode),
        cc.cudagraph_mode.name,
    )
    return True


def _patch_worker_load_model() -> None:
    """vllm-metamodels post7: install the hooks right after the weights are loaded when the
    engine compiles the model (torch.compile traces the hooks present at that moment and vLLM
    drops all guards afterwards).  Runs in every worker process (plugins load in
    ``WorkerWrapperBase.init_worker`` before the worker is constructed)."""
    try:
        from vllm.v1.worker.gpu_worker import Worker
    except Exception:  # pragma: no cover - no GPU worker in this process
        return
    if getattr(Worker.load_model, "_vllm_lens_wrapped", False):
        return
    orig = Worker.load_model

    def load_model(self, *args, **kwargs):
        orig(self, *args, **kwargs)
        try:
            from vllm_lens._compile_op import model_is_compiled

            compiled = model_is_compiled(getattr(self, "vllm_config", None))
        except Exception:  # noqa: BLE001
            compiled = False
        if compiled and hasattr(self, "install_hooks"):
            self.install_hooks()

    load_model._vllm_lens_wrapped = True  # type: ignore[attr-defined]
    Worker.load_model = load_model


def _patched_create_engine_config(self, *args, **kwargs):
    """Patch for ``EngineArgs.create_engine_config``.

    Injects our worker extension and forces eager mode *before* the
    ``VllmConfig`` is built, so the settings propagate through any
    engine creation path (``AsyncLLM.from_engine_args``,
    ``AsyncLLM.from_vllm_config``, ``vllm serve``, etc.) including
    across subprocess boundaries.  With ``VLLM_LENS_CUDA_GRAPHS=1`` eager
    mode is not forced (see ``_configure_cuda_graphs``).
    """
    global _cuda_graphs_enabled
    if not self.worker_extension_cls:
        self.worker_extension_cls = _WORKER_EXT
    if _env_truthy("VLLM_LENS_CUDA_GRAPHS"):
        _cuda_graphs_enabled = _configure_cuda_graphs(self)
    else:
        self.enforce_eager = True
        _cuda_graphs_enabled = False
    # vLLM >= 0.23 runs dense models on its "V2" GPU model runner by default.  The hooks
    # read the V1 runner's per-step state (``input_batch``, ``requests``, host
    # ``query_start_loc``); on V2 they would find nothing and silently capture / steer
    # nothing.  Default to V1 (an explicit user setting still wins, and is refused below).
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")

    assert _original_create_engine_config is not None
    config = _original_create_engine_config(self, *args, **kwargs)
    if getattr(config, "use_v2_model_runner", False):
        raise RuntimeError(
            "vllm-lens: the vLLM V2 model runner is active (VLLM_USE_V2_MODEL_RUNNER=1) but the "
            "hooks need the V1 runner's per-step state; unset VLLM_USE_V2_MODEL_RUNNER (the plugin "
            "defaults it to 0) or set VLLM_LENS_DISABLE=1 to run plain vLLM."
        )
    return config


def _check_graph_mode_request(
    steering_vectors: list[SteeringVector] | None,
    wants_activations: bool,
    max_tokens: int | None,
) -> None:
    """vllm-lens-metamodel: fail fast / warn for requests whose semantics change under CUDA graphs."""
    global _warned_capture_prompt_only
    if not _cuda_graphs_enabled:
        return
    for sv in steering_vectors or ():
        if sv.activations.dim() == 2:
            raise ValueError(
                "vllm-lens: 2-D (broadcast) steering vectors are not supported "
                "with CUDA graphs (VLLM_LENS_CUDA_GRAPHS=1): generated positions "
                "run inside replayed graphs where the hooks do not execute. Use "
                "3-D position-specific vectors on prompt positions, or run eager."
            )
    if wants_activations and (max_tokens is None or max_tokens > 1):
        if not _warned_capture_prompt_only:
            _warned_capture_prompt_only = True
            logger.warning(
                "vllm-lens: CUDA graphs are enabled, so output_residual_stream "
                "captures PROMPT positions only (generated positions run inside "
                "replayed graphs). Use enforce_eager to capture generated positions."
            )


def _check_layer_support(
    caps: dict[str, Any] | None,
    steering_vectors: Sequence[SteeringVector] | None,
    capture_layers: Any,
) -> None:
    """vllm-metamodels: refuse layer-output steering / capture on hyper-connection
    (multi-stream residual) architectures BEFORE the request reaches the engine,
    with a clear ``ValueError`` (the engine stays alive).  ``caps`` comes from the
    worker's ``lens_capabilities`` RPC; the worker-side validation and the runtime
    ``UnsupportedLayerOutputError`` remain as backstops."""
    if not caps or not caps.get("multi_stream"):
        return
    bad = sorted({int(l) for sv in steering_vectors or () for l in sv.layer_indices if l != EMBED_LAYER_INDEX})
    if bad:
        raise ValueError(
            f"vllm-lens: steering at decoder layer(s) {bad} is unsupported on this hyper-connection "
            "(multi-stream residual) architecture -- the residual stream at a layer boundary is a deferred "
            "fold of several tensors. Use layer_indices=[EMBED_LAYER_INDEX] (embedding-stream injection)."
        )
    if capture_layers is True or (
        isinstance(capture_layers, (list, tuple)) and any(int(l) != EMBED_LAYER_INDEX for l in capture_layers)
    ):
        raise ValueError(
            "vllm-lens: output_residual_stream at decoder-layer outputs is unsupported on this hyper-connection "
            "(multi-stream residual) architecture; only output_residual_stream=[EMBED_LAYER_INDEX] (the "
            "embedding stream) is defined here."
        )


def _pack_steering(
    payloads: dict[str, list[SteeringVector]],
) -> tuple[dict[str, Any] | None, dict[str, list[SteeringVector]]]:
    """vllm-lens-metamodel: split an offline call's steering into one block + the rest.

    A request is block-packable when it has exactly one SteeringVector with
    ``(1, 1, hidden)`` activations (one layer, one position) -- the
    per-request "steer this prompt's marker token" pattern, including
    ``mode="replace"`` and ``EMBED_LAYER_INDEX`` (the block carries per-entry
    layer / position / scale / norm_match / mode).  Those are stacked into
    one ``[n, hidden]`` CPU tensor for ``set_steering_block``; everything
    else goes through ``set_steering_data_many``.
    """
    keys: list[str] = []
    vecs: list[torch.Tensor] = []
    layers: list[int] = []
    positions: list[int] = []
    scales: list[float] = []
    nms: list[bool] = []
    modes: list[str] = []
    rest: dict[str, list[SteeringVector]] = {}
    for key, vectors in payloads.items():
        sv = vectors[0] if len(vectors) == 1 else None
        act = sv.activations if sv is not None else None
        if (
            act is not None
            and act.dim() == 3
            and act.shape[0] == 1
            and act.shape[1] == 1
            and (sv.position_indices is None or len(sv.position_indices) >= 1)
        ):
            keys.append(key)
            vecs.append(act[0, 0].detach().cpu())
            layers.append(int(sv.layer_indices[0]))
            positions.append(
                int(sv.position_indices[0]) if sv.position_indices is not None else 0
            )
            scales.append(float(sv.scale))
            nms.append(bool(sv.norm_match))
            modes.append(str(sv.mode))
        else:
            rest[key] = vectors
    if not keys:
        return None, rest
    block = {
        "keys": keys,
        "vecs": torch.stack(vecs).contiguous(),
        "layers": layers,
        "positions": positions,
        "scales": scales,
        "norm_match": nms,
        "modes": modes,
    }
    return block, rest


# ---------------------------------------------------------------------------
# Generate patch — install hooks and attach activations to output
# ---------------------------------------------------------------------------


async def _patched_generate(
    self: AsyncLLM,
    prompt: str,
    sampling_params: SamplingParams,
    request_id: str,
    **kwargs,
) -> AsyncIterator:
    """Wrap generate to install hooks, apply steering, and attach activations.

    On the first call that requests activations or steering, sends a
    one-time RPC to install forward hooks on every decoder layer.

    If ``apply_steering_vectors`` is present in ``extra_args``, the
    steering data is sent to workers via RPC *before* generation starts
    (tensors can't survive msgspec serialization in extra_args).

    When generation finishes, retrieves the captured activations from
    the worker and attaches them as ``output.activations``.
    """
    # In vLLM v1, the chat completion endpoint creates an
    # EngineCoreRequest with a *cloned* SamplingParams before calling
    # generate(). add_request() uses the clone from the
    # EngineCoreRequest, ignoring the separately-passed sampling_params.
    # We must read/modify the clone so our changes take effect.
    effective_params = sampling_params
    try:
        from vllm.v1.engine import EngineCoreRequest

        if isinstance(prompt, EngineCoreRequest) and prompt.sampling_params is not None:  # type: ignore[reportAttributeAccessIssue]
            effective_params = prompt.sampling_params  # type: ignore[reportAttributeAccessIssue]
    except ImportError:
        pass

    extra = effective_params.extra_args or {}
    wants_activations = extra.get("output_residual_stream") is not None
    # Extract steering data and remove from extra_args before vLLM
    # serialises the SamplingParams (tensors don't survive msgspec).
    steering_vectors = extra.pop("apply_steering_vectors", None)
    # When arriving via the OpenAI API (vllm_xargs), complex values
    # are JSON-encoded strings; decode and validate as SteeringVector.
    if isinstance(steering_vectors, str):
        steering_vectors = [
            SteeringVector.model_validate(d) for d in json.loads(steering_vectors)
        ]
    readouts = _coerce_readouts(extra.pop("apply_readout_vectors", None))
    if isinstance(extra.get(CAPTURE_POSITIONS_KEY), str) and extra[CAPTURE_POSITIONS_KEY].startswith(("{", "[")):
        extra[CAPTURE_POSITIONS_KEY] = json.loads(extra[CAPTURE_POSITIONS_KEY])  # vllm_xargs JSON string
    max_tokens = getattr(effective_params, "max_tokens", None)
    _check_graph_mode_request(steering_vectors, wants_activations, max_tokens)

    # Allow explicit prefix-cache bypass via extra_args.
    skip_kv_cache = extra.pop("skip_reading_prefix_cache", None)
    cache_salt = extra.pop(CACHE_SALT_KEY, None)  # vllm-metamodels: user salt mode, never reaches the engine

    wants_readout = bool(readouts)
    needs_hooks = wants_activations or steering_vectors is not None or wants_readout
    if needs_hooks or skip_kv_cache:
        # vllm-metamodels: with prefix caching + the salt patch, steering-only requests keep
        # reading the cache (steered blocks salted); capture / readout / early exit as before.
        _apply_kv_policy(
            effective_params, steering_vectors, wants_activations, readouts, extra.get(EARLY_EXIT_KEY),
            cache_salt, f"{request_id}", await _prefix_cache_state_async(self), bool(skip_kv_cache),
        )
    if needs_hooks and not getattr(self, "_hooks_installed", False):
        await self.collective_rpc("install_hooks")
        setattr(self, "_hooks_installed", True)
    if needs_hooks:
        caps = await _lens_capabilities_async(self)
        _check_layer_support(caps, steering_vectors, extra.get("output_residual_stream"))
        _check_readout_request(
            caps, readouts, extra.get(EARLY_EXIT_KEY), max_tokens, extra.get("output_residual_stream")
        )

    # Send steering / readout data to workers before the forward pass begins.
    if steering_vectors is not None:
        await self.collective_rpc(
            "set_steering_data",
            args=(request_id, pickle.dumps(steering_vectors)),
        )
    if wants_readout:
        await self.collective_rpc("set_readout_data", args=(request_id, pickle.dumps(readouts)))

    assert _original_generate is not None
    try:
        async for output in _original_generate(
            self, prompt, sampling_params, request_id, **kwargs
        ):
            if output.finished and (wants_activations or wants_readout):
                n_prompt = len(output.prompt_token_ids)
                n_gen = len(output.outputs[0].token_ids)
            if output.finished and wants_activations:
                states = await self.collective_rpc(
                    "get_captured_states", args=(request_id,)
                )
                activations = _merge_captured_states(states)
                if activations is not None:
                    _trim_activations(activations, n_prompt + n_gen - 1)
                    output.activations = activations
            if output.finished and wants_readout:
                res = await self.collective_rpc("get_readouts", args=(request_id,))
                blob = next((r for r in res if r is not None), None) if res else None
                if blob is not None:
                    readout = pickle.loads(blob)
                    _trim_readout(readout, n_prompt + n_gen - 1)
                    output.readout = readout
            yield output
    finally:
        if steering_vectors is not None:
            await self.collective_rpc("clear_steering_data", args=(request_id,))
        if wants_readout:
            await self.collective_rpc("clear_readout_data", args=(request_id,))
            await self.collective_rpc("clear_readouts", args=(request_id,))
        if wants_activations:
            await self.collective_rpc("clear_captured_states", args=(request_id,))


def _own_params(sp: Any) -> Any:
    """Shallow copy of a SamplingParams with its own ``extra_args`` dict (see _patched_llm_generate)."""
    try:
        sp2 = copy.copy(sp)
        sp2.extra_args = dict(sp.extra_args) if sp.extra_args else sp.extra_args
        return sp2
    except Exception:  # noqa: BLE001 - exotic params object: fall back to in-place (1.1.0 behaviour)
        return sp


def _lens_capabilities_sync(llm: Any) -> dict[str, Any]:
    """Worker capabilities merged with the prefix-cache facts (``prefix_caching``,
    ``kv_salt_active``; ``early_exit`` adjusted), cached per engine.  Installs the hooks
    first (several capabilities -- early exit, compile mode -- are only known afterwards)."""
    caps = getattr(llm, "_lens_caps", None)
    if caps is None:
        try:
            if not getattr(llm, "_hooks_installed", False):
                llm.collective_rpc("install_hooks")
                llm._hooks_installed = True
            caps = llm.collective_rpc("lens_capabilities")[0] or {}
        except Exception:  # noqa: BLE001 - older worker without the RPC
            caps = {}
        pc_enabled, salt_active = _prefix_cache_state_sync(llm)
        caps = _effective_caps(caps, pc_enabled, salt_active)
        llm._lens_caps = caps
    return caps


async def _lens_capabilities_async(llm: Any) -> dict[str, Any]:
    caps = getattr(llm, "_lens_caps", None)
    if caps is None:
        try:
            if not getattr(llm, "_hooks_installed", False):
                await llm.collective_rpc("install_hooks")
                setattr(llm, "_hooks_installed", True)
            caps = (await llm.collective_rpc("lens_capabilities"))[0] or {}
        except Exception:  # noqa: BLE001
            caps = {}
        pc_enabled, salt_active = await _prefix_cache_state_async(llm)
        caps = _effective_caps(caps, pc_enabled, salt_active)
        llm._lens_caps = caps
    return caps


# ---------------------------------------------------------------------------
# vllm-metamodels post7: shared-memory transport helpers (see _shm)
# ---------------------------------------------------------------------------


def _shm_supported(llm: Any) -> bool:
    caps = _lens_capabilities_sync(llm)
    return bool(caps.get("shm", False))


def _ship_block(block: dict[str, Any], tag: str) -> dict[str, Any]:
    """Replace ``block["vecs"]`` by a shared-memory descriptor when VLLM_LENS_SHM is on."""
    if not _shm.shm_mode():
        return block
    try:
        desc = _shm.put({"vecs": block["vecs"]}, tag=tag)
    except Exception:  # noqa: BLE001 - no /dev/shm: pickle the tensor as before
        logger.warning("vllm-lens: shared-memory block transport failed; pickling the block", exc_info=True)
        return block
    return {**block, "vecs": None, "shm": desc}


def _unpack_shm_capture(blob: dict[str, Any], mode: str, outputs: Sequence[Any]) -> dict[str, Any]:
    """Descriptor from ``get_captured_states_shm`` -> ``{request_id: activations}``.  ``mode`` ==
    ``"view"`` keeps zero-copy views (the mapping handle is attached to every output as
    ``lens_shm``); ``"copy"`` copies out and releases the mapping."""
    if "pickled" in blob:
        return blob["pickled"]
    desc, positions = blob.get("shm"), blob.get("positions", {})
    if desc is None:
        return {}
    tensors, handle = _shm.get(desc, copy=(mode != "view"))
    if handle is not None:
        for o in outputs:  # keep the mapping alive as long as any output (and its views) lives
            o.lens_shm = handle
    out: dict[str, Any] = {}
    for ext, t in tensors.items():
        acts: dict[str, Any] = {"residual_stream": t}
        if ext in positions:
            acts["positions"] = positions[ext]
        out[ext] = acts
    return out


# ---------------------------------------------------------------------------
# Offline (sync) LLM.generate patch
# ---------------------------------------------------------------------------


def _patched_llm_generate(
    self: LLM,
    prompts: Any,
    sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
    **kwargs,
) -> list:
    """Wrap ``LLM.generate`` to install hooks, apply steering, and attach activations.

    Same logic as the async variant but for the synchronous offline API.
    Because ``LLM.generate`` auto-assigns request IDs internally, steering
    data is keyed by a synthetic ``_steering_id`` stored in ``extra_args``
    (a lightweight string that survives msgspec serialization).
    """
    # vllm-metamodels post7: never mutate the caller's SamplingParams.  1.1.0 popped
    # ``apply_steering_vectors`` out of the caller's ``extra_args`` in place, so a params object
    # reused for a second ``generate()`` was silently UNSTEERED.  Work on shallow copies (the
    # tensors are shared, only the Struct + its extra_args dict are copied: ~0.3 us each).
    if isinstance(sampling_params, Sequence):
        params_list = [_own_params(sp) for sp in sampling_params]
        sampling_params = params_list
    elif sampling_params is not None:
        params_list = [_own_params(sampling_params)]
        sampling_params = params_list[0]
    else:
        params_list = []

    wants_activations = any(
        (sp.extra_args or {}).get("output_residual_stream") is not None
        for sp in params_list
    )

    # Extract steering vectors per-request.  We must pop them from
    # extra_args before vLLM serialises SamplingParams (tensors don't
    # survive msgspec), but keep them for the RPC call.
    steering_payloads: dict[str, list[SteeringVector]] = {}  # steering_id -> vectors
    readout_payloads: dict[str, list[ReadoutVector]] = {}  # readout_id -> vectors
    per_request: list[tuple[list[SteeringVector] | None, Any, list[ReadoutVector] | None, Any, int | None]] = []
    for idx, sp in enumerate(params_list):
        extra = sp.extra_args or {}
        vectors = extra.pop("apply_steering_vectors", None)
        if isinstance(vectors, str):
            vectors = [SteeringVector.model_validate(d) for d in json.loads(vectors)]
        readouts = _coerce_readouts(extra.pop("apply_readout_vectors", None))
        per_request.append(
            (vectors, extra.get("output_residual_stream"), readouts, extra.get(EARLY_EXIT_KEY), sp.max_tokens)
        )
        if vectors is not None:
            _check_graph_mode_request(vectors, False, None)
            steering_id = f"_steer_{idx}"
            steering_payloads[steering_id] = vectors
            if sp.extra_args is None:
                sp.extra_args = {}
            sp.extra_args["_steering_id"] = steering_id
        if readouts:
            readout_id = f"_read_{idx}"
            readout_payloads[readout_id] = readouts
            if sp.extra_args is None:
                sp.extra_args = {}
            sp.extra_args["_readout_id"] = readout_id
    if wants_activations:
        _check_graph_mode_request(
            None,
            True,
            max((sp.max_tokens or 0) for sp in params_list) if params_list else None,
        )

    # Pop skip_reading_prefix_cache / lens_cache_salt from extra_args for each request.
    any_skip_kv_cache = False
    cache_salts: list[Any] = []
    for sp in params_list:
        extra = sp.extra_args or {}
        if extra.pop("skip_reading_prefix_cache", None):
            any_skip_kv_cache = True
        cache_salts.append(extra.pop(CACHE_SALT_KEY, None))

    has_steering = len(steering_payloads) > 0
    wants_readout = len(readout_payloads) > 0
    needs_hooks = wants_activations or has_steering or wants_readout
    if needs_hooks or any_skip_kv_cache:
        # vllm-metamodels: per-request prefix-cache policy (see _apply_kv_policy / _kv_salt).
        pc_state = _prefix_cache_state_sync(self)
        call_nonce = uuid.uuid4().hex[:12]
        for idx, (sp, (vectors, cap, readouts, early_exit, _mt)) in enumerate(zip(params_list, per_request)):
            _apply_kv_policy(
                sp, vectors, cap is not None, readouts, early_exit, cache_salts[idx],
                f"{call_nonce}-{idx}", pc_state, any_skip_kv_cache,
            )

    if needs_hooks and not getattr(self, "_hooks_installed", False):
        self.collective_rpc("install_hooks")
        self._hooks_installed = True  # type: ignore[reportAttributeAccessIssue]
    if needs_hooks:
        caps = _lens_capabilities_sync(self)
        for vectors, cap, readouts, early_exit, max_tokens in per_request:
            if vectors is not None or cap is not None:
                _check_layer_support(caps, vectors, cap)
            if readouts or early_exit:
                _check_readout_request(caps, readouts, early_exit, max_tokens, cap)

    # Send steering data to workers before generation: one block RPC for the
    # single-position per-request vectors, one "many" RPC for the rest
    # (vllm-lens-metamodel; 1.1.0 did one RPC per request).
    if has_steering:
        block, rest = (
            _pack_steering(steering_payloads)
            if _env_truthy("VLLM_LENS_BLOCK_RPC", "1")
            else (None, steering_payloads)
        )
        if block is not None:
            self.collective_rpc("set_steering_block", args=(pickle.dumps(_ship_block(block, "steer")),))
        if rest:
            self.collective_rpc("set_steering_data_many", args=(pickle.dumps(rest),))
    if wants_readout:
        rblock, rrest = _pack_readouts(readout_payloads)
        if rblock is not None:
            self.collective_rpc("set_readout_block", args=(pickle.dumps(_ship_block(rblock, "read")),))
        if rrest:
            self.collective_rpc("set_readout_data_many", args=(pickle.dumps(rrest),))

    assert _original_llm_generate is not None
    try:
        outputs = _original_llm_generate(self, prompts, sampling_params, **kwargs)
    finally:
        # Clean up steering / readout data (also on error, so keys never leak).
        if has_steering:
            self.collective_rpc(
                "clear_steering_data_many", args=(list(steering_payloads),)
            )
        if wants_readout:
            self.collective_rpc("clear_readout_data_many", args=(list(readout_payloads),))

    fast = _env_truthy("VLLM_LENS_FAST_CAPTURE", "1")
    if wants_activations and fast:
        # vllm-metamodels: ONE RPC for every request of this call (per-PP-rank blobs); with
        # VLLM_LENS_SHM the blobs are shared-memory descriptors (post7).
        mode = _shm.shm_mode()
        if mode and _shm_supported(self):
            blobs = self.collective_rpc("get_captured_states_shm", args=([o.request_id for o in outputs],))
            parts = [_unpack_shm_capture(pickle.loads(b), mode, outputs) for b in blobs if b is not None]
        else:
            blobs = self.collective_rpc("get_captured_states_many", args=([o.request_id for o in outputs],))
            parts = [pickle.loads(b) for b in blobs if b is not None]
        for output in outputs:
            found = [p[output.request_id] for p in parts if output.request_id in p]
            if not found:
                continue
            activations = found[0]
            if len(found) > 1:  # PP: lower ranks hold earlier layers
                activations = {"residual_stream": torch.cat([f["residual_stream"] for f in found], dim=0)}
                if "positions" in found[0]:
                    activations["positions"] = found[0]["positions"]
            n_prompt = len(output.prompt_token_ids)
            n_gen = len(output.outputs[0].token_ids)
            _trim_activations(activations, n_prompt + n_gen - 1)
            output.activations = activations
    elif wants_activations:
        for output in outputs:
            req_id = output.request_id
            states = self.collective_rpc("get_captured_states", args=(req_id,))
            activations = _merge_captured_states(states)
            if activations is not None:
                n_prompt = len(output.prompt_token_ids)
                n_gen = len(output.outputs[0].token_ids)
                _trim_activations(activations, n_prompt + n_gen - 1)
                output.activations = activations
    if wants_readout:
        blobs = self.collective_rpc("get_readouts_many", args=([o.request_id for o in outputs],))
        parts = [pickle.loads(b) for b in blobs if b is not None]
        for output in outputs:
            found = [p[output.request_id] for p in parts if output.request_id in p]
            if not found:
                continue
            readout = found[0]
            n_prompt = len(output.prompt_token_ids)
            n_gen = len(output.outputs[0].token_ids)
            _trim_readout(readout, n_prompt + n_gen - 1)
            output.readout = readout

    return outputs


# ---------------------------------------------------------------------------
# Response builder patches for vllm serve (OpenAI-compatible API)
# ---------------------------------------------------------------------------


def _patched_completion_response(self, final_res_batch, *args, **kwargs):
    """Wrap the completion response builder to inject serialized activations."""
    assert _original_completion_response is not None
    response = _original_completion_response(self, final_res_batch, *args, **kwargs)
    for res in final_res_batch or ():
        activations = getattr(res, "activations", None)
        if activations is not None:
            response.activations = serialize_activations(activations)
            break
    for res in final_res_batch or ():
        readout = getattr(res, "readout", None)
        if readout is not None:
            try:
                response.readout = [{**r, "values": serialize_tensor(r["values"])} for r in readout]
            except Exception:  # noqa: BLE001 - response model without extra fields
                logger.warning("vllm-lens: could not attach readout to the completion response", exc_info=True)
            break
    return response


async def _patched_chat_full_generator(
    self, request, result_generator, *args, **kwargs
):
    """Wrap the chat completion full generator to inject serialized activations.

    The original method iterates ``result_generator`` internally, so we
    wrap it with a capturing async generator to grab the final
    ``RequestOutput`` (which has ``.activations`` attached by
    ``_patched_generate``).
    """
    assert _original_chat_full_generator is not None

    last_output = None

    async def _capturing(gen: AsyncIterator) -> AsyncIterator:
        nonlocal last_output
        async for output in gen:
            last_output = output
            yield output

    response = await _original_chat_full_generator(
        self, request, _capturing(result_generator), *args, **kwargs
    )

    # Only inject for successful responses (not ErrorResponse).
    if last_output is not None and hasattr(response, "model_dump"):
        activations = getattr(last_output, "activations", None)
        if activations is not None:
            response.activations = serialize_activations(activations)
        readout = getattr(last_output, "readout", None)
        if readout is not None:
            try:
                response.readout = [{**r, "values": serialize_tensor(r["values"])} for r in readout]
            except Exception:  # noqa: BLE001
                logger.warning("vllm-lens: could not attach readout to the chat response", exc_info=True)

    return response


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


def register() -> None:
    """Entry point called by vLLM's plugin system at engine startup.

    Patches ``EngineArgs.create_engine_config`` to inject the worker
    extension and eager mode, ``AsyncLLM.generate`` and ``LLM.generate``
    to retrieve per-request activations for both online and offline
    usage.  Also patches the OpenAI-compatible response builders so
    activations are included in HTTP responses from ``vllm serve``.

    Use ``extra_args={"output_residual_stream": True | list[int]}`` in
    SamplingParams to request activations.

    Set ``VLLM_LENS_DISABLE=1`` to make this a no-op.
    """
    global _original_create_engine_config
    global _original_generate, _original_llm_generate
    global _original_completion_response, _original_chat_full_generator

    if _env_truthy("VLLM_LENS_DISABLE"):
        logger.info("VLLM_LENS_DISABLE set; vllm-lens activation plugin inactive.")
        return
    # vLLM >= 0.27 refuses pickled collective_rpc payloads unless opted in; the
    # steering RPCs ship pickled SteeringVector tensors (trusted, same-user).
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    # vllm-metamodels: salt the block hashes of steered / early-exit requests (runs in the
    # scheduler process too -- vLLM loads general plugins in EngineCore.__init__).
    _kv_salt.install()
    # vllm-metamodels: compile-mode hooks must exist before the first (compiling) forward pass.
    _patch_worker_load_model()

    from vllm import LLM
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    _original_create_engine_config = EngineArgs.create_engine_config
    EngineArgs.create_engine_config = _patched_create_engine_config

    _original_generate = AsyncLLM.generate
    AsyncLLM.generate = _patched_generate  # type: ignore[reportAttributeAccessIssue]

    _original_llm_generate = LLM.generate
    LLM.generate = _patched_llm_generate

    # Patch OpenAI-compatible response builders so activations survive
    # HTTP serialization.  Wrapped in try/except because these modules
    # are only available when running as an API server.
    try:
        from vllm.entrypoints.openai.completion.serving import (
            OpenAIServingCompletion,
        )

        _original_completion_response = (
            OpenAIServingCompletion.request_output_to_completion_response
        )
        OpenAIServingCompletion.request_output_to_completion_response = (
            _patched_completion_response
        )
    except Exception:
        pass

    try:
        from vllm.entrypoints.openai.chat_completion.serving import (
            OpenAIServingChat,
        )

        _original_chat_full_generator = OpenAIServingChat.chat_completion_full_generator
        OpenAIServingChat.chat_completion_full_generator = _patched_chat_full_generator
    except Exception:
        pass
