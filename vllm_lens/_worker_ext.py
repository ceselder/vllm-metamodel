"""
Worker extension that captures residual-stream activations from
configurable layers during transformer forward passes, and optionally
applies steering vectors (activation additions) to modify the residual
stream in-flight.

Uses PyTorch forward hooks on each decoder layer for concurrency-safe,
per-request activation capture and steering.  Each hook checks the
request's ``extra_args["output_residual_stream"]`` to decide whether to
capture, and reads from ``_steering_data`` to apply any steering vectors.

vllm-lens-metamodel: the hook resolves each request's steering configs ONCE
(indexed lookups instead of a per-layer ``startswith`` scan over every
key), plans each forward pass once from host-side buffers (no device
syncs), skips passes / layers with nothing to do, applies a layer's
vectors with one ``index_add_``, and can run with CUDA graphs for decode
(``VLLM_LENS_CUDA_GRAPHS=1``, prompt-position steering only).  The
steering arithmetic is upstream's, with one deliberate change ported from
upstream 1.2.0 (#7): on fused-residual layers ``norm_match`` scales to the
norm of the FULL residual stream ``hidden_states + residual`` (see
``_apply_steering``).  ``mode="replace"`` and ``EMBED_LAYER_INDEX`` (the
embedding stream entering layer 0, applied in the layer-0 pre-hook) are
fork additions.

Fast hidden-state readout (vllm-metamodels 1.1.0.post4):

* capture gathers every capturing row's requested positions of a layer with
  ONE ``index_select`` and ONE pinned, asynchronous device->host copy per
  layer-step (``_capture_gather``; the 1.1.0 path did a blocking ``.cpu()``
  per request), honours ``extra_args["capture_positions"]`` (``"all"``,
  ``{"last": k}``, explicit list) and returns all requests of a
  ``generate()`` call in one RPC (``get_captured_states_many``);
* ``ReadoutVector`` (``apply_readout_vectors``): the hook computes
  cosine / dot products of the selected residual-stream rows with a
  per-request direction *in the worker* and returns float32 scalars only
  (``_readout_layer``, ``get_readouts_many``);
* early exit (``extra_args["lens_early_exit"]``): when every request of a
  forward pass is a ``max_tokens=1`` capture / readout request, the hook of
  the deepest requested layer raises ``_EarlyExit`` and the wrapped
  ``model_runner._model_forward`` returns a zero placeholder instead of
  running the remaining layers.
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import zstandard as zstd
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.models.utils import PPMissingLayer

from vllm_lens._helpers.types import (
    CAPTURE_POSITIONS_KEY,
    EARLY_EXIT_KEY,
    ReadoutVector,
    SteeringVector,
    normalize_positions,
)

if TYPE_CHECKING:
    from jaxtyping import Float
    from vllm.config import ParallelConfig

logger = logging.getLogger(__name__)

_ZSTD_COMPRESSOR = zstd.ZstdCompressor(level=1)


def _get_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    """Find the transformer decoder layers regardless of model architecture."""
    # Module.__getattr__ returns Tensor | Module, so pyright can't narrow
    # through chained attribute access.  Use Any for duck-typed traversal.
    m: Any = model
    if hasattr(m, "language_model") and hasattr(m.language_model, "model"):
        return m.language_model.model.layers
    if (
        hasattr(m, "model")
        and hasattr(m.model, "decoder")
        and hasattr(m.model.decoder, "layers")
    ):
        return m.model.decoder.layers
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        return m.model.layers
    raise AttributeError(
        f"Cannot find decoder layers on {type(model).__name__}. "
        "Expected model.language_model.model.layers, "
        "model.model.decoder.layers, or model.model.layers"
    )


def _find_steering_configs(
    extension: HiddenStatesExtension,
    internal_req_id: str,
    extra_args: dict[str, Any] | None,
) -> list[SteeringVector]:
    """Find all steering configs that apply to an internal request ID.

    Matches by ``"{external_id}-"`` prefix (async path: vLLM appends
    ``"-{random_suffix}"`` to external IDs) and by ``_steering_id``
    sentinel in ``extra_args`` (offline path).
    """
    results: list[SteeringVector] = []
    for external_id, configs in extension._steering_data.items():
        if internal_req_id.startswith(f"{external_id}-"):
            results.extend(configs)
    # Offline path stores a lightweight string key in extra_args
    if extra_args:
        steering_id = extra_args.get("_steering_id")
        if steering_id and steering_id in extension._steering_data:
            results.extend(extension._steering_data[steering_id])
    return results


# ---------------------------------------------------------------------------
# vllm-lens-metamodel: indexed per-request steering
#
# 1.1.0 called ``_find_steering_configs`` for every layer x every request on
# every forward pass (O(layers x requests x keys) ``startswith`` calls, plus
# two ``.item()`` device syncs per request per layer).  The code below keeps
# its exact matching semantics but resolves each request once, plans each
# forward pass once from host-side buffers, and lets idle passes / layers
# return immediately.
# ---------------------------------------------------------------------------

# Prune the per-request resolution cache when it grows past this many
# entries beyond the live batch (entries for finished requests).
_REQ_CACHE_SLACK = 4096
# Vectorised apply falls back to the sequential loop above this many
# (row, vector) pairs per steered row (broadcast vectors over long chunks).
_VEC_MAX_ENTRIES_PER_ROW = 4
_NO_POS = 1 << 62
_TRUTHY = ("1", "true", "yes", "on")

from vllm_lens._helpers.types import EMBED_LAYER_INDEX  # noqa: E402  (sentinel)


@dataclass(slots=True)
class _SteerEntry:
    """One ``set_steering_data`` key, summarised for O(1) hook-time decisions."""

    configs: list[SteeringVector]
    layers: frozenset[int]
    """Every layer index any config touches."""
    broadcast: bool
    """True if any config is 2-D (applies to every position, generated ones too)."""
    min_pos: int
    """Lowest absolute position any 3-D config can touch (``_NO_POS`` if none)."""
    max_pos: int
    """Highest absolute position any 3-D config can touch (-1 if none)."""
    seq: int
    """Insertion order, so multi-key matches keep the 1.1.0 (dict-order) ordering."""
    replace_layers: frozenset[int] = frozenset()
    """Layers at which some config uses ``mode="replace"`` (needs the residual
    half cloned on fused-residual layers)."""


def _index_configs(configs: list[SteeringVector], seq: int) -> _SteerEntry:
    layers: set[int] = set()
    replace_layers: set[int] = set()
    broadcast = False
    min_pos, max_pos = _NO_POS, -1
    for cfg in configs:
        layers.update(cfg.layer_indices)
        if cfg.mode == "replace":
            replace_layers.update(cfg.layer_indices)
        if cfg.activations.dim() == 2:
            broadcast = True
            continue
        n_pos = int(cfg.activations.shape[1])
        # _apply_steering stops at pi >= n_pos, so only the first n_pos count.
        pos = (
            list(cfg.position_indices[:n_pos])
            if cfg.position_indices is not None
            else list(range(n_pos))
        )
        if pos:
            min_pos, max_pos = min(min_pos, min(pos)), max(max_pos, max(pos))
    return _SteerEntry(
        configs,
        frozenset(layers),
        broadcast,
        min_pos,
        max_pos,
        seq,
        frozenset(replace_layers),
    )


def _prefix_keys(internal_req_id: str) -> Iterator[str]:
    """Yield every ``k`` such that ``internal_req_id.startswith(k + "-")``.

    vLLM turns an external request id into ``"{external_id}-{8 hex}"``, so
    the ``startswith`` scan over all keys matched exactly the keys that end
    right before a ``"-"`` in the internal id.  Enumerating those prefixes
    (a handful of dict lookups) is equivalent and independent of the number
    of registered keys.
    """
    i = internal_req_id.find("-")
    while i != -1:
        yield internal_req_id[:i]
        i = internal_req_id.find("-", i + 1)


def _resolve_entries(
    extension: HiddenStatesExtension,
    internal_req_id: str,
    extra_args: dict[str, Any] | None,
) -> list[_SteerEntry]:
    """Indexed equivalent of ``_find_steering_configs``: prefix matches first
    (in insertion order, like iterating the dict), then the ``_steering_id``
    sentinel match (offline path)."""
    index = extension._steering_index
    if not index:
        return []
    found: list[_SteerEntry] = []
    for key in _prefix_keys(internal_req_id):
        entry = index.get(key)
        if entry is not None:
            found.append(entry)
    if len(found) > 1:
        found.sort(key=lambda e: e.seq)
    if extra_args:
        steering_id = extra_args.get("_steering_id")
        if steering_id:
            entry = index.get(steering_id)
            if entry is not None:
                found.append(entry)
    return found


@dataclass(slots=True)
class _ReqPlan:
    """What one request wants from the hooks, resolved once and cached."""

    gen: int
    """``_steering_gen`` when resolved; a mismatch means steering data changed."""
    configs: list[SteeringVector]
    """All matching steering configs, flattened, in 1.1.0 order."""
    layers: frozenset[int]
    broadcast: bool
    min_pos: int
    max_pos: int
    cap_any: bool
    """``output_residual_stream`` present in extra_args."""
    cap_set: frozenset[int] | None
    """Layer set when ``output_residual_stream`` is a list, else None (= all layers)."""
    num_prompt: int
    """Prompt length; rows at/after it are generated positions."""
    replace_layers: frozenset[int] = frozenset()
    cap_pos: tuple[str, Any] = ("all", None)
    """Normalised ``capture_positions`` spec (fast capture path only)."""
    reads: tuple[_ReadEntry, ...] = ()
    """Readout vectors registered for this request (in order)."""
    early_exit: bool = False
    """Request opted into early exit and is eligible (max_tokens == 1, finite layer set)."""
    exit_layer: int = -1
    """Deepest layer this request needs (capture or readout); -1 if not eligible."""


@dataclass(slots=True)
class _ReadEntry:
    """One ``ReadoutVector`` as stored on the worker: its direction rows live in a
    float32 ``[n, hidden]`` device block shared by the RPC that registered it."""

    key: str
    seq: int
    """Index of this vector among the request's readout vectors (result order)."""
    block: torch.Tensor
    layer_rows: dict[int, int]
    """layer -> row of ``block`` holding that layer's direction."""
    spec: tuple[str, Any]
    cos: bool
    bias: float
    layers: frozenset[int] = frozenset()


@dataclass(slots=True)
class _HostBlock:
    """One layer-step's device->host copy: ``host`` (pinned) holds the rows of
    several requests back to back; ``segments`` = ``(internal_req_id, n_rows,
    abs_positions[, readout seq])`` in that order; ``event`` completes when the
    copy has landed (None on CPU)."""

    host: torch.Tensor
    event: Any
    layer: int
    segments: list[tuple]


class _EarlyExit(Exception):
    """Raised by the layer hook to stop the forward pass after the deepest
    requested layer; caught by the wrapped ``model_runner._model_forward``,
    which returns ``placeholder`` (zeros, ``[tokens, hidden]``) as the model
    output so logits / sampling still run (on garbage the caller ignores)."""

    def __init__(self, placeholder: torch.Tensor) -> None:
        super().__init__("vllm-lens early exit")
        self.placeholder = placeholder


def _select_positions(
    spec: tuple[str, Any], start: int, end: int, a0: int, num_prompt: int
) -> tuple[np.ndarray, np.ndarray]:
    """Rows of this pass's ``[tokens, hidden]`` tensor that ``spec`` selects for a
    request whose chunk occupies flat rows ``[start, end)`` = absolute positions
    ``[a0, a0 + n)``.  Returns ``(flat_row_indices, absolute_positions)``, both
    ascending.  ``"last": k`` = prompt positions ``>= num_prompt - k`` plus every
    generated position; an explicit list is absolute (negative = from the end
    of the prompt)."""
    n = end - start
    kind, arg = spec
    if kind == "all":
        return np.arange(start, end, dtype=np.int64), np.arange(a0, a0 + n, dtype=np.int64)
    if kind == "last":
        first = max(a0, num_prompt - int(arg), 0)
        if first >= a0 + n:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        return (
            np.arange(first - a0 + start, end, dtype=np.int64),
            np.arange(first, a0 + n, dtype=np.int64),
        )
    keep = sorted({(p if p >= 0 else num_prompt + p) for p in arg})
    keep = [p for p in keep if a0 <= p < a0 + n]
    return (
        np.array([p - a0 + start for p in keep], dtype=np.int64),
        np.array(keep, dtype=np.int64),
    )


def _to_host(t: torch.Tensor) -> tuple[torch.Tensor, Any]:
    """One asynchronous device->host copy into pinned memory (PyTorch's caching
    host allocator makes repeated allocations of the same size cheap); the
    returned CUDA event completes when the data has landed.  CPU tensors are
    cloned; a failed pin falls back to a blocking copy."""
    if t.device.type != "cuda":
        return t.clone(), None
    try:
        host = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
    except RuntimeError:  # pragma: no cover - no pinned memory available
        return t.cpu(), None
    host.copy_(t, non_blocking=True)
    ev = torch.cuda.Event()
    ev.record()
    return host, ev


@dataclass(slots=True)
class _StepPlan:
    """Everything the layer hooks need for ONE forward pass (built lazily by
    the first hook that runs in the pass, dropped by the first layer's
    pre-hook)."""

    qsl: list[int]
    """``query_start_loc`` as host ints."""
    abs_start: list[int]
    """Absolute position of each row's first token (== ``seq_lens[i] - n_query``)."""
    steer: dict[int, list[tuple[int, list[SteeringVector]]]]
    """layer -> [(row index, that row's configs)] for rows a vector can touch this pass."""
    cap_all: list[int] = field(default_factory=list)
    cap_by_layer: dict[int, list[int]] = field(default_factory=dict)
    ctx_id: int = 0
    replace_layers: set[int] = field(default_factory=set)
    """Layers where some scheduled row uses ``mode="replace"`` this pass."""
    cap_sel: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    """row -> (flat rows, absolute positions) selected for capture this pass (fast path)."""
    read_by_layer: dict[int, list[tuple[int, _ReadEntry, np.ndarray, np.ndarray]]] = field(
        default_factory=dict
    )
    """layer -> [(row, readout entry, flat rows, absolute positions)]."""
    exit_layer: int | None = None
    """Raise ``_EarlyExit`` after this layer's hook (every row of the pass is a
    readout-only request that needs nothing deeper); None = run to the end."""
    cap_concat: dict[tuple[int, ...], tuple[np.ndarray, list[tuple]]] = field(default_factory=dict)
    """Cache: capture row set -> (concatenated flat rows, segments)."""

    def capture_rows(self, layer_idx: int) -> list[int]:
        rows = self.cap_by_layer.get(layer_idx)
        if not self.cap_all:
            return rows or []
        if not rows:
            return self.cap_all
        return sorted(set(self.cap_all) | set(rows))


def _host_query_start_loc(runner: Any, meta: Any, num_reqs: int) -> list[int]:
    """``query_start_loc[:num_reqs+1]`` as host ints without a device sync.

    The GPU model runner keeps a pinned host mirror (``query_start_loc.np``)
    of the tensor it hands to the attention backends; fall back to one
    ``tolist()`` (a single sync per pass) on runners without it.
    """
    host = getattr(getattr(runner, "query_start_loc", None), "np", None)
    if host is not None:
        return [int(x) for x in host[: num_reqs + 1]]
    if meta is None:
        raise RuntimeError("vllm-lens: no host query_start_loc buffer and no attention metadata")
    q = meta.query_start_loc
    if isinstance(q, torch.Tensor):
        return [int(x) for x in q[: num_reqs + 1].tolist()]
    return [int(x) for x in q[: num_reqs + 1]]


def _host_abs_start(runner: Any, meta: Any, qsl: list[int], num_reqs: int) -> list[int]:
    """1.1.0's ``seq_lens[i] - n_query`` per row, from host memory when possible."""
    nct = getattr(getattr(runner, "input_batch", None), "num_computed_tokens_cpu", None)
    if nct is not None:
        return [int(x) for x in nct[:num_reqs]]
    seq_lens: Any = getattr(meta, "seq_lens", None) if meta is not None else None
    if seq_lens is None:
        return [0] * num_reqs  # fallback: treat as prefill from position 0
    if isinstance(seq_lens, torch.Tensor):
        sl = seq_lens[:num_reqs].tolist()
    else:
        sl = [int(x) for x in seq_lens[:num_reqs]]
    return [int(sl[i]) - (qsl[i + 1] - qsl[i]) for i in range(num_reqs)]


def _resolve_request(
    extension: HiddenStatesExtension, runner: Any, req_id: str, gen: int
) -> _ReqPlan | None:
    req_state = runner.requests.get(req_id)
    if req_state is None:
        return None
    sp = req_state.sampling_params
    extra = sp.extra_args if sp is not None else None
    entries = _resolve_entries(extension, req_id, extra)
    configs: list[SteeringVector] = []
    layers: set[int] = set()
    replace_layers: set[int] = set()
    broadcast = False
    min_pos, max_pos = _NO_POS, -1
    for e in entries:
        configs.extend(e.configs)
        layers.update(e.layers)
        replace_layers.update(e.replace_layers)
        broadcast |= e.broadcast
        min_pos, max_pos = min(min_pos, e.min_pos), max(max_pos, e.max_pos)
    cap = extra.get("output_residual_stream") if extra else None
    reads = _resolve_reads(extension, req_id, extra)
    if cap is not None or reads:
        extension._capture_live.add(req_id)
    num_prompt = getattr(req_state, "num_prompt_tokens", None)
    if num_prompt is None:
        num_prompt = len(req_state.prompt_token_ids or ())
    cap_pos: tuple[str, Any] = ("all", None)
    if extra and extra.get(CAPTURE_POSITIONS_KEY) is not None:
        try:
            cap_pos = normalize_positions(extra[CAPTURE_POSITIONS_KEY])
        except ValueError:
            logger.warning("vllm-lens: bad %s for %s, capturing all positions", CAPTURE_POSITIONS_KEY, req_id, exc_info=True)
    cap_set = frozenset(cap) if isinstance(cap, list) else None
    early_exit, exit_layer = False, -1
    if extra and extra.get(EARLY_EXIT_KEY):
        need: set[int] = set(cap_set) if cap_set is not None else set()
        for e in reads:
            need.update(e.layers)
        max_tokens = getattr(sp, "max_tokens", None)
        if max_tokens == 1 and need and not (cap is not None and cap_set is None):
            early_exit, exit_layer = True, max(0, max(need))
    return _ReqPlan(
        gen=gen,
        configs=configs,
        layers=frozenset(layers),
        broadcast=broadcast,
        min_pos=min_pos,
        max_pos=max_pos,
        cap_any=cap is not None,
        cap_set=cap_set,
        num_prompt=int(num_prompt),
        replace_layers=frozenset(replace_layers),
        cap_pos=cap_pos,
        reads=tuple(reads),
        early_exit=early_exit,
        exit_layer=exit_layer,
    )


def _resolve_reads(
    extension: HiddenStatesExtension, internal_req_id: str, extra_args: dict[str, Any] | None
) -> list[_ReadEntry]:
    """Readout entries for a request: same matching as steering (``"-"``-boundary
    prefixes of the internal id, then the ``_readout_id`` sentinel)."""
    index = getattr(extension, "_readout_index", None)
    if not index:
        return []
    found: list[_ReadEntry] = []
    for key in _prefix_keys(internal_req_id):
        found.extend(index.get(key, ()))
    if extra_args:
        rid = extra_args.get("_readout_id")
        if rid:
            found.extend(index.get(rid, ()))
    return found


def _build_step_plan(
    extension: HiddenStatesExtension, runner: Any, num_reqs: int
) -> _StepPlan | None:
    ctx = get_forward_context()
    attn_metadata = ctx.attn_metadata
    if attn_metadata is None:
        return None
    if isinstance(attn_metadata, list):
        attn_metadata = attn_metadata[0]
        if attn_metadata is None:
            return None
    # Hybrid models (e.g. Qwen3-Next with GatedDeltaNet) have multiple
    # attention metadata entries — some (like GDNAttentionMetadata) lack
    # query_start_loc.  Find one that has it.  vLLM >= 0.27 moved
    # query_start_loc off several backends' metadata entirely (FlashInfer,
    # FlashMLA sparse, ...); the model runner's persistent host buffers
    # (``runner.query_start_loc.np``, ``input_batch.num_computed_tokens_cpu``)
    # are the stable source, so the metadata is only a fallback here.
    meta = None
    for _meta in attn_metadata.values():
        if hasattr(_meta, "query_start_loc"):
            meta = _meta
            break
    if meta is None and getattr(getattr(runner, "query_start_loc", None), "np", None) is None:
        logger.warning(
            "No attention metadata with query_start_loc found (keys: %s) and the "
            "model runner has no host query_start_loc buffer. Skipping hook for "
            "this step.",
            list(attn_metadata.keys()),
        )
        return None

    qsl = _host_query_start_loc(runner, meta, num_reqs)
    abs_start = _host_abs_start(runner, meta, qsl, num_reqs)

    req_ids = runner.input_batch.req_ids
    cache = extension._req_plan_cache
    gen = extension._steering_gen
    prompt_only = extension._prompt_only
    should_capture = getattr(extension, "_should_capture", True)
    stats = extension._stats

    fast_capture = getattr(extension, "_fast_capture", True)
    steer: dict[int, list[tuple[int, list[SteeringVector]]]] = {}
    replace_layers: set[int] = set()
    cap_all: list[int] = []
    cap_by_layer: dict[int, list[int]] = {}
    cap_sel: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    read_by_layer: dict[int, list[tuple[int, _ReadEntry, np.ndarray, np.ndarray]]] = {}
    all_exit = bool(getattr(extension, "_early_exit_ok", False))
    exit_max = -1
    for i in range(num_reqs):
        req_id = req_ids[i]
        plan = cache.get(req_id)
        if plan is None or plan.gen != gen:
            plan = _resolve_request(extension, runner, req_id, gen)
            if plan is None:
                all_exit = False
                continue
            cache[req_id] = plan
        if plan.early_exit:
            exit_max = max(exit_max, plan.exit_layer)
        else:
            all_exit = False
        if not plan.configs and not plan.cap_any and not plan.reads:
            continue
        a0 = abs_start[i]
        if prompt_only and a0 >= plan.num_prompt:
            # Generated positions run inside replayed CUDA graphs (no hooks);
            # never touch them so behaviour does not depend on batch mix.
            stats["rows_skipped_generated"] += 1
            continue
        start, end = qsl[i], qsl[i + 1]
        if plan.configs and (
            plan.broadcast
            or (plan.max_pos >= a0 and plan.min_pos < a0 + (end - start))
        ):
            for layer_idx in plan.layers:
                steer.setdefault(layer_idx, []).append((i, plan.configs))
            if plan.replace_layers:
                replace_layers.update(plan.replace_layers)
        if plan.cap_any and should_capture:
            if fast_capture:
                sel = _select_positions(plan.cap_pos, start, end, a0, plan.num_prompt)
                if len(sel[0]) == 0:
                    sel = None
                else:
                    cap_sel[i] = sel
            else:
                sel = True  # legacy per-request .cpu() path captures the whole chunk
            if sel is not None:
                if plan.cap_set is None:
                    cap_all.append(i)
                else:
                    for layer_idx in plan.cap_set:
                        cap_by_layer.setdefault(layer_idx, []).append(i)
        if plan.reads and should_capture:
            for entry in plan.reads:
                idx, pos = _select_positions(entry.spec, start, end, a0, plan.num_prompt)
                if len(idx) == 0:
                    continue
                for layer_idx in entry.layers:
                    read_by_layer.setdefault(layer_idx, []).append((i, entry, idx, pos))

    if len(cache) > num_reqs + _REQ_CACHE_SLACK:
        live = runner.requests
        for stale in [k for k in cache if k not in live]:
            del cache[stale]
        extension._capture_live = {r for r in extension._capture_live if r in live}

    stats["steps_planned"] += 1
    if not steer and not cap_all and not cap_by_layer and not read_by_layer:
        stats["steps_idle"] += 1
    return _StepPlan(
        qsl,
        abs_start,
        steer,
        cap_all,
        cap_by_layer,
        id(ctx),
        replace_layers,
        cap_sel,
        read_by_layer,
        exit_max if (all_exit and num_reqs > 0 and exit_max >= 0) else None,
    )


def _step_is_idle(extension: HiddenStatesExtension, runner: Any, num_reqs: int) -> bool:
    """O(1)-ish test for a forward pass on which no hook can have work.

    True when the pass is uniform decode (one token per row), no registered
    vector broadcasts to generated positions, every positional vector lies
    behind every row's current position, every row has been seen before (so
    its capture wish is known) and no capture is in flight.  Uses only
    host-side buffers; returns False whenever unsure.
    """
    extension._refresh_aggregates()
    if extension._agg_broadcast:
        return False
    if extension._capture_live:
        live = runner.requests
        extension._capture_live = {r for r in extension._capture_live if r in live}
        if extension._capture_live:
            return False
    host = getattr(getattr(runner, "query_start_loc", None), "np", None)
    nct = getattr(getattr(runner, "input_batch", None), "num_computed_tokens_cpu", None)
    if host is None or nct is None:
        return False
    if int(host[num_reqs]) != num_reqs:  # some row computes more than one token
        return False
    min_start = int(nct[:num_reqs].min())
    return min_start > 0 and min_start > extension._agg_max_pos


def norm_match(
    residual: torch.Tensor,
    steering: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Scale a steering vector to match the L2 norm of the residual stream.

    Norm matching approach from the Activation Oracles paper
    (arXiv:2512.15674):

        h'_i = h_i + ‖h_i‖ · v_i / ‖v_i‖

    This rescales the steering vector so its magnitude matches the
    residual before addition, ensuring activations of varying provenance
    are automatically scaled to a consistent magnitude.
    """
    r_norm = residual.float().norm(dim=-1, keepdim=True)
    v_norm = steering.float().norm(dim=-1, keepdim=True)
    return (steering * (r_norm / (v_norm + eps))).to(residual.dtype)


def _apply_steering(
    configs: list[SteeringVector],
    layer_idx: int,
    target: torch.Tensor,
    start: int,
    end: int,
    abs_start: int,
    residual: torch.Tensor | None,
) -> None:
    """Apply all matching steering vectors to a token slice *in-place*.

    ``target`` is the (already-cloned) output tensor.  ``start``/``end``
    are batch-relative indices, ``abs_start`` is the absolute sequence
    position of the first token in ``target[start:end]``.

    ``residual`` is the second half of a fused-residual layer output (Qwen,
    Llama, Gemma, ...: the layer returns ``(hidden_states, residual)`` and
    the TRUE residual stream is ``hidden_states + residual``), or ``None``
    for layers that return the stream as one tensor (and for the embedding
    stream).  Ported from upstream 1.2.0 (#7): ``norm_match`` scales to the
    norm of the full stream ``target + residual`` -- 1.1.0 used ``target``
    alone, i.e. the MLP-delta half, so the injected magnitude on
    fused-residual models was ``scale · ‖hidden_states‖`` instead of
    ``scale · ‖h‖`` (a ratio of ~0.12 on Qwen3.6-27B layer 1).  With the
    port, ``SteeringVector(norm_match=True, scale=c)`` is exactly the
    activation-oracle injection ``h' = h + c · ‖h‖ · v/‖v‖``.
    ``mode="replace"`` on a fused layer overwrites BOTH halves (``target[rel]
    = scale·v``, ``residual[rel] = 0``) so the full stream equals ``scale·v``
    exactly; the caller must therefore pass a *cloned* residual whenever a
    replace config is present (see ``_hook_inner``).  Required (no default)
    so a forgotten reference fails at the call instead of silently scaling to
    the MLP-delta-half norm.
    """
    n_tokens = end - start
    for cfg in configs:
        if layer_idx not in cfg.layer_index_map:
            continue
        act_idx = cfg.layer_index_map[layer_idx]
        vec = cfg.activations[act_idx].to(target.dtype)  # (hidden,) or (n_pos, hidden)

        if vec.dim() == 1:
            # 2D: broadcast to all positions
            v = vec.unsqueeze(0)
            if cfg.norm_match:
                ref = target[start:end]
                if residual is not None:
                    ref = ref + residual[start:end]
                v = norm_match(ref, v)
            target[start:end] = target[start:end] + v * cfg.scale
        else:
            # 3D: position-specific
            pos_indices = (
                cfg.position_indices
                if cfg.position_indices is not None
                else list(range(vec.shape[0]))
            )
            abs_end = abs_start + n_tokens
            for pi, abs_pos in enumerate(pos_indices):
                if pi >= vec.shape[0]:
                    break
                if abs_pos < abs_start or abs_pos >= abs_end:
                    continue
                rel = abs_pos - abs_start + start
                v = vec[pi]
                if cfg.norm_match:
                    ref = target[rel]
                    if residual is not None:
                        ref = ref + residual[rel]
                    v = norm_match(ref, v)
                if cfg.mode == "replace":
                    target[rel] = v * cfg.scale
                    if residual is not None:
                        residual[rel] = 0
                else:
                    target[rel] = target[rel] + v * cfg.scale


def _apply_layer_vectorized(
    todo: list[tuple[int, list[SteeringVector]]],
    layer_idx: int,
    target: torch.Tensor,
    plan: _StepPlan,
    residual: torch.Tensor | None,
) -> bool:
    """vllm-lens-metamodel: apply every (row, vector) pair of this layer/pass at once.

    Gathers the vectors ``_apply_steering`` would add into one ``[n, hidden]``
    tensor and adds them with a single ``index_add_`` (norm-matching in the
    same batched op); ``mode="replace"`` rows go through one ``index_copy_``
    (plus ``index_fill_(0)`` on the fused ``residual`` half, see
    ``_apply_steering``).  Returns False without touching ``target`` when the
    batch is not vectorisable with identical semantics -- a row would receive
    several vectors, or a broadcast vector covers a multi-token chunk -- so
    the caller runs ``_apply_steering`` row by row.  With ``norm_match=False``
    the result is bit-identical to the sequential path (one multiply and one
    add per element, same dtype); with ``norm_match=True`` the per-row norms
    (of the FULL residual stream ``target + residual``) come from a batched
    reduction and may differ by float32 rounding.
    """
    qsl, abs_start = plan.qsl, plan.abs_start
    rows: list[int] = []
    vecs: list[torch.Tensor] = []
    scales: list[float] = []
    nms: list[bool] = []
    reps: list[bool] = []
    limit = _VEC_MAX_ENTRIES_PER_ROW * len(todo) + 64
    for i, configs in todo:
        start, end, a0 = qsl[i], qsl[i + 1], abs_start[i]
        n_tokens = end - start
        for cfg in configs:
            layer_index_map = cfg.layer_index_map
            if layer_idx not in layer_index_map:
                continue
            is_replace = cfg.mode == "replace"
            vec = cfg.activations[layer_index_map[layer_idx]]
            if vec.dim() == 1:
                if n_tokens != 1:
                    return False
                rows.append(start)
                vecs.append(vec)
            else:
                pos_indices = (
                    cfg.position_indices
                    if cfg.position_indices is not None
                    else range(vec.shape[0])
                )
                for pi, abs_pos in enumerate(pos_indices):
                    if pi >= vec.shape[0]:
                        break
                    if a0 <= abs_pos < a0 + n_tokens:
                        rows.append(abs_pos - a0 + start)
                        vecs.append(vec[pi])
                    else:
                        continue
                    scales.append(cfg.scale)
                    nms.append(cfg.norm_match)
                    reps.append(is_replace)
                continue
            scales.append(cfg.scale)
            nms.append(cfg.norm_match)
            reps.append(is_replace)
        if len(rows) > limit:
            return False
    if not rows:
        return True  # nothing in range this pass (e.g. a later prefill chunk)
    if len(set(rows)) != len(rows):
        return False

    device = target.device
    idx = torch.tensor(rows, dtype=torch.long, device=device)
    v = torch.stack(vecs).to(target.dtype)
    if any(nms):
        ref = target.index_select(0, idx)
        if residual is not None:  # fused-residual layer: norm of the FULL stream
            ref = ref + residual.index_select(0, idx)
        r_norm = ref.float().norm(dim=-1, keepdim=True)
        v_norm = v.float().norm(dim=-1, keepdim=True)
        matched = (v * (r_norm / (v_norm + 1e-6))).to(target.dtype)
        if all(nms):
            v = matched
        else:
            v = torch.where(torch.tensor(nms, device=device).unsqueeze(1), matched, v)
    if all(s == scales[0] for s in scales):
        if scales[0] != 1.0:
            v = v * scales[0]
    else:
        v = (v.float() * torch.tensor(scales, device=device).unsqueeze(1)).to(
            target.dtype
        )
    if not any(reps):
        target.index_add_(0, idx, v)
        return True
    if all(reps):
        rep_idx, rep_v = idx, v
    else:
        mask = torch.tensor(reps, device=device)
        rep_idx, rep_v = idx[mask], v[mask]
        target.index_add_(0, idx[~mask], v[~mask])
    target.index_copy_(0, rep_idx, rep_v)
    if residual is not None:
        residual.index_fill_(0, rep_idx, 0)
    return True


class UnsupportedLayerOutputError(RuntimeError):
    """A decoder layer returned a hyper-connection / multi-stream output
    (e.g. DeepSeek-V4's ``(x, residual[T, hc, D], post_mix, res_mix)``): the
    residual stream at this layer boundary is a deferred fold of several
    tensors, so steering or capturing "the layer output" is undefined.  Use
    ``EMBED_LAYER_INDEX`` (the embedding stream entering layer 0, a plain
    ``[T, D]`` tensor on every architecture).  Raised, never swallowed."""


def _split_layer_output(
    output: torch.Tensor | tuple, layer_idx: int
) -> tuple[torch.Tensor, torch.Tensor | None, tuple]:
    """``(stream_half, residual_half_or_None, rest)`` for a decoder layer output.

    Plain tensor -> ``(t, None, ())``; fused-residual 2-tuple ``(h, r)`` with
    ``r.shape == h.shape`` -> ``(h, r, ())``; ``(h, None)`` -> ``(h, None, (None,))``.
    Anything else (more than two elements, or a second element whose shape
    differs from the first) is a multi-stream output ->
    :class:`UnsupportedLayerOutputError`.
    """
    if not isinstance(output, tuple):
        return output, None, ()
    if len(output) == 0 or not isinstance(output[0], torch.Tensor):
        raise UnsupportedLayerOutputError(
            f"vllm-lens: layer {layer_idx} returned an unexpected output {type(output).__name__} of length {len(output)}"
        )
    multi = len(output) > 2
    residual = None
    if len(output) == 2 and isinstance(output[1], torch.Tensor):
        residual = output[1]
        multi = residual.shape != output[0].shape
    if multi:
        shapes = [tuple(o.shape) if isinstance(o, torch.Tensor) else type(o).__name__ for o in output]
        raise UnsupportedLayerOutputError(
            f"vllm-lens: layer {layer_idx} returned a {len(output)}-tuple {shapes}: a hyper-connection / "
            "multi-stream residual (deferred fold), on which layer-output steering and capture are "
            "undefined. Inject / capture at the embedding stream instead (layer_indices=[EMBED_LAYER_INDEX], "
            "output_residual_stream=[EMBED_LAYER_INDEX])."
        )
    return output[0], residual, output[1:]


class EmbedInjectionError(RuntimeError):
    """``EMBED_LAYER_INDEX`` steering could not be applied on this pass.

    Raised (not warned) out of the layer-0 pre-hook: silently skipping the
    injection is the worst possible failure mode for a training run.
    """


def _find_hidden_states_arg(
    args: tuple, kwargs: dict[str, Any] | None, total_tokens: int
) -> torch.Tensor:
    """Locate the hidden states entering decoder layer 0 among the layer's
    positional AND keyword inputs.

    vLLM model code is inconsistent: ``Qwen2Model``/``LlamaModel`` call
    ``layer(positions, hidden_states, residual)`` positionally while
    ``Qwen3NextModel`` (Qwen3.5 / Qwen3.6) calls ``layer(positions=...,
    hidden_states=..., residual=...)`` by keyword, so both must be searched.
    A candidate is a 2-D floating tensor whose dim 0 covers this pass's
    scheduled tokens (``>=``: vLLM may pad the token dim for sequence
    parallelism).  At layer 0 ``residual`` is ``None`` on every fused-residual
    architecture, so exactly one candidate is expected; if several match, a
    keyword literally named ``hidden_states`` wins; anything else raises
    :class:`EmbedInjectionError`.
    """
    cands: list[tuple[str, torch.Tensor]] = []
    for i, a in enumerate(args):
        if _is_hidden_candidate(a, total_tokens):
            cands.append((f"args[{i}]", a))
    for k, a in (kwargs or {}).items():
        if _is_hidden_candidate(a, total_tokens):
            cands.append((k, a))
    if len(cands) == 1:
        return cands[0][1]
    named = [t for k, t in cands if k == "hidden_states"]
    if len(cands) > 1 and len(named) == 1:
        return named[0]
    seen = [
        f"{'kw:' if k in (kwargs or {}) else ''}{k}={tuple(a.shape)}/{a.dtype}"
        if isinstance(a, torch.Tensor)
        else f"{k}={type(a).__name__}"
        for k, a in [*((f"args[{i}]", a) for i, a in enumerate(args)), *(kwargs or {}).items()]
    ]
    raise EmbedInjectionError(
        f"vllm-lens: EMBED_LAYER_INDEX steering needs exactly one "
        f"[>= {total_tokens} tokens, hidden] floating input on decoder layer 0, "
        f"found {len(cands)} candidate(s) {[k for k, _ in cands]} among layer "
        f"inputs {seen}"
    )


def _is_hidden_candidate(a: Any, total_tokens: int) -> bool:
    return (
        isinstance(a, torch.Tensor)
        and a.dim() == 2
        and a.is_floating_point()
        and a.shape[0] >= total_tokens
    )


def _apply_embed(
    todo: list[tuple[int, list[SteeringVector]]],
    target: torch.Tensor,
    plan: _StepPlan,
    stats: dict[str, int],
) -> None:
    """Apply EMBED_LAYER_INDEX configs to the embedding stream *in place*.

    Vectorised: one ``index_copy_`` for all replace rows and one
    ``index_add_`` for all add rows.  Rows are prompt positions resolved
    against this pass's chunk offsets, so chunked prefill is handled and
    decode rows are never touched (their absolute positions lie past every
    registered marker position).
    """
    qsl, abs_start = plan.qsl, plan.abs_start
    rep_rows: list[int] = []
    rep_vecs: list[torch.Tensor] = []
    add_rows: list[int] = []
    add_vecs: list[torch.Tensor] = []
    for i, configs in todo:
        start, end, a0 = qsl[i], qsl[i + 1], abs_start[i]
        n_tokens = end - start
        for cfg in configs:
            lim = cfg.layer_index_map
            if EMBED_LAYER_INDEX not in lim:
                continue
            vec = cfg.activations[lim[EMBED_LAYER_INDEX]]
            if vec.dim() == 1:  # broadcast add over this chunk
                v = vec.to(target.dtype) * cfg.scale
                target[start:end] += v
                continue
            pos_indices = (
                cfg.position_indices
                if cfg.position_indices is not None
                else range(vec.shape[0])
            )
            for pi, abs_pos in enumerate(pos_indices):
                if pi >= vec.shape[0]:
                    break
                if not (a0 <= abs_pos < a0 + n_tokens):
                    continue
                rel = abs_pos - a0 + start
                v = vec[pi]
                if cfg.norm_match:
                    v = norm_match(target[rel], v)
                v = (v.to(target.dtype)) * cfg.scale
                if cfg.mode == "replace":
                    rep_rows.append(rel)
                    rep_vecs.append(v)
                else:
                    add_rows.append(rel)
                    add_vecs.append(v)
    dev = target.device
    if rep_rows:
        if len(set(rep_rows)) != len(rep_rows):
            for r, v in zip(rep_rows, rep_vecs):  # duplicates: last wins, in order
                target[r] = v
        else:
            target.index_copy_(
                0, torch.tensor(rep_rows, dtype=torch.long, device=dev),
                torch.stack(rep_vecs),
            )
        stats["rows_replaced"] += len(rep_rows)
    if add_rows:
        target.index_add_(
            0, torch.tensor(add_rows, dtype=torch.long, device=dev),
            torch.stack(add_vecs),
        )
        stats["rows_steered"] += len(add_rows)
    if rep_rows or add_rows:
        stats["embed_apply_steps"] += 1


def _hook_inner(
    extension: HiddenStatesExtension,
    layer_idx: int,
    output: torch.Tensor | tuple[torch.Tensor, ...],
) -> torch.Tensor | tuple[torch.Tensor, ...] | None:
    """Core hook logic, separated so _make_hook can wrap it in try/except."""
    if extension._step_idle:
        return None
    if not is_forward_context_available():
        return None

    runner = extension.model_runner
    num_reqs = runner.input_batch.num_reqs
    if num_reqs == 0:
        return None
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return None  # CUDA-graph capture (dummy inputs): never bake anything in

    # --- Phase 1: one plan per forward pass, shared by all layer hooks ------
    plan = extension._step_plan
    if plan is None or plan.ctx_id != id(get_forward_context()):
        plan = extension._step_plan = _build_step_plan(extension, runner, num_reqs)
        if plan is None:
            return None
    todo = plan.steer.get(layer_idx)
    cap_rows = plan.capture_rows(layer_idx)
    reads = plan.read_by_layer.get(layer_idx)
    do_exit = plan.exit_layer is not None and layer_idx >= plan.exit_layer
    if not todo and not cap_rows and not reads:
        if do_exit:
            raise _EarlyExit(torch.zeros_like(output[0] if isinstance(output, tuple) else output))
        return None
    query_start_loc = plan.qsl

    # --- Phase 2: apply steering ------------------------------------
    # Multi-stream (hyper-connection) layer outputs are refused here -- loudly.
    stream, residual, rest = _split_layer_output(output, layer_idx)
    modified_output: torch.Tensor | tuple[torch.Tensor, ...] | None = None
    if todo:
        target = stream.clone()
        if isinstance(output, tuple):
            if residual is not None and layer_idx in plan.replace_layers:
                # Fused-residual layer: the true stream is output[0] + output[1].
                # norm_match reads it; mode="replace" also zeroes the residual
                # half, so clone that half only when a replace row is scheduled.
                residual = residual.clone()
                rest = (residual, *rest[1:])
            modified_output = (target, *rest)
        else:
            modified_output = target

        extension._stats["steer_layer_steps"] += 1
        extension._stats["rows_steered"] += len(todo)
        if extension._vectorized and _apply_layer_vectorized(
            todo, layer_idx, target, plan, residual
        ):
            extension._stats["vectorized_layer_steps"] += 1
        else:
            for i, configs in todo:
                _apply_steering(
                    configs,
                    layer_idx,
                    target,
                    query_start_loc[i],
                    query_start_loc[i + 1],
                    plan.abs_start[i],
                    residual,
                )

    # --- Phase 3: capture activations (rank 0 only) -----------------
    if cap_rows or reads:
        if modified_output is not None:
            stream, residual, _ = _split_layer_output(modified_output, layer_idx)
    if cap_rows:
        if extension._fast_capture:
            _capture_gather(extension, runner, layer_idx, stream, residual, plan, cap_rows)
        else:
            hidden_states: Float[torch.Tensor, "total_tokens hidden_dim"]  # type: ignore[reportUndefinedVariable]
            hidden_states = stream + residual if residual is not None else stream
            _capture_rows(extension, runner, layer_idx, hidden_states, query_start_loc, cap_rows)

    # --- Phase 4: in-engine readout (projection onto per-request directions) --
    if reads:
        _readout_layer(extension, runner, layer_idx, stream, residual, reads)

    # --- Phase 5: early exit: nothing deeper is needed by any row of this pass --
    if do_exit:
        raise _EarlyExit(torch.zeros_like(stream))

    return modified_output


def _capture_gather(
    extension: HiddenStatesExtension,
    runner: Any,
    layer_idx: int,
    stream: torch.Tensor,
    residual: torch.Tensor | None,
    plan: _StepPlan,
    cap_rows: list[int],
) -> None:
    """vllm-metamodels fast capture: ONE ``index_select`` over every capturing
    row's selected positions (``plan.cap_sel``) and ONE asynchronous pinned
    device->host copy per layer-step, split per request lazily at retrieval
    (``_flush_host_blocks``).  On fused-residual layers the stream
    ``hidden_states + residual`` is formed on the selected rows only."""
    t0 = time.perf_counter()
    key = tuple(cap_rows)
    cached = plan.cap_concat.get(key)
    if cached is None:
        req_ids = runner.input_batch.req_ids
        parts: list[np.ndarray] = []
        segments: list[tuple] = []
        for i in cap_rows:
            sel = plan.cap_sel.get(i)
            if sel is None:
                continue
            parts.append(sel[0])
            segments.append((req_ids[i], int(len(sel[0])), sel[1]))
        if not parts:
            return
        cached = (np.concatenate(parts), segments)
        plan.cap_concat[key] = cached
    idx_np, segments = cached
    idx = torch.from_numpy(idx_np).to(stream.device)
    sel_rows = stream.index_select(0, idx)
    if residual is not None:
        sel_rows += residual.index_select(0, idx)
    host, ev = _to_host(sel_rows)
    extension._cap_blocks.append(_HostBlock(host, ev, layer_idx, segments))
    st = extension._stats
    st["capture_layer_steps"] += 1
    st["capture_rows"] += int(len(idx_np))
    st["hook_capture_s"] += time.perf_counter() - t0


_READOUT_CHUNK = 8192  # rows per float32 chunk in _readout_layer (8192 x 5120 x 4 B = 168 MB)


def _readout_layer(
    extension: HiddenStatesExtension,
    runner: Any,
    layer_idx: int,
    stream: torch.Tensor,
    residual: torch.Tensor | None,
    todo: list[tuple[int, _ReadEntry, np.ndarray, np.ndarray]],
) -> None:
    """Compute ``metric(h_pos, v_req) + bias`` for every (row, readout entry,
    position) of this layer-step and ship only the float32 scalars to the host.
    Rows are grouped by (direction block, metric); each group is one
    ``index_select`` of hidden rows + one of directions, reduced in float32
    chunks of ``_READOUT_CHUNK`` rows (bounded temporary memory)."""
    t0 = time.perf_counter()
    req_ids = runner.input_batch.req_ids
    groups: dict[tuple[int, bool], dict[str, list]] = {}
    for i, entry, idx, pos in todo:
        row = entry.layer_rows.get(layer_idx)
        if row is None:
            continue
        g = groups.setdefault((id(entry.block), entry.cos), {"block": entry.block, "idx": [], "vrow": [], "bias": [], "seg": []})
        g["idx"].append(idx)
        g["vrow"].append(np.full(len(idx), row, dtype=np.int64))
        g["bias"].append(np.full(len(idx), entry.bias, dtype=np.float32))
        g["seg"].append((req_ids[i], int(len(idx)), pos, entry.seq))
    dev = stream.device
    n_rows = 0
    for (_bid, use_cos), g in groups.items():
        idx_np = np.concatenate(g["idx"])
        idx = torch.from_numpy(idx_np).to(dev)
        vrow = torch.from_numpy(np.concatenate(g["vrow"])).to(dev)
        block: torch.Tensor = g["block"]
        n = int(len(idx_np))
        n_rows += n
        out = torch.empty(n, dtype=torch.float32, device=dev)
        for s0 in range(0, n, _READOUT_CHUNK):
            s1 = min(n, s0 + _READOUT_CHUNK)
            sub = idx[s0:s1]
            h = stream.index_select(0, sub)
            if residual is not None:
                h = h + residual.index_select(0, sub)
            h = h.float()
            v = block.index_select(0, vrow[s0:s1])
            d = (h * v).sum(-1)
            if use_cos:
                d = d / (h.norm(dim=-1) * v.norm(dim=-1)).clamp_min(1e-6)
            out[s0:s1] = d
        bias_np = np.concatenate(g["bias"])
        if np.any(bias_np != 0.0):
            out += torch.from_numpy(bias_np).to(dev)
        host, ev = _to_host(out)
        extension._read_blocks.append(_HostBlock(host, ev, layer_idx, g["seg"]))
    st = extension._stats
    st["readout_layer_steps"] += 1
    st["readout_rows"] += n_rows
    st["hook_readout_s"] += time.perf_counter() - t0


def _capture_rows(
    extension: HiddenStatesExtension,
    runner: Any,
    layer_idx: int,
    hidden_states: torch.Tensor,
    query_start_loc: list[int],
    cap_rows: list[int],
) -> None:
    """Copy each capturing row's slice of ``hidden_states`` to the CPU store
    (``layer_idx`` may be ``EMBED_LAYER_INDEX`` for the embedding stream)."""
    req_ids = runner.input_batch.req_ids
    for i in cap_rows:
        req_id = req_ids[i]
        start = query_start_loc[i]
        end = query_start_loc[i + 1]
        # Blocking .cpu() benchmarked faster than non_blocking + event sync
        activation: Float[torch.Tensor, "seq_len hidden_dim"] = hidden_states[  # type: ignore[reportUndefinedVariable]
            start:end
        ].cpu()

        if req_id not in extension._captured_states:
            extension._captured_states[req_id] = {}
        layer_states = extension._captured_states[req_id]
        if layer_idx not in layer_states:
            layer_states[layer_idx] = []
        layer_states[layer_idx].append(activation)


def _make_hook(extension: HiddenStatesExtension, layer_idx: int) -> Callable:
    """Create a forward hook closure for a specific layer index."""

    def hook(
        _module: torch.nn.Module,
        _input: object,
        output: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | None:
        """Forward hook: apply steering vectors then capture activations.

        Returns the modified output if any steering was applied, ``None``
        otherwise (so PyTorch leaves the original output untouched).
        """
        try:
            return _hook_inner(extension, layer_idx, output)
        except _EarlyExit:
            raise
        except UnsupportedLayerOutputError:
            extension._stats["errors"] += 1
            extension._stats["unsupported_layer_output"] += 1
            logger.error(
                "vllm-lens: layer %d has a multi-stream output; steering/capture "
                "there is undefined -- raising rather than silently skipping",
                layer_idx,
            )
            raise
        except Exception:
            extension._stats["errors"] += 1
            logger.warning(
                "vllm-lens hook error on layer %d, skipping", layer_idx, exc_info=True
            )
            return None

    return hook


def _make_pre_hook(extension: HiddenStatesExtension, layer_idx: int) -> Callable:
    """vllm-lens-metamodel: pre-hook on this rank's first decoder layer
    (registered with ``with_kwargs=True``).

    A new forward pass begins: drop the previous pass's plan and decide
    whether this pass is *idle* (``_step_is_idle``), in which case every
    layer hook returns on a single flag check and the pass costs nothing
    else.  Otherwise build the pass's plan here (the layer hooks reuse it)
    and, when this is GLOBAL layer 0, apply / capture ``EMBED_LAYER_INDEX``
    configs on the hidden states ENTERING the layer -- the forward hooks only
    ever see layer OUTPUTS.  Embedding-injection failures are counted AND
    re-raised (``EmbedInjectionError``): a silently skipped injection would
    corrupt a training run, so it must be loud.
    """
    is_layer0 = layer_idx == 0

    def pre_hook(
        _module: torch.nn.Module, args: tuple, kwargs: dict[str, Any]
    ) -> None:
        extension._step_plan = None
        extension._step_idle = False
        try:
            if not is_forward_context_available():
                return
            runner = extension.model_runner
            num_reqs = runner.input_batch.num_reqs
            if not num_reqs:
                return
            if _step_is_idle(extension, runner, num_reqs):
                extension._step_idle = True
                extension._stats["steps_fast_idle"] += 1
                return
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                return  # CUDA-graph capture (dummy inputs): never bake anything in
            plan = extension._step_plan = _build_step_plan(extension, runner, num_reqs)
        except Exception:
            extension._stats["errors"] += 1
            logger.warning("vllm-lens pre-hook error, running full path", exc_info=True)
            extension._step_idle = False
            extension._step_plan = None
            return
        if plan is None or not is_layer0:
            return
        todo = plan.steer.get(EMBED_LAYER_INDEX)
        cap_rows = plan.cap_by_layer.get(EMBED_LAYER_INDEX)
        reads = plan.read_by_layer.get(EMBED_LAYER_INDEX)
        if not todo and not cap_rows and not reads:
            return
        try:
            target = _find_hidden_states_arg(args, kwargs, plan.qsl[num_reqs])
            if todo:
                _apply_embed(todo, target, plan, extension._stats)
            if cap_rows:  # post-injection embedding stream, explicit layer -1 only
                if extension._fast_capture:
                    _capture_gather(extension, runner, EMBED_LAYER_INDEX, target, None, plan, cap_rows)
                else:
                    _capture_rows(
                        extension, runner, EMBED_LAYER_INDEX, target, plan.qsl, cap_rows
                    )
            if reads:
                _readout_layer(extension, runner, EMBED_LAYER_INDEX, target, None, reads)
        except Exception:
            extension._stats["errors"] += 1
            extension._stats["embed_errors"] += 1
            logger.error(
                "vllm-lens: EMBED_LAYER_INDEX injection failed on layer 0 -- "
                "raising rather than silently skipping",
                exc_info=True,
            )
            raise

    return pre_hook


_MULTI_STREAM_MSG = (
    "vllm-lens: {what} is unsupported on this hyper-connection (multi-stream residual) "
    "architecture: decoder layers return a deferred fold of several tensors, so the "
    "residual stream at a layer boundary is not a single [tokens, hidden] tensor. "
    "Inject with layer_indices=[EMBED_LAYER_INDEX] (mode='replace' or 'add' on the "
    "embedding stream) and capture with output_residual_stream=[EMBED_LAYER_INDEX]. "
    "Override the detection with VLLM_LENS_MULTI_STREAM=0/1."
)


def _detect_multi_stream(extension: HiddenStatesExtension) -> bool:
    """Hyper-connection architectures (DeepSeek-V4 mHC: ``hc_mult`` > 1) keep
    several residual streams per layer boundary; ``VLLM_LENS_MULTI_STREAM``
    overrides the config-based detection."""
    env = os.environ.get("VLLM_LENS_MULTI_STREAM")
    if env is not None:
        return env.strip().lower() in _TRUTHY
    try:
        cfg = getattr(extension, "vllm_config", None) or extension.model_runner.vllm_config
        hf = cfg.model_config.hf_config
        for c in (hf, getattr(hf, "text_config", None)):
            hc = getattr(c, "hc_mult", None)
            if hc is not None:
                return int(hc) > 1
    except Exception:  # pragma: no cover - defensive against config drift
        return False
    return False


def _new_stats() -> dict[str, int]:
    return {
        "steps_fast_idle": 0,
        "steps_planned": 0,
        "steps_idle": 0,
        "steer_layer_steps": 0,
        "vectorized_layer_steps": 0,
        "rows_steered": 0,
        "rows_replaced": 0,
        "embed_apply_steps": 0,
        "embed_errors": 0,
        "unsupported_layer_output": 0,
        "rows_skipped_generated": 0,
        "capture_layer_steps": 0,
        "capture_rows": 0,
        "hook_capture_s": 0.0,
        "readout_layer_steps": 0,
        "readout_rows": 0,
        "hook_readout_s": 0.0,
        "early_exits": 0,
        "retrieval_s": 0.0,
        "errors": 0,
    }


class HiddenStatesExtension:
    """Mixin injected into vLLM's GPU Worker at runtime.

    Configured via the ``worker_extension_cls`` engine arg. vLLM dynamically
    adds this class as a base of Worker
    (``Worker.__bases__ += (HiddenStatesExtension,)``), so ``self`` is the
    Worker instance and its methods are callable via
    ``collective_rpc("method_name")``.

    It doesn't extend Worker directly — vLLM handles that injection.
    """

    if TYPE_CHECKING:
        model_runner: Any  # Provided by Worker at runtime
        rank: int
        parallel_config: ParallelConfig
        vllm_config: Any

    # Per-request captured activations:
    # internal_req_id → { layer_idx → [tensor, ...] }
    _captured_states: dict[
        str,
        dict[int, list[Float[torch.Tensor, "seq_len hidden_dim"]]],  # type: ignore[reportUndefinedVariable]
    ] = {}
    _hooks_installed: bool = False

    # Per-request steering configs:
    # key (external_req_id or _steering_id) → list of SteeringVector
    _steering_data: dict[str, list[SteeringVector]] = {}

    # Whether this rank should capture activations (only TP rank 0).
    _should_capture: bool = True

    # vllm-lens-metamodel: index over _steering_data + per-pass state.
    _steering_index: dict[str, _SteerEntry] = {}
    _steering_gen: int = 0  # bumped on every set/clear; invalidates _req_plan_cache
    _steering_seq: int = 0
    _req_plan_cache: dict[str, _ReqPlan] = {}  # internal_req_id -> _ReqPlan
    _capture_live: set[str] = set()
    _step_plan: _StepPlan | None = None
    _step_idle: bool = False
    _agg_gen: int = -1
    _agg_broadcast: bool = False
    _agg_max_pos: int = -1
    _agg_embed: bool = False
    _multi_stream: bool = False  # hyper-connection architecture (hc_mult > 1): embed-only
    _stats: dict[str, Any] = _new_stats()
    _vectorized: bool = True
    _prompt_only: bool = (
        False  # CUDA graphs active for decode: hooks only see prompt rows
    )
    # vllm-metamodels fast readout state.
    _fast_capture: bool = True  # gather + pinned async D2H per layer-step (VLLM_LENS_FAST_CAPTURE)
    _cap_blocks: list[_HostBlock] = []  # pending capture copies, flushed at retrieval
    _read_blocks: list[_HostBlock] = []  # pending readout copies
    _captured_positions: dict[str, dict[int, list[np.ndarray]]] = {}  # req -> layer -> [abs pos per pass]
    _readout_index: dict[str, list[_ReadEntry]] = {}  # key -> readout entries
    _readouts: dict[str, dict[tuple[int, int], list[tuple[np.ndarray, torch.Tensor]]]] = {}
    """internal_req_id -> {(readout seq, layer): [(abs positions, float32 values) per pass]}"""
    _early_exit_ok: bool = False
    _early_exit_reason: str = "hooks not installed"

    def install_hooks(self) -> None:
        """Register a forward hook on every decoder layer. Idempotent.

        Hooks are installed on **all** TP ranks because steering must
        modify hidden states everywhere.  Activation *capture* is gated
        to rank 0 only via ``_should_capture``.

        Requires ``enforce_eager=True`` in engine args (the plugin default)
        — otherwise ``@support_torch_compile`` would compile the forward
        graph and hooks won't fire — or, with ``VLLM_LENS_CUDA_GRAPHS=1``,
        compilation mode NONE with ``cudagraph_mode=FULL_DECODE_ONLY``.
        """
        if self._hooks_installed:
            return
        self._hooks_installed = True
        # Reset to instance-level dicts (class-level defaults are shared)
        self._captured_states = {}
        self._steering_data = {}
        self._steering_index = {}
        self._steering_gen = 0
        self._steering_seq = 0
        self._req_plan_cache = {}
        self._capture_live = set()
        self._step_plan = None
        self._step_idle = False
        self._agg_gen = -1
        self._stats = _new_stats()
        self._vectorized = (
            os.environ.get("VLLM_LENS_VECTORIZED", "1").strip().lower() in _TRUTHY
        )
        self._fast_capture = (
            os.environ.get("VLLM_LENS_FAST_CAPTURE", "1").strip().lower() in _TRUTHY
        )
        self._cap_blocks = []
        self._read_blocks = []
        self._captured_positions = {}
        self._readout_index = {}
        self._readouts = {}
        self._early_exit_ok, self._early_exit_reason = self._early_exit_supported()
        if self._early_exit_ok:
            self._wrap_model_forward()
        else:
            logger.info("vllm-lens: early exit unavailable (%s)", self._early_exit_reason)

        # Only rank 0 captures — residual streams are replicated across
        # TP ranks after all-reduce, so the data is identical.
        tp_size = self.parallel_config.tensor_parallel_size
        self._should_capture = tp_size <= 1 or self.rank % tp_size == 0

        self._multi_stream = _detect_multi_stream(self)
        if self._multi_stream:
            logger.info(
                "vllm-lens: hyper-connection (multi-stream residual) architecture detected: "
                "steering and capture are restricted to the embedding stream "
                "(EMBED_LAYER_INDEX); layer-output requests are rejected."
            )
        self._prompt_only = self._cuda_graphs_active()
        if self._prompt_only:
            logger.info(
                "vllm-lens: CUDA graphs active for decode batches; steering and "
                "activation capture apply to prompt positions only."
            )
            sched = getattr(
                getattr(self, "vllm_config", None), "scheduler_config", None
            )
            mnbt = getattr(sched, "max_num_batched_tokens", None)
            mml = getattr(sched, "max_model_len", None)
            if mnbt is not None and mml is not None and mnbt < mml:
                logger.warning(
                    "vllm-lens: chunked prefill can split a prompt (max_num_batched_tokens=%d "
                    "< max_model_len=%d). A 1-token final chunk is dispatched as a decode "
                    "graph where hooks do not run; raise max_num_batched_tokens above your "
                    "longest prompt (times concurrency) to guarantee prompt coverage.",
                    mnbt,
                    mml,
                )

        # Hooks must be installed on ALL ranks so steering vectors are
        # applied everywhere (not just rank 0).
        layers = _get_layers(self.model_runner.model)
        first = True
        for layer_idx, layer in enumerate(layers):
            if isinstance(layer, PPMissingLayer):
                continue
            if first:
                # with_kwargs: Qwen3.5/3.6 (Qwen3NextModel) pass hidden_states
                # by keyword, Qwen2/Llama positionally -- see _find_hidden_states_arg
                layer.register_forward_pre_hook(
                    _make_pre_hook(self, layer_idx), with_kwargs=True
                )
                first = False
            layer.register_forward_hook(_make_hook(self, layer_idx))

    def lens_capabilities(self) -> dict[str, Any]:
        """vllm-metamodels: what this engine supports (queried once by the plugin
        after ``install_hooks``): ``multi_stream`` (layer-output steering /
        capture undefined -> embedding stream only), ``prompt_only`` (CUDA
        graphs active), ``num_layers``."""
        return {
            "multi_stream": bool(self._multi_stream),
            "prompt_only": bool(self._prompt_only),
            "num_layers": len(_get_layers(self.model_runner.model)),
            "hooks_installed": bool(self._hooks_installed),
            "fast_capture": bool(self._fast_capture),
            "readout": True,
            "early_exit": bool(self._early_exit_ok),
            "early_exit_reason": self._early_exit_reason,
        }

    # ------------------------------------------------------------------
    # vllm-metamodels: early exit
    # ------------------------------------------------------------------

    def _early_exit_supported(self) -> tuple[bool, str]:
        """Early exit needs: PP == 1 (the placeholder replaces the whole model
        output), no prefix caching (skipped layers leave stale KV blocks that a
        later request could reuse), no aux-hidden-state (EAGLE-3) outputs, a
        generative (not pooling) model, and an overridable
        ``model_runner._model_forward``.  ``VLLM_LENS_EARLY_EXIT=0`` disables it."""
        if os.environ.get("VLLM_LENS_EARLY_EXIT", "1").strip().lower() not in _TRUTHY:
            return False, "disabled by VLLM_LENS_EARLY_EXIT"
        runner = self.model_runner
        if not callable(getattr(runner, "_model_forward", None)):
            return False, "model runner has no _model_forward to wrap"
        try:
            cfg = getattr(self, "vllm_config", None) or runner.vllm_config
        except Exception:  # pragma: no cover - defensive
            return False, "no vllm_config"
        if getattr(cfg.parallel_config, "pipeline_parallel_size", 1) != 1:
            return False, "pipeline parallelism > 1"
        if getattr(cfg.cache_config, "enable_prefix_caching", False):
            return False, "enable_prefix_caching=True (skipped layers would leave reusable garbage KV blocks)"
        if getattr(runner, "use_aux_hidden_state_outputs", False):
            return False, "aux hidden-state outputs (EAGLE-3) enabled"
        if getattr(runner, "is_pooling_model", False):
            return False, "pooling model"
        return True, "ok"

    def _wrap_model_forward(self) -> None:
        """Wrap ``model_runner._model_forward`` so an ``_EarlyExit`` raised by a
        layer hook becomes a normal return of the zero placeholder (the runner
        then computes logits from it and samples a meaningless token)."""
        runner = self.model_runner
        if getattr(runner, "_lens_early_exit_wrapped", False):
            return
        orig = runner._model_forward
        ext = self  # look the stats dict up at call time: steering_stats(reset=True) replaces it

        def _model_forward(*args: Any, **kwargs: Any) -> Any:
            try:
                return orig(*args, **kwargs)
            except _EarlyExit as e:
                ext._stats["early_exits"] += 1
                return e.placeholder

        runner._model_forward = _model_forward
        runner._lens_early_exit_wrapped = True

    def _cuda_graphs_active(self) -> bool:
        """True when decode batches may run as CUDA-graph replays (hooks silent)."""
        try:
            cfg = getattr(self, "vllm_config", None) or self.model_runner.vllm_config
            if cfg.model_config.enforce_eager:
                return False
            mode = cfg.compilation_config.cudagraph_mode
            return mode is not None and getattr(mode, "name", str(mode)) != "NONE"
        except Exception:  # pragma: no cover - defensive against config drift
            return False

    def _refresh_aggregates(self) -> None:
        """Per-``_steering_gen`` summary of all keys, for the idle fast path."""
        if self._agg_gen == self._steering_gen:
            return
        entries = list(self._steering_index.values())
        self._agg_broadcast = any(e.broadcast for e in entries)
        self._agg_max_pos = max((e.max_pos for e in entries), default=-1)
        self._agg_embed = any(EMBED_LAYER_INDEX in e.layers for e in entries)
        self._agg_gen = self._steering_gen

    # ------------------------------------------------------------------
    # Steering data management (called via collective_rpc)
    # ------------------------------------------------------------------

    def _prepare_vectors(self, sv_list: list[SteeringVector]) -> list[SteeringVector]:
        """Validate layer indices, move activations to the model device/dtype."""
        device = next(self.model_runner.model.parameters()).device
        dtype = next(self.model_runner.model.parameters()).dtype

        num_layers = len(_get_layers(self.model_runner.model))
        vectors: list[SteeringVector] = []

        for sv in sv_list:
            for idx in sv.layer_indices:
                if idx != EMBED_LAYER_INDEX and (idx < 0 or idx >= num_layers):
                    raise ValueError(
                        f"layer_index {idx} out of range [0, {num_layers}) "
                        f"(or EMBED_LAYER_INDEX={EMBED_LAYER_INDEX})"
                    )
                if idx != EMBED_LAYER_INDEX and self._multi_stream:
                    raise ValueError(_MULTI_STREAM_MSG.format(what=f"steering at layer {idx}"))
            if self._prompt_only and sv.activations.dim() == 2:
                raise ValueError(
                    "2-D (broadcast) steering vectors apply to generated positions, "
                    "which are computed inside replayed CUDA graphs where vllm-lens "
                    "hooks do not run. Either run with enforce_eager=True (unset "
                    "VLLM_LENS_CUDA_GRAPHS) or use 3-D position-specific vectors "
                    "on prompt positions."
                )

            vectors.append(
                sv.model_copy(
                    update={
                        "activations": sv.activations.to(device=device, dtype=dtype)
                    }
                )
            )
        return vectors

    def _store(self, key: str, vectors: list[SteeringVector]) -> None:
        self._steering_seq += 1
        self._steering_data[key] = vectors
        self._steering_index[key] = _index_configs(vectors, self._steering_seq)

    def set_steering_data(self, key: str, pickled_data: bytes) -> None:
        """Receive and store steering vectors for a request.

        Called via ``collective_rpc`` before generation begins.  Unpickles
        the list of ``SteeringVector`` instances, validates layer indices
        against the model, moves activation tensors to GPU in the model's
        dtype, and stores them keyed by *key* (an external request ID or a
        synthetic ``_steering_id``).
        """
        sv_list: list[SteeringVector] = pickle.loads(pickled_data)
        self._store(key, self._prepare_vectors(sv_list))
        self._steering_gen += 1

    def set_steering_data_many(self, pickled_data: bytes) -> int:
        """vllm-lens-metamodel: ``set_steering_data`` for MANY keys in one RPC.

        ``pickled_data`` is a pickled ``dict[str, list[SteeringVector]]``.
        Returns the number of keys stored.
        """
        payload: dict[str, list[SteeringVector]] = pickle.loads(pickled_data)
        for key, sv_list in payload.items():
            self._store(key, self._prepare_vectors(sv_list))
        self._steering_gen += 1
        return len(payload)

    def set_steering_block(self, pickled_data: bytes) -> int:
        """vllm-lens-metamodel: one single-position vector for MANY keys from ONE tensor.

        ``pickled_data`` is a pickled dict::

            {"keys": [str], "vecs": Tensor(n, hidden), "layers": [int],
             "positions": [int], "scales": [float], "norm_match": [bool],
             "modes": [str]}          # optional, default "add"

        Key ``i`` behaves exactly like ``set_steering_data(key_i, [SteeringVector(
        activations=vecs[i].view(1, 1, hidden), layer_indices=[layers[i]],
        scale=scales[i], norm_match=norm_match[i], position_indices=[positions[i]],
        mode=modes[i])])`` but the whole block is moved to the model device/dtype
        in one copy and each entry's activations are a view into it.  ``layers``
        may contain ``EMBED_LAYER_INDEX``.
        """
        d = pickle.loads(pickled_data)
        keys: list[str] = list(d["keys"])
        vecs: torch.Tensor = d["vecs"]
        if vecs.dim() != 2 or vecs.shape[0] != len(keys):
            raise ValueError(f"vecs must be (n_keys, hidden), got {tuple(vecs.shape)}")
        device = next(self.model_runner.model.parameters()).device
        dtype = next(self.model_runner.model.parameters()).dtype
        num_layers = len(_get_layers(self.model_runner.model))
        layers = [int(x) for x in d["layers"]]
        for idx in layers:
            if idx != EMBED_LAYER_INDEX and (idx < 0 or idx >= num_layers):
                raise ValueError(
                    f"layer_index {idx} out of range [0, {num_layers}) "
                    f"(or EMBED_LAYER_INDEX={EMBED_LAYER_INDEX})"
                )
            if idx != EMBED_LAYER_INDEX and self._multi_stream:
                raise ValueError(_MULTI_STREAM_MSG.format(what=f"steering at layer {idx}"))
        positions = [int(x) for x in d["positions"]]
        scales = [float(x) for x in d["scales"]]
        nms = [bool(x) for x in d["norm_match"]]
        modes = [str(x) for x in d.get("modes", ["add"] * len(keys))]
        if len(modes) != len(keys) or any(m not in ("add", "replace") for m in modes):
            raise ValueError(f"modes must be n_keys entries of 'add'|'replace', got {modes[:5]}")
        block = vecs.to(device=device, dtype=dtype).contiguous()
        for i, key in enumerate(keys):
            sv = SteeringVector.model_construct(
                activations=block[i : i + 1].unsqueeze(0),  # (1, 1, hidden) view
                layer_indices=[layers[i]],
                scale=scales[i],
                norm_match=nms[i],
                position_indices=[positions[i]],
                mode=modes[i],
            )
            self._store(key, [sv])
        self._steering_gen += 1
        return len(keys)

    def clear_steering_data(self, key: str) -> None:
        """Remove steering data for a completed request."""
        self._steering_data.pop(key, None)
        self._steering_index.pop(key, None)
        self._steering_gen += 1

    def clear_steering_data_many(self, keys: list[str]) -> None:
        """vllm-lens-metamodel: ``clear_steering_data`` for many keys in one RPC."""
        for key in keys:
            self._steering_data.pop(key, None)
            self._steering_index.pop(key, None)
        self._steering_gen += 1

    def set_fast_capture(self, enabled: bool) -> bool:
        """vllm-metamodels: toggle the gather + pinned-copy capture path (default on,
        ``VLLM_LENS_FAST_CAPTURE``); off = 1.1.0's per-request ``.cpu()`` slices."""
        self._flush_host_blocks()
        self._fast_capture = bool(enabled)
        self._req_plan_cache.clear()
        return self._fast_capture

    def set_vectorized(self, enabled: bool) -> bool:
        """vllm-lens-metamodel: toggle the vectorised apply (default on, ``VLLM_LENS_VECTORIZED``)."""
        self._vectorized = bool(enabled)
        return self._vectorized

    def steering_stats(self, reset: bool = False) -> dict[str, int]:
        """vllm-lens-metamodel: hook counters (passes skipped by the idle fast path,
        passes planned / planned-but-idle, layer-steps steered / vectorised,
        rows steered / replaced, embedding-injection passes and failures, rows
        skipped as generated under CUDA graphs, errors)."""
        out = dict(self._stats)
        if reset:
            self._stats = _new_stats()
        return out

    # ------------------------------------------------------------------
    # vllm-metamodels: readout vectors (called via collective_rpc)
    # ------------------------------------------------------------------

    def _check_layer(self, idx: int, num_layers: int, what: str) -> None:
        if idx != EMBED_LAYER_INDEX and (idx < 0 or idx >= num_layers):
            raise ValueError(
                f"layer_index {idx} out of range [0, {num_layers}) (or EMBED_LAYER_INDEX={EMBED_LAYER_INDEX})"
            )
        if idx != EMBED_LAYER_INDEX and self._multi_stream:
            raise ValueError(_MULTI_STREAM_MSG.format(what=f"{what} at layer {idx}"))

    def _store_reads(self, payload: dict[str, list[ReadoutVector]]) -> int:
        """Stack every direction of ``payload`` into ONE float32 device block and
        register per-key ``_ReadEntry`` views into it."""
        device = next(self.model_runner.model.parameters()).device
        num_layers = len(_get_layers(self.model_runner.model))
        rows: list[torch.Tensor] = []
        pending: list[tuple[str, int, dict[int, int], tuple[str, Any], bool, float, frozenset[int]]] = []
        for key, rvs in payload.items():
            for seq, rv in enumerate(rvs):
                if rv.activations.dim() != 2 or rv.activations.shape[0] != len(rv.layer_indices):
                    raise ValueError(f"readout activations must be (n_layers, hidden) matching layer_indices, got {tuple(rv.activations.shape)}")
                layer_rows: dict[int, int] = {}
                for li, layer in enumerate(rv.layer_indices):
                    self._check_layer(int(layer), num_layers, "readout")
                    layer_rows[int(layer)] = len(rows)
                    rows.append(rv.activations[li].detach().float().reshape(-1))
                pending.append((key, seq, layer_rows, normalize_positions(rv.positions), rv.metric == "cos", float(rv.bias), frozenset(layer_rows)))
        if not rows:
            return 0
        block = torch.stack(rows).to(device=device, dtype=torch.float32).contiguous()
        for key, seq, layer_rows, spec, use_cos, bias, layers in pending:
            entry = _ReadEntry(key, seq, block, layer_rows, spec, use_cos, bias, layers)
            if seq == 0:
                self._readout_index[key] = [entry]
            else:
                self._readout_index[key].append(entry)
        self._steering_gen += 1  # invalidates cached request plans
        return len(payload)

    def set_readout_data(self, key: str, pickled_data: bytes) -> int:
        """Register ``list[ReadoutVector]`` for one key (external request id or ``_readout_id``)."""
        return self._store_reads({key: pickle.loads(pickled_data)})

    def set_readout_data_many(self, pickled_data: bytes) -> int:
        """``set_readout_data`` for many keys in one RPC (pickled ``dict[str, list[ReadoutVector]]``)."""
        return self._store_reads(pickle.loads(pickled_data))

    def set_readout_block(self, pickled_data: bytes) -> int:
        """One single-layer direction per key from ONE tensor::

            {"keys": [str], "vecs": Tensor(n, hidden), "layers": [int],
             "positions": [spec], "metric": [str], "bias": [float]}

        Key ``i`` behaves like ``ReadoutVector(activations=vecs[i].view(1, hidden),
        layer_indices=[layers[i]], positions=positions[i], metric=metric[i], bias=bias[i])``.
        """
        d = pickle.loads(pickled_data)
        keys: list[str] = list(d["keys"])
        vecs: torch.Tensor = d["vecs"]
        if vecs.dim() != 2 or vecs.shape[0] != len(keys):
            raise ValueError(f"vecs must be (n_keys, hidden), got {tuple(vecs.shape)}")
        device = next(self.model_runner.model.parameters()).device
        num_layers = len(_get_layers(self.model_runner.model))
        layers = [int(x) for x in d["layers"]]
        for idx in layers:
            self._check_layer(idx, num_layers, "readout")
        specs = [normalize_positions(p) for p in d.get("positions", ["all"] * len(keys))]
        metrics = [str(m) for m in d.get("metric", ["cos"] * len(keys))]
        if any(m not in ("cos", "dot") for m in metrics):
            raise ValueError(f"metric must be 'cos'|'dot', got {metrics[:5]}")
        biases = [float(b) for b in d.get("bias", [0.0] * len(keys))]
        block = vecs.to(device=device, dtype=torch.float32).contiguous()
        for i, key in enumerate(keys):
            self._readout_index[key] = [
                _ReadEntry(key, 0, block, {layers[i]: i}, specs[i], metrics[i] == "cos", biases[i], frozenset([layers[i]]))
            ]
        self._steering_gen += 1
        return len(keys)

    def clear_readout_data(self, key: str) -> None:
        self._readout_index.pop(key, None)
        self._steering_gen += 1

    def clear_readout_data_many(self, keys: list[str]) -> None:
        for key in keys:
            self._readout_index.pop(key, None)
        self._steering_gen += 1

    # ------------------------------------------------------------------
    # vllm-metamodels: host blocks -> per-request results
    # ------------------------------------------------------------------

    def _flush_host_blocks(self) -> None:
        """Wait for pending device->host copies and split them per request."""
        for blk in self._cap_blocks:
            if blk.event is not None:
                blk.event.synchronize()
            off = 0
            for req_id, n, pos in blk.segments:
                self._captured_states.setdefault(req_id, {}).setdefault(blk.layer, []).append(blk.host[off : off + n])
                self._captured_positions.setdefault(req_id, {}).setdefault(blk.layer, []).append(pos)
                off += n
        self._cap_blocks.clear()
        for blk in self._read_blocks:
            if blk.event is not None:
                blk.event.synchronize()
            off = 0
            for req_id, n, pos, seq in blk.segments:
                self._readouts.setdefault(req_id, {}).setdefault((seq, blk.layer), []).append((pos, blk.host[off : off + n]))
                off += n
        self._read_blocks.clear()

    def _pop_activations(self, req_id: str) -> dict[str, Any]:
        layer_dict = self._captured_states.pop(req_id)
        pos_dict = self._captured_positions.pop(req_id, None)
        sorted_indices = sorted(layer_dict.keys())
        per_layer = [torch.cat(layer_dict[idx], dim=0) for idx in sorted_indices]
        acts: dict[str, Any] = {"residual_stream": torch.stack(per_layer, dim=0)}
        if pos_dict and sorted_indices[0] in pos_dict:
            acts["positions"] = [int(p) for p in np.concatenate(pos_dict[sorted_indices[0]])]
        return acts

    def _pop_readouts(self, req_id: str) -> list[dict[str, Any]]:
        per = self._readouts.pop(req_id)
        out: list[dict[str, Any]] = []
        for seq in sorted({s for s, _ in per}):
            layers = sorted(l for s, l in per if s == seq)
            vals = torch.stack([torch.cat([v for _, v in per[(seq, l)]]) for l in layers])
            positions = [int(p) for p in np.concatenate([p for p, _ in per[(seq, layers[0])]])]
            out.append({"values": vals, "positions": positions, "layers": layers})
        return out

    @staticmethod
    def _by_external(internal_ids: list[str], external_ids: list[str]) -> dict[str, list[str]]:
        """external id -> internal ids with the ``"{external}-"`` prefix (vLLM appends
        ``-{8 hex}``); one dict pass instead of a scan per external id."""
        wanted = set(external_ids)
        out: dict[str, list[str]] = {}
        for rid in internal_ids:
            ext = rid.rsplit("-", 1)[0]
            if ext in wanted:
                out.setdefault(ext, []).append(rid)
            else:  # unusual suffixes: exact prefix semantics as a fallback
                for e in wanted:
                    if rid.startswith(f"{e}-"):
                        out.setdefault(e, []).append(rid)
                        break
        return out

    def get_captured_states_many(self, external_req_ids: list[str]) -> bytes:
        """vllm-metamodels: ``get_captured_states`` for every request of a
        ``generate()`` call in ONE RPC.  Returns a pickled ``{external_id:
        {"residual_stream": Tensor(n_layers, n_pos, hidden), "positions": [int]}}``
        (uncompressed: activations do not compress; ``positions`` present on the
        fast path).  Removes the requests' data."""
        t0 = time.perf_counter()
        self._flush_host_blocks()
        by_ext = self._by_external(list(self._captured_states), external_req_ids)
        out: dict[str, Any] = {}
        for ext, rids in by_ext.items():
            out[ext] = self._pop_activations(rids[0])
            for extra in rids[1:]:
                self._captured_states.pop(extra, None)
                self._captured_positions.pop(extra, None)
        blob = pickle.dumps(out, protocol=pickle.HIGHEST_PROTOCOL)
        self._stats["retrieval_s"] += time.perf_counter() - t0
        return blob

    def get_readouts(self, external_req_id: str) -> bytes | None:
        """Readout results of one request (async path): pickled list, see ``_pop_readouts``."""
        self._flush_host_blocks()
        prefix = f"{external_req_id}-"
        for rid in list(self._readouts):
            if rid.startswith(prefix):
                return pickle.dumps(self._pop_readouts(rid), protocol=pickle.HIGHEST_PROTOCOL)
        return None

    def get_readouts_many(self, external_req_ids: list[str]) -> bytes:
        """Readout results for many requests in ONE RPC: pickled ``{external_id: [
        {"values": Tensor(n_layers, n_pos) float32, "positions": [int], "layers": [int]}
        per ReadoutVector]}``."""
        t0 = time.perf_counter()
        self._flush_host_blocks()
        by_ext = self._by_external(list(self._readouts), external_req_ids)
        out = {ext: self._pop_readouts(rids[0]) for ext, rids in by_ext.items()}
        for rids in by_ext.values():
            for extra in rids[1:]:
                self._readouts.pop(extra, None)
        blob = pickle.dumps(out, protocol=pickle.HIGHEST_PROTOCOL)
        self._stats["retrieval_s"] += time.perf_counter() - t0
        return blob

    def clear_readouts(self, external_req_id: str) -> None:
        self._flush_host_blocks()
        prefix = f"{external_req_id}-"
        for rid in list(self._readouts):
            if rid.startswith(prefix):
                del self._readouts[rid]

    def clear_captured_states(self, external_req_id: str) -> None:
        """Remove captured activations without returning them.

        Called in the ``finally`` block of ``_patched_generate`` to clean
        up leaked state when a request is aborted or the client disconnects
        before ``get_captured_states`` is called.  On normal completion this
        is a no-op because ``get_captured_states`` already ``.pop()``-ed
        the entry.
        """
        self._flush_host_blocks()
        prefix = f"{external_req_id}-"
        for req_id in list(self._captured_states):
            if req_id.startswith(prefix):
                del self._captured_states[req_id]
                self._captured_positions.pop(req_id, None)
                logger.debug("Cleared leaked activations for %s", req_id)

    def get_captured_states(self, external_req_id: str) -> bytes | None:
        """Retrieve captured activations for a specific request.

        Matches by ``"{external_req_id}-"`` prefix because vLLM internally
        transforms the user-provided ``request_id`` into
        ``"{request_id}-{random_suffix}"``. So ``"req-0"`` matches
        ``"req-0-a1b2c3d4"`` but NOT ``"req-00-b5c6d7e8"``.

        Moves tensors to CPU and serializes via pickle for safe ZMQ
        transport.

        Returns a dict when deserialized::

            {
                "activations": {
                    "residual_stream": Tensor,  # (n_layers, total_pos, d_model)
                }
            }

        Layers are stacked in ascending order along dim 0.
        Removes the request's data after retrieval.
        """
        self._flush_host_blocks()
        prefix = f"{external_req_id}-"
        for req_id in list(self._captured_states):
            if req_id.startswith(prefix):
                return _ZSTD_COMPRESSOR.compress(
                    pickle.dumps({"activations": self._pop_activations(req_id)})
                )
        return None

    def _debug_captured_states_count(self) -> int:
        """Return the number of entries in _captured_states (for testing)."""
        return len(self._captured_states)
