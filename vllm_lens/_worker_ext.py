"""
Worker extension that captures residual-stream activations from
configurable layers during transformer forward passes, and optionally
applies steering vectors (activation additions) to modify the residual
stream in-flight.

Uses PyTorch forward hooks on each decoder layer for concurrency-safe,
per-request activation capture and steering.  Each hook checks the
request's ``extra_args["output_residual_stream"]`` to decide whether to
capture, and reads from ``_steering_data`` to apply any steering vectors.

vllm-lens-port: the hook resolves each request's steering configs ONCE
(indexed lookups instead of a per-layer ``startswith`` scan over every
key), plans each forward pass once from host-side buffers (no device
syncs), skips passes / layers with nothing to do, applies a layer's
vectors with one ``index_add_``, and can run with CUDA graphs for decode
(``VLLM_LENS_CUDA_GRAPHS=1``, prompt-position steering only).  The
steering arithmetic (``norm_match``, ``_apply_steering``) is unchanged.
"""

from __future__ import annotations

import logging
import os
import pickle
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import zstandard as zstd
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.models.utils import PPMissingLayer

from vllm_lens._helpers.types import SteeringVector

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
# vllm-lens-port: indexed per-request steering
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


def _index_configs(configs: list[SteeringVector], seq: int) -> _SteerEntry:
    layers: set[int] = set()
    broadcast = False
    min_pos, max_pos = _NO_POS, -1
    for cfg in configs:
        layers.update(cfg.layer_indices)
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
    return _SteerEntry(configs, frozenset(layers), broadcast, min_pos, max_pos, seq)


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
    q = meta.query_start_loc
    if isinstance(q, torch.Tensor):
        return [int(x) for x in q[: num_reqs + 1].tolist()]
    return [int(x) for x in q[: num_reqs + 1]]


def _host_abs_start(runner: Any, meta: Any, qsl: list[int], num_reqs: int) -> list[int]:
    """1.1.0's ``seq_lens[i] - n_query`` per row, from host memory when possible."""
    nct = getattr(getattr(runner, "input_batch", None), "num_computed_tokens_cpu", None)
    if nct is not None:
        return [int(x) for x in nct[:num_reqs]]
    seq_lens: Any = getattr(meta, "seq_lens", None)
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
    broadcast = False
    min_pos, max_pos = _NO_POS, -1
    for e in entries:
        configs.extend(e.configs)
        layers.update(e.layers)
        broadcast |= e.broadcast
        min_pos, max_pos = min(min_pos, e.min_pos), max(max_pos, e.max_pos)
    cap = extra.get("output_residual_stream") if extra else None
    if cap is not None:
        extension._capture_live.add(req_id)
    num_prompt = getattr(req_state, "num_prompt_tokens", None)
    if num_prompt is None:
        num_prompt = len(req_state.prompt_token_ids or ())
    return _ReqPlan(
        gen=gen,
        configs=configs,
        layers=frozenset(layers),
        broadcast=broadcast,
        min_pos=min_pos,
        max_pos=max_pos,
        cap_any=cap is not None,
        cap_set=frozenset(cap) if isinstance(cap, list) else None,
        num_prompt=int(num_prompt),
    )


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
    # query_start_loc.  Find one that has it.
    meta = None
    for _meta in attn_metadata.values():
        if hasattr(_meta, "query_start_loc"):
            meta = _meta
            break
    if meta is None:
        logger.warning(
            "No attention metadata with query_start_loc found "
            "(keys: %s). Skipping hook for this step.",
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

    steer: dict[int, list[tuple[int, list[SteeringVector]]]] = {}
    cap_all: list[int] = []
    cap_by_layer: dict[int, list[int]] = {}
    for i in range(num_reqs):
        req_id = req_ids[i]
        plan = cache.get(req_id)
        if plan is None or plan.gen != gen:
            plan = _resolve_request(extension, runner, req_id, gen)
            if plan is None:
                continue
            cache[req_id] = plan
        if not plan.configs and not plan.cap_any:
            continue
        a0 = abs_start[i]
        if prompt_only and a0 >= plan.num_prompt:
            # Generated positions run inside replayed CUDA graphs (no hooks);
            # never touch them so behaviour does not depend on batch mix.
            stats["rows_skipped_generated"] += 1
            continue
        if plan.configs and (
            plan.broadcast
            or (plan.max_pos >= a0 and plan.min_pos < a0 + (qsl[i + 1] - qsl[i]))
        ):
            for layer_idx in plan.layers:
                steer.setdefault(layer_idx, []).append((i, plan.configs))
        if plan.cap_any and should_capture:
            if plan.cap_set is None:
                cap_all.append(i)
            else:
                for layer_idx in plan.cap_set:
                    cap_by_layer.setdefault(layer_idx, []).append(i)

    if len(cache) > num_reqs + _REQ_CACHE_SLACK:
        live = runner.requests
        for stale in [k for k in cache if k not in live]:
            del cache[stale]
        extension._capture_live = {r for r in extension._capture_live if r in live}

    stats["steps_planned"] += 1
    if not steer and not cap_all and not cap_by_layer:
        stats["steps_idle"] += 1
    return _StepPlan(qsl, abs_start, steer, cap_all, cap_by_layer, id(ctx))


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
) -> None:
    """Apply all matching steering vectors to a token slice *in-place*.

    ``target`` is the (already-cloned) output tensor.  ``start``/``end``
    are batch-relative indices, ``abs_start`` is the absolute sequence
    position of the first token in ``target[start:end]``.
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
                v = norm_match(target[start:end], v)
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
                    v = norm_match(target[rel], v)
                target[rel] = target[rel] + v * cfg.scale


def _apply_layer_vectorized(
    todo: list[tuple[int, list[SteeringVector]]],
    layer_idx: int,
    target: torch.Tensor,
    plan: _StepPlan,
) -> bool:
    """vllm-lens-port: apply every (row, vector) pair of this layer/pass at once.

    Gathers the vectors ``_apply_steering`` would add into one ``[n, hidden]``
    tensor and adds them with a single ``index_add_`` (norm-matching in the
    same batched op).  Returns False without touching ``target`` when the
    batch is not vectorisable with identical semantics -- a row would receive
    several vectors, or a broadcast vector covers a multi-token chunk -- so
    the caller runs ``_apply_steering`` row by row.  With ``norm_match=False``
    the result is bit-identical to the sequential path (one multiply and one
    add per element, same dtype); with ``norm_match=True`` the per-row norms
    come from a batched reduction and may differ by float32 rounding.
    """
    qsl, abs_start = plan.qsl, plan.abs_start
    rows: list[int] = []
    vecs: list[torch.Tensor] = []
    scales: list[float] = []
    nms: list[bool] = []
    limit = _VEC_MAX_ENTRIES_PER_ROW * len(todo) + 64
    for i, configs in todo:
        start, end, a0 = qsl[i], qsl[i + 1], abs_start[i]
        n_tokens = end - start
        for cfg in configs:
            layer_index_map = cfg.layer_index_map
            if layer_idx not in layer_index_map:
                continue
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
                continue
            scales.append(cfg.scale)
            nms.append(cfg.norm_match)
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
        r_norm = target.index_select(0, idx).float().norm(dim=-1, keepdim=True)
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
    target.index_add_(0, idx, v)
    return True


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
    if not todo and not cap_rows:
        return None
    query_start_loc = plan.qsl

    # --- Phase 2: apply steering ------------------------------------
    modified_output: torch.Tensor | tuple[torch.Tensor, ...] | None = None
    if todo:
        if isinstance(output, tuple):
            modified_output = (output[0].clone(), *output[1:])
            target = modified_output[0]
        else:
            modified_output = output.clone()
            target = modified_output

        extension._stats["steer_layer_steps"] += 1
        extension._stats["rows_steered"] += len(todo)
        if extension._vectorized and _apply_layer_vectorized(
            todo, layer_idx, target, plan
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
                )

    # --- Phase 3: capture activations (rank 0 only) -----------------
    if cap_rows:
        capture_src = modified_output if modified_output is not None else output
        hidden_states: Float[torch.Tensor, "total_tokens hidden_dim"]  # type: ignore[reportUndefinedVariable]
        if isinstance(capture_src, tuple):
            if capture_src[1] is not None:
                hidden_states = capture_src[0] + capture_src[1]
            else:
                hidden_states = capture_src[0]
        else:
            hidden_states = capture_src

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

    return modified_output


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
        except Exception:
            extension._stats["errors"] += 1
            logger.warning(
                "vllm-lens hook error on layer %d, skipping", layer_idx, exc_info=True
            )
            return None

    return hook


def _make_pre_hook(extension: HiddenStatesExtension) -> Callable:
    """vllm-lens-port: pre-hook on this rank's first decoder layer.

    A new forward pass begins: drop the previous pass's plan and decide
    whether this pass is *idle* (``_step_is_idle``), in which case every
    layer hook returns on a single flag check and the pass costs nothing
    else.
    """

    def pre_hook(_module: torch.nn.Module, _input: object) -> None:
        extension._step_plan = None
        extension._step_idle = False
        try:
            if not is_forward_context_available():
                return
            runner = extension.model_runner
            num_reqs = runner.input_batch.num_reqs
            if num_reqs and _step_is_idle(extension, runner, num_reqs):
                extension._step_idle = True
                extension._stats["steps_fast_idle"] += 1
        except Exception:
            extension._stats["errors"] += 1
            logger.warning("vllm-lens pre-hook error, running full path", exc_info=True)
            extension._step_idle = False

    return pre_hook


def _new_stats() -> dict[str, int]:
    return {
        "steps_fast_idle": 0,
        "steps_planned": 0,
        "steps_idle": 0,
        "steer_layer_steps": 0,
        "vectorized_layer_steps": 0,
        "rows_steered": 0,
        "rows_skipped_generated": 0,
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

    # vllm-lens-port: index over _steering_data + per-pass state.
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
    _stats: dict[str, int] = _new_stats()
    _vectorized: bool = True
    _prompt_only: bool = (
        False  # CUDA graphs active for decode: hooks only see prompt rows
    )

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

        # Only rank 0 captures — residual streams are replicated across
        # TP ranks after all-reduce, so the data is identical.
        tp_size = self.parallel_config.tensor_parallel_size
        self._should_capture = tp_size <= 1 or self.rank % tp_size == 0

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
                layer.register_forward_pre_hook(_make_pre_hook(self))
                first = False
            layer.register_forward_hook(_make_hook(self, layer_idx))

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
        entries = self._steering_index.values()
        self._agg_broadcast = any(e.broadcast for e in entries)
        self._agg_max_pos = max((e.max_pos for e in entries), default=-1)
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
                if idx < 0 or idx >= num_layers:
                    raise ValueError(
                        f"layer_index {idx} out of range [0, {num_layers})"
                    )
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
        """vllm-lens-port: ``set_steering_data`` for MANY keys in one RPC.

        ``pickled_data`` is a pickled ``dict[str, list[SteeringVector]]``.
        Returns the number of keys stored.
        """
        payload: dict[str, list[SteeringVector]] = pickle.loads(pickled_data)
        for key, sv_list in payload.items():
            self._store(key, self._prepare_vectors(sv_list))
        self._steering_gen += 1
        return len(payload)

    def set_steering_block(self, pickled_data: bytes) -> int:
        """vllm-lens-port: one single-position vector for MANY keys from ONE tensor.

        ``pickled_data`` is a pickled dict::

            {"keys": [str], "vecs": Tensor(n, hidden), "layers": [int],
             "positions": [int], "scales": [float], "norm_match": [bool]}

        Key ``i`` behaves exactly like ``set_steering_data(key_i, [SteeringVector(
        activations=vecs[i].view(1, 1, hidden), layer_indices=[layers[i]],
        scale=scales[i], norm_match=norm_match[i], position_indices=[positions[i]])])``
        but the whole block is moved to the model device/dtype in one copy and
        each entry's activations are a view into it.
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
            if idx < 0 or idx >= num_layers:
                raise ValueError(f"layer_index {idx} out of range [0, {num_layers})")
        positions = [int(x) for x in d["positions"]]
        scales = [float(x) for x in d["scales"]]
        nms = [bool(x) for x in d["norm_match"]]
        block = vecs.to(device=device, dtype=dtype).contiguous()
        for i, key in enumerate(keys):
            sv = SteeringVector.model_construct(
                activations=block[i : i + 1].unsqueeze(0),  # (1, 1, hidden) view
                layer_indices=[layers[i]],
                scale=scales[i],
                norm_match=nms[i],
                position_indices=[positions[i]],
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
        """vllm-lens-port: ``clear_steering_data`` for many keys in one RPC."""
        for key in keys:
            self._steering_data.pop(key, None)
            self._steering_index.pop(key, None)
        self._steering_gen += 1

    def set_vectorized(self, enabled: bool) -> bool:
        """vllm-lens-port: toggle the vectorised apply (default on, ``VLLM_LENS_VECTORIZED``)."""
        self._vectorized = bool(enabled)
        return self._vectorized

    def steering_stats(self, reset: bool = False) -> dict[str, int]:
        """vllm-lens-port: hook counters (passes skipped by the idle fast path,
        passes planned / planned-but-idle, layer-steps steered / vectorised,
        rows steered, rows skipped as generated under CUDA graphs, errors)."""
        out = dict(self._stats)
        if reset:
            self._stats = _new_stats()
        return out

    def clear_captured_states(self, external_req_id: str) -> None:
        """Remove captured activations without returning them.

        Called in the ``finally`` block of ``_patched_generate`` to clean
        up leaked state when a request is aborted or the client disconnects
        before ``get_captured_states`` is called.  On normal completion this
        is a no-op because ``get_captured_states`` already ``.pop()``-ed
        the entry.
        """
        prefix = f"{external_req_id}-"
        for req_id in list(self._captured_states):
            if req_id.startswith(prefix):
                del self._captured_states[req_id]
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
        prefix = f"{external_req_id}-"
        for req_id in list(self._captured_states):
            if req_id.startswith(prefix):
                layer_dict = self._captured_states.pop(req_id)
                sorted_indices = sorted(layer_dict.keys())
                per_layer: list[Float[torch.Tensor, "total_pos hidden_dim"]] = [  # type: ignore[reportUndefinedVariable]
                    torch.cat(layer_dict[idx], dim=0) for idx in sorted_indices
                ]
                stacked: Float[torch.Tensor, "n_layers total_pos hidden_dim"] = (  # type: ignore[reportUndefinedVariable]
                    torch.stack(per_layer, dim=0)
                )
                return _ZSTD_COMPRESSOR.compress(
                    pickle.dumps(
                        {
                            "activations": {"residual_stream": stacked},
                        }
                    )
                )
        return None

    def _debug_captured_states_count(self) -> int:
        """Return the number of entries in _captured_states (for testing)."""
        return len(self._captured_states)
