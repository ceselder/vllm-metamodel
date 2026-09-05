"""
Worker extension that captures residual-stream activations from
configurable layers during transformer forward passes, and optionally
applies steering vectors (activation additions) to modify the residual
stream in-flight.

Uses PyTorch forward hooks on each decoder layer for concurrency-safe,
per-request activation capture and steering.  Each hook checks the
request's ``extra_args["output_residual_stream"]`` to decide whether to
capture, and reads from ``_steering_data`` to apply any steering vectors.
"""

from __future__ import annotations

import json
import logging
import pickle
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cloudpickle
import torch
import zstandard as zstd
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.models.utils import PPMissingLayer

from vllm_lens._helpers.types import Hook, HookContext, SteeringVector

if TYPE_CHECKING:
    from jaxtyping import Float
    from vllm.config import ParallelConfig

logger = logging.getLogger(__name__)

_DTYPE_LIST = [
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.int64,
    torch.int32,
    torch.int16,
    torch.int8,
    torch.float64,
]
_DTYPE_TO_IDX_MAP = {d: i for i, d in enumerate(_DTYPE_LIST)}


def _dtype_to_idx(dtype: torch.dtype) -> int:
    return _DTYPE_TO_IDX_MAP.get(dtype, 0)


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


def _prefix_keys(internal_req_id: str) -> Iterator[str]:
    """Every ``k`` with ``internal_req_id.startswith(k + "-")``.

    vLLM turns an external request id into ``"{external_id}-{8 hex}"``, so the
    ``startswith`` scan over all registered keys matched exactly the keys that
    end right before a ``"-"`` in the internal id.  Enumerating those prefixes
    (a handful of dict lookups) is equivalent and independent of how many keys
    are registered -- the scan was O(keys) per request per layer per step.
    """
    i = internal_req_id.find("-")
    while i != -1:
        yield internal_req_id[:i]
        i = internal_req_id.find("-", i + 1)


def _lookup_keyed(
    store: dict[str, list],
    seq: dict[str, int],
    internal_req_id: str,
    sentinel: str | None,
) -> list:
    """Dict-lookup equivalent of the ``startswith`` scan over ``store`` (matches in
    insertion order, like iterating the dict) followed by the ``sentinel`` key."""
    found = [k for k in _prefix_keys(internal_req_id) if k in store]
    if len(found) > 1:
        found.sort(key=lambda k: seq.get(k, 0))
    results: list = []
    for k in found:
        results.extend(store[k])
    if sentinel and sentinel in store:
        results.extend(store[sentinel])
    return results


def _find_steering_configs(
    extension: HiddenStatesExtension,
    internal_req_id: str,
    extra_args: dict[str, Any] | None,
) -> list[SteeringVector]:
    """Find all steering configs that apply to an internal request ID.

    Matches by ``"{external_id}-"`` prefix (async path: vLLM appends
    ``"-{random_suffix}"`` to external IDs) and by ``_steering_id``
    sentinel in ``extra_args`` (offline path).  Indexed: dict lookups on the
    id's ``-``-boundary prefixes instead of a scan over every key.
    """
    sentinel = extra_args.get("_steering_id") if extra_args else None
    return _lookup_keyed(extension._steering_data, extension._steering_seq, internal_req_id, sentinel)


def _find_hook_configs(
    extension: HiddenStatesExtension,
    internal_req_id: str,
    extra_args: dict[str, Any] | None,
) -> list[Hook]:
    """Find all hook definitions that apply to an internal request ID.

    Checks three sources (in order):
    1. Per-request hooks keyed by external ID prefix (async path).
    2. Per-request hooks keyed by ``_hook_id`` sentinel (offline path).
    3. Persistent hooks (apply to every request).
    """
    results = _find_hook_configs_no_persistent(extension, internal_req_id, extra_args)
    results.extend(extension._persistent_hooks)
    return results


def _find_hook_configs_no_persistent(
    extension: HiddenStatesExtension,
    internal_req_id: str,
    extra_args: dict[str, Any] | None,
) -> list[Hook]:
    """Find per-request hook definitions only (excludes persistent hooks)."""
    sentinel = extra_args.get("_hook_id") if extra_args else None
    return _lookup_keyed(extension._hook_data, extension._hook_seq, internal_req_id, sentinel)


# ---------------------------------------------------------------------------
# Per-request resolution cache + one plan per forward pass
#
# Before: every layer hook, on every forward pass, re-resolved every request's
# steering vectors and hooks (two scans over all registered keys) and paid two
# device syncs (``query_start_loc[i].item()``) per request -- O(layers x
# requests x keys) Python + O(layers x requests) syncs per decode step.
# Now: each request is resolved once (cached until any steering/hook data
# changes), each pass is planned once from the runner's HOST buffers, and a
# pre-hook on the first layer flags passes on which no hook can have work.
# ---------------------------------------------------------------------------

_REQ_CACHE_SLACK = 4096
_NO_POS = 1 << 62
_TRUTHY = ("1", "true", "yes", "on")


@dataclass(slots=True)
class _ReqPlan:
    """What one request wants from the hooks, resolved once and cached."""

    gen: int
    steering: list[SteeringVector]
    steer_layers: frozenset[int]
    broadcast: bool
    """Some vector is 2-D (every position, generated ones too)."""
    min_pos: int
    max_pos: int
    hooks: list[Hook]
    """Per-request hooks (persistent hooks are global)."""
    capture: Any
    """``None`` | ``True`` (all layers) | ``frozenset`` of layers."""
    num_prompt: int


@dataclass(slots=True)
class _StepPlan:
    """Everything the hooks of ONE forward pass need (built by the first hook that
    runs in the pass; dropped by the first layer's pre-hook)."""

    ctx_id: int
    qsl: list[int]
    """``query_start_loc`` as host ints."""
    abs_start: list[int]
    """Absolute position of each row's first token."""
    plans: list[_ReqPlan | None]
    prompt_only: bool

    def active(self, i: int) -> bool:
        """Under decode-only CUDA graphs generated positions run inside replayed
        graphs; never touch them eagerly either, so results do not depend on
        batch composition."""
        rp = self.plans[i]
        return rp is not None and not (self.prompt_only and self.abs_start[i] >= rp.num_prompt)


def _parse_capture(extra: dict[str, Any] | None) -> Any:
    if not extra:
        return None
    ors = extra.get("output_residual_stream")
    if ors is None:
        return None
    if isinstance(ors, str):  # vllm_xargs passes values as strings; parse JSON lists
        try:
            ors = json.loads(ors)
        except (json.JSONDecodeError, ValueError):
            return True
    if isinstance(ors, list):
        return frozenset(int(x) for x in ors)
    return True if ors else None


def _resolve_request(extension: HiddenStatesExtension, runner: Any, req_id: str) -> _ReqPlan | None:
    cache = extension._req_plan_cache
    gen = extension._gen
    plan = cache.get(req_id)
    if plan is not None and plan.gen == gen:
        return plan
    req_state = runner.requests.get(req_id)
    if req_state is None:
        return None
    sp = req_state.sampling_params
    extra = sp.extra_args if sp is not None else None
    steering = _find_steering_configs(extension, req_id, extra)
    layers: set[int] = set()
    broadcast = False
    min_pos, max_pos = _NO_POS, -1
    for cfg in steering:
        layers.update(cfg.layer_indices)
        if cfg.activations.dim() == 2:
            broadcast = True
            continue
        n_pos = int(cfg.activations.shape[1])
        pos = list(cfg.position_indices[:n_pos]) if cfg.position_indices is not None else list(range(n_pos))
        if pos:
            min_pos, max_pos = min(min_pos, min(pos)), max(max_pos, max(pos))
    num_prompt = getattr(req_state, "num_prompt_tokens", None)
    if num_prompt is None:
        num_prompt = len(req_state.prompt_token_ids or ())
    plan = _ReqPlan(
        gen, steering, frozenset(layers), broadcast, min_pos, max_pos,
        _find_hook_configs_no_persistent(extension, req_id, extra), _parse_capture(extra), int(num_prompt),
    )
    cache[req_id] = plan
    if len(cache) > runner.input_batch.num_reqs + _REQ_CACHE_SLACK:
        live = runner.requests
        for stale in [k for k in cache if k not in live]:
            del cache[stale]
    return plan


def _host_query_start_loc(runner: Any, meta: Any, num_reqs: int) -> list[int]:
    """``query_start_loc[:num_reqs+1]`` as host ints without a device sync when the
    runner keeps its pinned host mirror; one ``tolist()`` (a single sync) otherwise."""
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
    """``seq_lens[i] - n_query`` per row, from host memory when possible."""
    nct = getattr(getattr(runner, "input_batch", None), "num_computed_tokens_cpu", None)
    if nct is not None:
        return [int(x) for x in nct[:num_reqs]]
    seq_lens: Any = getattr(meta, "seq_lens", None) if meta is not None else None
    if seq_lens is None:
        return [0] * num_reqs  # fallback: treat as prefill from position 0
    sl = seq_lens[:num_reqs].tolist() if isinstance(seq_lens, torch.Tensor) else [int(x) for x in seq_lens[:num_reqs]]
    return [int(sl[i]) - (qsl[i + 1] - qsl[i]) for i in range(num_reqs)]


def _get_step_plan(extension: HiddenStatesExtension, runner: Any, num_reqs: int) -> _StepPlan | None:
    ctx = get_forward_context()
    plan = extension._step_plan
    if plan is not None and plan.ctx_id == id(ctx):
        return plan
    attn_metadata = ctx.attn_metadata
    if attn_metadata is None:
        return None
    if isinstance(attn_metadata, list):
        attn_metadata = attn_metadata[0]
        if attn_metadata is None:
            return None
    # Hybrid models (e.g. Qwen3-Next with GatedDeltaNet) have multiple attention
    # metadata entries -- some (like GDNAttentionMetadata) lack query_start_loc.
    # vLLM >= 0.27 moved query_start_loc off several backends' metadata entirely;
    # the runner's host buffers are the stable source, the metadata a fallback.
    meta = None
    for _meta in attn_metadata.values():
        if hasattr(_meta, "query_start_loc"):
            meta = _meta
            break
    if meta is None and getattr(getattr(runner, "query_start_loc", None), "np", None) is None:
        logger.warning(
            "No attention metadata with query_start_loc found (keys: %s) and the model runner "
            "has no host query_start_loc buffer. Skipping hook for this step.",
            list(attn_metadata.keys()),
        )
        return None
    qsl = _host_query_start_loc(runner, meta, num_reqs)
    abs_start = _host_abs_start(runner, meta, qsl, num_reqs)
    req_ids = runner.input_batch.req_ids
    plans = [_resolve_request(extension, runner, req_ids[i]) for i in range(num_reqs)]
    plan = _StepPlan(id(ctx), qsl, abs_start, plans, bool(getattr(extension, "_prompt_only", False)))
    extension._step_plan = plan
    return plan


def _begin_pass(extension: HiddenStatesExtension) -> None:
    """Called from the first layer's pre-hook: drop the previous pass's plan and
    decide whether this pass is *idle* -- no row has hooks or capture, and every
    steering vector lies behind every row's current position (the common decode
    step of prompt-position steering).  Idle passes cost one flag check per layer."""
    extension._step_plan = None
    extension._step_idle = False
    try:
        if not is_forward_context_available():
            return
        runner = extension.model_runner
        num_reqs = runner.input_batch.num_reqs
        if not num_reqs:
            return
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return
        if extension._persistent_hooks:
            return
        plan = _get_step_plan(extension, runner, num_reqs)
        if plan is None:
            return
        for i, rp in enumerate(plan.plans):
            if rp is None:
                continue
            if rp.hooks or rp.capture is not None:
                return
            if rp.steering and plan.active(i) and (rp.broadcast or rp.max_pos >= plan.abs_start[i]):
                return
        extension._step_idle = True
        extension._idle_passes += 1
    except Exception:
        extension._step_idle = False
        extension._step_plan = None
        logger.warning("vllm-lens pre-hook error, running full path", exc_info=True)


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
    norm_ref: torch.Tensor,
) -> None:
    """Apply all matching steering vectors to a token slice *in-place*.

    ``target`` is the (already-cloned) output tensor.  ``start``/``end``
    are batch-relative indices, ``abs_start`` is the absolute sequence
    position of the first token in ``target[start:end]``.

    ``norm_ref`` is the tensor whose per-token L2 norm ``norm_match`` should
    match.  For fused-residual models (e.g. Qwen3) the layer returns
    ``(hidden_states, residual)`` and ``target`` is only ``hidden_states``
    (the MLP-delta half); the *true* residual stream is
    ``hidden_states + residual``, so ``norm_ref`` must be that full stream,
    not the MLP-delta half, or the steering vector is scaled to a far smaller
    norm than HF uses.  For non-fused / plain-tensor layers the caller passes
    ``norm_ref = target``.  Required (no default) so a forgotten reference
    fails at the call instead of silently scaling to the wrong
    MLP-delta-half norm.
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
                v = norm_match(norm_ref[start:end], v)
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
                    v = norm_match(norm_ref[rel], v)
                target[rel] = target[rel] + v * cfg.scale


def _apply_hook_delta(
    output: torch.Tensor | tuple[torch.Tensor, ...],
    modified_output: torch.Tensor | tuple[torch.Tensor, ...] | None,
    hook_hidden: torch.Tensor,
    start: int,
    end: int,
    result: torch.Tensor,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Write a post-hook's modification into the layer output.

    Applies ``result - hook_hidden[start:end]`` as a delta onto
    ``modified_output`` (cloning the original ``output`` lazily on first
    write) and updates ``hook_hidden`` in place so later hooks in the same
    forward pass observe the change.  Returns the (possibly newly created)
    ``modified_output``.
    """
    delta = result - hook_hidden[start:end]
    if modified_output is None:
        if isinstance(output, tuple):
            modified_output = (output[0].clone(), output[1])
        else:
            modified_output = output.clone()
    if isinstance(modified_output, tuple):
        modified_output[0][start:end] = modified_output[0][start:end] + delta
    else:
        modified_output[start:end] = modified_output[start:end] + delta
    hook_hidden[start:end] = result
    return modified_output


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

    plan = _get_step_plan(extension, runner, num_reqs)
    if plan is None:
        return None
    req_ids = runner.input_batch.req_ids
    qsl, abs_start, plans = plan.qsl, plan.abs_start, plan.plans

    # --- Phase 1: rows a steering vector can touch at this layer ------
    steer_rows: list[int] = []
    for i, rp in enumerate(plans):
        if rp is None or not rp.steering or layer_idx not in rp.steer_layers or not plan.active(i):
            continue
        if rp.broadcast or (rp.max_pos >= abs_start[i] and rp.min_pos < abs_start[i] + qsl[i + 1] - qsl[i]):
            steer_rows.append(i)

    # --- Phase 2: apply steering ------------------------------------
    modified_output: torch.Tensor | tuple[torch.Tensor, ...] | None = None
    if steer_rows:
        if isinstance(output, tuple):
            modified_output = (output[0].clone(), output[1])
            target = modified_output[0]
            # Fused-residual layers: true residual stream is output[0]+output[1].
            # norm_match must reference the full stream, not the MLP-delta half.
            norm_ref = output[0] + output[1] if output[1] is not None else output[0]
        else:
            modified_output = output.clone()
            target = modified_output
            norm_ref = target
        for i in steer_rows:
            _apply_steering(plans[i].steering, layer_idx, target, qsl[i], qsl[i + 1], abs_start[i], norm_ref)  # type: ignore[union-attr]

    # --- Phase 2.5: run generic (post) hooks -------------------------
    # Per-request and persistent hooks are stored in separate context
    # dicts (per-request contexts are cleaned up after each request;
    # persistent ones accumulate).  Within each dict, contexts are keyed
    # by the hook's position in its category list — so a pre-hook and a
    # post-hook at different positions never share a HookContext, and the
    # returned result index ("0", "1", ...) is stable regardless of how
    # many pre/post hooks a request mixes.  Pre-hooks are handled in
    # _pre_hook_inner using the same position keys.
    persistent_hooks = extension._persistent_hooks
    persistent_here = any(not h.pre and h.has_layer(layer_idx) for h in persistent_hooks)
    hook_rows = [
        i for i, rp in enumerate(plans)
        if plan.active(i) and (persistent_here or any(not h.pre and h.has_layer(layer_idx) for h in rp.hooks))  # type: ignore[union-attr]
    ]

    if hook_rows:
        # Compute hidden_states (summed if tuple) same as Phase 3 does.
        hook_src = modified_output if modified_output is not None else output
        if isinstance(hook_src, tuple):
            hook_hidden = (
                hook_src[0] + hook_src[1] if hook_src[1] is not None else hook_src[0]
            )
        else:
            hook_hidden = hook_src
        # Clone to avoid aliasing — hooks read/write this independently.
        hook_hidden = hook_hidden.clone()

        def _run_post_category(
            hooks: list[Hook],
            store: dict[str, dict[int, HookContext]],
            req_id: str,
            start: int,
            end: int,
        ) -> None:
            """Run the post-hooks in one category list at this layer.

            Contexts live in ``store[req_id][position]``, created lazily.
            """
            nonlocal modified_output
            for pos, hook in enumerate(hooks):
                if hook.pre or not hook.has_layer(layer_idx):
                    continue
                ctxs = store.setdefault(req_id, {})
                ctx = ctxs.get(pos)
                if ctx is None:
                    ctx = HookContext()
                    ctxs[pos] = ctx
                ctx.layer_idx = layer_idx
                ctx.seq_len = end - start
                ctx.model = runner.model
                ctx._prefetched = extension._prefetched_params

                result = hook.fn(ctx, hook_hidden[start:end])
                if result is not None:
                    modified_output = _apply_hook_delta(
                        output, modified_output, hook_hidden, start, end, result
                    )

        for i in hook_rows:
            req_id = req_ids[i]
            start, end = qsl[i], qsl[i + 1]
            # Persistent hooks fire first (base layer); per-request hooks
            # see the persistent-modified state.
            _run_post_category(
                persistent_hooks,
                extension._persistent_hook_contexts,
                req_id,
                start,
                end,
            )
            _run_post_category(
                plans[i].hooks, extension._hook_contexts, req_id, start, end  # type: ignore[union-attr]
            )

    # --- Phase 3: capture activations (rank 0 only) -----------------
    if getattr(extension, "_should_capture", True):
        cap_rows = [
            i for i, rp in enumerate(plans)
            if rp is not None and rp.capture is not None and plan.active(i)
            and (rp.capture is True or layer_idx in rp.capture)
        ]
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
            for i in cap_rows:
                req_id = req_ids[i]
                activation: Float[torch.Tensor, "seq_len hidden_dim"] = hidden_states[  # type: ignore[reportUndefinedVariable]
                    qsl[i] : qsl[i + 1]
                ].cpu()
                if req_id not in extension._captured_states:
                    extension._captured_states[req_id] = {}
                layer_states = extension._captured_states[req_id]
                if layer_idx not in layer_states:
                    layer_states[layer_idx] = []
                layer_states[layer_idx].append(activation)

    return modified_output


def _pre_hook_inner(
    extension: HiddenStatesExtension,
    layer_idx: int,
    input_tensor: torch.Tensor,
) -> torch.Tensor | None:
    """Run pre-hooks (hook.pre=True) on the layer input.

    Only runs generic hooks — steering and activation capture are
    post-hook operations and are not affected.  On this rank's first
    decoder layer it also begins the forward pass (``_begin_pass``).
    """
    if layer_idx == extension._first_layer_idx:
        _begin_pass(extension)
    if extension._step_idle:
        return None
    if not is_forward_context_available():
        return None

    runner = extension.model_runner
    num_reqs = runner.input_batch.num_reqs
    if num_reqs == 0:
        return None
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return None

    persistent_hooks = extension._persistent_hooks
    plan = _get_step_plan(extension, runner, num_reqs)
    if plan is None:
        return None
    persistent_here = any(h.pre and h.has_layer(layer_idx) for h in persistent_hooks)
    rows = [
        i for i, rp in enumerate(plan.plans)
        if plan.active(i) and (persistent_here or any(h.pre and h.has_layer(layer_idx) for h in rp.hooks))  # type: ignore[union-attr]
    ]
    if not rows:
        return None
    req_ids = runner.input_batch.req_ids
    qsl = plan.qsl

    # Pre-hooks share the same context stores as post-hooks, keyed by the
    # hook's position in its category list.  A hook at a given position is
    # either pre or post (never both), so pre and post never collide on the
    # same key — this is what lets a request mix pre- and post-hooks safely.
    modified = False
    working = input_tensor

    def _run_pre_category(
        hooks: list[Hook],
        store: dict[str, dict[int, HookContext]],
        req_id: str,
        start: int,
        end: int,
    ) -> None:
        """Run the pre-hooks in one category list at this layer."""
        nonlocal working, modified
        for pos, hook in enumerate(hooks):
            if not hook.pre or not hook.has_layer(layer_idx):
                continue
            ctxs = store.setdefault(req_id, {})
            hctx = ctxs.get(pos)
            if hctx is None:
                hctx = HookContext()
                ctxs[pos] = hctx
            hctx.layer_idx = layer_idx
            hctx.seq_len = end - start
            hctx.model = runner.model
            hctx._prefetched = extension._prefetched_params

            result = hook.fn(hctx, working[start:end])
            if result is not None:
                if not modified:
                    working = input_tensor.clone()
                    modified = True
                working[start:end] = result

    for i in rows:
        req_id = req_ids[i]
        start, end = qsl[i], qsl[i + 1]
        _run_pre_category(
            persistent_hooks, extension._persistent_hook_contexts, req_id, start, end
        )
        _run_pre_category(plan.plans[i].hooks, extension._hook_contexts, req_id, start, end)  # type: ignore[union-attr]

    return working if modified else None


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
            logger.warning(
                "vllm-lens hook error on layer %d, skipping", layer_idx, exc_info=True
            )
            return None

    return hook


def _make_pre_hook(extension: HiddenStatesExtension, layer_idx: int) -> Callable:
    """Create a forward pre-hook closure for a specific layer index.

    vLLM decoder layers have signature
    ``forward(positions, hidden_states, residual)`` — the hidden states
    are at ``args[1]``, not ``args[0]``.
    """

    def hook(
        _module: torch.nn.Module,
        args: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...] | None:
        """Forward pre-hook: run user pre-hooks on the layer input."""
        try:
            # hidden_states is args[1] (args[0] is positions).
            hidden = args[1]
            result = _pre_hook_inner(extension, layer_idx, hidden)
            if result is not None:
                return args[:1] + (result,) + args[2:]
            return None
        except Exception:
            logger.warning(
                "vllm-lens pre-hook error on layer %d, skipping",
                layer_idx,
                exc_info=True,
            )
            return None

    return hook


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

    # Per-request hook definitions:
    # key (external_req_id or _hook_id) → list of Hook
    _hook_data: dict[str, list[Hook]] = {}

    # Persistent hooks (apply to every request, not auto-cleaned):
    _persistent_hooks: list[Hook] = []

    # Per-request hook contexts, keyed by internal request ID then by the
    # hook's position in the per-request hook list:
    # internal_req_id → { hook_position → HookContext }
    _hook_contexts: dict[str, dict[int, HookContext]] = {}

    # Persistent hook contexts (separate from per-request to avoid cleanup
    # conflicts), keyed the same way by position in the persistent hook list:
    # internal_req_id → { hook_position → HookContext }
    _persistent_hook_contexts: dict[str, dict[int, HookContext]] = {}

    # Whether this rank should capture activations (only TP rank 0).
    _should_capture: bool = True

    # Indexed dispatch: insertion order of keys (the scan iterated the dict in
    # insertion order), a generation counter bumped on every set/clear that
    # invalidates the per-request resolution cache, and per-pass state.
    _steering_seq: dict[str, int] = {}
    _hook_seq: dict[str, int] = {}
    _seq_counter: int = 0
    _gen: int = 0
    _req_plan_cache: dict[str, _ReqPlan] = {}
    _step_plan: _StepPlan | None = None
    _step_idle: bool = False
    _idle_passes: int = 0
    _first_layer_idx: int = 0
    # Decode batches replay CUDA graphs (VLLM_LENS_CUDA_GRAPHS=1): hooks only ever
    # see prompt positions and must never touch generated ones.
    _prompt_only: bool = False

    def _cuda_graphs_active(self) -> bool:
        try:
            cfg = getattr(self, "vllm_config", None) or self.model_runner.vllm_config
            if cfg.model_config.enforce_eager:
                return False
            mode = cfg.compilation_config.cudagraph_mode
            return mode is not None and getattr(mode, "name", str(mode)) != "NONE"
        except Exception:  # noqa: BLE001 - defensive against config drift
            return False

    def _bump(self, key: str | None = None, seq: dict[str, int] | None = None) -> None:
        """Record a key's insertion order and invalidate cached request plans."""
        if key is not None and seq is not None and key not in seq:
            self._seq_counter += 1
            seq[key] = self._seq_counter
        self._gen += 1

    def install_hooks(self) -> None:
        """Register a forward hook on every decoder layer. Idempotent.

        Hooks are installed on **all** TP ranks because steering must
        modify hidden states everywhere.  Activation *capture* is gated
        to rank 0 only via ``_should_capture``.

        Requires ``enforce_eager=True`` in engine args — otherwise
        ``@support_torch_compile`` would compile the forward graph and
        hooks won't fire.
        """
        if self._hooks_installed:
            return
        self._hooks_installed = True
        # Reset to instance-level dicts (class-level defaults are shared).
        # Do NOT reset _persistent_hooks — they may have been set via
        # set_persistent_hooks() before the first generate call.
        self._captured_states = {}
        self._steering_data = {}
        self._hook_data = {}
        if not isinstance(self.__dict__.get("_persistent_hooks"), list):
            self._persistent_hooks = []
        self._hook_contexts = {}
        self._persistent_hook_contexts = {}
        self._steering_seq = {}
        self._hook_seq = {}
        self._seq_counter = 0
        self._gen = 0
        self._req_plan_cache = {}
        self._step_plan = None
        self._step_idle = False
        self._idle_passes = 0
        self._prompt_only = self._cuda_graphs_active()
        if self._prompt_only:
            logger.info(
                "vllm-lens: CUDA graphs active for decode batches; steering, hooks and "
                "activation capture apply to prompt positions only."
            )

        # Only rank 0 captures — residual streams are replicated across
        # TP ranks after all-reduce, so the data is identical.
        tp_size = self.parallel_config.tensor_parallel_size
        self._should_capture = tp_size <= 1 or self.rank % tp_size == 0

        # Hooks must be installed on ALL ranks so steering vectors are
        # applied everywhere (not just rank 0).
        layers = _get_layers(self.model_runner.model)
        first = True
        for layer_idx, layer in enumerate(layers):
            if isinstance(layer, PPMissingLayer):
                continue
            if first:
                self._first_layer_idx = layer_idx
                first = False
            layer.register_forward_pre_hook(_make_pre_hook(self, layer_idx))
            layer.register_forward_hook(_make_hook(self, layer_idx))

    # ------------------------------------------------------------------
    # Steering data management (called via collective_rpc)
    # ------------------------------------------------------------------

    def set_steering_data(self, key: str, pickled_data: bytes) -> None:
        """Receive and store steering vectors for a request.

        Called via ``collective_rpc`` before generation begins.  Unpickles
        the list of ``SteeringVector`` instances, validates layer indices
        against the model, moves activation tensors to GPU in the model's
        dtype, and stores them keyed by *key* (an external request ID or a
        synthetic ``_steering_id``).
        """
        sv_list: list[SteeringVector] = pickle.loads(pickled_data)

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
                    "2-D (broadcast) steering vectors apply to generated positions, which are "
                    "computed inside replayed CUDA graphs where vllm-lens hooks do not run. Either "
                    "run with enforce_eager=True (unset VLLM_LENS_CUDA_GRAPHS) or use 3-D "
                    "position-specific vectors on prompt positions."
                )

            vectors.append(
                sv.model_copy(
                    update={
                        "activations": sv.activations.to(device=device, dtype=dtype)
                    }
                )
            )

        self._steering_data[key] = vectors
        self._bump(key, self._steering_seq)

    def clear_steering_data(self, key: str) -> None:
        """Remove steering data for a completed request."""
        self._steering_data.pop(key, None)
        self._steering_seq.pop(key, None)
        self._bump()

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

    def _build_payload(
        self, internal_req_id: str
    ) -> dict[str, dict[str, "Float[torch.Tensor, 'n_layers total_pos hidden_dim']"]]:  # type: ignore[reportUndefinedVariable]
        """Materialise the stacked-tensor payload for one internal request id.

        Pops the entry from ``_captured_states`` so successive calls do not
        re-emit the same data. Shared by :meth:`get_captured_states` and
        :meth:`get_captured_states_batch`.
        """
        layer_dict = self._captured_states.pop(internal_req_id)
        sorted_indices = sorted(layer_dict.keys())
        per_layer: list[Float[torch.Tensor, "total_pos hidden_dim"]] = [  # type: ignore[reportUndefinedVariable]
            torch.cat(layer_dict[idx], dim=0) for idx in sorted_indices
        ]
        stacked: Float[torch.Tensor, "n_layers total_pos hidden_dim"] = (  # type: ignore[reportUndefinedVariable]
            torch.stack(per_layer, dim=0)
        )
        if stacked.is_cuda:
            stacked = stacked.cpu()
        return {"activations": {"residual_stream": stacked}}

    def get_captured_states(self, external_req_id: str) -> bytes | None:
        """Retrieve captured activations for a specific request.

        Matches by ``"{external_req_id}-"`` prefix because vLLM internally
        transforms the user-provided ``request_id`` into
        ``"{request_id}-{random_suffix}"``. So ``"req-0"`` matches
        ``"req-0-a1b2c3d4"`` but NOT ``"req-00-b5c6d7e8"``.

        Moves tensors to CPU and serializes via pickle + zstd for safe ZMQ
        transport (the compression matters most when the response crosses
        the network in the OpenAI/Inspect HTTP path).

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
                payload = self._build_payload(req_id)
                return _ZSTD_COMPRESSOR.compress(pickle.dumps(payload))
        return None

    def get_captured_states_batch(self, external_req_ids: list[str]) -> bytes | None:
        """Retrieve captured activations for many requests in one RPC.

        Equivalent to calling :meth:`get_captured_states` once per id, but
        emits a single payload covering every request that has data. At
        large batch sizes the per-request ``collective_rpc`` roundtrip is
        the dominant cost on the offline ``LLM.generate`` path; batching it
        cuts N round-trips to one.

        Returns ``pickle.dumps({external_req_id: payload, ...})`` where
        each ``payload`` is ``{"activations": {"residual_stream": Tensor}}``.
        Missing or unmatched ids are simply absent; ``None`` if nothing
        matched at all.

        We deliberately don't ``zstd``-compress here: this RPC only fires
        on the offline ``LLM.generate`` path, which is in-process IPC,
        not HTTP. ``get_captured_states`` keeps zstd for the OpenAI/Inspect
        path where the response crosses the network.
        """
        if not external_req_ids:
            return None
        out: dict[str, dict[str, Any]] = {}
        # Walk live state once and bucket by external id; matches the
        # ``"{external_req_id}-"`` prefix rule from get_captured_states.
        for req_id in list(self._captured_states):
            for external_req_id in external_req_ids:
                if req_id.startswith(f"{external_req_id}-"):
                    out[external_req_id] = self._build_payload(req_id)
                    break
        if not out:
            return None
        return pickle.dumps(out, protocol=pickle.HIGHEST_PROTOCOL)

    def _debug_captured_states_count(self) -> int:
        """Return the number of entries in _captured_states (for testing)."""
        return len(self._captured_states)

    # ------------------------------------------------------------------
    # Hook data management (called via collective_rpc)
    # ------------------------------------------------------------------

    def set_hook_data(self, key: str, pickled_data: bytes) -> None:
        """Receive and store hook definitions for a request.

        Called via ``collective_rpc`` before generation begins.  Unpickles
        the list of ``Hook`` instances (using cloudpickle for the callable
        ``fn``), validates layer indices against the model, and stores them
        keyed by *key* (an external request ID or ``_hook_id`` sentinel).
        """
        hooks: list[Hook] = cloudpickle.loads(pickled_data)
        num_layers = len(_get_layers(self.model_runner.model))
        for hook in hooks:
            for idx in hook.layer_indices:
                if idx < 0 or idx >= num_layers:
                    raise ValueError(
                        f"layer_index {idx} out of range [0, {num_layers})"
                    )
        self._hook_data[key] = hooks
        self._bump(key, self._hook_seq)

    def get_hook_results(self, external_req_id: str) -> bytes | None:
        """Retrieve hook results (``ctx.saved`` dicts) for a request.

        Returns from ALL ranks (including PP ranks that own different
        layers).  The plugin merges results across ranks.
        Matches by ``"{external_req_id}-"`` prefix on ``_hook_contexts``.
        Returns ``{str(hook_position): ctx.saved}`` pickled, where
        ``hook_position`` indexes the per-request hook list.
        """
        prefix = f"{external_req_id}-"
        for req_id in list(self._hook_contexts):
            if req_id.startswith(prefix):
                contexts = self._hook_contexts.pop(req_id)
                saved_dicts = {str(pos): ctx.saved for pos, ctx in contexts.items()}
                return pickle.dumps(saved_dicts)
        return None

    def clear_hook_data(self, key: str) -> None:
        """Remove hook definitions for a completed request."""
        self._hook_data.pop(key, None)
        self._hook_seq.pop(key, None)
        self._bump()

    def clear_hook_contexts(self, external_req_id: str) -> None:
        """Remove hook contexts for a completed or aborted request.

        Prefix-match cleanup, same pattern as ``clear_captured_states``.
        """
        prefix = f"{external_req_id}-"
        for req_id in list(self._hook_contexts):
            if req_id.startswith(prefix):
                del self._hook_contexts[req_id]

    # ------------------------------------------------------------------
    # Persistent hook management (called via collective_rpc)
    # ------------------------------------------------------------------

    def set_persistent_hooks(self, pickled_data: bytes) -> None:
        """Append hooks that apply to every subsequent request.

        Accepts cloudpickle'd ``list[Hook]``.  Validates layer indices.
        Appends to existing persistent hooks (call ``clear_persistent_hooks``
        first for a clean slate).  Also ensures forward hooks are installed
        on the model layers.
        """
        self.install_hooks()
        hooks: list[Hook] = cloudpickle.loads(pickled_data)
        num_layers = len(_get_layers(self.model_runner.model))
        for hook in hooks:
            for idx in hook.layer_indices:
                if idx < 0 or idx >= num_layers:
                    raise ValueError(
                        f"layer_index {idx} out of range [0, {num_layers})"
                    )
        self._persistent_hooks.extend(hooks)
        self._bump()

    def get_all_hook_results(self) -> bytes | None:
        """Retrieve accumulated persistent hook contexts from all requests.

        Returns from ALL ranks (for PP support).  Does NOT clear — call
        ``clear_persistent_hooks`` explicitly.

        Returns pickled ``{internal_req_id: {hook_idx_str: ctx.saved}}``.
        """
        if not self._persistent_hook_contexts:
            return None
        results: dict[str, dict[str, dict[str, Any]]] = {}
        for req_id, contexts in self._persistent_hook_contexts.items():
            results[req_id] = {str(pos): ctx.saved for pos, ctx in contexts.items()}
        return pickle.dumps(results)

    def clear_persistent_hooks(self) -> None:
        """Remove persistent hooks and all accumulated contexts."""
        self._persistent_hooks = []
        self._persistent_hook_contexts = {}
        self._bump()

    def clear_persistent_hook_results(self) -> None:
        """Drop accumulated persistent-hook contexts, keeping hooks registered.

        ``get_all_hook_results`` never drains, so results pile up across
        requests. This clears them without unregistering the hooks (so a
        fitted lens does not need re-uploading), letting a client bound
        accumulation by clearing between turns.
        """
        self._persistent_hook_contexts = {}

    # ------------------------------------------------------------------
    # Parameter prefetch (called via collective_rpc — all ranks in sync)
    # ------------------------------------------------------------------

    _prefetched_params: dict[str, torch.Tensor] = {}

    def prefetch_parameters(self, names: list[str]) -> None:
        """Pre-fetch and gather parameters across TP and PP ranks.

        Safe to call PP collectives here because ``collective_rpc``
        runs on all ranks simultaneously.  Results are stored in
        ``_prefetched_params`` for use by ``HookContext.get_parameter``.
        """
        import torch.distributed as dist

        from vllm.distributed.parallel_state import get_pp_group, get_tp_group
        from vllm.model_executor.models.utils import PPMissingLayer

        model = self.model_runner.model
        tp_group = get_tp_group()
        pp_group = get_pp_group()

        for name in names:
            # Traverse to find the parameter.
            obj: Any = model
            parts = name.split(".")
            is_local = True
            for attr in parts:
                obj = getattr(obj, attr)
                if isinstance(obj, PPMissingLayer):
                    is_local = False
                    break

            param: torch.Tensor | None = None
            if is_local:
                local_t = torch.as_tensor(obj)

                # TP gather if sharded; otherwise reuse the existing tensor.
                module: Any = model
                for attr in parts[:-1]:
                    module = getattr(module, attr)
                tp_size = getattr(module, "tp_size", 1)
                if tp_size > 1:
                    gathered = [torch.empty_like(local_t) for _ in range(tp_size)]
                    dist.all_gather(gathered, local_t, group=tp_group.device_group)
                    gather_dim = getattr(module, "gather_dim", 0)
                    param = torch.cat(gathered, dim=gather_dim)
                else:
                    param = local_t  # no copy — reference to existing parameter

            # PP broadcast — safe here because all ranks are in this RPC.
            if pp_group.world_size > 1:
                has_it = torch.tensor(
                    [1 if is_local else 0], device="cuda", dtype=torch.int32
                )
                all_has = [torch.zeros_like(has_it) for _ in range(pp_group.world_size)]
                dist.all_gather(all_has, has_it, group=pp_group.device_group)
                source_pp = next(i for i, t in enumerate(all_has) if t.item() == 1)
                source_global = pp_group.ranks[source_pp]

                if param is None:
                    # Receive shape + dtype.
                    meta = torch.zeros(3, device="cuda", dtype=torch.int64)
                    dist.broadcast(meta, src=source_global, group=pp_group.device_group)
                    ndim = int(meta[0].item())
                    dtype = _DTYPE_LIST[int(meta[1].item())]
                    shape_t = torch.zeros(ndim, device="cuda", dtype=torch.int64)
                    dist.broadcast(
                        shape_t, src=source_global, group=pp_group.device_group
                    )
                    shape = tuple(int(s) for s in shape_t.tolist())
                    param = torch.empty(shape, device="cuda", dtype=dtype)
                else:
                    meta = torch.tensor(
                        [param.ndim, _dtype_to_idx(param.dtype), 0],
                        device="cuda",
                        dtype=torch.int64,
                    )
                    dist.broadcast(meta, src=source_global, group=pp_group.device_group)
                    shape_t = torch.tensor(
                        list(param.shape),
                        device="cuda",
                        dtype=torch.int64,
                    )
                    dist.broadcast(
                        shape_t, src=source_global, group=pp_group.device_group
                    )

                dist.broadcast(param, src=source_global, group=pp_group.device_group)

            assert param is not None, f"Parameter {name!r} not found on any rank"
            self._prefetched_params[name] = param

    def clear_prefetched_params(self) -> None:
        """Remove all pre-fetched parameters."""
        self._prefetched_params = {}
