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
``VLLM_LENS_BLOCK_RPC=0``
    Disable packing the offline call's single-position vectors into one
    ``set_steering_block`` RPC (falls back to ``set_steering_data_many``).
``VLLM_LENS_DISABLE=1``
    Make the plugin a no-op (plain vLLM), e.g. for baselines.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from collections.abc import AsyncIterator, Callable, Sequence
from typing import TYPE_CHECKING, Any

import torch
import zstandard as zstd

from vllm_lens._helpers._serialize import serialize_activations
from vllm_lens._helpers.types import EMBED_LAYER_INDEX, SteeringVector

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
    return {"residual_stream": merged}


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
    if rs is not None and rs.shape[1] > expected_len:
        activations["residual_stream"] = rs[:, :expected_len, :]
    ids = activations.get("input_ids")
    if ids is not None and len(ids) > expected_len:
        activations["input_ids"] = ids[:expected_len]


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

    if cc.mode not in (None, CompilationMode.NONE):
        logger.warning(
            "vllm-lens: compilation mode %s would compile the decoder layers and "
            "the forward hooks would not fire; forcing enforce_eager=True. For "
            "CUDA graphs use compilation_config mode=0 (NONE) with "
            "cudagraph_mode=FULL_DECODE_ONLY.",
            cc.mode,
        )
        engine_args.enforce_eager = True
        return False

    cc.mode = CompilationMode.NONE
    if cc.cudagraph_mode is None:
        cc.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
    elif cc.cudagraph_mode not in (CUDAGraphMode.NONE, CUDAGraphMode.FULL_DECODE_ONLY):
        logger.warning(
            "vllm-lens: cudagraph_mode %s would run prefill batches inside CUDA "
            "graphs where the hooks do not fire; overriding to FULL_DECODE_ONLY.",
            cc.cudagraph_mode.name,
        )
        cc.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
    engine_args.compilation_config = cc

    if engine_args.enforce_eager or cc.cudagraph_mode == CUDAGraphMode.NONE:
        return False
    logger.info(
        "vllm-lens: CUDA graphs enabled (compilation mode NONE, cudagraph_mode "
        "%s). Hooks run only in forward passes that contain prompt tokens: "
        "steering and activation capture are prompt-position only.",
        cc.cudagraph_mode.name,
    )
    return True


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

    assert _original_create_engine_config is not None
    return _original_create_engine_config(self, *args, **kwargs)


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
    """vllm-metamodel: refuse layer-output steering / capture on hyper-connection
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
    _check_graph_mode_request(
        steering_vectors,
        wants_activations,
        getattr(effective_params, "max_tokens", None),
    )

    # Allow explicit prefix-cache bypass via extra_args.
    skip_kv_cache = extra.pop("skip_reading_prefix_cache", None)

    needs_hooks = wants_activations or steering_vectors is not None
    if needs_hooks or skip_kv_cache:
        # Hooks rely on forward passes firing; prefix-cached tokens skip
        # computation entirely, so force a fresh prefill for this request.
        effective_params.skip_reading_prefix_cache = True
    if needs_hooks and not getattr(self, "_hooks_installed", False):
        await self.collective_rpc("install_hooks")
        setattr(self, "_hooks_installed", True)
    if needs_hooks:
        _check_layer_support(
            await _lens_capabilities_async(self), steering_vectors, extra.get("output_residual_stream")
        )

    # Send steering data to workers before the forward pass begins.
    if steering_vectors is not None:
        await self.collective_rpc(
            "set_steering_data",
            args=(request_id, pickle.dumps(steering_vectors)),
        )

    assert _original_generate is not None
    try:
        async for output in _original_generate(
            self, prompt, sampling_params, request_id, **kwargs
        ):
            if output.finished and wants_activations:
                states = await self.collective_rpc(
                    "get_captured_states", args=(request_id,)
                )
                activations = _merge_captured_states(states)
                if activations is not None:
                    n_prompt = len(output.prompt_token_ids)
                    n_gen = len(output.outputs[0].token_ids)
                    _trim_activations(activations, n_prompt + n_gen - 1)
                    output.activations = activations
            yield output
    finally:
        if steering_vectors is not None:
            await self.collective_rpc("clear_steering_data", args=(request_id,))
        if wants_activations:
            await self.collective_rpc("clear_captured_states", args=(request_id,))


def _lens_capabilities_sync(llm: Any) -> dict[str, Any]:
    caps = getattr(llm, "_lens_caps", None)
    if caps is None:
        try:
            caps = llm.collective_rpc("lens_capabilities")[0] or {}
        except Exception:  # noqa: BLE001 - older worker without the RPC
            caps = {}
        llm._lens_caps = caps
    return caps


async def _lens_capabilities_async(llm: Any) -> dict[str, Any]:
    caps = getattr(llm, "_lens_caps", None)
    if caps is None:
        try:
            caps = (await llm.collective_rpc("lens_capabilities"))[0] or {}
        except Exception:  # noqa: BLE001
            caps = {}
        llm._lens_caps = caps
    return caps


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
    if isinstance(sampling_params, Sequence):
        params_list = list(sampling_params)
    elif sampling_params is not None:
        params_list = [sampling_params]
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
    per_request: list[tuple[list[SteeringVector] | None, Any]] = []
    for idx, sp in enumerate(params_list):
        extra = sp.extra_args or {}
        vectors = extra.pop("apply_steering_vectors", None)
        if isinstance(vectors, str):
            vectors = [SteeringVector.model_validate(d) for d in json.loads(vectors)]
        per_request.append((vectors, extra.get("output_residual_stream")))
        if vectors is not None:
            _check_graph_mode_request(vectors, False, None)
            steering_id = f"_steer_{idx}"
            steering_payloads[steering_id] = vectors
            if sp.extra_args is None:
                sp.extra_args = {}
            sp.extra_args["_steering_id"] = steering_id
    if wants_activations:
        _check_graph_mode_request(
            None,
            True,
            max((sp.max_tokens or 0) for sp in params_list) if params_list else None,
        )

    # Pop skip_reading_prefix_cache from extra_args for each request.
    any_skip_kv_cache = False
    for sp in params_list:
        if (sp.extra_args or {}).pop("skip_reading_prefix_cache", None):
            any_skip_kv_cache = True

    has_steering = len(steering_payloads) > 0
    needs_hooks = wants_activations or has_steering
    if needs_hooks or any_skip_kv_cache:
        for sp in params_list:
            sp.skip_reading_prefix_cache = True

    if needs_hooks and not getattr(self, "_hooks_installed", False):
        self.collective_rpc("install_hooks")
        self._hooks_installed = True  # type: ignore[reportAttributeAccessIssue]
    if needs_hooks:
        caps = _lens_capabilities_sync(self)
        for vectors, cap in per_request:
            if vectors is not None or cap is not None:
                _check_layer_support(caps, vectors, cap)

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
            self.collective_rpc("set_steering_block", args=(pickle.dumps(block),))
        if rest:
            self.collective_rpc("set_steering_data_many", args=(pickle.dumps(rest),))

    assert _original_llm_generate is not None
    try:
        outputs = _original_llm_generate(self, prompts, sampling_params, **kwargs)
    finally:
        # Clean up steering data (also on error, so keys never leak).
        if has_steering:
            self.collective_rpc(
                "clear_steering_data_many", args=(list(steering_payloads),)
            )

    if wants_activations:
        for output in outputs:
            req_id = output.request_id
            states = self.collective_rpc("get_captured_states", args=(req_id,))
            activations = _merge_captured_states(states)
            if activations is not None:
                n_prompt = len(output.prompt_token_ids)
                n_gen = len(output.outputs[0].token_ids)
                _trim_activations(activations, n_prompt + n_gen - 1)
                output.activations = activations

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
