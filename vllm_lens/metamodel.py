"""vllm-metamodels convenience helpers: one-call reward scoring / readout on a vLLM engine.

These are thin wrappers over the per-request ``extra_args`` API (``apply_readout_vectors``,
``lens_early_exit``, ``capture_positions``) so a meta-model training loop can score a batch
of texts against a batch of directions in a single line::

    from vllm_lens.metamodel import readout_scores
    values, positions = readout_scores(llm, token_ids, directions, layer=42, positions={"last": 5})
    reward = values.max(dim=-1).values          # [n]  max cosine over the last 5 tokens

Everything here is prompt-position only, so it works under CUDA graphs and on the same
engine that generates rollouts (pass ``lora_request=None`` to read the clean base model).
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

import torch

from vllm_lens._helpers.types import ReadoutVector

logger = logging.getLogger("vllm_lens.metamodel")

PositionSpec = Any  # "all" | {"last": k} | list[int]


def capabilities(llm: Any) -> dict[str, Any]:
    """The worker's ``lens_capabilities`` dict (``early_exit``, ``early_exit_reason``, ``multi_stream``, ...)
    merged with the plugin's prefix-cache facts (``prefix_caching``, ``kv_salt_active``; ``early_exit`` is
    False when prefix caching is on but the scheduler process lacks the block-hash salt patch).
    Returns ``{}`` if the engine has no vllm-lens worker extension."""
    try:
        from vllm_lens._activations_plugin import _lens_capabilities_sync

        return dict(_lens_capabilities_sync(llm))
    except Exception as e:  # noqa
        logger.warning("lens_capabilities failed (%s); assuming no early exit", e)
        return {}


def readout_scores(
    llm: Any,
    prompt_token_ids: Sequence[Sequence[int]],
    directions: torch.Tensor,
    layer: int | Sequence[int],
    positions: PositionSpec = {"last": 5},
    metric: str = "cos",
    bias: float | Sequence[float] = 0.0,
    early_exit: bool = True,
    lora_request: Any = None,
    sampling_params_cls: Any = None,
) -> tuple[torch.Tensor, list[list[int]]]:
    """Score ``prompt_token_ids[i]`` against ``directions[i]`` at ``layer`` and return
    ``(values, positions)``: ``values`` is a float32 tensor ``[n, n_layers, n_pos]`` of
    ``metric(h, d) + bias`` (positions padded with NaN when texts are shorter than the
    requested window), ``positions`` the absolute positions read per text.

    One prefill-only ``generate()`` call (``max_tokens=1``); with ``early_exit=True`` the
    engine stops after the deepest requested layer when it supports it (see
    :func:`capabilities`), otherwise the flag is dropped with a warning and the full model
    runs.  ``directions``: ``[n, hidden]`` (or ``[n, n_layers, hidden]`` for several layers).
    """
    if sampling_params_cls is None:
        from vllm import SamplingParams as sampling_params_cls  # type: ignore
    layers = [int(layer)] if isinstance(layer, int) else [int(x) for x in layer]
    n = len(prompt_token_ids)
    if directions.dim() == 2:
        directions = directions.unsqueeze(1)
    if directions.shape[0] != n or directions.shape[1] != len(layers):
        raise ValueError(f"directions must be [n={n}, n_layers={len(layers)}, hidden], got {tuple(directions.shape)}")
    biases = [float(bias)] * n if isinstance(bias, (int, float)) else [float(b) for b in bias]
    if early_exit and not capabilities(llm).get("early_exit", False):
        logger.warning("engine does not support lens_early_exit (%s); running the full model",
                       capabilities(llm).get("early_exit_reason", "unknown"))
        early_exit = False
    params = []
    for i in range(n):
        rv = ReadoutVector(activations=directions[i].detach().float().cpu(), layer_indices=layers,
                           positions=positions, metric=metric, bias=biases[i])
        extra: dict[str, Any] = {"apply_readout_vectors": [rv]}
        if early_exit:
            extra["lens_early_exit"] = True
        params.append(sampling_params_cls(max_tokens=1, temperature=0.0, extra_args=extra))
    prompts = [{"prompt_token_ids": list(map(int, ids))} for ids in prompt_token_ids]
    outs = llm.generate(prompts, params, lora_request=lora_request, use_tqdm=False)
    vals, poss = [], []
    for out in outs:
        r = getattr(out, "readout", None)
        if not r:
            raise RuntimeError("request returned no readout — is the vllm-lens plugin active?")
        vals.append(torch.as_tensor(r[0]["values"], dtype=torch.float32))
        poss.append(list(r[0]["positions"]))
    n_pos = max(v.shape[-1] for v in vals)
    out_t = torch.full((n, len(layers), n_pos), float("nan"), dtype=torch.float32)
    for i, v in enumerate(vals):
        out_t[i, :, : v.shape[-1]] = v
    return out_t, poss


def readout_max(llm: Any, prompt_token_ids: Sequence[Sequence[int]], directions: torch.Tensor, layer: int,
                positions: PositionSpec = {"last": 5}, **kw: Any) -> torch.Tensor:
    """``readout_scores`` reduced to one reward per text: the max over the read positions (NaN-safe)."""
    values, _ = readout_scores(llm, prompt_token_ids, directions, layer, positions=positions, **kw)
    return torch.nan_to_num(values[:, 0, :], nan=-float("inf")).max(dim=-1).values


# ---------------------------------------------------------------------------
# LoRA merge-on-publish (vllm-metamodels): serve the current adapter as plain merged weights
# ---------------------------------------------------------------------------


def _rpc0(llm: Any, name: str, *args: Any, **kw: Any) -> Any:
    res = llm.collective_rpc(name, args=args, kwargs=kw or None)
    return res[0] if isinstance(res, (list, tuple)) else res


def merge_lora(llm: Any, adapter_dir: str | None = None, tensors: dict[str, torch.Tensor] | None = None,
               scaling: float | None = None, keep_base: str = "auto") -> dict[str, Any]:
    """Merge a PEFT LoRA adapter INTO the served weights on every worker (in place), so generation
    runs without LoRA kernels; call again with the next adapter to replace it.  ``adapter_dir`` is a
    PEFT directory (``adapter_config.json`` + ``adapter_model.safetensors``); alternatively pass the
    PEFT-style ``tensors`` (``base_model.model.<module>.lora_A.weight`` / ``lora_B.weight``) with an
    explicit ``scaling`` (``alpha / r`` or ``alpha / sqrt(r)`` for rsLoRA).

    ``keep_base``: ``"gpu"`` keeps a bf16 copy of the LoRA-targeted weights on the device (exact
    single-rounding merges, exact unmerge; costs ~ the size of those weights), ``"cpu"`` keeps it in
    pinned host memory (publish streams it back), ``"none"`` keeps no copy and subtracts the previous
    adapter instead (<= 1/2 ulp drift per publish), ``"auto"`` = gpu if it fits else cpu.

    After a merge the *base* served by this engine IS the merged policy: clean-base readout /
    reward scoring needs :func:`unmerge_lora` first (a copy in gpu/cpu mode), and LoRA requests are
    applied on top of the merged weights.  Prefix caches are reset (weights changed).
    """
    import pickle

    if adapter_dir is None and tensors is None:
        raise ValueError("merge_lora needs adapter_dir or tensors")
    if tensors is not None:
        if scaling is None:
            raise ValueError("merge_lora(tensors=...) needs an explicit scaling")
        payload = pickle.dumps({k: v.detach().cpu() for k, v in tensors.items()}, protocol=pickle.HIGHEST_PROTOCOL)
        out = _rpc0(llm, "lens_merge_lora", None, payload, float(scaling), keep_base)
    else:
        out = _rpc0(llm, "lens_merge_lora", adapter_dir, None, scaling, keep_base)
    _reset_prefix_cache(llm)
    return out


def unmerge_lora(llm: Any, release: bool = False, how: str = "auto") -> dict[str, Any]:
    """Restore the base weights on every worker (exact when a base copy exists; ``how`` = ``"auto"`` |
    ``"copy"`` | ``"subtract"``); ``release`` frees the copies."""
    out = _rpc0(llm, "lens_unmerge_lora", release, how)
    _reset_prefix_cache(llm)
    return out


def lora_status(llm: Any) -> dict[str, Any]:
    """``{merged, n_modules, n_params, mode, base_where, base_bytes, publishes, last_publish_s, tp_size}``."""
    return _rpc0(llm, "lens_lora_status")


def _reset_prefix_cache(llm: Any) -> None:
    try:
        eng = getattr(llm, "llm_engine", None)
        if eng is not None and hasattr(eng, "reset_prefix_cache"):
            eng.reset_prefix_cache()
    except Exception:  # noqa: BLE001
        logger.debug("reset_prefix_cache failed", exc_info=True)
