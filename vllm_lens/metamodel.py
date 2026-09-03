"""vllm-metamodel convenience helpers: one-call reward scoring / readout on a vLLM engine.

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
    """The worker's ``lens_capabilities`` dict (``early_exit``, ``early_exit_reason``, ``multi_stream``, ...).
    Returns ``{}`` if the engine has no vllm-lens worker extension."""
    try:
        caps = llm.collective_rpc("lens_capabilities")
        return (caps[0] if isinstance(caps, (list, tuple)) else caps) or {}
    except Exception as e:  # noqa
        logger.warning("lens_capabilities RPC failed (%s); assuming no early exit", e)
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
