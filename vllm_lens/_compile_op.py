"""vllm-metamodels: torch.compile-compatible layer hooks (1.1.0.post7).

vLLM compiles the decoder stack with ``torch.compile(fullgraph=True)`` and drops every
guard afterwards, so a Python forward hook is either (a) inlined into the compiled graph
when it is present at trace time -- its body must then be traceable -- or (b) ignored for
ever when it is registered later.  The fork's eager hooks (``_worker_ext._make_hook``) do
dict lookups, numpy, host syncs and data-dependent indexing: untraceable, which is why
every release so far forced ``compilation mode NONE`` (17 % slower than compiled vLLM on
0.27.1, 4 % on 0.19).

This module makes each hook body a single call to an *opaque custom op*::

    torch.ops.vllm_lens.lens_layer_(stream, residual, layer_idx)   # mutates in place

Dynamo records the call and never looks inside; Inductor treats it as a black box with
side effects on its tensor arguments (``mutates_args``); at run time the op's Python
implementation runs the fork's normal per-pass planning and injection / capture /
readout logic *in place* on the layer's output tensors.  During CUDA-graph capture the
implementation returns immediately, so replayed decode graphs contain no injection
kernels -- the same prompt-position-only semantics as ``VLLM_LENS_CUDA_GRAPHS=1``.
Prefill batches run the compiled code eagerly (``cudagraph_mode=FULL_DECODE_ONLY`` is
forced: under PIECEWISE graphs a small prefill batch would replay a graph without the
op's kernels and lose the injection silently).

The hooks must exist before the first forward pass (the profiling run compiles the
model), so the plugin wraps ``Worker.load_model`` to call ``install_hooks`` right after
the weights are loaded whenever the engine config compiles the model.  vLLM's compile
cache directory hashes the contents of every file Dynamo traced through, and the hook
closures below are traced, so a graph compiled without them (or with an older version of
this file) is never reused.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from vllm_lens._helpers.types import EMBED_LAYER_INDEX

_ACTIVE: Any = None
"""The ``HiddenStatesExtension`` whose plan the op consults (one worker per process)."""

OP_NAME = "vllm_lens::lens_layer_"


@torch.library.custom_op(OP_NAME, mutates_args={"stream", "residual"})
def lens_layer_(stream: torch.Tensor, residual: Optional[torch.Tensor], layer: int) -> None:
    """Run the vllm-lens hook logic for ``layer`` in place on the layer's output halves
    (``layer == EMBED_LAYER_INDEX``: the hidden states ENTERING layer 0 -- begins a pass)."""
    ext = _ACTIVE
    if ext is None:
        return
    ext._op_dispatch(int(layer), stream, residual)


@lens_layer_.register_fake
def _lens_layer_fake(stream: torch.Tensor, residual: Optional[torch.Tensor], layer: int) -> None:
    return None


def make_post_hook(layer_idx: int):
    """Forward hook whose body Dynamo can inline: one op call on the output tensor(s)."""

    def hook(_module: torch.nn.Module, _args: tuple, output: Any) -> None:
        if isinstance(output, tuple):
            torch.ops.vllm_lens.lens_layer_(output[0], output[1], layer_idx)
        else:
            torch.ops.vllm_lens.lens_layer_(output, None, layer_idx)
        return None

    return hook


def make_pre_hook():
    """Forward pre-hook (``with_kwargs=True``) on this rank's first decoder layer: begins the
    pass and, on global layer 0, applies EMBED_LAYER_INDEX configs to the entering hidden
    states.  ``hidden_states`` is the keyword (Qwen3-Next style) or the second positional
    argument (Llama / Qwen2 style: ``layer(positions, hidden_states, residual)``); the op
    implementation verifies the tensor covers this pass's tokens and raises otherwise."""

    def pre_hook(_module: torch.nn.Module, args: tuple, kwargs: dict) -> None:
        hs = kwargs["hidden_states"] if "hidden_states" in kwargs else args[1]
        torch.ops.vllm_lens.lens_layer_(hs, None, EMBED_LAYER_INDEX)
        return None

    return pre_hook


def model_is_compiled(vllm_config: Any) -> bool:
    """True when vLLM will run the decoder stack through torch.compile (any compilation mode
    other than NONE and not enforce_eager) -- then only op-based hooks can fire."""
    try:
        if vllm_config.model_config.enforce_eager:
            return False
        mode = vllm_config.compilation_config.mode
    except Exception:  # noqa: BLE001 - defensive against config drift
        return False
    if mode is None:
        return False
    name = getattr(mode, "name", None)
    if name is not None:
        return name != "NONE"
    return int(mode) != 0


def compiled_submodule(model: torch.nn.Module) -> Any:
    """The ``@support_torch_compile``-decorated module inside ``model`` (has ``compiled`` /
    ``do_not_compile``), or None."""
    for m in model.modules():
        if hasattr(m, "do_not_compile") and hasattr(m, "compiled"):
            return m
    return None
