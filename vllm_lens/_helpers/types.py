"""Pydantic models for vllm-lens steering vectors."""

from __future__ import annotations

from typing import Any, Literal, Self

import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    field_serializer,
    field_validator,
    model_validator,
)

from vllm_lens._helpers._serialize import deserialize_tensor, serialize_tensor

EMBED_LAYER_INDEX = -1
"""Sentinel ``layer_indices`` value: target the EMBEDDING stream (the hidden
states entering decoder layer 0) instead of a decoder layer's output.  See
``SteeringVector.mode`` — embedding replacement is the injection for
NLA-style metamodels and for hyper-connection architectures (DeepSeek-V4)
where decoder-layer outputs are multi-stream tuples.  Applied during prefill
only, keeping decode-only CUDA graphs legal."""


class SteeringVector(BaseModel):
    """A steering vector that modifies the residual stream during inference.

    Supports automatic serialization/deserialization of ``torch.Tensor``
    activations for JSON transport (HTTP API) and direct ``torch.Tensor``
    values for in-process usage (offline ``LLM`` / ``AsyncLLMEngine``).

    Example (offline)::

        sv = SteeringVector(
            activations=torch.randn(1, 4096),
            layer_indices=[18],
            scale=2.0,
        )

    Example (JSON round-trip)::

        data = sv.model_dump()          # base64-encoded activations
        sv2 = SteeringVector.model_validate(data)  # decoded back to tensor
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    activations: torch.Tensor
    """Steering activations.  Shape ``(n_layers, hidden_dim)`` for broadcast
    or ``(n_layers, n_positions, hidden_dim)`` for position-specific."""

    layer_indices: list[int]
    """Which model layers this steering vector applies to.  Length must
    match ``activations.shape[0]``."""

    scale: float = 1.0
    """Scalar multiplier applied to the steering vector before addition."""

    norm_match: bool = False
    """If True, scale the steering vector so the added magnitude equals the
    residual stream's per-token L2 norm (times ``scale``):
    ``h' = h + scale · ‖h‖ · v/‖v‖`` -- the Activation Oracles injection.
    Does NOT renormalize ``h'`` back to ``‖h‖``.  ``‖h‖`` is the norm of the
    FULL residual stream at that position (on fused-residual architectures
    the layer's ``hidden_states + residual``, not the ``hidden_states`` half
    alone -- upstream #7, ported in vllm-metamodel 1.1.0.post2; 1.1.0 used
    the half and under-injected by ~8x on Qwen-style models).  With
    ``mode="replace"``: ``h' = scale · ‖h‖ · v/‖v‖``."""

    position_indices: list[int] | None = None
    """Absolute token positions for 3D activations.  ``None`` means broadcast
    (2D) or sequential ``0..n_positions-1`` (3D)."""

    mode: Literal["add", "replace"] = "add"
    """``"add"`` (default) adds ``scale * v`` to the hidden state.
    ``"replace"`` OVERWRITES the hidden row with ``scale * v`` (or, with
    ``norm_match=True``, ``scale * ‖h_orig‖ · v/‖v‖``).  Replacement is the
    injection used by NLA-style metamodels ("replace the marker token's
    embedding with α·v/‖v‖") and is the only well-defined injection on
    architectures whose decoder-layer outputs are not a single residual
    tensor (e.g. hyper-connection / multi-stream models like DeepSeek-V4) —
    target the embedding stream via ``EMBED_LAYER_INDEX`` there.  On a
    regular layer of a fused-residual model the FULL stream is replaced
    (both the ``hidden_states`` and the ``residual`` half are rewritten).
    Requires 3D (position-specific) activations."""

    @field_validator("activations", mode="before")
    @classmethod
    def _deserialize_activations(cls, v: Any) -> torch.Tensor:
        """Accept base64 dicts (from JSON transport) or raw tensors."""
        if isinstance(v, dict) and "data" in v:
            return deserialize_tensor(v)
        if isinstance(v, torch.Tensor):
            return v
        raise ValueError(
            f"activations must be a torch.Tensor or a base64 dict, got {type(v)}"
        )

    @field_serializer("activations")
    def _serialize_activations(self, v: torch.Tensor, _info: Any) -> dict[str, Any]:
        """Serialize tensor to base64 dict for JSON transport."""
        return serialize_tensor(v)

    @model_validator(mode="after")
    def _check_shape(self) -> Self:
        """Validate activation tensor shape matches layer_indices."""
        if self.activations.dim() not in (2, 3):
            raise ValueError(
                f"activations must be 2D or 3D, got {self.activations.dim()}D"
            )
        if self.activations.shape[0] != len(self.layer_indices):
            raise ValueError(
                f"activations dim 0 ({self.activations.shape[0]}) must match "
                f"len(layer_indices) ({len(self.layer_indices)})"
            )
        if self.mode == "replace" and self.activations.dim() != 3:
            raise ValueError(
                "mode='replace' requires 3D (position-specific) activations — "
                "broadcast replacement would overwrite every token"
            )
        return self

    @property
    def layer_index_map(self) -> dict[int, int]:
        """Maps actual model layer index to index into ``activations`` dim-0."""
        return {li: i for i, li in enumerate(self.layer_indices)}
