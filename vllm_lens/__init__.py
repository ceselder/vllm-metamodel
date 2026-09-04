from importlib.metadata import PackageNotFoundError, version

from vllm_lens._helpers._serialize import (
    decode_activations,
    deserialize_tensor,
    serialize_activations,
    serialize_tensor,
)
from vllm_lens.metamodel import capabilities, lora_status, merge_lora, readout_max, readout_scores, unmerge_lora
from vllm_lens._helpers.types import (
    CAPTURE_POSITIONS_KEY,
    EARLY_EXIT_KEY,
    EMBED_LAYER_INDEX,
    ReadoutVector,
    SteeringVector,
)

try:
    __version__ = version("vllm-lens")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "decode_activations",
    "deserialize_tensor",
    "serialize_activations",
    "serialize_tensor",
    "CAPTURE_POSITIONS_KEY",
    "EARLY_EXIT_KEY",
    "EMBED_LAYER_INDEX",
    "ReadoutVector",
    "capabilities",
    "lora_status",
    "merge_lora",
    "readout_max",
    "readout_scores",
    "unmerge_lora",
    "SteeringVector",
    "__version__",
]
