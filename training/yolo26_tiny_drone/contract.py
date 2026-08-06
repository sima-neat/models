"""Deployment contract for the one-class YOLO26n-P2 tiny-drone detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

INPUT_NAME = "images"
CLASS_NAMES = ("drone",)
STRIDES = (4, 8, 16, 32)
DEFAULT_MODEL_SIZE = 640


@dataclass(frozen=True)
class DeploymentContract:
    """Static tensor ABI shared by training, export, compilation, and Neat."""

    model_size: int = DEFAULT_MODEL_SIZE

    def __post_init__(self) -> None:
        if self.model_size < max(STRIDES) or self.model_size % max(STRIDES) != 0:
            raise ValueError(f"model size must be at least {max(STRIDES)} and divisible by {max(STRIDES)}")

    @property
    def input_shape(self) -> tuple[int, int, int, int]:
        return (1, 3, self.model_size, self.model_size)

    @property
    def feature_sizes(self) -> tuple[int, ...]:
        return tuple(self.model_size // stride for stride in STRIDES)

    @property
    def outputs(self) -> tuple[tuple[str, tuple[int, int, int, int]], ...]:
        boxes = tuple((f"bbox_{index}", (1, 4, size, size)) for index, size in enumerate(self.feature_sizes))
        classes = tuple(
            (f"class_logit_{index}", (1, len(CLASS_NAMES), size, size)) for index, size in enumerate(self.feature_sizes)
        )
        return boxes + classes

    def output_names(self) -> list[str]:
        return [name for name, _shape in self.outputs]

    def output_shapes(self, batch_size: int = 1) -> list[tuple[int, int, int, int]]:
        return [(batch_size, channels, height, width) for _name, (_batch, channels, height, width) in self.outputs]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_size": self.model_size,
            "input_name": INPUT_NAME,
            "input_shape": list(self.input_shape),
            "outputs": [{"name": name, "shape": list(shape)} for name, shape in self.outputs],
            "strides": list(STRIDES),
        }

    @classmethod
    def from_dict(cls, document: dict[str, Any] | None) -> DeploymentContract:
        if not document:
            return cls()
        if not isinstance(document, dict):
            raise TypeError("serialized deployment contract must be an object")
        contract = cls(model_size=int(document.get("model_size", DEFAULT_MODEL_SIZE)))
        if document != contract.to_dict():
            raise ValueError("serialized deployment contract is incomplete or inconsistent")
        return contract


DEFAULT_CONTRACT = DeploymentContract()

# Default-shape aliases keep the existing 640 contract readable to callers and
# tests while variable-resolution workflows pass a DeploymentContract explicitly.
MODEL_SIZE = DEFAULT_CONTRACT.model_size
INPUT_SHAPE = DEFAULT_CONTRACT.input_shape
FEATURE_SIZES = DEFAULT_CONTRACT.feature_sizes
OUTPUTS = DEFAULT_CONTRACT.outputs

BBOX_OUTPUTS = OUTPUTS[: len(STRIDES)]
CLASS_OUTPUTS = OUTPUTS[len(STRIDES) :]


def output_names() -> list[str]:
    return DEFAULT_CONTRACT.output_names()


def output_shapes(batch_size: int = 1) -> list[tuple[int, int, int, int]]:
    return DEFAULT_CONTRACT.output_shapes(batch_size)
