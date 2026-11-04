"""Public raw-head interface of the one-class YOLO26n-P2 QAT export."""

from __future__ import annotations

from dataclasses import dataclass

INPUT_NAME = "images"
STRIDES = (4, 8, 16, 32)


@dataclass(frozen=True)
class DeploymentContract:
    """NCHW float32 interface; internal INT8 storage is not the public dtype."""

    model_size: int = 1600

    def __post_init__(self) -> None:
        if self.model_size < max(STRIDES) or self.model_size % max(STRIDES):
            raise ValueError("model size must be at least 32 and divisible by 32")

    @property
    def input_shape(self) -> tuple[int, int, int, int]:
        return (1, 3, self.model_size, self.model_size)

    @property
    def outputs(self) -> tuple[tuple[str, tuple[int, int, int, int]], ...]:
        # Keep all boxes before all logits: two roles can have equal byte sizes.
        return tuple(
            (f"{role}_{index}", (1, channels, self.model_size // stride, self.model_size // stride))
            for role, channels in (("bbox", 4), ("class_logit", 1))
            for index, stride in enumerate(STRIDES)
        )

    def output_names(self) -> list[str]:
        return [name for name, _ in self.outputs]


DEFAULT_CONTRACT = DeploymentContract()
