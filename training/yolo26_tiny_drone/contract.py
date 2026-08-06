"""Deployment contract for the one-class YOLO26n-P2 tiny-drone detector."""

from __future__ import annotations

MODEL_SIZE = 640
INPUT_NAME = "images"
INPUT_SHAPE = (1, 3, MODEL_SIZE, MODEL_SIZE)
CLASS_NAMES = ("drone",)
STRIDES = (4, 8, 16, 32)
FEATURE_SIZES = tuple(MODEL_SIZE // stride for stride in STRIDES)

BBOX_OUTPUTS = tuple((f"bbox_{index}", (1, 4, size, size)) for index, size in enumerate(FEATURE_SIZES))
CLASS_OUTPUTS = tuple(
    (f"class_logit_{index}", (1, len(CLASS_NAMES), size, size)) for index, size in enumerate(FEATURE_SIZES)
)
OUTPUTS = BBOX_OUTPUTS + CLASS_OUTPUTS


def output_names() -> list[str]:
    return [name for name, _shape in OUTPUTS]


def output_shapes(batch_size: int = 1) -> list[tuple[int, int, int, int]]:
    return [(batch_size, channels, height, width) for _name, (_batch, channels, height, width) in OUTPUTS]
