"""Public tensor contract for the TorchVision SSDLite320-MobileNetV3 model."""

from __future__ import annotations

MODEL_SIZE = 320
INPUT_NAME = "images"
INPUT_SHAPE = (1, 3, MODEL_SIZE, MODEL_SIZE)
NUM_CLASSES = 91
FEATURE_SIZES = (20, 10, 5, 3, 2, 1)
ANCHORS_PER_LOCATION = (6, 6, 6, 6, 6, 6)
CONVOLUTION_COUNT = 98
TOTAL_ANCHORS = sum(size * size * anchors for size, anchors in zip(FEATURE_SIZES, ANCHORS_PER_LOCATION))

OUTPUTS = tuple(
    (f"ssd_{role}_{level}", (1, anchors * columns, size, size))
    for role, columns in (("loc", 4), ("confidence", NUM_CLASSES))
    for level, (size, anchors) in enumerate(zip(FEATURE_SIZES, ANCHORS_PER_LOCATION))
)


def output_names() -> list[str]:
    return [name for name, _shape in OUTPUTS]


def output_shapes(batch_size: int = 1) -> list[tuple[int, int, int, int]]:
    return [(batch_size, channels, height, width) for _name, (_batch, channels, height, width) in OUTPUTS]
