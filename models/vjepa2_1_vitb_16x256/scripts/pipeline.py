"""V-JEPA 2.1 ViT-B's interface, shared by conversion, compile and the application: the clip shape, the frame
preprocessing and the token layout. It needs only NumPy and Pillow, and no model weights.

The encoder takes one clip of FRAMES RGB frames at FRAME_SIZE x FRAME_SIZE and returns one feature vector per
spatio-temporal patch: TUBELET frames x PATCH x PATCH pixels each, TOKENS = 8 x 16 x 16 in all. Its 12 transformer
blocks run as three components compiled apart, BLOCKS_PER_COMPONENT each: encoder_0 also embeds the patches, and
encoder_2 also applies the final norm. Each component's tokens [1, FEATURES, 1, TOKENS] are the next one's input_tokens.
"""

from __future__ import annotations

import numpy as np

FRAMES = 16
FRAME_SIZE = 256
TUBELET = 2
PATCH = 16
FEATURES = 768
GRID = (FRAMES // TUBELET, FRAME_SIZE // PATCH, FRAME_SIZE // PATCH)
TOKENS = GRID[0] * GRID[1] * GRID[2]
# The clip's tubelets stacked down the height: channels are (frame in tubelet, colour).
INPUT_SHAPE = (1, 3 * TUBELET, GRID[0] * FRAME_SIZE, FRAME_SIZE)
OUTPUT_SHAPE = (1, FEATURES, 1, TOKENS)
COMPONENTS = ("encoder_0", "encoder_1", "encoder_2")
BLOCKS_PER_COMPONENT = 4
TOKEN_SHAPE = (1, FEATURES, 1, TOKENS)
INPUT_SHAPES = {
    "encoder_0": {"video": INPUT_SHAPE},
    "encoder_1": {"input_tokens": TOKEN_SHAPE},
    "encoder_2": {"input_tokens": TOKEN_SHAPE},
}
OUTPUT_SHAPES = {
    "encoder_0": {"tokens": TOKEN_SHAPE},
    "encoder_1": {"tokens": TOKEN_SHAPE},
    "encoder_2": {"features": OUTPUT_SHAPE},
}
# ImageNet normalization on 0-1 RGB, as V-JEPA's evaluation transforms use.
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def tensor_name(component: str, name: str) -> str:
    """A component's tensor in the side-by-side ONNX that conversion writes."""
    return f"{component}_{name}"


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """An RGB uint8 HWC frame resized bilinearly so its shorter side is FRAME_SIZE, centre-cropped to
    FRAME_SIZE x FRAME_SIZE and normalized, as float32 HWC."""
    from PIL import Image

    height, width = frame.shape[:2]
    scale = FRAME_SIZE / min(height, width)
    size = (max(FRAME_SIZE, round(width * scale)), max(FRAME_SIZE, round(height * scale)))
    resized = np.asarray(Image.fromarray(frame).resize(size, Image.Resampling.BILINEAR), np.float32) / 255
    top, left = (size[1] - FRAME_SIZE) // 2, (size[0] - FRAME_SIZE) // 2
    return (resized[top : top + FRAME_SIZE, left : left + FRAME_SIZE] - MEAN) / STD


def stack_clip(frames: np.ndarray) -> np.ndarray:
    """FRAMES preprocessed HWC frames [FRAMES, H, W, 3] as the encoder's NCHW input INPUT_SHAPE: tubelet t fills
    rows t * H to (t + 1) * H, and its channels are (frame in tubelet, colour)."""
    frames = np.asarray(frames, np.float32)
    if frames.shape != (FRAMES, FRAME_SIZE, FRAME_SIZE, 3):
        raise ValueError(f"expected {(FRAMES, FRAME_SIZE, FRAME_SIZE, 3)} frames, got {frames.shape}")
    x = frames.reshape(GRID[0], TUBELET, FRAME_SIZE, FRAME_SIZE, 3)
    return np.ascontiguousarray(x.transpose(1, 4, 0, 2, 3).reshape(INPUT_SHAPE))


def token_features(features: np.ndarray) -> np.ndarray:
    """The encoder's output [1, FEATURES, 1, TOKENS] as [GRID..., FEATURES]: (tubelet, row, column, feature)."""
    return np.asarray(features, np.float32).reshape(FEATURES, TOKENS).T.reshape(*GRID, FEATURES)
