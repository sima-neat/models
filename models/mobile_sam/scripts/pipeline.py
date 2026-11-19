"""MobileSAM application helpers for a whole encoder and reusable prompt decoder.

Prepare an image once and retain its [1,256,64,64] embedding. Each subsequent point/box prompt runs only
prompt_decoder. Inputs have five user-point slots plus the single optional learned padding token;
unused slots never participate as attention keys. Fourier encoding is CPU float32.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

COMPONENTS = ("image_encoder", "prompt_decoder")
IMAGE_SIZE = 1024
EMBEDDING_SHAPE = (1, 256, 64, 64)
# Five user points plus the one learned padding token upstream adds to prompts without a box.
MAX_POINTS = 5
POINTS = MAX_POINTS + 1
# One-hot point types in this order; segment_anything labels them -1, 0, 1, 2 and 3.
POINT_TYPES = ("padding", "background", "foreground", "box_top_left", "box_bottom_right")
LOW_RES_MASK_SIZE = 256
MASKS = 4
INPUT_SHAPES = {
    "image_encoder": {"image": (1, 3, IMAGE_SIZE, IMAGE_SIZE)},
    "prompt_decoder": {
        "image_embeddings": EMBEDDING_SHAPE,
        "point_coords": (1, 2, 1, POINTS),
        "point_types": (1, len(POINT_TYPES), 1, POINTS),
    },
}
# Keep the full exported/source ONNX interface unchanged. Surgery alone moves Fourier encoding to CPU.
COMPILED_INPUT_SHAPES = {
    **INPUT_SHAPES,
    "prompt_decoder": {
        "image_embeddings": EMBEDDING_SHAPE,
        "point_encoding": (1, 256, 1, POINTS),
        "point_types": INPUT_SHAPES["prompt_decoder"]["point_types"],
    },
}
PROMPT_ENCODING_METADATA = "mobile_sam.prompt_encoding"
PROMPT_ENCODING_FILE = "prompt-encoding.json"


@lru_cache(maxsize=1)
def _fmaf():
    """Linux libm's single-rounding float32 multiply-add, loaded once, not a float64 emulation."""
    function = ctypes.CDLL("libm.so.6").fmaf
    function.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
    function.restype = ctypes.c_float
    return function


@dataclass(frozen=True)
class PromptEncoding:
    """The source model's small Fourier table, shipped as a checksum-bound package supporting file."""

    gaussian: np.ndarray
    phase_scale: np.float32
    source_sha256: str

    @classmethod
    def from_metadata(cls, metadata: dict[str, str]) -> PromptEncoding:
        try:
            encoded = metadata[PROMPT_ENCODING_METADATA]
            if not isinstance(encoded, str) or len(encoded.encode("utf-8")) > 65536:
                raise ValueError("prompt encoding metadata exceeds 64 KiB or is not a JSON string")
            value = json.loads(encoded)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("decoder lacks valid MobileSAM prompt encoding metadata; rebuild old MPKs") from error
        keys = {"version", "source_sha256", "gaussian", "gaussian_sha256", "phase_scale"}
        if (
            not isinstance(value, dict)
            or set(value) != keys
            or type(value["version"]) is not int
            or value["version"] != 1
        ):
            raise ValueError("unsupported MobileSAM prompt encoding metadata")
        source = value["source_sha256"]
        if not isinstance(source, str) or len(source) != 64 or any(c not in "0123456789abcdef" for c in source):
            raise ValueError("invalid prompt encoding source SHA256")
        matrix = value["gaussian"]
        if (
            not isinstance(matrix, list)
            or len(matrix) != 2
            or any(not isinstance(row, list) or len(row) != 128 for row in matrix)
            or any(type(item) not in (int, float) for row in matrix for item in row)
            or type(value["phase_scale"]) not in (int, float)
        ):
            raise ValueError("expected numeric Gaussian [2,128] and scalar phase scale")
        try:
            gaussian = np.asarray(matrix, dtype=np.float32)
            scale = np.asarray(value["phase_scale"], dtype=np.float32)
        except (ValueError, TypeError, OverflowError) as error:
            raise ValueError("invalid prompt encoding coefficients") from error
        if gaussian.shape != (2, 128) or not np.isfinite(gaussian).all() or scale.shape != () or not np.isfinite(scale):
            raise ValueError("expected finite float32 Gaussian [2,128] and scalar phase scale")
        if scale != np.float32(2 * np.pi):
            raise ValueError("unsupported Fourier phase scale")
        digest = hashlib.sha256(gaussian.astype("<f4").tobytes()).hexdigest()
        if digest != value["gaussian_sha256"]:
            raise ValueError("prompt encoding Gaussian SHA256 mismatch")
        gaussian.setflags(write=False)
        return cls(gaussian, np.float32(scale), source)

    @classmethod
    def from_package(cls, decoder: Path, resources: Path | None = None) -> PromptEncoding:
        """Read the supporting file from an explicit root or the standard flat/component package layout.

        Registry scripts verify its declared SHA-256 before use. Standalone callers can pass --resources when
        keeping component archives apart; no checkpoint, registry connection or MPK archive rewrite is needed.
        """
        roots = (Path(resources),) if resources is not None else (decoder.parent, decoder.parent.parent)
        files = [root / PROMPT_ENCODING_FILE for root in roots if (root / PROMPT_ENCODING_FILE).is_file()]
        if len(files) != 1:
            raise ValueError("expected one prompt-encoding.json in the package; provide --resources or rebuild it")
        path = files[0]
        if path.stat().st_size > 65536:
            raise ValueError("prompt encoding supporting file exceeds 64 KiB")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict) or set(value) != {PROMPT_ENCODING_METADATA}:
                raise ValueError("unexpected prompt encoding supporting-file fields")
            return cls.from_metadata({key: json.dumps(item, allow_nan=False) for key, item in value.items()})
        except (OSError, TypeError, ValueError) as error:
            raise ValueError(f"invalid prompt encoding supporting file: {path}") from error

    def __call__(self, coords: np.ndarray) -> np.ndarray:
        """Source FP32 MatMul then scale/Sin/Cos, without coordinate clipping or quantized phase lookup.

        The validated FP32 source/export rounds its first product, then fuses the second multiply/add. Other BLAS
        and float64 emulation can round differently at large phases; libm.fmaf preserves single rounding.
        """
        coords = np.asarray(coords)
        if coords.shape != INPUT_SHAPES["prompt_decoder"]["point_coords"] or coords.dtype != np.float32:
            raise ValueError("expected float32 point coordinates [1,2,1,6]")
        if not np.isfinite(coords).all():
            raise ValueError("prompt coordinates must be finite")
        # Repeated padding coordinates need the same encoding. Deduplicate raw bits so signed zeros are kept;
        # this changes neither arithmetic nor real-point capacity and avoids recomputing unused slots.
        bits, inverse = np.unique(
            np.ascontiguousarray(coords[0, :, 0, :].T).view(np.uint32), axis=0, return_inverse=True
        )
        points = bits.view(np.float32)
        fmaf = _fmaf()
        with np.errstate(over="ignore", invalid="ignore"):
            first = points[:, :1] * self.gaussian[0]
            projected = np.fromiter(
                (
                    fmaf(y, coefficient, initial)
                    for y, row in zip(points[:, 1], first)
                    for coefficient, initial in zip(self.gaussian[1], row)
                ),
                dtype=np.float32,
                count=len(points) * 128,
            ).reshape(len(points), 128)
            phase = np.multiply(projected, self.phase_scale, dtype=np.float32)
        if not np.isfinite(phase).all():
            raise ValueError("prompt Fourier phase overflows float32")
        encoding = np.concatenate((np.sin(phase), np.cos(phase)), axis=-1)
        return np.ascontiguousarray(encoding[inverse].T[None, :, None, :])


OUTPUT_SHAPES = {
    "image_encoder": {"image_embeddings": EMBEDDING_SHAPE},
    "prompt_decoder": {
        "masks": (1, MASKS, LOW_RES_MASK_SIZE, LOW_RES_MASK_SIZE),
        "iou_predictions": (1, MASKS),
    },
}
# segment_anything's pixel normalization, on 0-255 RGB.
PIXEL_MEAN = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
PIXEL_STD = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)


def tensor_name(component: str, name: str) -> str:
    """A component's tensor in the side-by-side ONNX that surgery.py builds."""
    return f"{component}_{name}"


def resized_size(height: int, width: int) -> tuple[int, int]:
    """The (height, width) segment_anything resizes an image to: its longer side becomes IMAGE_SIZE."""
    if height <= 0 or width <= 0:
        raise ValueError(f"expected positive image dimensions, got {(height, width)}")
    scale = IMAGE_SIZE / max(height, width)
    return int(height * scale + 0.5), int(width * scale + 0.5)


def preprocess(image: np.ndarray) -> np.ndarray:
    """An RGB uint8 HWC image as the encoder's float32 HWC input: resized bilinearly so its longer side is
    IMAGE_SIZE, normalized, then padded with zeros on the right and bottom."""
    from PIL import Image

    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"expected RGB uint8 HWC pixels, got {image.shape} {image.dtype}")
    height, width = resized_size(*image.shape[:2])
    resized = np.asarray(Image.fromarray(image).resize((width, height), Image.Resampling.BILINEAR), np.float32)
    padded = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), np.float32)
    padded[:height, :width] = (resized - PIXEL_MEAN) / PIXEL_STD
    return padded


def prompt_inputs(points: np.ndarray, types: list[str], original_size: tuple[int, int]) -> dict[str, np.ndarray]:
    """The decoder's inputs for up to MAX_POINTS real points, given as original-image (x, y) pixels.

    Coordinates are scaled to the resized image, shifted to pixel centres and normalized to [-1, 1], as
    segment_anything's prompt encoder does before its positional encoding. A no-box prompt gets exactly one
    active learned padding token; every unused slot has all-zero types and is excluded from attention.
    """
    points = np.asarray(points, np.float64).reshape(-1, 2)
    if len(points) != len(types) or not 0 < len(points) <= MAX_POINTS:
        raise ValueError(f"expected 1 to {MAX_POINTS} points with one type each, got {len(points)} and {len(types)}")
    if not np.isfinite(points).all():
        raise ValueError("prompt coordinates must be finite")
    if types.count("box_top_left") != types.count("box_bottom_right") or types.count("box_top_left") > 1:
        raise ValueError("a box prompt requires exactly one top-left and one bottom-right corner")
    unknown = sorted(set(types) - set(POINT_TYPES[1:]))
    if unknown:
        raise ValueError(f"unknown point types {unknown}; use {', '.join(POINT_TYPES[1:])}")
    height, width = original_size
    resized_height, resized_width = resized_size(height, width)
    scaled = points * (resized_width / width, resized_height / height)
    # Public SAM places clicks before box corners. Canonical ordering also minimizes floating reduction drift.
    order = [index for index, kind in enumerate(types) if not kind.startswith("box_")]
    order += [types.index(kind) for kind in ("box_top_left", "box_bottom_right") if kind in types]
    coords = np.zeros((POINTS, 2), np.float32)
    coords[: len(points)] = (scaled[order].astype(np.float32) + 0.5) / IMAGE_SIZE * 2 - 1
    if not np.isfinite(coords).all():
        raise ValueError("normalized prompt coordinates must be finite float32 values")
    one_hot = np.zeros((POINTS, len(POINT_TYPES)), np.float32)
    one_hot[np.arange(len(points)), [POINT_TYPES.index(types[index]) for index in order]] = 1
    if "box_top_left" not in types:
        one_hot[len(points), 0] = 1
    return {
        "point_coords": np.ascontiguousarray(coords.T[None, :, None, :]),
        "point_types": np.ascontiguousarray(one_hot.T[None, :, None, :]),
    }


def mask_layout(masks: np.ndarray, iou_predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Read the decoder's NCHW/NHWC masks and scores without copying a compatible float32 array."""
    masks = np.asarray(masks, np.float32)
    if masks.shape in ((LOW_RES_MASK_SIZE, LOW_RES_MASK_SIZE, MASKS), (1, LOW_RES_MASK_SIZE, LOW_RES_MASK_SIZE, MASKS)):
        masks = masks.reshape(LOW_RES_MASK_SIZE, LOW_RES_MASK_SIZE, MASKS).transpose(2, 0, 1)
    elif masks.shape in (
        (MASKS, LOW_RES_MASK_SIZE, LOW_RES_MASK_SIZE),
        (1, MASKS, LOW_RES_MASK_SIZE, LOW_RES_MASK_SIZE),
    ):
        masks = masks.reshape(MASKS, LOW_RES_MASK_SIZE, LOW_RES_MASK_SIZE)
    else:
        raise ValueError(f"unexpected mask output shape {masks.shape}")
    scores = np.asarray(iou_predictions, np.float32).ravel()
    if scores.size != MASKS:
        raise ValueError(f"expected {MASKS} IoU predictions, got {scores.size}")
    if not np.isfinite(masks).all() or not np.isfinite(scores).all():
        raise ValueError("decoder masks and IoU predictions must be finite")
    return masks, scores


def resize_mask_logits(logits: np.ndarray, original_size: tuple[int, int]) -> np.ndarray:
    """SAM's bilinear align_corners=False upsample, unpad and resize for one selected mask.

    OpenCV's linear float resize uses the same pixel-centre geometry. Pillow's bilinear downsampling
    uses a wider filter, so it must not stand in for this operation. Threshold only after both resizes.
    """
    import cv2

    height, width = original_size
    resized_height, resized_width = resized_size(height, width)
    logits = np.asarray(logits, np.float32)
    if logits.shape != (LOW_RES_MASK_SIZE, LOW_RES_MASK_SIZE) or not np.isfinite(logits).all():
        raise ValueError(f"expected finite {LOW_RES_MASK_SIZE}x{LOW_RES_MASK_SIZE} mask logits")
    upsampled = cv2.resize(logits, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(upsampled[:resized_height, :resized_width], (width, height), interpolation=cv2.INTER_LINEAR)


def image_mask(logits: np.ndarray, original_size: tuple[int, int]) -> np.ndarray:
    """One selected mask as independently owned binary pixels on the original image."""
    return resize_mask_logits(logits, original_size) > 0


def postprocess_masks(masks: np.ndarray, original_size: tuple[int, int]) -> np.ndarray:
    """All four masks' logits in image coordinates; applications needing one mask should call image_mask."""
    masks = np.asarray(masks, np.float32)
    if masks.shape == (1, MASKS, LOW_RES_MASK_SIZE, LOW_RES_MASK_SIZE):
        masks = masks[0]
    if masks.shape != (MASKS, LOW_RES_MASK_SIZE, LOW_RES_MASK_SIZE):
        raise ValueError(f"expected {MASKS}x{LOW_RES_MASK_SIZE}x{LOW_RES_MASK_SIZE} mask logits")
    return np.stack([resize_mask_logits(mask, original_size) for mask in masks])


def selected_mask(
    masks: np.ndarray, scores: np.ndarray, original_size: tuple[int, int], *, multimask: bool
) -> np.ndarray:
    """One final binary mask: the best of masks 1-3 for a click, or mask 0 for a box/multiple points."""
    masks, scores = mask_layout(masks, scores)
    index = 1 + int(np.argmax(scores[1:])) if multimask else 0
    return image_mask(masks[index], original_size)
