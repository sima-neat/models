"""FAN's CPU steps and the AFW faces, shared by compile.py and accuracy.py.

The crop and the decode follow ibug's FANPredictor (MIT), in NumPy and OpenCV rather than PyTorch.

Crop: the face box's centre and mean side, the side enlarged by 1 / 0.55, a square around the centre, zero-padded
where it leaves the image, resized to 256x256 RGB in [0, 1].
Decode: each heatmap's peak; a disc of radius 0.1 * sqrt(64 * 64) around it; the centroid of the clamped heatmap in
that disc, at pixel centres (index + 0.5); then mapped back from the 64x64 heatmap to the image through the crop.
"""

from __future__ import annotations

import tarfile
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

CROP_RATIO = 0.55
INPUT_SIZE = 256
HEATMAP_SIZE = 64
LANDMARKS = 68
RADIUS = 0.1
DATASET_ID = "afw-faces-default"

# Integer peak-relative offsets. The support is fixed by FAN, not by the image.
_OFFSETS = np.arange(-int(RADIUS * HEATMAP_SIZE), int(RADIUS * HEATMAP_SIZE) + 1)
_DY, _DX = np.meshgrid(_OFFSETS, _OFFSETS, indexing="ij")
_DISC = _DY * _DY + _DX * _DX <= (RADIUS * HEATMAP_SIZE) ** 2


def checked_box(value: np.ndarray) -> np.ndarray:
    """A finite, nondegenerate [left, top, right, bottom] rectangle; negative origins are valid."""
    box = np.asarray(value, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
        raise ValueError("face/crop box must contain four finite coordinates with positive width and height")
    return box


def crop_box(face_box: np.ndarray) -> np.ndarray:
    """[left, top, right, bottom] of the square crop around a [left, top, right, bottom] face box, in whole pixels."""
    left, top, right, bottom = checked_box(face_box)
    centre = np.array([(left + right) / 2.0, (top + bottom) / 2.0])
    size = ((bottom - top) + (right - left)) / 2.0 / CROP_RATIO
    start = np.round(centre - size / 2.0)
    result = np.concatenate([start, np.round(start + size) + 1])
    if not np.isfinite(result).all() or np.abs(result).max() > np.iinfo(np.int32).max:
        raise ValueError("face crop exceeds signed 32-bit image coordinates")
    return result.astype(int)


def crop_rgb(image_rgb: np.ndarray, face_box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The exact resized [256, 256, 3] uint8 RGB crop, before normalization, and its box in the image."""
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or min(image_rgb.shape[:2]) <= 0:
        raise ValueError("face crop requires a nonempty HWC RGB image")
    if image_rgb.dtype != np.uint8:
        raise ValueError("face crop requires uint8 RGB pixels")
    box = crop_box(face_box)
    left, top, right, bottom = box
    pad = [max(0, -top), max(0, bottom - image_rgb.shape[0]), max(0, -left), max(0, right - image_rgb.shape[1])]
    padded = cv2.copyMakeBorder(image_rgb, *pad, cv2.BORDER_CONSTANT, value=(0, 0, 0)) if any(pad) else image_rgb
    patch = padded[top + pad[0] : bottom + pad[0], left + pad[2] : right + pad[2]]
    return cv2.resize(patch, (INPUT_SIZE, INPUT_SIZE)), box


def crop(image_rgb: np.ndarray, face_box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The model input [256, 256, 3] float32 in [0, 1] for one face, and its crop box in the image."""
    resized, box = crop_rgb(image_rgb, face_box)
    return resized.astype(np.float32) / np.float32(255), box


def _spatial_argmax(heatmaps: np.ndarray) -> np.ndarray:
    """First spatial peak per channel, without packing a full HWC-backed view into CHW."""
    hwc = heatmaps.transpose(1, 2, 0)
    if not hwc.flags.c_contiguous:
        return heatmaps.reshape(LANDMARKS, -1).argmax(axis=1)
    # Cache-local reductions for Neat's HWC output. Two first-index argmaxes
    # preserve the original flatten/argmax tie rule across and within blocks.
    block_size = 64
    tiles = hwc.reshape(HEATMAP_SIZE * HEATMAP_SIZE // block_size, block_size, LANDMARKS)
    maxima = tiles.max(axis=1)
    blocks = maxima.argmax(axis=0)
    selected = tiles[blocks, :, np.arange(LANDMARKS)]
    return blocks * block_size + selected.argmax(axis=1)


def decode(heatmaps: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[68, 64, 64] heatmaps of one face -> its [68, 2] landmarks in the image and [68] peak scores."""
    heatmaps = np.asarray(heatmaps, dtype=np.float32)
    if heatmaps.shape != (LANDMARKS, HEATMAP_SIZE, HEATMAP_SIZE) or not np.isfinite(heatmaps).all():
        raise ValueError("heatmaps must be finite [68, 64, 64] values")
    box = checked_box(box)
    flat = _spatial_argmax(heatmaps)
    rows = flat[:, None, None] // HEATMAP_SIZE + _DY
    columns = flat[:, None, None] % HEATMAP_SIZE + _DX
    valid = _DISC & (rows >= 0) & (rows < HEATMAP_SIZE) & (columns >= 0) & (columns < HEATMAP_SIZE)
    values = heatmaps[
        np.arange(LANDMARKS)[:, None, None], rows.clip(0, HEATMAP_SIZE - 1), columns.clip(0, HEATMAP_SIZE - 1)
    ]
    masked = np.where(valid, values, 0).clip(min=0)
    total = np.maximum(masked.sum(axis=(1, 2)), np.finfo(np.float32).eps)
    xs = (masked.sum(axis=1) * (columns[:, 0].astype(np.float32) + 0.5)).sum(axis=1) / total
    ys = (masked.sum(axis=2) * (rows[:, :, 0].astype(np.float32) + 0.5)).sum(axis=1) / total
    scores = heatmaps[np.arange(LANDMARKS), flat // HEATMAP_SIZE, flat % HEATMAP_SIZE]
    left, top, right, bottom = box
    landmarks = np.stack([xs * (right - left) / HEATMAP_SIZE + left, ys * (bottom - top) / HEATMAP_SIZE + top], axis=1)
    return landmarks, scores


# The Dataset Registry's afw-faces-default archive: cal_data/ holds face crops for calibration; test_data/ holds face
# crops, each with an iBUG .pts file of its 68 annotated points. Each crop's bounds are its face box.
CALIBRATION_SPLIT, TEST_SPLIT = "cal_data", "test_data"


def parse_pts(text: str) -> np.ndarray:
    """An iBUG .pts file's 68 (x, y) points."""
    points = np.array(text.split("{")[1].split("}")[0].split(), dtype=np.float64)
    if points.size != 2 * LANDMARKS or not np.isfinite(points).all():
        raise ValueError(f"expected {LANDMARKS} points, found {points.size / 2:g}")
    return points.reshape(LANDMARKS, 2)


def _files(dataset: Path, split: str) -> dict[str, bytes]:
    """A split's files by name, from an extracted directory or the archive; macOS metadata files are skipped."""

    def wanted(name: str) -> bool:
        parts = PurePosixPath(name).parts
        return len(parts) >= 2 and parts[-2] == split and not parts[-1].startswith(".")

    if dataset.is_dir():
        return {
            path.name: path.read_bytes()
            for path in sorted((dataset / split).glob("*"))
            if path.is_file() and not path.name.startswith(".")
        }
    found = {}
    with tarfile.open(dataset, "r:gz") as archive:
        for member in archive.getmembers():
            name = member.name.removeprefix("./")
            if member.isfile() and wanted(name):
                stream = archive.extractfile(member)
                if stream is not None:
                    found[PurePosixPath(name).name] = stream.read()
    return dict(sorted(found.items()))


def decode_rgb(content: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("unreadable image")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def whole_image_box(image: np.ndarray) -> np.ndarray:
    """A face crop's own bounds as its face box."""
    return np.array([0, 0, image.shape[1], image.shape[0]], dtype=np.float64)


def test_faces(dataset: Path) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Each annotated test face: its name, RGB image and 68 annotated points."""
    files = _files(dataset, TEST_SPLIT)
    faces = [
        (name, decode_rgb(content), parse_pts(files[name[:-4] + ".pts"].decode("utf-8")))
        for name, content in files.items()
        if name.lower().endswith(".jpg") and name[:-4] + ".pts" in files
    ]
    if not faces:
        raise ValueError(f"{dataset} has no annotated faces under {TEST_SPLIT}/")
    return faces


def calibration_samples(dataset: Path, count: int) -> np.ndarray:
    """The calibration faces, cropped as the model takes them: [count, 256, 256, 3] float32 RGB in [0, 1]."""
    images = [
        decode_rgb(content)
        for name, content in _files(dataset, CALIBRATION_SPLIT).items()
        if name.lower().endswith(".jpg")
    ]
    if len(images) != count:
        raise ValueError(f"calibration requires {count} faces, but {dataset} has {len(images)}")
    return np.stack([crop(image, whole_image_box(image))[0] for image in images])


def nme(landmarks: np.ndarray, truth: np.ndarray) -> float:
    """Mean point error over the inter-ocular distance: the outer eye corners, points 36 and 45."""
    if landmarks.shape != (LANDMARKS, 2) or truth.shape != (LANDMARKS, 2):
        raise ValueError("NME requires two [68, 2] point arrays")
    distance = np.linalg.norm(truth[36] - truth[45])
    if not np.isfinite(landmarks).all() or not np.isfinite(truth).all() or not np.isfinite(distance) or distance <= 0:
        raise ValueError("NME requires finite points and a positive inter-ocular distance")
    return float(np.linalg.norm(landmarks - truth, axis=1).mean() / distance)


def evaluate(faces, infer):
    """Shared MPK/FP32 scoring; infer takes one normalized HWC crop and returns CHW heatmaps."""
    errors = []
    for name, image, truth in faces:
        patch, box = crop(image, whole_image_box(image))
        try:
            points, _ = decode(infer(patch), box)
            errors.append(nme(points, truth))
        except ValueError as error:
            raise ValueError(f"{name}: {error}") from error
    if not errors:
        raise ValueError("accuracy requires at least one annotated face")
    return {"nme": float(np.mean(errors))}, {"median_nme": float(np.median(errors))}


def dataset_path(value: str) -> Path:
    """Validate the standalone scripts' one pinned Dataset Registry source argument."""
    name, separator, directory = value.partition("=")
    if name != DATASET_ID or not separator or not directory.strip():
        raise ValueError(f"--dataset must be {DATASET_ID}=<directory>")
    return Path(directory)
