"""Tile DOTA images as Ultralytics' DOTA split does, letterbox each tile as YOLO26 inference does, and select the
INT8 calibration tiles."""

from __future__ import annotations

import math
import tarfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath

import numpy as np

IMAGE_SIZE = 1024
FILL = 114
# Ultralytics' DOTA split cuts 1024-pixel windows that overlap by 200 pixels.
TILE_STEP = IMAGE_SIZE - 200
# The dota-v1-0 cal variant lists its images in cal.txt, beside cal/images and cal/labelTxt.
CALIBRATION_LIST = "cal.txt"


def letterbox_geometry(width: int, height: int) -> tuple[float, tuple[int, int], tuple[int, int, int, int]]:
    """Return Ultralytics' LetterBox gain, resized (width, height) and (left, top, right, bottom) padding."""
    gain = min(IMAGE_SIZE / height, IMAGE_SIZE / width)
    resized = (round(width * gain), round(height * gain))
    pad_x, pad_y = (IMAGE_SIZE - resized[0]) / 2, (IMAGE_SIZE - resized[1]) / 2
    # The 0.1 offsets put an odd padding pixel on the right or bottom, as Ultralytics does.
    return gain, resized, (round(pad_x - 0.1), round(pad_y - 0.1), round(pad_x + 0.1), round(pad_y + 0.1))


def decode_image(content: bytes) -> np.ndarray:
    """Decode an encoded image to RGB, ignoring EXIF orientation as the DOTA annotations do."""
    import cv2

    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError("could not decode image")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def letterbox(image: np.ndarray) -> np.ndarray:
    """Scale an RGB image to fit IMAGE_SIZE, up or down, and pad it to a centered square."""
    import cv2

    height, width = image.shape[:2]
    _, resized, (left, top, right, bottom) = letterbox_geometry(width, height)
    if resized != (width, height):
        image = cv2.resize(image, resized, interpolation=cv2.INTER_LINEAR)
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(FILL, FILL, FILL))


def preprocess_tile(tile: np.ndarray) -> np.ndarray:
    """Return the model input for one RGB tile: letterboxed and scaled to [0, 1], HWC float32."""
    return letterbox(tile).astype(np.float32) / np.float32(255)


def tile_windows(width: int, height: int) -> list[tuple[int, int, int, int]]:
    """Return the (x0, y0, x1, y1) windows of Ultralytics' DOTA split for one image, in its order.

    A side up to IMAGE_SIZE gets one window from 0, which the tile crop ends at the image edge. A longer side gets
    windows every TILE_STEP pixels, the last moved back to end at the edge. Every window of an image then covers
    the same share of it, so Ultralytics' coverage filter keeps them all.
    """

    def starts(size: int) -> list[int]:
        if size <= IMAGE_SIZE:
            return [0]
        values = [TILE_STEP * index for index in range(math.ceil((size - IMAGE_SIZE) / TILE_STEP + 1))]
        values[-1] = min(values[-1], size - IMAGE_SIZE)
        return values

    return [(x, y, x + IMAGE_SIZE, y + IMAGE_SIZE) for x in starts(width) for y in starts(height)]


def label_centres(content: bytes) -> np.ndarray:
    """Return the centre of every object in a DOTA label file: the mean of its four corners."""
    rows = [line.split()[:8] for line in content.decode().splitlines() if len(line.split()) >= 9]
    return np.asarray(rows, dtype=np.float64).reshape(-1, 4, 2).mean(axis=1)


def busiest_window(width: int, height: int, centres: np.ndarray) -> tuple[int, int, int, int]:
    """Return the first window that holds the most object centres."""
    windows = tile_windows(width, height)
    x, y = centres.T
    counts = [np.count_nonzero((x >= x0) & (x < x1) & (y >= y0) & (y < y1)) for x0, y0, x1, y1 in windows]
    return windows[int(np.argmax(counts))]


def _dataset_files(dataset: Path, wanted: Callable[[str], bool]) -> dict[str, bytes]:
    """Return the files wanted() accepts by their path below the dataset root.

    The root is the given directory, or the single top-level directory of the dataset-registry tarball.
    """
    if dataset.is_dir():
        paths = (path for path in sorted(dataset.rglob("*")) if path.is_file())
        return {name: path.read_bytes() for path in paths if wanted(name := path.relative_to(dataset).as_posix())}
    if tarfile.is_tarfile(dataset):
        files = {}
        with tarfile.open(dataset) as archive:
            # Reading members in archive order streams the compressed tarball once.
            for member in archive:
                name = PurePosixPath(*PurePosixPath(member.name).parts[1:]).as_posix()
                if member.isfile() and wanted(name):
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise ValueError(f"could not read {member.name} from the dataset")
                    files[name] = stream.read()
        return files
    raise ValueError(f"unsupported calibration dataset input: {dataset}")


def calibration_names(dataset: Path, count: int) -> list[str]:
    """Return the first count image names of cal.txt."""
    listing = _dataset_files(dataset, lambda name: name == CALIBRATION_LIST)
    if CALIBRATION_LIST not in listing:
        raise ValueError(f"the calibration dataset has no {CALIBRATION_LIST}")
    names = listing[CALIBRATION_LIST].decode().split()
    if count < 1 or len(names) < count:
        raise ValueError(f"cannot select {count} calibration images from {len(names)} listed images")
    return names[:count]


def calibration_tiles(dataset: Path, count: int) -> np.ndarray:
    """Return one tile from each of the first count cal.txt images as an NHWC float32 batch.

    Each image gives its window with the most labelled object centres, so calibration sees objects rather than
    empty ground or water.
    """
    names = calibration_names(dataset, count)
    stems = {PurePosixPath(name).stem for name in names}
    files = _dataset_files(
        dataset,
        lambda name: name.startswith(("cal/images/", "cal/labelTxt/")) and PurePosixPath(name).stem in stems,
    )
    tiles = []
    for name in names:
        image = decode_image(files[f"cal/images/{name}"])
        height, width = image.shape[:2]
        labels = files[f"cal/labelTxt/{PurePosixPath(name).stem}.txt"]
        x0, y0, x1, y1 = busiest_window(width, height, label_centres(labels))
        tiles.append(preprocess_tile(image[y0:y1, x0:x1]))
    return np.stack(tiles)
