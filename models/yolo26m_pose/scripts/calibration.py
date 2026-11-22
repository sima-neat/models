"""Letterbox images the way Ultralytics YOLO26 inference does, and select the INT8 calibration images."""

from __future__ import annotations

import json
import random
import tarfile
from pathlib import Path, PurePosixPath

import numpy as np
from model_inputs.images import image_names, read_images

IMAGE_SIZE = 640
FILL = 114
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".webp")
# Calibration draws only from cal2017, so no scored val2017 image calibrates the model.
CALIBRATION_SPLIT = "cal2017"
# Only images with at least one person whose keypoints are labelled calibrate the pose head.
KEYPOINT_ANNOTATIONS = "person_keypoints_cal2017.json"
CALIBRATION_SEED = 20260930


def letterbox_geometry(width: int, height: int) -> tuple[float, tuple[int, int], tuple[int, int, int, int]]:
    """Return Ultralytics' LetterBox gain, resized (width, height) and (left, top, right, bottom) padding."""
    gain = min(IMAGE_SIZE / height, IMAGE_SIZE / width)
    resized = (round(width * gain), round(height * gain))
    pad_x, pad_y = (IMAGE_SIZE - resized[0]) / 2, (IMAGE_SIZE - resized[1]) / 2
    # The 0.1 offsets put an odd padding pixel on the right or bottom, as Ultralytics does.
    return gain, resized, (round(pad_x - 0.1), round(pad_y - 0.1), round(pad_x + 0.1), round(pad_y + 0.1))


def decode_image(content: bytes) -> np.ndarray:
    """Decode an encoded image to RGB, ignoring EXIF orientation as the COCO annotations do."""
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


def preprocess_image(content: bytes) -> np.ndarray:
    """Return the model input for one encoded image: letterboxed RGB scaled to [0, 1], HWC float32."""
    return letterbox(decode_image(content)).astype(np.float32) / np.float32(255)


def _keypoint_annotations(dataset: Path) -> dict:
    content = None
    if dataset.is_dir():
        paths = sorted(dataset.rglob(KEYPOINT_ANNOTATIONS))
        content = paths[0].read_bytes() if paths else None
    elif tarfile.is_tarfile(dataset):
        with tarfile.open(dataset) as archive:
            for member in archive.getmembers():
                if member.isfile() and PurePosixPath(member.name).name == KEYPOINT_ANNOTATIONS:
                    stream = archive.extractfile(member)
                    content = stream.read() if stream else None
                    break
    else:
        raise ValueError(f"unsupported image dataset input: {dataset}")
    if content is None:
        raise ValueError(f"dataset has no {KEYPOINT_ANNOTATIONS}")
    return json.loads(content)


def keypoint_image_files(dataset: Path) -> set[str]:
    """Return the file names of the images with at least one person whose keypoints are labelled."""
    annotations = _keypoint_annotations(dataset)
    labelled = {annotation["image_id"] for annotation in annotations["annotations"] if annotation["num_keypoints"] > 0}
    return {image["file_name"] for image in annotations["images"] if image["id"] in labelled}


def calibration_names(dataset: Path, count: int) -> list[str]:
    """Return a seeded sample of the dataset's keypoint-labelled cal2017 images, in sampling order."""
    labelled = keypoint_image_files(dataset)
    names = [
        name
        for name in image_names(dataset, IMAGE_SUFFIXES)
        if PurePosixPath(name).parent.name == CALIBRATION_SPLIT and PurePosixPath(name).name in labelled
    ]
    if count < 1 or len(names) < count:
        raise ValueError(
            f"cannot select {count} calibration images from {len(names)} keypoint-labelled {CALIBRATION_SPLIT} images"
        )
    return random.Random(CALIBRATION_SEED).sample(names, count)


def calibration_images(dataset: Path, count: int) -> np.ndarray:
    """Return the selected calibration images as one NHWC float32 batch."""
    names = calibration_names(dataset, count)
    images = read_images(dataset, names)
    return np.stack([preprocess_image(images[name]) for name in names])
