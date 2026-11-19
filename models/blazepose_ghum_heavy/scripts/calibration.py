"""Crop a person from a labelled box for BlazePose, and select the INT8 calibration people.

MediaPipe's pose_detection_to_roi.pbtxt and pose_landmark_by_roi_cpu.pbtxt (tag v0.10.26) build a square from two
keypoints of the person detector, rotate it upright, scale it by 1.25, repeat the image's edge outside it
(ImageToTensorCalculator's default BORDER_REPLICATE), and resize it bilinearly to 256x256 in [0, 1]. A labelled
dataset gives a box instead: this crop centres on the box, takes 1.25 times its longer side and does not rotate.
"""

from __future__ import annotations

import json
import random
import tarfile
from collections.abc import Callable, Sequence
from pathlib import Path, PurePosixPath

import numpy as np

INPUT_SIZE = 256
ROI_SCALE = 1.25
# Calibration draws only from cal2017, so no scored val2017 person calibrates the model.
CALIBRATION_SPLIT = "cal2017"
KEYPOINT_ANNOTATIONS = "person_keypoints_cal2017.json"
CALIBRATION_SEED = 20260930


def crop_square(box: Sequence[float]) -> tuple[float, float, float]:
    """The left, top and side of the square MediaPipe crops around a person's x, y, width, height box."""
    x, y, width, height = box
    side = ROI_SCALE * max(width, height)
    if side <= 0:
        raise ValueError(f"empty person box {list(box)}")
    return x + (width - side) / 2, y + (height - side) / 2, side


def crop(image: np.ndarray, left: float, top: float, side: float) -> np.ndarray:
    """The model input for one square of an RGB image: 256x256 HWC float32 in [0, 1], its corners on the square's."""
    import cv2

    scale = INPUT_SIZE / side
    matrix = np.array([[scale, 0, -left * scale], [0, scale, -top * scale]])
    pixels = cv2.warpAffine(
        image, matrix, (INPUT_SIZE, INPUT_SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    return pixels.astype(np.float32) / np.float32(255)


def decode_image(content: bytes) -> np.ndarray:
    """Decode an encoded image to RGB, ignoring EXIF orientation as the COCO annotations do."""
    import cv2

    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError("could not decode image")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _read_files(dataset: Path, wanted: Callable[[PurePosixPath], bool]) -> dict[PurePosixPath, bytes]:
    """The files of an extracted directory or a .tar.gz archive whose relative path wanted accepts."""
    if dataset.is_dir():
        paths = (PurePosixPath(path.relative_to(dataset).as_posix()) for path in dataset.rglob("*") if path.is_file())
        return {path: (dataset / path).read_bytes() for path in paths if wanted(path)}
    found = {}
    with tarfile.open(dataset, "r|gz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            stream = archive.extractfile(member) if member.isfile() and wanted(path) else None
            if stream is not None:
                found[path] = stream.read()
    return found


def calibration_people(dataset: Path, count: int) -> list[dict]:
    """A seeded sample of the cal2017 people accuracy.py would score: non-crowd, with labelled keypoints."""
    annotations = list(_read_files(dataset, lambda path: path.name == KEYPOINT_ANNOTATIONS).values())
    if len(annotations) != 1:
        raise ValueError(f"expected one {KEYPOINT_ANNOTATIONS} in {dataset}, found {len(annotations)}")
    coco = json.loads(annotations[0])
    files = {image["id"]: image["file_name"] for image in coco["images"]}
    people = [
        {**person, "file_name": files[person["image_id"]]}
        for person in coco["annotations"]
        if person["num_keypoints"] and not person["iscrowd"]
    ]
    if count < 1 or len(people) < count:
        raise ValueError(
            f"cannot select {count} calibration people from {len(people)} labelled {CALIBRATION_SPLIT} people"
        )
    return random.Random(CALIBRATION_SEED).sample(people, count)


def calibration_images(dataset: Path, count: int) -> np.ndarray:
    """The selected calibration people, each cropped as the model is fed, as one NHWC float32 batch."""
    people = calibration_people(dataset, count)
    names = {person["file_name"] for person in people}
    images = {
        path.name: content
        for path, content in _read_files(
            dataset, lambda path: path.parent.name == CALIBRATION_SPLIT and path.name in names
        ).items()
    }
    if set(images) != names:
        raise ValueError(f"{dataset} lacks {CALIBRATION_SPLIT} images {sorted(names - set(images))}")
    return np.stack([
        crop(decode_image(images[person["file_name"]]), *crop_square(person["bbox"])) for person in people
    ])
