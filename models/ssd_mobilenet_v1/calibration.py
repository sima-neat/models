#!/usr/bin/env python3
"""Build and validate the immutable SSD-MobileNet V1 COCO calibration asset."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ARCHIVE_SHA256 = "ae4d46084839c7659c679a170366e8e2d143b1c42c911697bb59c5639e3bce20"
CALIBRATION_ARRAY_SHA256 = "ac4447c0b7a1b41ecd5c7d59528226dbc7b8f2e0d1340dd14dc285b0c316e25f"
VALIDATION_ARRAY_SHA256 = "c285dbc293b9cf24a25140bafbaad33768f7acef9014db21606d594a5bf2a6a1"
IMAGE_SIZE = 300
ANNOTATIONS_SHA256 = "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f"
CALIBRATION_IMAGE_IDS = (
    139,
    14038,
    16228,
    17899,
    18380,
    25096,
    33114,
    35279,
    39405,
    40471,
    54593,
    59386,
    60507,
    66231,
    67616,
    74209,
    76547,
    84241,
    92091,
    94944,
    102805,
    104669,
    125572,
    132544,
    134882,
    135561,
    139099,
    140583,
    159977,
    160556,
    171382,
    172595,
    173008,
    175364,
    177489,
    178028,
    179265,
    181753,
    184324,
    186422,
    189475,
    190236,
    190753,
    192670,
    193162,
    226147,
    226417,
    227399,
    229358,
    229997,
    231822,
    231831,
    245513,
    245576,
    246968,
    253452,
    254814,
    257169,
    259690,
    261888,
    273232,
    273711,
    286182,
    289393,
    298396,
    301421,
    303713,
    306893,
    312237,
    322163,
    323751,
    325031,
    326082,
    327769,
    338905,
    345027,
    349480,
    350148,
    356612,
    357816,
    357888,
    366884,
    367095,
    376322,
    379441,
    379533,
    381971,
    385997,
    387098,
    393569,
    396200,
    397133,
    405195,
    407614,
    407868,
    411530,
    413689,
    416885,
    419974,
    425221,
    425361,
    435081,
    435206,
    435208,
    441247,
    453341,
    455937,
    458663,
    470779,
    471789,
    472678,
    490936,
    491008,
    491216,
    492362,
    494913,
    500464,
    524850,
    530836,
    530975,
    540502,
    542625,
    543581,
    562818,
    569917,
    571313,
    572678,
    575970,
)
VALIDATION_IMAGE_IDS = (
    37740,
    37777,
    55528,
    59598,
    78266,
    84362,
    91615,
    92053,
    106389,
    119233,
    127092,
    156076,
    213255,
    268729,
    279278,
    279714,
    286553,
    288685,
    290179,
    294783,
    345466,
    397354,
    411938,
    428454,
    430073,
    438774,
    460494,
    546219,
    546823,
    565776,
    569030,
    572462,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def select_image_ids(annotation_path: Path) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if sha256_file(annotation_path) != ANNOTATIONS_SHA256:
        raise ValueError(f"COCO annotations SHA-256 mismatch: {annotation_path}")
    document = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories_by_image: defaultdict[int, set[int]] = defaultdict(set)
    for annotation in document["annotations"]:
        if not annotation.get("iscrowd", 0) and float(annotation.get("area", 0)) > 0:
            categories_by_image[int(annotation["image_id"])].add(int(annotation["category_id"]))

    def balanced_selection(candidates: list[int], count: int) -> tuple[int, ...]:
        coverage: Counter[int] = Counter()
        remaining = set(candidates)
        selected: list[int] = []
        for _ in range(count):
            image_id = max(
                remaining,
                key=lambda candidate: (
                    sum(1.0 / (1 + coverage[category]) for category in categories_by_image[candidate]),
                    len(categories_by_image[candidate]),
                    -candidate,
                ),
            )
            selected.append(image_id)
            remaining.remove(image_id)
            coverage.update(categories_by_image[image_id])
        return tuple(sorted(selected))

    all_image_ids = sorted(categories_by_image)
    calibration = balanced_selection(all_image_ids, 128)
    calibration_set = set(calibration)
    validation = balanced_selection([image_id for image_id in all_image_ids if image_id not in calibration_set], 32)
    return calibration, validation


def _validate_array(name: str, value: np.ndarray, count: int, expected_sha256: str | None) -> np.ndarray:
    array = np.asarray(value)
    expected_shape = (count, IMAGE_SIZE, IMAGE_SIZE, 3)
    if array.shape != expected_shape or array.dtype != np.float32:
        raise ValueError(f"{name} must be {expected_shape}/float32, found {array.shape}/{array.dtype}")
    if not np.isfinite(array).all() or float(array.min()) < -1.0 or float(array.max()) > 1.0:
        raise ValueError(f"{name} must contain finite RGB values normalized to [-1, 1]")
    digest = array_sha256(array)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"{name} SHA-256 mismatch: {digest}")
    return array


def validate_arrays(
    calibration: np.ndarray,
    validation: np.ndarray,
    calibration_ids: np.ndarray,
    validation_ids: np.ndarray,
    *,
    check_hashes: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    calibration_ids = np.asarray(calibration_ids)
    validation_ids = np.asarray(validation_ids)
    if calibration_ids.dtype != np.int64 or tuple(calibration_ids.tolist()) != CALIBRATION_IMAGE_IDS:
        raise ValueError("calibration image IDs do not match the pinned split")
    if validation_ids.dtype != np.int64 or tuple(validation_ids.tolist()) != VALIDATION_IMAGE_IDS:
        raise ValueError("validation image IDs do not match the pinned split")
    if set(calibration_ids.tolist()) & set(validation_ids.tolist()):
        raise ValueError("calibration and validation image IDs must be disjoint")
    expected_calibration = CALIBRATION_ARRAY_SHA256 if check_hashes else None
    expected_validation = VALIDATION_ARRAY_SHA256 if check_hashes else None
    return (
        _validate_array("calibration", calibration, len(CALIBRATION_IMAGE_IDS), expected_calibration),
        _validate_array("validation", validation, len(VALIDATION_IMAGE_IDS), expected_validation),
    )


def validate_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if sha256_file(path) != ARCHIVE_SHA256:
        raise ValueError(f"calibration archive SHA-256 mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        expected_files = ["calibration", "validation", "calibration_image_id", "validation_image_id"]
        if archive.files != expected_files:
            raise ValueError(f"calibration archive fields must be {expected_files}, found {archive.files}")
        return validate_arrays(
            archive["calibration"],
            archive["validation"],
            archive["calibration_image_id"],
            archive["validation_image_id"],
        )


def _preprocess_image(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as source:
        image = source.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32)
    # This is algebraically image / 127.5 - 1, matching the removed TF
    # preprocessor and the deployed CVU bilinear-resize path.
    return pixels * np.float32(1.0 / 127.5) - np.float32(1.0)


def _load_split(directory: Path, image_ids: tuple[int, ...]) -> np.ndarray:
    images = []
    for image_id in image_ids:
        path = directory / f"{image_id:012d}.jpg"
        if not path.is_file():
            raise FileNotFoundError(f"missing pinned COCO image: {path}")
        images.append(_preprocess_image(path))
    return np.stack(images).astype(np.float32, copy=False)


def regenerate(annotation_path: Path, image_dir: Path, destination: Path) -> Path:
    selected_calibration, selected_validation = select_image_ids(annotation_path)
    if selected_calibration != CALIBRATION_IMAGE_IDS or selected_validation != VALIDATION_IMAGE_IDS:
        raise RuntimeError("COCO category-balanced split no longer matches its pinned IDs")
    calibration = _load_split(image_dir, CALIBRATION_IMAGE_IDS)
    validation = _load_split(image_dir, VALIDATION_IMAGE_IDS)
    validate_arrays(
        calibration,
        validation,
        np.asarray(CALIBRATION_IMAGE_IDS, dtype=np.int64),
        np.asarray(VALIDATION_IMAGE_IDS, dtype=np.int64),
        check_hashes=False,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        calibration=calibration,
        validation=validation,
        calibration_image_id=np.asarray(CALIBRATION_IMAGE_IDS, dtype=np.int64),
        validation_image_id=np.asarray(VALIDATION_IMAGE_IDS, dtype=np.int64),
    )
    return destination


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = regenerate(args.annotations, args.images, args.output)
    print(f"archive={sha256_file(output)} size={output.stat().st_size}")
    with np.load(output, allow_pickle=False) as archive:
        print(f"calibration={array_sha256(archive['calibration'])}")
        print(f"validation={array_sha256(archive['validation'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
