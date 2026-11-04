"""Read the registered, ordered 128-frame calibration archive without stacking it.

The archived pixel and preprocessed-tensor hashes preserve the QAT qualification
input. This reader neither selects new frames nor downloads or republishes data.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from contract import INPUT_NAME, DeploymentContract
from PIL import Image

ARCHIVE_ROOT = PurePosixPath("yolo26_tiny_drone_calibration_v1")
MANIFEST_NAME = str(ARCHIVE_ROOT / "manifest.json")
EXPECTED_IMAGES = 128
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024


def preprocess_rgb_image(image: Image.Image, contract: DeploymentContract) -> np.ndarray:
    """Preserve the original PIL bilinear, centered 114-fill, RGB / 255 recipe."""
    width, height = image.size
    scale = min(contract.model_size / width, contract.model_size / height)
    resized_width, resized_height = max(1, round(width * scale)), max(1, round(height * scale))
    left, top = (contract.model_size - resized_width) // 2, (contract.model_size - resized_height) // 2
    resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (contract.model_size, contract.model_size), (114, 114, 114))
    canvas.paste(resized, (left, top))
    return np.asarray(canvas, dtype=np.float32) / np.float32(255.0)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _members(stream: tarfile.TarFile) -> dict[str, tarfile.TarInfo]:
    result, total = {}, 0
    for member in stream.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or not member.isfile() or member.name in result:
            raise ValueError(f"unsafe calibration member: {member.name}")
        total += member.size
        if total > MAX_ARCHIVE_BYTES:
            raise ValueError("calibration archive exceeds its uncompressed size limit")
        result[member.name] = member
    return result


def _read(stream: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    source = stream.extractfile(member)
    if source is None:
        raise ValueError(f"cannot read calibration member: {member.name}")
    return source.read()


class CalibrationSamples:
    """A re-iterable InputValues collection, validated completely before SDK use.

    Each iteration yields one batch-1 NHWC float32 tensor. Unlike a one-shot
    generator this supports the compiler's repeated passes, with only one
    image tensor retained by this reader (the compiler owns its own retention).
    """

    def __init__(self, archive: Path, contract: DeploymentContract, expected_tensor_sha256: str):
        self.archive, self.contract = archive, contract
        self.expected_tensor_sha256 = expected_tensor_sha256
        self.archive_sha256 = _digest(archive)
        self.images: list[dict[str, str]] = []
        self._validate()

    def _validate(self) -> None:
        with tarfile.open(self.archive, "r:gz") as stream:
            members = _members(stream)
            if MANIFEST_NAME not in members:
                raise ValueError(f"calibration archive is missing {MANIFEST_NAME}")
            manifest = json.loads(_read(stream, members[MANIFEST_NAME]))
            images = manifest.get("images")
            preprocessing = f"rgb_letterbox_{self.contract.model_size}_fill_114_divide_255_nhwc_float32"
            if (
                manifest.get("schema_version") != 1
                or manifest.get("count") != EXPECTED_IMAGES
                or manifest.get("preprocessing") != preprocessing
                or manifest.get("tensor_sha256") != self.expected_tensor_sha256
                or not isinstance(images, list)
                or len(images) != EXPECTED_IMAGES
            ):
                raise ValueError("calibration manifest does not match the deployment contract")
            names = {MANIFEST_NAME}
            for index, image in enumerate(images):
                if not isinstance(image, dict):
                    raise ValueError(f"calibration image {index} is not an object")
                path = PurePosixPath(str(image.get("file", "")))
                sequence, digest = str(image.get("sequence", "")), str(image.get("sha256", ""))
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or not sequence
                    or path.parent != ARCHIVE_ROOT / "images" / sequence
                    or len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest)
                    or str(path) in names
                    or str(path) not in members
                ):
                    raise ValueError(f"invalid, duplicate or missing calibration image {index}")
                names.add(str(path))
            if names != set(members):
                raise ValueError("calibration archive has unexpected members")
            self.images = images
            tensor_digest = hashlib.sha256()
            for sample in self._samples(stream, members):
                tensor_digest.update(memoryview(sample).cast("B"))
            if tensor_digest.hexdigest() != self.expected_tensor_sha256:
                raise ValueError("calibration tensor digest mismatch")
        if _digest(self.archive) != self.archive_sha256:
            raise ValueError("calibration archive changed during validation")

    def _samples(self, stream: tarfile.TarFile, members: dict[str, tarfile.TarInfo]) -> Iterator[np.ndarray]:
        for image in self.images:
            payload = _read(stream, members[image["file"]])
            if hashlib.sha256(payload).hexdigest() != image["sha256"]:
                raise ValueError(f"calibration image digest mismatch: {image['file']}")
            with Image.open(io.BytesIO(payload)) as source:
                sample = preprocess_rgb_image(source.convert("RGB"), self.contract)
            if sample.shape != (self.contract.model_size, self.contract.model_size, 3) or sample.dtype != np.float32:
                raise ValueError(f"invalid preprocessed calibration image: {image['file']}")
            yield np.ascontiguousarray(sample)

    def __len__(self) -> int:
        return EXPECTED_IMAGES

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if _digest(self.archive) != self.archive_sha256:
            raise ValueError("calibration archive changed after validation")
        with tarfile.open(self.archive, "r:gz") as stream:
            members = _members(stream)
            for sample in self._samples(stream, members):
                yield {INPUT_NAME: sample[np.newaxis, ...]}
