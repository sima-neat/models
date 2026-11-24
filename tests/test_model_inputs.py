"""Tests for the image listing, image reading, and 720p frame that model scripts share."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import numpy as np
import pytest
from model_inputs.frame import FILL, FRAME_HEIGHT, FRAME_WIDTH, letterbox_to_720p, read_rgb
from model_inputs.images import image_names, read_images

FILES = {
    "coco/cal2017/b.jpg": b"b",
    "coco/cal2017/a.JPG": b"a",
    # macOS tar writes this companion beside an image with extended attributes. It holds no image.
    "coco/cal2017/._a.JPG": b"\x00\x05\x16\x07",
    "coco/README.md": b"notes",
}


def _write_directory(root: Path) -> Path:
    for name, content in FILES.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(content)
    return root


def _write_archive(path: Path, prefix: str = "") -> Path:
    with tarfile.open(path, "w:gz") as archive:
        for name, content in FILES.items():
            member = tarfile.TarInfo(prefix + name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    return path


@pytest.fixture(params=["directory", "archive", "archive with ./ names"])
def dataset(request: pytest.FixtureRequest, tmp_path: Path) -> Path:
    if request.param == "directory":
        return _write_directory(tmp_path / "dataset")
    return _write_archive(tmp_path / "dataset.tar.gz", "./" if request.param.endswith("names") else "")


def test_image_names_are_the_sorted_images_without_macos_companions(dataset: Path) -> None:
    assert image_names(dataset, (".jpg",)) == ["coco/cal2017/a.JPG", "coco/cal2017/b.jpg"]


def test_read_images_returns_each_named_image(dataset: Path) -> None:
    names = ["coco/cal2017/b.jpg", "coco/cal2017/a.JPG"]

    assert read_images(dataset, names) == {"coco/cal2017/a.JPG": b"a", "coco/cal2017/b.jpg": b"b"}


def test_read_images_reports_a_missing_image(dataset: Path) -> None:
    with pytest.raises(ValueError, match="missing 1 requested images, such as coco/cal2017/c.jpg"):
        read_images(dataset, ["coco/cal2017/a.JPG", "coco/cal2017/c.jpg"])


def test_a_file_that_is_no_archive_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "images.txt"
    path.write_text("not an archive", encoding="utf-8")

    with pytest.raises(ValueError, match="neither a dataset directory nor a tar archive"):
        image_names(path, (".jpg",))
    with pytest.raises(ValueError, match="neither a dataset directory nor a tar archive"):
        read_images(path, ["a.jpg"])


def test_read_rgb_swaps_opencvs_channel_order(tmp_path: Path) -> None:
    import cv2

    path = tmp_path / "blue.png"
    assert cv2.imwrite(str(path), np.full((2, 3, 3), (255, 0, 0), np.uint8))

    image = read_rgb(path)

    assert image.shape == (2, 3, 3)
    assert image[0, 0].tolist() == [0, 0, 255]


def test_read_rgb_rejects_a_file_opencv_cannot_decode(tmp_path: Path) -> None:
    path = tmp_path / "broken.jpg"
    path.write_bytes(b"not an image")

    with pytest.raises(SystemExit, match="could not read .*broken.jpg"):
        read_rgb(path)


def test_letterbox_centers_the_image_in_a_720p_frame() -> None:
    frame = letterbox_to_720p(np.full((480, 640, 3), 200, np.uint8))

    assert frame.shape == (FRAME_HEIGHT, FRAME_WIDTH, 3)
    # A 4:3 image scales to 960x720, which leaves 160 fill columns on each side.
    assert (frame[:, :160] == FILL).all() and (frame[:, 1120:] == FILL).all()
    assert (frame[:, 160:1120] == 200).all()
