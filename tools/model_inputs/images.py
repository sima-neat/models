"""Find and read the images of a dataset, from its extracted directory or its tar archive.

A model's calibration.py imports this module, and compile.py and accuracy.py import calibration.py. It is therefore
a compile fingerprint input of every ONNX build, and it runs on a DevKit too, so it needs only the Python standard
library.
"""

from __future__ import annotations

import tarfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath


def image_names(dataset: Path, suffixes: tuple[str, ...]) -> list[str]:
    """The dataset's image paths relative to its root, sorted.

    Args:
        dataset: An extracted dataset directory, or its tar archive with any compression.
        suffixes: The lower-case file suffixes that mark an image, such as (".jpg", ".jpeg").

    Raises:
        ValueError: dataset is neither a directory nor a tar archive.
    """
    if dataset.is_dir():
        names = [path.relative_to(dataset).as_posix() for path in dataset.rglob("*") if path.is_file()]
    elif tarfile.is_tarfile(dataset):
        with tarfile.open(dataset) as archive:
            names = [_member_name(member) for member in archive.getmembers() if member.isfile()]
    else:
        raise ValueError(f"{dataset} is neither a dataset directory nor a tar archive")
    return sorted(name for name in names if _is_image(name, suffixes))


def read_images(dataset: Path, names: Iterable[str]) -> dict[str, bytes]:
    """Each named image's encoded bytes. An archive is read through once, because seeking in one is slow.

    Raises:
        ValueError: dataset is neither a directory nor a tar archive, or it holds no file of a requested name.
    """
    wanted = set(names)
    if dataset.is_dir():
        found = {name: (dataset / name).read_bytes() for name in wanted if (dataset / name).is_file()}
    elif tarfile.is_tarfile(dataset):
        found = {}
        with tarfile.open(dataset, "r|*") as archive:
            for member in archive:
                name = _member_name(member)
                if name in wanted and member.isfile():
                    found[name] = archive.extractfile(member).read()
    else:
        raise ValueError(f"{dataset} is neither a dataset directory nor a tar archive")
    missing = wanted - set(found)
    if missing:
        raise ValueError(f"{dataset} is missing {len(missing)} requested images, such as {sorted(missing)[0]}")
    return found


def _member_name(member: tarfile.TarInfo) -> str:
    """An archive member's path as a directory would give it: tar may write a leading "./"."""
    return member.name.removeprefix("./")


def _is_image(name: str, suffixes: tuple[str, ...]) -> bool:
    # macOS tar writes a "._<name>" companion beside a file that carries extended attributes. It has the image's
    # suffix and holds no image.
    return name.lower().endswith(suffixes) and not PurePosixPath(name).name.startswith("._")
