"""Download official COCO validation data and the frozen training calibration subset.

Usage: python download_data.py --output /path/to/coco

Local alternative explicitly requested when Artifactory credentials are absent.
Validation is the complete official 5,000-image split. Calibration downloads
only its 128 selected training images, verified by calibration_samples.py.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import urllib.request
import zipfile
import zlib

from calibration import verified_images

ROOT_URL = "https://s3.amazonaws.com/images.cocodataset.org/"
SOURCES = {
    "val2017.zip": ("zips/val2017.zip", "4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05", 815585330),
    "annotations_trainval2017.zip": ("annotations/annotations_trainval2017.zip", "113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268", 252907541),
}


def download(directory):
    directory.mkdir(parents=True, exist_ok=True)
    provenance = {}
    for name, (relative, expected, size) in SOURCES.items():
        path = directory / name
        if not path.exists():
            temporary = path.with_suffix(".partial")
            urllib.request.urlretrieve(ROOT_URL + relative, temporary)
            temporary.replace(path)
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if path.stat().st_size != size or actual != expected:
            raise ValueError(f"official COCO archive checksum mismatch: {path}")
        provenance[name] = {"url": ROOT_URL + relative, "sha256": actual, "size_bytes": size}
    with zipfile.ZipFile(directory / "val2017.zip") as archive:
        names = [name for name in archive.namelist() if re.fullmatch(r"val2017/\d{12}\.jpg", name)]
        if len(names) != 5000:
            raise ValueError("expected all 5000 official validation images")
        for name in names:
            path = directory / name
            if not path.exists():
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(archive.read(name))
            info = archive.getinfo(name)
            content = path.read_bytes()
            if len(content) != info.file_size or zlib.crc32(content) != info.CRC:
                raise ValueError(f"cached validation image differs from the official archive: {path}")
    with zipfile.ZipFile(directory / "annotations_trainval2017.zip") as archive:
        content = archive.read("annotations/instances_val2017.json")
    if hashlib.sha256(content).hexdigest() != "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f":
        raise ValueError("validation annotation checksum mismatch")
    (directory / "annotations").mkdir(exist_ok=True)
    (directory / "annotations/instances_val2017.json").write_bytes(content)
    verified_images(directory / "cal128", download_missing=True)
    (directory / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    download(parser.parse_args().output)
