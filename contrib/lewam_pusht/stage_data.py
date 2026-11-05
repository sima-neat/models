"""Stage the original PushT evaluation source pack without inventing runtime data.

The frozen pack contains real start/goal observations, not native rollouts or
calibration. It cannot be passed directly to the staged accuracy adapter: reset
seeds, source RNG/feature traces and environment identities still need recording.
This command never uploads or creates Dataset Registry pins.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

import numpy as np

PINS = {
    "dataset-schema.json": "a7b707a9e5bc27dd88ef487e92610baa41c504eecfc701afc66a738d34e493e5",
    "selection.json": "07f15a12ac7c649758ac26783e3632ed7aba91c6e15f6ebe3b46bb0dc33e305a",
    "episodes.npz": "c5617c0948d10b767d1498ebe89f6f5d24b00dd24e4ad8e2e5d3ed2832308999",
}
DATASET_REVISION = "63346c94f4dbe62b4744261c6c99d63cbff3f6e6"
DATASET_ARCHIVE = "a075dcd49120844a1720cd3f650519f934a632b5c3e9a5a4e413ba8b020aaa43"


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def validate_pack(directory: Path) -> dict:
    for name, expected in PINS.items():
        path = directory / name
        if path.is_symlink() or digest(path) != expected:
            raise ValueError(f"Frozen evaluation source changed: {name}")
    selection = json.loads((directory / "selection.json").read_text())
    rows = selection["episodes"]
    if (
        selection["held_out"] is not False
        or selection["dataset_archive_sha256"] != DATASET_ARCHIVE
        or selection["pack_sha256"] != PINS["episodes.npz"]
        or len(rows) != 150
        or [row["seed"] for row in rows] != [42] * 50 + [0] * 50 + [1] * 50
        or len({row["episode"] for row in rows}) != 148
        or len({(row["episode"], row["start"], row["goal_step"]) for row in rows}) != 150
        or any(row["goal_step"] - row["start"] != 25 for row in rows)
    ):
        raise ValueError("Frozen selection protocol changed")
    shapes = {
        "pixels": ((224, 224, 3), np.uint8),
        "state": ((7,), np.float32),
        "proprio": ((4,), np.float32),
        "action": ((2,), np.float32),
    }
    expected = {f"{i:03}_{point}_{kind}" for i in range(150) for point in ("start", "goal") for kind in shapes}
    with np.load(directory / "episodes.npz", allow_pickle=False) as archive:
        if len(archive.files) != len(expected) or set(archive.files) != expected:
            raise ValueError("Unexpected evaluation source arrays")
        for name in expected:
            shape, dtype = shapes[name.rsplit("_", 1)[1]]
            value = archive[name]
            if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
                raise ValueError(f"Invalid evaluation source array: {name}")
    return selection


def stage(directory: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    validate_pack(directory)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pusht-stage-", dir=output.parent) as temporary:
        staged = Path(temporary) / "evaluation"
        staged.mkdir()
        for name, expected in PINS.items():
            shutil.copyfile(directory / name, staged / name)
            if digest(staged / name) != expected:
                raise ValueError(f"Evaluation source changed during copy: {name}")
        # Fixed metadata and gzip header make the Dataset Registry input reproducible.
        path = staged / "pusht-eval-source.tar.gz"
        with path.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT) as archive:
                for name in sorted(PINS):
                    member = tarfile.TarInfo(name)
                    member.size = (staged / name).stat().st_size
                    member.mode, member.mtime = 0o644, 0
                    with (staged / name).open("rb") as source:
                        archive.addfile(member, source)
        receipt = {
            "schema": "lewam.evaluation-source-stage.v1",
            "status": "STAGED_NOT_PUBLISHED",
            "scope": "Original source fixtures only; not calibration or a qualified runtime evaluation bundle",
            "held_out": False,
            "selection_rows": 150,
            "unique_episodes": 148,
            "dataset_revision": DATASET_REVISION,
            "dataset_archive_sha256": DATASET_ARCHIVE,
            "files": {
                name: {"sha256": digest(staged / name), "size_bytes": (staged / name).stat().st_size}
                for name in sorted(PINS)
            },
            "archive": {"file": path.name, "sha256": digest(path), "size_bytes": path.stat().st_size},
            "runtime_adapter_missing": [
                "source reset seeds",
                "source time/horizon/RNG traces",
                "physics/rendering/RNG package identities",
                "source-adapter parity",
            ],
        }
        (staged / "evaluation-stage.json").write_text(json.dumps(receipt, indent=2) + "\n")
        staged.rename(output)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stage(args.evaluation_pack, args.output)


if __name__ == "__main__":
    main()
