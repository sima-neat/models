"""Read hash-bound, real-source calibration tensors from Dataset Registry.

The producer must preserve frame/episode identity and the original eight-step
flow trajectories. Synthetic probes or evaluation episodes are not calibration.
No hidden network access, pickle loading, or fallback data generation is allowed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from validation import COMPONENTS, SHAPES, digest

CHECKPOINT_SHA256 = "e649a703b224e4144fa068b73ae6ab3735cf60b81ed2ffe36898c1e26474d5fb"
SELECTION_SHA256 = "07f15a12ac7c649758ac26783e3632ed7aba91c6e15f6ebe3b46bb0dc33e305a"


def read_calibration(directory: Path, component: str, count: int) -> np.ndarray:
    if component not in COMPONENTS or type(count) is not int or count < 1:
        raise ValueError("Known component and positive integer sample count required")
    manifest = json.loads((directory / "calibration.json").read_text())
    if manifest.get("schema") != "lewam.calibration.v1" or manifest.get("source") != "source_trajectories":
        raise ValueError("Real source-trajectory calibration manifest required")
    if manifest.get("checkpoint_sha256") != CHECKPOINT_SHA256:
        raise ValueError("Calibration belongs to different weights")
    episodes = manifest.get("episode_ids", [])
    evaluation = manifest.get("evaluation_episode_ids", [])
    if not episodes or not evaluation or set(episodes) & set(evaluation):
        raise ValueError("Explicit disjoint calibration and evaluation episodes required")
    if any(type(value) is not int or value < 0 for value in episodes + evaluation):
        raise ValueError("Episode identities must be nonnegative integers")
    selection_path = directory / "selection.json"
    if selection_path.is_symlink() or digest(selection_path) != SELECTION_SHA256:
        raise ValueError("Frozen evaluation selection bytes changed")
    selected = json.loads(selection_path.read_text())["episodes"]
    if set(evaluation) != {row["episode"] for row in selected}:
        raise ValueError("Calibration exclusions differ from the actual frozen evaluation episodes")
    if set(manifest.get("components", {})) != set(COMPONENTS):
        raise ValueError("Calibration must cover exactly four components")
    row = manifest["components"][component]
    path = directory / f"{component}.npz"
    if path.is_symlink() or digest(path) != row["sha256"]:
        raise ValueError(f"Changed {component} calibration archive")
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != {"inputs", "episode_ids"}:
            raise ValueError("Calibration archive must bind every input to its source episode")
        value = saved["inputs"]
        ids = saved["episode_ids"]
    if len(value) < count:
        raise ValueError("Not enough real calibration samples; no repetition or padding")
    if value.dtype != np.float32 or value.shape[1:] != SHAPES[component][0][1:]:
        raise ValueError("Calibration inputs must use the declared FP32 NCHW logical interface")
    if not np.isfinite(value).all() or ids.shape != (len(value),) or ids.dtype.kind not in "iu":
        raise ValueError("Malformed or nonfinite calibration data")
    if not set(ids.tolist()) <= set(episodes):
        raise ValueError("Calibration sample has undeclared episode identity")
    return np.ascontiguousarray(value[:count].transpose(0, 2, 3, 1))
