"""Benchmark one compiled SuperGlue matcher MPK with Neat Core on one keypoint pair and save its report.

SuperGlue takes keypoints, not images. With --dataset yfcc100m-keypoints-val=<extracted directory> the benchmark
repeats two random.Random(0) images of that dataset, the SuperPoint keypoints accuracy.py matches; without it, a
fixed seeded pair of 512 keypoints. The Sinkhorn iterations and match extraction run in the application.
"""

import argparse
import random
from pathlib import Path

import model_report
import numpy as np
import pyneat

KEYPOINTS_DATASET = "yfcc100m-keypoints-val"
KEYPOINTS, DESCRIPTOR_SIZE = 512, 256
# The MPK's channels-last inputs, in its order.
INPUTS = ("keypoints0", "scores0", "descriptors0", "keypoints1", "scores1", "descriptors1")
# The fixed pair's image size: the longer side SuperPoint resizes YFCC images to.
FIXED_WIDTH, FIXED_HEIGHT = 1600, 1200


def normalize_keypoints(keypoints: np.ndarray, width: int, height: int) -> np.ndarray:
    """Map pixel (x, y) keypoints to SuperGlue's frame: centred on the image, divided by 0.7 times its longer side."""
    size = np.array([width, height], dtype=np.float64)
    return ((np.asarray(keypoints, dtype=np.float64) - size / 2) / (size.max() * 0.7)).astype(np.float32)


def dataset_images(dataset: Path) -> tuple[str, list]:
    """Two random.Random(0) images of the keypoint dataset: their names and (keypoints, scores, descriptors, size)."""
    names = (dataset / "images.txt").read_text(encoding="utf-8").split()
    arrays = {name: np.load(dataset / f"{name}.npy", allow_pickle=False)
              for name in ("image_sizes", "keypoints", "scores", "descriptors")}
    if arrays["keypoints"].shape != (len(names), KEYPOINTS, 2):
        raise SystemExit(f"{dataset} is not {KEYPOINTS_DATASET} with {KEYPOINTS} keypoints per image")
    chosen = random.Random(0).sample(range(len(names)), 2)
    images = [(arrays["keypoints"][i], arrays["scores"][i], arrays["descriptors"][i], tuple(arrays["image_sizes"][i]))
              for i in chosen]
    return f"{KEYPOINTS_DATASET} {names[chosen[0]]} and {names[chosen[1]]}", images


def fixed_images() -> tuple[str, list]:
    """Two seeded sets of 512 keypoints in a 1600x1200 image, with scores and unit descriptors."""
    rng = np.random.RandomState(0)
    images = []
    for _ in range(2):
        keypoints = rng.uniform((0, 0), (FIXED_WIDTH, FIXED_HEIGHT), (KEYPOINTS, 2))
        descriptors = rng.normal(0, 1, (KEYPOINTS, DESCRIPTOR_SIZE))
        descriptors /= np.linalg.norm(descriptors, axis=1, keepdims=True)
        images.append((keypoints, rng.uniform(0.005, 1, KEYPOINTS), descriptors, (FIXED_WIDTH, FIXED_HEIGHT)))
    return f"a seeded pair of {FIXED_WIDTH}x{FIXED_HEIGHT} images", images


def pair_tensors(images: list) -> list[np.ndarray]:
    """The pair as the MPK's channels-last inputs, each image's keypoints normalized by its size."""
    values = {}
    for side, (keypoints, scores, descriptors, (width, height)) in enumerate(images):
        values[f"keypoints{side}"] = normalize_keypoints(keypoints, width, height).reshape(1, 1, KEYPOINTS, 2)
        values[f"scores{side}"] = np.asarray(scores, np.float32).reshape(1, 1, KEYPOINTS, 1)
        values[f"descriptors{side}"] = np.asarray(descriptors, np.float32).reshape(1, 1, KEYPOINTS, DESCRIPTOR_SIZE)
    return [np.ascontiguousarray(values[name]) for name in INPUTS]


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mpk", required=True, help="compiled model archive")
parser.add_argument("--dataset", help=f"{KEYPOINTS_DATASET}=<extracted directory>")
parser.add_argument("--output", required=True, help="directory for model-report.json")
args = parser.parse_args()
if args.dataset:
    dataset_id, dataset_dir = args.dataset.split("=", 1)
    if dataset_id != KEYPOINTS_DATASET:
        raise SystemExit(f"--dataset must be {KEYPOINTS_DATASET}, got {dataset_id}")
    workload, images = dataset_images(Path(dataset_dir))
else:
    workload, images = fixed_images()

options = pyneat.ModelOptions()
options.preprocess.kind = pyneat.InputKind.Tensor
# The inputs are keypoint features, not images: Neat only quantizes and tessellates them.
options.preprocess.resize.enable = pyneat.AutoFlag.Off
options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
options.preprocess.normalize.enable = pyneat.AutoFlag.Off
model = pyneat.Model(args.mpk, options)
# Request 1000 samples and a 60 s minimum measurement duration.
benchmark_options = pyneat.BenchmarkOptions()
benchmark_options.num_samples = 1000
benchmark_options.min_duration_ms = 60000
benchmark_options.include_mla_layer_stats = True
benchmark_options.include_mla_stage_stats = True
benchmark_options.inputs = [pyneat.Tensor.from_numpy(value, copy=True) for value in pair_tensors(images)]
report = model_report.from_performance(
    model.benchmark(benchmark_options),
    {"model": model},
    workload=f"{workload}, {KEYPOINTS} keypoints each, on the tensor route",
)
report.save(args.output)
