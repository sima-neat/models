"""Benchmark LightGlue's matcher MPK with Neat Core on one yfcc100m-keypoints pair and save its report.

The inputs are a fixed val pair's 512 normalized keypoints and descriptors per image, on Neat's tensor route; Core
times the MPK on them. Match extraction on the CPU is not part of the measurement.
"""

import argparse
import random
from pathlib import Path

import model_report
import numpy as np
import pyneat
from calibration import INPUT_SHAPES, Pair, dataset_root, pair_inputs, read_pairs


def tensor_options() -> pyneat.ModelOptions:
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    # The script normalizes the keypoints; Neat only quantizes and tessellates the inputs.
    options.preprocess.resize.enable = pyneat.AutoFlag.Off
    options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
    options.preprocess.normalize.enable = pyneat.AutoFlag.Off
    return options


def input_tensors(inputs: dict[str, np.ndarray]) -> list[pyneat.Tensor]:
    """The matcher's inputs as Neat tensors in EV74 memory. Neat reads a [1, 512, C] input as height 1, width 512 and
    C channels, so each array goes in as [1, 1, 512, C]."""
    return [
        pyneat.Tensor.from_numpy(inputs[name].reshape(1, 1, *shape[1:]), copy=True, memory=pyneat.TensorMemory.EV74)
        for name, shape in INPUT_SHAPES.items()
    ]


def benchmark_pair(dataset: Path) -> tuple[Pair, dict[str, np.ndarray]]:
    """One val pair, chosen with a fixed seed so every run measures the same input."""
    root = dataset_root(dataset)
    pair = random.Random(0).choice(sorted(read_pairs(root), key=lambda item: (item.name0, item.name1)))
    return pair, pair_inputs(root, pair)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", required=True, help="compiled model archive")
    parser.add_argument("--dataset", required=True, help="yfcc100m-keypoints-val=<extracted directory>")
    parser.add_argument("--output", required=True, help="directory for model-report.json")
    args = parser.parse_args()
    dataset_id, dataset_dir = args.dataset.split("=", 1)
    pair, inputs = benchmark_pair(Path(dataset_dir))

    model = pyneat.Model(args.mpk, tensor_options())
    # Request 1000 samples and a 60 s minimum measurement duration.
    options = pyneat.BenchmarkOptions()
    options.num_samples = 1000
    options.min_duration_ms = 60000
    options.include_mla_layer_stats = True
    options.include_mla_stage_stats = True
    options.inputs = input_tensors(inputs)
    report = model_report.from_performance(
        model.benchmark(options),
        {"model": model},
        workload=f"{dataset_id} pair {pair.name0} and {pair.name1}, 512 SuperPoint keypoints per image, tensor inputs",
    )
    report.save(args.output)


if __name__ == "__main__":
    main()
