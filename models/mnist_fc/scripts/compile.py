#!/usr/bin/env python3
"""Compile the converted MNIST-MLP ONNX source into a Modalix MPK."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
from afe.apis.defines import CalibrationMethod, QuantizationParams, bfloat16_scheme, gen2_target, quantization_scheme
from afe.apis.loaded_net import load_model
from afe.apis.model import Model as AfeModel
from afe.core.utils import convert_data_generator_to_iterable
from afe.ir.tensor_type import ScalarType
from afe.load.importers.general_importer import onnx_source
from calibration import calibration_samples
from model_registry.model import load_model as load_metadata
from sima_utils.data.data_generator import DataGenerator

LOG = logging.getLogger("mnist_fc.compile")
MODEL_DIR = Path(__file__).resolve().parents[1]
MODEL_ID = "mnist_fc"
COMPONENT_ID = "model"
INPUT_SHAPE = (1, 1, 28, 28)


def _metadata() -> dict[str, Any]:
    return load_metadata(MODEL_DIR)


def _variant(metadata: dict[str, Any], variant_id: str) -> dict[str, Any]:
    matches = [item for item in metadata["build_profiles"] if item["id"] == variant_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one build profile named {variant_id!r}")
    profile = matches[0]
    binding = next(item for item in profile["components"] if item["id"] == COMPONENT_ID)
    return {**profile, "parameters": binding["parameters"]}


def compile_model(source: Path, component_dir: Path, variant_id: str, dataset: Path | None) -> AfeModel:
    variant = _variant(_metadata(), variant_id)
    parameters = variant["parameters"]
    precision = str(parameters.get("precision"))
    if variant["target"] != "modalix" or precision not in {"w16a16", "w8a8"}:
        raise ValueError("this recipe supports only the declared Modalix BF16 and INT8 variants")

    input_name = str(parameters["input_name"])
    input_shape = tuple(int(value) for value in parameters["input_shape"])
    if input_shape != INPUT_SHAPE:
        raise ValueError(f"unsupported MNIST input shape: {input_shape}")

    if precision == "w8a8" and dataset is None:
        raise ValueError("the INT8 variant requires --dataset from dataset-registry")

    component_dir.mkdir(parents=True, exist_ok=False)
    importer = onnx_source(str(source), {input_name: input_shape}, {input_name: ScalarType.float32})
    loaded = load_model(importer, target=gen2_target)
    if precision == "w8a8":
        sample_count = int(parameters["calibration_samples"])
        LOG.info("Loading %d MNIST training images for calibration", sample_count)
        samples = calibration_samples(dataset, sample_count)
        activation_scheme = quantization_scheme(asymmetric=True, per_channel=False, bits=8)
        weight_scheme = quantization_scheme(asymmetric=False, per_channel=True, bits=8)
    else:
        # BF16 does not derive quantization ranges from representative data.
        # The API still requires an iterable, so provide one deterministic NHWC sample.
        samples = np.zeros((1, 28, 28, 1), dtype=np.float32)
        activation_scheme = weight_scheme = bfloat16_scheme()
    data = convert_data_generator_to_iterable(DataGenerator({input_name: samples}))
    model_name = f"{MODEL_ID}_{variant_id}"
    quantized = loaded.quantize(
        data,
        QuantizationParams(
            calibration_method=CalibrationMethod.from_str("mse"),
            activation_quantization_scheme=activation_scheme,
            weight_quantization_scheme=weight_scheme,
        ),
        model_name=model_name,
    )
    quantized.save(model_name=model_name, output_directory=str(component_dir))
    saved = AfeModel.load(model_name=f"{model_name}.sima", network_directory=str(component_dir))
    saved.compile(output_path=str(component_dir))

    archives = sorted(component_dir.glob("*_mpk.tar.gz"))
    if len(archives) != 1:
        raise RuntimeError(f"expected exactly one MPK archive, found {len(archives)} in {component_dir}")
    LOG.info("Compiled MPK: %s", archives[0])
    return saved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-profile", required=True)
    parser.add_argument("--compile-invocation", required=True)
    parser.add_argument("--dataset", type=Path, help="dataset-registry calibration archive (INT8 only)")
    args = parser.parse_args()
    if args.compile_invocation != args.build_profile:
        parser.error("a build profile compiles in one invocation named after it")
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    source = args.model.expanduser().resolve()
    if not source.is_file():
        parser.error(f"model does not exist: {source}")
    dataset = args.dataset.expanduser().resolve() if args.dataset else None
    compile_model(source, args.output_dir.expanduser().resolve() / COMPONENT_ID, args.build_profile, dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
