#!/usr/bin/env python3
"""Compile the pinned ResNet-50 ONNX source into a Modalix MPK."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from afe.apis.defines import (
    CalibrationMethod,
    QuantizationParams,
    bfloat16_scheme,
    gen2_target,
    quantization_scheme,
)
from afe.apis.loaded_net import load_model
from afe.apis.model import Model as AfeModel
from afe.core.utils import convert_data_generator_to_iterable
from afe.ir.tensor_type import ScalarType
from afe.load.importers.general_importer import onnx_source
from model_registry.accuracy import load_calibration_samples
from sima_utils.data.data_generator import DataGenerator

LOG = logging.getLogger("resnet_50.compile")
MODEL_DIR = Path(__file__).resolve().parents[1]


def _variant(variant_id: str) -> dict[str, Any]:
    metadata = yaml.safe_load((MODEL_DIR / "model.yaml").read_text(encoding="utf-8"))
    matches = [item for item in metadata["build_profiles"] if item["id"] == variant_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one build profile named {variant_id!r}")
    profile = matches[0]
    binding = next(item for item in profile["components"] if item["id"] == "model")
    return {**profile, "parameters": binding["parameters"]}


def compile_model(
    source: Path,
    output_dir: Path,
    variant_id: str,
    dataset: Path | None = None,
) -> Path:
    variant = _variant(variant_id)
    parameters = variant["parameters"]
    precision = str(parameters.get("precision"))
    if variant["target"] != "modalix" or precision not in {"bf16", "int8"}:
        raise ValueError("this recipe supports only the declared Modalix BF16 and INT8 variants")

    input_name = str(parameters["input_name"])
    input_shape = tuple(int(value) for value in parameters["input_shape"])
    if input_shape != (1, 3, 224, 224):
        raise ValueError(f"unsupported ResNet-50 input shape: {input_shape}")

    output_dir.mkdir(parents=True, exist_ok=False)
    importer = onnx_source(str(source), {input_name: input_shape}, {input_name: ScalarType.float32})
    loaded = load_model(importer, target=gen2_target)

    if precision == "int8":
        if dataset is None:
            raise ValueError("the INT8 variant requires --dataset from dataset-registry")
        sample_count = int(parameters.get("calibration_samples", 0))
        LOG.info("Loading %d representative calibration samples", sample_count)
        samples = load_calibration_samples(MODEL_DIR, dataset, sample_count, output_layout="NHWC")
        activation_scheme = quantization_scheme(asymmetric=True, per_channel=False, bits=8)
        weight_scheme = quantization_scheme(asymmetric=False, per_channel=True, bits=8)
    else:
        # BF16 does not derive quantization ranges from representative data.
        # The API still requires an iterable, so provide one deterministic sample.
        samples = np.zeros((1, 224, 224, 3), dtype=np.float32)
        activation_scheme = weight_scheme = bfloat16_scheme()

    data = convert_data_generator_to_iterable(DataGenerator({input_name: samples}))
    model_name = f"resnet_50_{variant_id}"
    quantized = loaded.quantize(
        data,
        QuantizationParams(
            calibration_method=CalibrationMethod.from_str("mse"),
            activation_quantization_scheme=activation_scheme,
            weight_quantization_scheme=weight_scheme,
        ),
        model_name=model_name,
    )
    quantized.save(model_name=model_name, output_directory=str(output_dir))
    saved = AfeModel.load(model_name=f"{model_name}.sima", network_directory=str(output_dir))
    saved.compile(output_path=str(output_dir))

    archives = sorted(output_dir.glob("*_mpk.tar.gz"))
    if len(archives) != 1:
        raise RuntimeError(f"expected exactly one MPK archive, found {len(archives)} in {output_dir}")
    return archives[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-profile", required=True)
    parser.add_argument("--component", required=True)
    parser.add_argument("--dataset", type=Path, help="dataset-registry calibration archive")
    args = parser.parse_args()
    if args.component != "model":
        parser.error(f"unsupported component: {args.component}")
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    source = args.model.expanduser().resolve()
    if not source.is_file():
        parser.error(f"model does not exist: {source}")
    dataset = args.dataset.expanduser().resolve() if args.dataset else None
    archive = compile_model(source, args.output_dir.expanduser().resolve(), args.build_profile, dataset)
    LOG.info("Compiled MPK: %s", archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
