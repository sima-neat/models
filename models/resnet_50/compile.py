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
    TensorDRAMLayout,
    TensorTessellateParameters,
    bfloat16_scheme,
    gen2_target,
    quantization_scheme,
)
from afe.apis.loaded_net import load_model
from afe.core.utils import convert_data_generator_to_iterable
from afe.ir.node import node_is_tuple
from afe.ir.tensor_type import ScalarType
from afe.load.importers.general_importer import onnx_source
from sima_utils.data.data_generator import DataGenerator

LOG = logging.getLogger("resnet_50.compile")
MODEL_DIR = Path(__file__).resolve().parent


def _variant(variant_id: str) -> dict[str, Any]:
    metadata = yaml.safe_load((MODEL_DIR / "model.yaml").read_text(encoding="utf-8"))
    matches = [item for item in metadata["compile_variants"] if item["id"] == variant_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one compile variant named {variant_id!r}")
    return matches[0]


def _tessellation(quantized_model: Any) -> dict[str, TensorTessellateParameters]:
    mla = quantized_model._net.nodes["MLA_0"]
    input_layout = TensorTessellateParameters(
        tile_shape=(0, 0, 0, 0), enable_mla=True, dram_layout=TensorDRAMLayout.HWC
    )
    output_layout = TensorTessellateParameters(
        tile_shape=(0, 0, 0, 0), enable_mla=True, dram_layout=TensorDRAMLayout.HWC16
    )
    parameters = {name: input_layout for name in mla.input_names}
    output_node = mla.ir.nodes[mla.ir.output_node_name]
    output_names = output_node.input_node_names if node_is_tuple(output_node) else [output_node.name]
    parameters.update({f"{name}_output": output_layout for name in output_names})
    return parameters


def compile_model(source: Path, output_dir: Path, variant_id: str) -> Path:
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
        # This deterministic input exercises the INT8 compile path, but is not
        # representative calibration data and does not qualify model accuracy.
        sample_count = int(parameters.get("calibration_samples", 1))
        if sample_count < 1:
            raise ValueError("calibration_samples must be at least 1")
        samples = np.random.default_rng(1).normal(
            0.0, 1.0, (sample_count, 224, 224, 3)
        ).astype(np.float32)
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
    tessellation = _tessellation(quantized) if parameters["mla_tessellation"] else None
    quantized.compile(output_path=str(output_dir), tessellate_parameters=tessellation)

    archives = sorted(output_dir.glob("*_mpk.tar.gz"))
    if len(archives) != 1:
        raise RuntimeError(f"expected exactly one MPK archive, found {len(archives)} in {output_dir}")
    return archives[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    source = args.model.expanduser().resolve()
    if not source.is_file():
        parser.error(f"model does not exist: {source}")
    archive = compile_model(source, args.output_dir.expanduser().resolve(), args.variant)
    LOG.info("Compiled MPK: %s", archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
