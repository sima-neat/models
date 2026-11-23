#!/usr/bin/env python3
"""Compile the converted FAN 2dfan2_alt face landmark ONNX into a Modalix MPK."""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Any

from afe.apis.defines import (
    CalibrationMethod,
    HistogramPercentileMethod,
    QuantizationParams,
    SkipCalibration,
    bfloat16_scheme,
    gen2_target,
    quantization_scheme,
)
from afe.apis.loaded_net import load_model
from afe.apis.model import Model as AfeModel
from afe.core.utils import convert_data_generator_to_iterable
from afe.ir.tensor_type import ScalarType
from afe.load.importers.general_importer import onnx_source
from landmarks import INPUT_SIZE, calibration_samples
from model_registry.model import load_model as load_metadata
from sima_utils.data.data_generator import DataGenerator

LOG = logging.getLogger("fld_68landmarks.compile")
MODEL_DIR = Path(__file__).resolve().parents[1]
MODEL_ID = "fld_68landmarks"
COMPONENT_ID = "model"
INPUT_SHAPE = (1, 3, INPUT_SIZE, INPUT_SIZE)
PRECISIONS = {"w16a16", "w8a8", "w8a16"}
CALIBRATION_METHODS = {"mse", "min_max", "entropy", "percentile"}


def calibration_method(parameters: dict[str, Any]) -> Any:
    """The declared calibration_method (default mse); percentile takes calibration_percentile, in percent."""
    name = str(parameters.get("calibration_method", "mse"))
    if name not in CALIBRATION_METHODS:
        raise ValueError(f"unsupported calibration method: {name}")
    if name == "percentile":
        value = float(parameters.get("calibration_percentile", float("nan")))
        if not math.isfinite(value) or not 0 < value <= 100:
            raise ValueError("calibration_percentile must be finite and in (0,100]")
        return HistogramPercentileMethod(percentile_value=value, num_bins=1024)
    return CalibrationMethod.from_str(name)


def _quantization(parameters: dict[str, Any]) -> QuantizationParams:
    """Keep each precision's calibration and per-token policy explicit."""
    precision = str(parameters["precision"])
    if precision == "w16a16":
        return QuantizationParams(
            calibration_method=SkipCalibration(),
            activation_quantization_scheme=bfloat16_scheme(),
            weight_quantization_scheme=bfloat16_scheme(),
            per_token_quant=False,
        )
    if precision == "w8a8":
        return QuantizationParams(
            calibration_method=calibration_method(parameters),
            activation_quantization_scheme=quantization_scheme(asymmetric=True, per_channel=False, bits=8),
            weight_quantization_scheme=quantization_scheme(asymmetric=False, per_channel=True, bits=8),
            per_token_quant=False,
        )
    if precision == "w8a16":
        # Preserve the qualified FAN calibration and per-token INT8-weight route.
        return QuantizationParams(
            calibration_method=calibration_method(parameters),
            activation_quantization_scheme=bfloat16_scheme(),
            weight_quantization_scheme=quantization_scheme(asymmetric=False, per_channel=True, bits=8),
            per_token_quant=True,
        )
    raise ValueError(f"unsupported precision: {precision}")


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
    if variant["target"] != "modalix" or precision not in PRECISIONS:
        raise ValueError("this recipe supports only the declared Modalix BF16, INT8 and INT8-weight variants")

    input_name = str(parameters["input_name"])
    input_shape = tuple(int(value) for value in parameters["input_shape"])
    if input_shape != INPUT_SHAPE:
        raise ValueError(f"unsupported FAN input shape: {input_shape}")
    calibrated = precision != "w16a16"
    if calibrated and dataset is None:
        raise ValueError(f"the {precision} variant requires --dataset from dataset-registry")

    component_dir.mkdir(parents=True, exist_ok=False)
    importer = onnx_source(str(source), {input_name: input_shape}, {input_name: ScalarType.float32})
    loaded = load_model(importer, target=gen2_target)
    if calibrated:
        sample_count = int(parameters["calibration_samples"])
        LOG.info("Loading %d AFW calibration faces", sample_count)
        samples = calibration_samples(dataset, sample_count)
        data = convert_data_generator_to_iterable(DataGenerator({input_name: samples}))
    else:
        # BF16 has no ranges to fit; do not run observers on a dummy image.
        data = None
    model_name = f"{MODEL_ID}_{variant_id}"
    quantized = loaded.quantize(
        calibration_data=data,
        quantization_config=_quantization(parameters),
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
    parser.add_argument("--dataset", type=Path, help="dataset-registry calibration archive (INT8 builds)")
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
