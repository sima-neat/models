#!/usr/bin/env python3
"""Compile the pinned ResNeXt-101 32x8d ONNX source into a Modalix MPK and measure its numerical fidelity."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from afe.apis.defines import (
    CalibrationMethod,
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
from calibration import calibration_samples, fidelity_names, preprocess_image
from model_inputs.images import read_images
from model_registry.compile_script import compiler_log_level, component_parameters, parse_arguments
from model_registry.fidelity import compare_with_onnx, write_fidelity_report
from model_registry.model import load_model as load_metadata
from sima_utils.data.data_generator import DataGenerator

LOG = logging.getLogger("resnext_101_32x8d.compile")
MODEL_DIR = Path(__file__).resolve().parents[1]
MODEL_ID = "resnext_101_32x8d"
COMPONENT_ID = "model"


def _quantization(precision: str) -> QuantizationParams:
    """The compiler's activation and weight formats for a precision. Only INT8 activations need calibration."""
    if precision == "w8a8":
        return QuantizationParams(
            # MSE picks each activation range from the calibration images.
            calibration_method=CalibrationMethod.from_str("mse"),
            activation_quantization_scheme=quantization_scheme(asymmetric=True, per_channel=False, bits=8),
            weight_quantization_scheme=quantization_scheme(asymmetric=False, per_channel=True, bits=8),
        )
    if precision == "w8a16":
        return QuantizationParams(
            calibration_method=SkipCalibration(),
            activation_quantization_scheme=bfloat16_scheme(),
            weight_quantization_scheme=quantization_scheme(asymmetric=False, per_channel=True, bits=8),
            # Each token's activations get their own INT8 scale before an INT8-weight operation.
            per_token_quant=True,
        )
    if precision == "w16a16":
        return QuantizationParams(
            calibration_method=SkipCalibration(),
            activation_quantization_scheme=bfloat16_scheme(),
            weight_quantization_scheme=bfloat16_scheme(),
        )
    raise ValueError(f"unsupported precision {precision!r}")


def compile_model(source: Path, component_dir: Path, profile_id: str, dataset: Path | None) -> AfeModel:
    parameters = component_parameters(MODEL_DIR, profile_id, COMPONENT_ID)
    precision = str(parameters.get("precision"))
    if precision not in {"w16a16", "w8a8", "w8a16"}:
        raise ValueError("this recipe supports only the declared BF16 and INT8 build profiles")

    input_name = str(parameters["input_name"])
    input_shape = tuple(int(value) for value in parameters["input_shape"])
    if input_shape != (1, 3, 224, 224):
        raise ValueError(f"unsupported ResNeXt-101 32x8d input shape: {input_shape}")

    component_dir.mkdir(parents=True, exist_ok=False)
    importer = onnx_source(str(source), {input_name: input_shape}, {input_name: ScalarType.float32})
    loaded = load_model(importer, target=gen2_target)

    data = None
    if precision == "w8a8":
        if dataset is None:
            raise ValueError("the INT8 build requires --dataset from dataset-registry")
        sample_count = int(parameters["calibration_samples"])
        LOG.info("Loading %d representative calibration samples", sample_count)
        data = convert_data_generator_to_iterable(
            DataGenerator({input_name: calibration_samples(dataset, sample_count)})
        )

    model_name = f"{MODEL_ID}_{profile_id}"
    quantized = loaded.quantize(data, _quantization(precision), model_name=model_name, log_level=compiler_log_level())
    quantized.save(model_name=model_name, output_directory=str(component_dir))
    saved = AfeModel.load(model_name=f"{model_name}.sima", network_directory=str(component_dir))
    saved.compile(output_path=str(component_dir), log_level=compiler_log_level())

    archives = sorted(component_dir.glob("*_mpk.tar.gz"))
    if len(archives) != 1:
        raise RuntimeError(f"expected exactly one MPK archive, found {len(archives)} in {component_dir}")
    LOG.info("Compiled MPK: %s", archives[0])
    return saved


def fidelity_samples(dataset: Path) -> list[np.ndarray]:
    """The preprocessed calibration images model.yaml declares for the fidelity comparison."""
    selection = load_metadata(MODEL_DIR)["numerical_fidelity"]["selection"]
    names = fidelity_names(dataset, int(selection["seed"]), int(selection["offset"]), int(selection["samples"]))
    images = read_images(dataset, names)
    return [preprocess_image(images[name]) for name in names]


def main() -> int:
    arguments = parse_arguments(__doc__, MODEL_DIR)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    component_dir = arguments.output_dir / COMPONENT_ID
    candidate = compile_model(arguments.model, component_dir, arguments.build_profile, arguments.dataset)
    input_name = str(component_parameters(MODEL_DIR, arguments.build_profile, COMPONENT_ID)["input_name"])
    compared = compare_with_onnx(
        arguments.model, candidate, fidelity_samples(arguments.fidelity_dataset), input_name, ["logits"]
    )
    write_fidelity_report(MODEL_DIR, arguments.build_profile, compared, arguments.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
