#!/usr/bin/env python3
"""Compile the pinned Depth Anything V2 Small ONNX source into a Modalix MPK and measure its numerical fidelity."""

from __future__ import annotations

import json
import logging
import math
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from afe.apis.defines import (
    CalibrationMethod,
    InputName,
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
from calibration import HEIGHT, WIDTH, calibration_samples, fidelity_names, preprocess_image, read_images
from model_registry.compile_script import component_parameters, fidelity_report, parse_arguments
from model_registry.model import load_model as load_metadata
from sima_utils.data.data_generator import DataGenerator
from surgery import prepare_model

LOG = logging.getLogger("depth_anything_v2_small.compile")
MODEL_DIR = Path(__file__).resolve().parents[1]
MODEL_ID = "depth_anything_v2_small"
COMPONENT_ID = "model"


def quantization(precision: str) -> QuantizationParams:
    if precision not in {"w16a16", "w8a8", "w8a16"}:
        raise ValueError(f"unsupported precision: {precision}")
    return QuantizationParams(
        calibration_method=CalibrationMethod.from_str("mse") if precision == "w8a8" else SkipCalibration(),
        activation_quantization_scheme=(
            quantization_scheme(asymmetric=True, per_channel=False, bits=8)
            if precision == "w8a8"
            else bfloat16_scheme()
        ),
        weight_quantization_scheme=(
            bfloat16_scheme()
            if precision == "w16a16"
            else quantization_scheme(asymmetric=False, per_channel=True, bits=8)
        ),
        per_token_quant=precision == "w8a16",
    )


def audit_mpk(archive: Path) -> None:
    """Require one MLA segment, I/O transforms, and the full-resolution float32 depth output."""
    with tarfile.open(archive, "r:gz") as stream:
        names = stream.getnames()
        manifests = [name for name in names if name.endswith("_mpk.json")]
        if len(manifests) != 1:
            raise RuntimeError("expected one MPK manifest")
        manifest = json.load(stream.extractfile(manifests[0]))
    plugins = sorted(manifest["plugins"], key=lambda plugin: plugin["sequence"])
    processors = [str(plugin.get("processor", "")).upper() for plugin in plugins]
    if processors.count("MLA") != 1 or set(processors) - {"MLA", "EV74"}:
        raise RuntimeError(f"expected one MLA segment and I/O transforms, found {processors}")
    if len([name for name in names if name.endswith(".elf")]) != 1 or any(".so" in name for name in names):
        raise RuntimeError("expected one ELF without host-compute libraries")
    input_bytes = math.prod((1, 3, HEIGHT, WIDTH)) * 4
    if [(value["name"], value["size"]) for value in manifest["input_nodes"]] != [("pixel_values", input_bytes)]:
        raise RuntimeError("unexpected MPK image input")
    if [value["size"] for value in plugins[-1]["output_nodes"]] != [HEIGHT * WIDTH * 4]:
        raise RuntimeError("expected one float32 depth map")


def compile_model(source: Path, component_dir: Path, variant_id: str, dataset: Path | None) -> AfeModel:
    parameters = component_parameters(MODEL_DIR, variant_id, COMPONENT_ID)
    precision = parameters["precision"]
    bf16_regions = parameters.get("bf16_regions", [])
    if bf16_regions and precision != "w8a8":
        raise ValueError("BF16 regions require W8A8 base precision")
    if bool(bf16_regions) != ("mixed" in variant_id.split("_")):
        raise ValueError("a build profile must have the mixed token exactly when it has BF16 regions")
    config = quantization(precision)

    input_name = str(parameters["input_name"])
    input_shape = tuple(int(value) for value in parameters["input_shape"])
    if input_shape != (1, 3, HEIGHT, WIDTH):
        raise ValueError(f"unsupported Depth Anything V2 input shape: {input_shape}")

    data = None
    if precision == "w8a8":
        if dataset is None:
            raise ValueError("the INT8 variant requires --dataset from dataset-registry")
        sample_count = int(parameters["calibration_samples"])
        LOG.info("Loading %d representative calibration samples", sample_count)
        samples = calibration_samples(dataset, sample_count)
        data = convert_data_generator_to_iterable(DataGenerator({input_name: samples}))

    component_dir.mkdir(parents=True, exist_ok=False)
    prepared = prepare_model(source, component_dir.parent / f"{MODEL_ID}_prepared.onnx", bf16_regions)
    LOG.info("Prepared graph: %s", prepared)
    importer = onnx_source(str(prepared), {input_name: input_shape}, {input_name: ScalarType.float32})
    model_name = f"{MODEL_ID}_{variant_id}"
    try:
        quantized = load_model(importer, target=gen2_target).quantize(
            calibration_data=data, quantization_config=config, model_name=model_name
        )
        quantized.compile(output_path=str(component_dir))
    except SystemExit as error:
        # The compiler can report a fatal error through SystemExit(0).
        raise RuntimeError(f"Model Compiler failed for {model_name}") from error

    archives = sorted(component_dir.glob("*_mpk.tar.gz"))
    if len(archives) != 1:
        raise RuntimeError(f"expected exactly one MPK archive, found {len(archives)} in {component_dir}")
    audit_mpk(archives[0])
    LOG.info("Compiled MPK: %s", archives[0])
    return quantized


def measure_fidelity(
    source: Path, candidate: AfeModel, dataset: Path, config: dict[str, Any], variant_id: str
) -> dict[str, Any]:
    """Compare the quantized graph's relative depth with the ONNX source on the declared validation images."""
    input_name = component_parameters(MODEL_DIR, variant_id, COMPONENT_ID)["input_name"]
    selection = config["selection"]
    names = fidelity_names(dataset, int(selection["seed"]), int(selection["offset"]), int(selection["samples"]))
    images = read_images(dataset, names)
    reference = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
    compared = []
    for name in names:
        sample = preprocess_image(images[name])
        expected = reference.run(None, {input_name: sample.transpose(2, 0, 1)[None]})[0]
        outputs = candidate.execute(inputs={InputName(input_name): sample[None]}, use_jax=True)
        compared.append({"predicted_depth": (expected, np.asarray(outputs[0]).reshape(expected.shape))})
    return fidelity_report(
        MODEL_ID,
        variant_id,
        config,
        compared,
        reference="ONNX source (ONNX Runtime)",
        candidate="Quantized AFE graph (JAX execute)",
    )


def main() -> int:
    arguments = parse_arguments(__doc__, MODEL_DIR)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    fidelity = load_metadata(MODEL_DIR)["numerical_fidelity"]
    component_dir = arguments.output_dir / COMPONENT_ID
    candidate = compile_model(arguments.model, component_dir, arguments.build_profile, arguments.dataset)
    LOG.info("Measuring numerical fidelity on %d images", fidelity["selection"]["samples"])
    report = measure_fidelity(arguments.model, candidate, arguments.fidelity_dataset, fidelity, arguments.build_profile)
    (arguments.output_dir / "fidelity.json").write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    LOG.info("Fidelity: %s", report["metrics"][0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
