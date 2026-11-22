#!/usr/bin/env python3
"""Compile the pinned ViT-B/16 ONNX source, after surgery.py, into a Modalix MPK and measure its numerical fidelity."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from afe.apis.defines import (
    InputName,
    QuantizationParams,
    SkipCalibration,
    bfloat16_scheme,
    gen2_target,
    quantization_scheme,
)
from afe.apis.loaded_net import load_model
from afe.apis.model import Model as AfeModel
from afe.ir.tensor_type import ScalarType
from afe.load.importers.general_importer import onnx_source
from calibration import fidelity_names, preprocess_image, read_images
from model_registry.compile_script import component_parameters, fidelity_report, parse_arguments
from model_registry.model import load_model as load_metadata
from surgery import prepare_model

LOG = logging.getLogger("vit_b_16.compile")
MODEL_DIR = Path(__file__).resolve().parents[1]
MODEL_ID = "vit_b_16"
COMPONENT_ID = "model"


def _quantization(precision: str) -> QuantizationParams:
    """The compiler's activation and weight formats for a precision. Both keep BF16 activations, which need no
    calibration."""
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


def compile_model(source: Path, component_dir: Path, profile_id: str) -> AfeModel:
    parameters = component_parameters(MODEL_DIR, profile_id, COMPONENT_ID)
    precision = str(parameters.get("precision"))
    if precision not in {"w16a16", "w8a16"}:
        raise ValueError("this recipe supports only the declared w16a16 and w8a16 build profiles")

    input_name = str(parameters["input_name"])
    input_shape = tuple(int(value) for value in parameters["input_shape"])
    if input_shape != (1, 3, 224, 224):
        raise ValueError(f"unsupported ViT-B/16 input shape: {input_shape}")

    component_dir.mkdir(parents=True, exist_ok=False)
    model_name = f"{MODEL_ID}_{profile_id}"
    with tempfile.TemporaryDirectory(prefix=f"{MODEL_ID}-", dir=os.environ.get("RUNNER_TEMP")) as temporary:
        graph = prepare_model(source, Path(temporary))
        importer = onnx_source(str(graph), {input_name: input_shape}, {input_name: ScalarType.float32})
        loaded = load_model(importer, target=gen2_target)
        quantized = loaded.quantize(None, _quantization(precision), model_name=model_name)
    quantized.save(model_name=model_name, output_directory=str(component_dir))
    saved = AfeModel.load(model_name=f"{model_name}.sima", network_directory=str(component_dir))
    saved.compile(output_path=str(component_dir))

    archives = sorted(component_dir.glob("*_mpk.tar.gz"))
    if len(archives) != 1:
        raise RuntimeError(f"expected exactly one MPK archive, found {len(archives)} in {component_dir}")
    LOG.info("Compiled MPK: %s", archives[0])
    return saved


def measure_fidelity(
    source: Path, candidate: AfeModel, dataset: Path, config: dict[str, Any], profile_id: str
) -> dict[str, Any]:
    """Compare the quantized graph's logits with the ONNX source on the declared calibration images."""
    input_name = str(component_parameters(MODEL_DIR, profile_id, COMPONENT_ID)["input_name"])
    selection = config["selection"]
    names = fidelity_names(dataset, int(selection["seed"]), int(selection["offset"]), int(selection["samples"]))
    images = read_images(dataset, names)
    reference = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
    compared = []
    for name in names:
        sample = preprocess_image(images[name])
        expected = reference.run(None, {input_name: sample.transpose(2, 0, 1)[None]})[0]
        outputs = candidate.execute(inputs={InputName(input_name): sample[None]}, use_jax=True)
        compared.append({"logits": (expected, np.asarray(outputs[0]).reshape(expected.shape))})
    return fidelity_report(
        MODEL_ID,
        profile_id,
        config,
        compared,
        reference="ONNX source (ONNX Runtime)",
        candidate="Quantized AFE graph (JAX execute)",
    )


def main() -> int:
    arguments = parse_arguments(__doc__, MODEL_DIR)
    # The compiler's imports already give the root logger a handler, which a plain basicConfig would keep.
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", force=True)

    fidelity = load_metadata(MODEL_DIR)["numerical_fidelity"]
    component_dir = arguments.output_dir / COMPONENT_ID
    candidate = compile_model(arguments.model, component_dir, arguments.build_profile)
    LOG.info("Measuring numerical fidelity on %d images", fidelity["selection"]["samples"])
    report = measure_fidelity(arguments.model, candidate, arguments.fidelity_dataset, fidelity, arguments.build_profile)
    (arguments.output_dir / "fidelity.json").write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    LOG.info("Fidelity: %s", report["metrics"][0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
