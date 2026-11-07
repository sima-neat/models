#!/usr/bin/env python3
"""Compile the pinned ResNeXt-101 64x4d ONNX source into a Modalix MPK and measure its numerical fidelity."""

from __future__ import annotations

import argparse
import json
import logging
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
from calibration import calibration_samples, fidelity_names, preprocess_image, read_images
from model_registry.model import load_model as load_metadata
from sima_utils.data.data_generator import DataGenerator

LOG = logging.getLogger("resnext_101_64x4d.compile")
MODEL_DIR = Path(__file__).resolve().parents[1]
MODEL_ID = "resnext_101_64x4d"
COMPONENT_ID = "model"


def _parameters(profile_id: str) -> dict[str, Any]:
    model = load_metadata(MODEL_DIR)
    profiles = [item for item in model["build_profiles"] if item["id"] == profile_id]
    if len(profiles) != 1:
        raise ValueError(f"expected exactly one build profile named {profile_id!r}")
    return next(item["parameters"] for item in profiles[0]["components"] if item["id"] == COMPONENT_ID)


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
    parameters = _parameters(profile_id)
    precision = str(parameters.get("precision"))
    if precision not in {"w16a16", "w8a8", "w8a16"}:
        raise ValueError("this recipe supports only the declared BF16 and INT8 build profiles")

    input_name = str(parameters["input_name"])
    input_shape = tuple(int(value) for value in parameters["input_shape"])
    if input_shape != (1, 3, 224, 224):
        raise ValueError(f"unsupported ResNeXt-101 64x4d input shape: {input_shape}")

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
    quantized = loaded.quantize(data, _quantization(precision), model_name=model_name)
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
    input_name = str(_parameters(profile_id)["input_name"])
    selection = config["selection"]
    names = fidelity_names(dataset, int(selection["seed"]), int(selection["offset"]), int(selection["samples"]))
    images = read_images(dataset, names)
    reference = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
    totals = {"count": 0, "dot": 0.0, "expected": 0.0, "actual": 0.0, "error": 0.0, "absolute": 0.0, "max": 0.0}
    for name in names:
        sample = preprocess_image(images[name])
        expected = reference.run(None, {input_name: sample.transpose(2, 0, 1)[None]})[0].astype(np.float64)
        outputs = candidate.execute(inputs={InputName(input_name): sample[None]}, use_jax=True)
        actual = np.asarray(outputs[0], dtype=np.float64).reshape(expected.shape)
        if not (np.isfinite(expected).all() and np.isfinite(actual).all()):
            raise RuntimeError(f"logits for {name} contain non-finite values")
        error = actual - expected
        totals["count"] += expected.size
        totals["dot"] += float(np.dot(expected.ravel(), actual.ravel()))
        totals["expected"] += float(np.sum(expected**2))
        totals["actual"] += float(np.sum(actual**2))
        totals["error"] += float(np.sum(error**2))
        totals["absolute"] += float(np.sum(np.abs(error)))
        totals["max"] = max(totals["max"], float(np.max(np.abs(error))))
    count = totals["count"]
    expected_rms = max(np.sqrt(totals["expected"] / count), np.finfo(np.float64).eps)
    norm = np.sqrt(totals["expected"] * totals["actual"])
    return {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "build_profile_id": profile_id,
        "status": config["status"],
        "reference": "ONNX source (ONNX Runtime)",
        "candidate": "Quantized AFE graph (JAX execute)",
        "dataset": {key: config["dataset"][key] for key in ("id", "variant", "sha256")},
        "selection": selection,
        "samples": len(names),
        "metrics": [
            {
                "name": "logits",
                "cosine_similarity": float(np.clip(totals["dot"] / norm if norm else 1.0, -1.0, 1.0)),
                "normalized_rmse": float(np.sqrt(totals["error"] / count) / expected_rms),
                "mean_absolute_error": float(totals["absolute"] / count),
                "max_absolute_error": totals["max"],
            }
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-profile", required=True)
    parser.add_argument("--compile-invocation", required=True)
    parser.add_argument("--dataset", type=Path, help="dataset-registry calibration archive")
    parser.add_argument("--fidelity-dataset", type=Path, help="dataset-registry fidelity archive")
    args = parser.parse_args()
    if args.compile_invocation != args.build_profile:
        parser.error("a build profile compiles in one invocation named after it")
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    source = args.model.expanduser().resolve()
    if not source.is_file():
        parser.error(f"model does not exist: {source}")
    fidelity = load_metadata(MODEL_DIR).get("numerical_fidelity")
    if fidelity is not None and args.fidelity_dataset is None:
        parser.error("model.yaml declares numerical_fidelity; pass --fidelity-dataset")
    output_dir = args.output_dir.expanduser().resolve()
    component_dir = output_dir / COMPONENT_ID
    dataset = args.dataset.expanduser().resolve() if args.dataset else None
    candidate = compile_model(source, component_dir, args.build_profile, dataset)
    if fidelity is not None:
        LOG.info("Measuring numerical fidelity on %d images", fidelity["selection"]["samples"])
        fidelity_dataset = args.fidelity_dataset.expanduser().resolve()
        report = measure_fidelity(source, candidate, fidelity_dataset, fidelity, args.build_profile)
        (output_dir / "fidelity.json").write_text(
            json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        LOG.info("Fidelity: %s", report["metrics"][0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
