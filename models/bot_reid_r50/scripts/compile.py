#!/usr/bin/env python3
"""Compile the converted BoT ReID R50 ONNX into a Modalix MPK on one MLA segment."""

from __future__ import annotations

import argparse
import json
import logging
import tarfile
from pathlib import Path
from typing import Any

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
from calibration import HEIGHT, WIDTH, calibration_samples
from model_registry.model import load_model as load_metadata
from sima_utils.data.data_generator import DataGenerator

LOG = logging.getLogger("bot_reid_r50.compile")
MODEL_DIR = Path(__file__).resolve().parents[1]
MODEL_ID = "bot_reid_r50"
COMPONENT_ID = "model"
INPUT_NAME = "images"
INPUT_SHAPE = (1, 3, HEIGHT, WIDTH)


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
            per_token_quant=True,
        )
    if precision == "w16a16":
        return QuantizationParams(
            calibration_method=SkipCalibration(),
            activation_quantization_scheme=bfloat16_scheme(),
            weight_quantization_scheme=bfloat16_scheme(),
        )
    raise ValueError(f"unsupported precision {precision!r}")


def _read_mpk_manifest(archive: Path) -> dict[str, Any]:
    with tarfile.open(archive, "r:gz") as package:
        for member in package.getmembers():
            if member.isfile() and member.name.endswith("_mpk.json"):
                stream = package.extractfile(member)
                if stream is not None:
                    return json.loads(stream.read())
    raise RuntimeError(f"{archive} holds no MPK manifest")


def audit_mpk(archive: Path) -> None:
    """Require the model on one MLA segment, with no host code."""
    manifest = _read_mpk_manifest(archive)
    processors = [str(plugin.get("processor", "")).upper() for plugin in manifest.get("plugins", [])]
    if processors.count("MLA") != 1 or {"A65", "APU"} & set(processors):
        raise RuntimeError(f"invalid MPK: plugins run on {processors}")


def compile_model(source: Path, component_dir: Path, profile_id: str, dataset: Path | None) -> Path:
    parameters = _parameters(profile_id)
    precision = str(parameters.get("precision"))
    if precision not in {"w16a16", "w8a8", "w8a16"}:
        raise ValueError("this recipe supports only the declared BF16 and INT8 build profiles")

    component_dir.mkdir(parents=True, exist_ok=False)
    importer = onnx_source(str(source), {INPUT_NAME: INPUT_SHAPE}, {INPUT_NAME: ScalarType.float32})
    loaded = load_model(importer, target=gen2_target)

    data = None
    if precision == "w8a8":
        if dataset is None:
            raise ValueError("the INT8 build requires --dataset from dataset-registry")
        sample_count = int(parameters["calibration_samples"])
        LOG.info("Loading %d training images for calibration", sample_count)
        data = convert_data_generator_to_iterable(
            DataGenerator({INPUT_NAME: calibration_samples(dataset, sample_count)})
        )

    model_name = f"{MODEL_ID}_{profile_id}"
    quantized = loaded.quantize(data, _quantization(precision), model_name=model_name)
    quantized.save(model_name=model_name, output_directory=str(component_dir))
    saved = AfeModel.load(model_name=f"{model_name}.sima", network_directory=str(component_dir))
    saved.compile(output_path=str(component_dir))

    archives = sorted(component_dir.glob("*_mpk.tar.gz"))
    if len(archives) != 1:
        raise RuntimeError(f"expected exactly one MPK archive, found {len(archives)} in {component_dir}")
    audit_mpk(archives[0])
    LOG.info("Compiled MPK: %s", archives[0])
    return archives[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-profile", required=True)
    parser.add_argument("--compile-invocation", required=True)
    parser.add_argument("--dataset", type=Path, help="dataset-registry calibration archive")
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
