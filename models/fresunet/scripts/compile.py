#!/usr/bin/env python3
"""Compile the pinned FresUNet ONNX into a Modalix MPK, calibrated on the OSCD train cities."""

from __future__ import annotations

import argparse
import json
import logging
import tarfile
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from afe.apis.defines import (
    BiasCorrectionType,
    HistogramMSEMethod,
    InputName,
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
from model_registry.model import load_model as load_metadata
from oscd import CHANNELS, INPUT_NAME, SIZE, cities, city_input, dataset_root
from sima_utils.data.data_generator import DataGenerator

LOG = logging.getLogger("fresunet.compile")
MODEL_DIR = Path(__file__).resolve().parents[1]
MODEL_ID = "fresunet"
COMPONENT_ID = "model"
INPUT_SHAPE = (1, CHANNELS, SIZE, SIZE)
PRECISIONS = {"w16a16", "w8a8", "w8a16"}


def _parameters(profile_id: str) -> dict[str, Any]:
    model = load_metadata(MODEL_DIR)
    profiles = [item for item in model["build_profiles"] if item["id"] == profile_id]
    if len(profiles) != 1:
        raise ValueError(f"expected exactly one build profile named {profile_id!r}")
    return next(item["parameters"] for item in profiles[0]["components"] if item["id"] == COMPONENT_ID)


def quantization(precision: str, bins: int) -> QuantizationParams:
    """The compiler's activation and weight formats. Every precision takes its ranges from calibration."""
    calibration = HistogramMSEMethod(bins)
    int8_weights = quantization_scheme(asymmetric=False, per_channel=True, bits=8)
    if precision == "w8a8":
        return QuantizationParams(
            calibration_method=calibration,
            activation_quantization_scheme=quantization_scheme(asymmetric=True, per_channel=False, bits=8),
            weight_quantization_scheme=int8_weights,
            biascorr_type=BiasCorrectionType.REGULAR,
        )
    if precision == "w8a16":
        return QuantizationParams(
            calibration_method=calibration,
            activation_quantization_scheme=bfloat16_scheme(),
            weight_quantization_scheme=int8_weights,
        )
    if precision == "w16a16":
        return QuantizationParams(
            calibration_method=calibration,
            activation_quantization_scheme=bfloat16_scheme(),
            weight_quantization_scheme=bfloat16_scheme(),
        )
    raise ValueError(f"unsupported precision {precision!r}")


@contextmanager
def opened(path: Path) -> Iterator[Path]:
    """The OSCD root, holding train.txt and test.txt, from an extracted directory or a Dataset Registry archive."""
    if path.is_dir():
        yield dataset_root(path)
        return
    with tempfile.TemporaryDirectory(prefix="oscd-") as directory:
        with tarfile.open(path, "r:*") as archive:
            # The data filter refuses absolute paths, links out of the directory and special files.
            archive.extractall(directory, filter="data")
        yield dataset_root(Path(directory))


def calibration_inputs(dataset: Path, samples: int) -> np.ndarray:
    """The first train cities in train.txt order, [samples, 544, 544, 26] channels last."""
    with opened(dataset) as root:
        names = cities(root, "train")
        if len(names) < samples:
            raise ValueError(f"calibration requires {samples} train cities, the dataset has {len(names)}")
        return np.stack([city_input(root, name) for name in names[:samples]])


def processors(archive: Path) -> list[str]:
    """The processor of each plugin in the MPK's manifest, in order."""
    with tarfile.open(archive) as tar:
        member = next(item for item in tar.getmembers() if item.name.endswith("_mpk.json"))
        manifest = json.load(tar.extractfile(member))
    return [str(plugin.get("processor", "")).upper() for plugin in manifest.get("plugins", [])]


def audit_mpk(archive: Path) -> None:
    """Require the whole network on one MLA segment, with no host code."""
    found = processors(archive)
    if found.count("MLA") != 1 or {"A65", "APU"} & set(found):
        raise RuntimeError(f"invalid FresUNet MPK: plugins run on {found}")


def compile_model(source: Path, component_dir: Path, profile_id: str, dataset: Path | None) -> Path:
    parameters = _parameters(profile_id)
    precision = str(parameters.get("precision"))
    if precision not in PRECISIONS:
        raise ValueError(f"this recipe supports {sorted(PRECISIONS)}, not {precision!r}")
    if dataset is None:
        raise ValueError("every FresUNet build calibrates: pass --dataset from the Dataset Registry")

    component_dir.mkdir(parents=True, exist_ok=False)
    importer = onnx_source(str(source), {INPUT_NAME: INPUT_SHAPE}, {INPUT_NAME: ScalarType.float32})
    loaded = load_model(importer, target=gen2_target)
    samples = int(parameters["calibration_samples"])
    LOG.info("Preparing %d OSCD train cities for calibration", samples)
    data = convert_data_generator_to_iterable(
        DataGenerator({InputName(INPUT_NAME): calibration_inputs(dataset, samples)})
    )

    model_name = f"{MODEL_ID}_{profile_id}"
    params = quantization(precision, int(parameters["calibration_bins"]))
    quantized = loaded.quantize(data, params, model_name=model_name)
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
