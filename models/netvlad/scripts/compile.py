#!/usr/bin/env python3
"""Compile NetVLAD from its converted weights into one Modalix MPK and measure its descriptor fidelity."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import tarfile
import tempfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from afe.apis.defines import (
    HistogramMSEMethod,
    InputName,
    QuantizationParams,
    bfloat16_scheme,
    gen2_target,
    quantization_scheme,
)
from afe.apis.loaded_net import load_model
from afe.apis.model import L2CachingMode
from afe.core.configs import QuantizationPrecision
from afe.core.utils import convert_data_generator_to_iterable
from afe.load.importers.general_importer import onnx_source
from calibration import load_images, select_images
from model_registry.model import load_model as load_metadata
from sima_utils.data.data_generator import DataGenerator
from surgery import (
    DESCRIPTOR_DIM,
    ENCODER_CONFIG,
    INTERFACE,
    annotate_int8_encoder,
    export_netvlad,
    load_netvlad,
    reference_netvlad,
)

# Importing afe installs a root handler at WARNING, so the recipe logs through its own INFO logger.
LOG = logging.getLogger("netvlad.compile")
LOG.setLevel(logging.INFO)
MODEL_DIR = Path(__file__).resolve().parents[1]
MODEL_ID = "netvlad"
COMPONENT_ID = "model"
INPUT, OUTPUT = list(INTERFACE)
INPUT_SHAPE = INTERFACE[INPUT]
# MPK manifests record each public tensor's size in bytes; the MPK keeps the ONNX float32 interface.
INPUT_BYTES = {INPUT: math.prod(INPUT_SHAPE) * 4}
OUTPUT_BYTES = [DESCRIPTOR_DIM * 4]
ENCODER_CONVOLUTIONS = sum(item != "M" for item in ENCODER_CONFIG)


def build_parameters(metadata: dict[str, Any], profile_id: str) -> dict[str, Any]:
    profiles = [profile for profile in metadata["build_profiles"] if profile["id"] == profile_id]
    if len(profiles) != 1:
        raise ValueError(f"model.yaml declares no build profile {profile_id!r}")
    return next(item["parameters"] for item in profiles[0]["components"] if item["id"] == COMPONENT_ID)


def quantization(precision: str, calibration_bins: int) -> QuantizationParams:
    """The compiler's activation and weight formats for a precision.

    Every build calibrates. AFE has no BF16 Sqrt, so even with BF16 activations the VLAD norm's Sqrt runs as an INT8
    lookup table, and without calibration data quantize() fails for missing calibration information. Histogram MSE
    measures its input range, and the W8A8 encoder's ranges, on the calibration images.
    """
    if precision in ("w8a8", "w8a16"):
        # The W8A8 build quantizes as W8A16 too, and its INT8 marks make the encoder W8A8 (bf16_regions). A BF16 mark
        # on the pooling instead would store the WPCA's 134M weights in BF16, twice the bytes the MLA reads per frame.
        return QuantizationParams(
            calibration_method=HistogramMSEMethod(calibration_bins),
            activation_quantization_scheme=bfloat16_scheme(),
            weight_quantization_scheme=quantization_scheme(asymmetric=False, per_channel=True, bits=8),
            # Each token's activations get their own INT8 scale before an INT8-weight operation.
            per_token_quant=True,
        )
    if precision == "w16a16":
        return QuantizationParams(
            calibration_method=HistogramMSEMethod(calibration_bins),
            activation_quantization_scheme=bfloat16_scheme(),
            weight_quantization_scheme=bfloat16_scheme(),
        )
    raise ValueError(f"unsupported precision: {precision}")


def bf16_regions(profile_id: str, parameters: dict[str, Any]) -> list[str]:
    """The regions a W8A8 build runs above W8A8, checking that the build profile's ID marks it mixed.

    W8A8 needs the pooling in BF16: with INT8 activations from conv5_3 on, the descriptor keeps 0.41 cosine.
    """
    regions = parameters.get("bf16_regions", [])
    if (parameters["precision"] == "w8a8") != (regions == ["pooling"]):
        raise ValueError(f"a W8A8 build, and only a W8A8 build, runs bf16_regions: [pooling]; found {regions}")
    if bool(regions) != ("mixed" in profile_id.split("_")):
        raise ValueError(f"build profile {profile_id!r} must have the mixed token exactly when it has bf16_regions")
    return regions


def require_bf16_pooling(quantized: Any) -> None:
    """Fail unless AFE ran exactly the encoder's convolutions before conv5_3 in INT8, as the marks ask."""
    # The quantized model keeps its network private, and AFE renames every node, so count by operation.
    counts = Counter(
        type(node.ir.operation).__name__
        for node in quantized._net.iter_nodes_recursive()
        if getattr(node.ir, "quant_config", None) is not None
        and node.ir.quant_config.quantization_precision.value == QuantizationPrecision.INT_8
    )
    LOG.info("INT8 operations: %s", dict(sorted(counts.items())))
    if counts["ConvAddActivationOp"] != ENCODER_CONVOLUTIONS - 1:
        raise RuntimeError(
            f"expected the encoder's {ENCODER_CONVOLUTIONS - 1} convolutions before conv5_3 in INT8, "
            f"found {dict(counts)}"
        )


def calibration_names(dataset: Path, selection: dict[str, Any], sample_count: int) -> tuple[str, ...]:
    """The first images of the fidelity seed's shuffle, which the fidelity comparison must start after."""
    if int(selection["offset"]) < sample_count:
        raise ValueError("the fidelity selection must begin after the calibration images")
    return select_images(dataset, int(selection["seed"]), 0, sample_count)


def _read_mpk_manifest(archive: Path) -> tuple[list[str], dict[str, Any]]:
    with tarfile.open(archive, "r:gz") as stream:
        members = stream.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
                raise RuntimeError(f"unsafe MPK member: {member.name}")
        names = [member.name for member in members]
        manifests = [name for name in names if name.endswith("_mpk.json")]
        if len(manifests) != 1:
            raise RuntimeError(f"expected one MPK manifest, found {manifests}")
        source = stream.extractfile(manifests[0])
        if source is None:
            raise RuntimeError(f"cannot read MPK manifest: {manifests[0]}")
        return names, json.load(source)


def audit_mpk(archive: Path) -> None:
    """Require the whole network on one MLA segment, with no host code, and the declared interface."""
    names, manifest = _read_mpk_manifest(archive)
    elfs = [name for name in names if name.endswith(".elf")]
    shared_objects = [name for name in names if name.endswith(".so") or ".so." in name]
    plugins = sorted(manifest.get("plugins", []), key=lambda item: item.get("sequence", 0))
    mla_plugins = [plugin for plugin in plugins if str(plugin.get("processor", "")).upper() == "MLA"]
    host_plugins = [plugin for plugin in plugins if str(plugin.get("processor", "")).upper() in {"A65", "APU"}]
    if len(elfs) != 1 or len(mla_plugins) != 1 or shared_objects or host_plugins:
        raise RuntimeError(
            f"invalid MPK: MLA={len(mla_plugins)}, ELF={len(elfs)}, host={len(host_plugins)}, "
            f"shared_objects={shared_objects}"
        )
    inputs = {node.get("name"): node.get("size") for node in manifest.get("input_nodes", [])}
    if inputs != INPUT_BYTES:
        raise RuntimeError(f"MPK input interface mismatch: {inputs}")
    # The last EV74 kernel converts the MLA's output back to float32, so the audit checks where the descriptor comes
    # from and its size rather than one kernel name.
    terminal = plugins[-1] if plugins else {}
    processor = str(terminal.get("processor", "")).upper()
    outputs = [node.get("size") for node in terminal.get("output_nodes", [])]
    if processor != "EV74" or outputs != OUTPUT_BYTES:
        raise RuntimeError(f"MPK output interface mismatch: {processor} returns output bytes {outputs}")


def compile_model(
    source: Path, output_dir: Path, profile_id: str, parameters: dict[str, Any], samples: np.ndarray
) -> tuple[Path, Any]:
    precision = str(parameters["precision"])
    component_dir = output_dir / COMPONENT_ID
    component_dir.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix=f"{MODEL_ID}-", dir=os.environ.get("RUNNER_TEMP")) as temporary:
        build_dir = Path(temporary)
        importer = onnx_source(str(source), shape_dict={INPUT: tuple(INPUT_SHAPE)}, layout="NCHW")
        loaded = load_model(importer, target=gen2_target)
        LOG.info("Quantizing %s with %d sample(s)", precision, len(samples))
        quantized = loaded.quantize(
            calibration_data=convert_data_generator_to_iterable(DataGenerator({InputName(INPUT): samples})),
            quantization_config=quantization(precision, int(parameters["calibration_bins"])),
            model_name=f"{MODEL_ID}_{profile_id}",
            log_level=logging.INFO,
        )
        if parameters.get("bf16_regions"):
            require_bf16_pooling(quantized)
        try:
            quantized.compile(
                output_path=str(build_dir / "mpk"),
                batch_size=1,
                compress=True,
                log_level=logging.INFO,
                tessellate_parameters=None,
                l2_caching_mode=L2CachingMode.NONE,
                deployable=True,
            )
        except SystemExit as exc:
            # AFE reports a compile error by raising SystemExit without a code, which would exit with status 0.
            raise RuntimeError("the Model Compiler failed to compile the model; its log above names the layer") from exc
        archives = sorted((build_dir / "mpk").glob("*_mpk.tar.gz"))
        if len(archives) != 1:
            raise RuntimeError(f"expected one MPK, found {len(archives)}")
        archive = component_dir / archives[0].name
        shutil.copyfile(archives[0], archive)
    # Audit the kept copy, so a rejected MPK stays in the output for diagnosis while the build still fails.
    audit_mpk(archive)
    return archive, quantized


def measure_fidelity(
    reference: Callable[[np.ndarray], np.ndarray],
    candidate: Any,
    dataset: Path,
    config: dict[str, Any],
    profile_id: str,
) -> dict[str, Any]:
    """Compare the quantized graph's descriptors with Patch-NetVLAD's own forward on the declared images."""
    selection = config["selection"]
    names = select_images(dataset, int(selection["seed"]), int(selection["offset"]), int(selection["samples"]))
    totals = {"count": 0, "dot": 0.0, "expected": 0.0, "actual": 0.0, "error": 0.0, "absolute": 0.0, "max": 0.0}
    for index, (name, sample) in enumerate(zip(names, load_images(dataset, names), strict=True), start=1):
        LOG.info("Fidelity image %d/%d: %s", index, len(names), name)
        expected = reference(sample.transpose(2, 0, 1)[None])
        outputs = candidate.execute(inputs={InputName(INPUT): sample[None]}, use_jax=True)
        actual = np.asarray(outputs[0], dtype=np.float64).reshape(expected.shape)
        if not (np.isfinite(expected).all() and np.isfinite(actual).all()):
            raise RuntimeError(f"descriptors for {name} contain non-finite values")
        error = actual - expected
        totals["count"] += expected.size
        totals["dot"] += float(np.dot(expected, actual))
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
        "reference": "VGG16 encoder (PyTorch) and Patch-NetVLAD pooling in float64",
        "candidate": "Quantized AFE graph (JAX execute)",
        "dataset": {key: config["dataset"][key] for key in ("id", "variant", "sha256")},
        "selection": selection,
        "samples": len(names),
        "metrics": [
            {
                "name": OUTPUT,
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
    weights = args.model.expanduser().resolve()
    if weights.suffix != ".pth":
        parser.error("--model must be the verified weights.pth from the conversion workflow")
    if not weights.is_file():
        parser.error(f"model does not exist: {weights}")
    if args.dataset is None:
        parser.error("every build calibrates; pass the dataset-registry --dataset")
    if args.fidelity_dataset is None:
        parser.error("model.yaml declares numerical_fidelity; pass --fidelity-dataset")
    for dataset in (args.dataset, args.fidelity_dataset):
        if not dataset.exists():
            parser.error(f"dataset does not exist: {dataset}")
    metadata = load_metadata(MODEL_DIR)
    parameters = build_parameters(metadata, args.build_profile)
    regions = bf16_regions(args.build_profile, parameters)
    fidelity = metadata["numerical_fidelity"]
    dataset = args.dataset.expanduser().resolve()
    samples = load_images(
        dataset, calibration_names(dataset, fidelity["selection"], int(parameters["calibration_samples"]))
    )
    output_dir = args.output_dir.expanduser().resolve()

    with tempfile.TemporaryDirectory(prefix=f"{MODEL_ID}-onnx-", dir=os.environ.get("RUNNER_TEMP")) as temporary:
        source = export_netvlad(weights, Path(temporary) / "model.onnx")
        if regions:
            marks = annotate_int8_encoder(source)
            LOG.info(
                "Marked %d encoder operations INT8 and conv5_3 BF16 (%d)",
                marks["int8"],
                marks["bfloat16"],
            )
        archive, quantized = compile_model(source, output_dir, args.build_profile, parameters, samples)
    LOG.info("Compiled MPK: %s", archive)
    reference = reference_netvlad(*load_netvlad(weights))
    report = measure_fidelity(
        reference, quantized, args.fidelity_dataset.expanduser().resolve(), fidelity, args.build_profile
    )
    (output_dir / "fidelity.json").write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    LOG.info("Fidelity: %s", report["metrics"][0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
