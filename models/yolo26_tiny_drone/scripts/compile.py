#!/usr/bin/env python3
"""Compile the pinned 1600px QDQ export, without surgery or a serialized snapshot.

The pinned compiler's QDQ import retains exported scales and zero points;
calibration supplies ranges only where the export has no QDQ hints. Disable
weight-changing PTQ optimizations explicitly. The documented MLA layout mapping
keeps HWC/HWC16 conversion on MLA rather than introducing external layout work.

The immutable export records Ultralytics 1367566337fb8056223a1aeb469360747f1b1bcd
and swml-qat dee091582acc7d776a4d4a582f8e25655f2ffd90. Its model-specific
training/export recipe is preserved in Models commit 91b5c07 and draft PR355;
training is not a dependency of this recipe.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from afe.apis.defines import (
    BiasCorrectionType,
    CalibrationMethod,
    RequantizationMode,
    TensorDRAMLayout,
    TensorTessellateParameters,
    default_quantization,
    gen2_target,
    quantization_scheme,
)
from afe.apis.loaded_net import load_model
from afe.apis.model import L2CachingMode
from afe.load.importers.general_importer import onnx_source
from calibration import CalibrationSamples
from contract import DEFAULT_CONTRACT, INPUT_NAME, DeploymentContract
from model_registry.artifact import verify_mpk
from model_registry.dependencies import resolve_model_compiler
from model_registry.fetch import sha256_file
from model_registry.model import load_model as load_metadata
from onnx_audit import audit_onnx
from precision_audit import audit_quantized_model

MODEL_DIR = Path(__file__).resolve().parents[1]
MODEL_ID = MODEL_DIR.name
COMPILER_VERSION = resolve_model_compiler(MODEL_DIR.parents[1])["version"]


def _parameters(profile_id: str) -> dict[str, Any]:
    profiles = [profile for profile in load_metadata(MODEL_DIR)["build_profiles"] if profile["id"] == profile_id]
    if len(profiles) != 1:
        raise ValueError(f"model.yaml declares no build profile {profile_id!r}")
    return next(component["parameters"] for component in profiles[0]["components"] if component["id"] == "model")


def _quantization() -> Any:
    """Use exported QAT hints; do not equalize weights or run PTQ bias correction."""
    return (
        default_quantization.with_activation_quantization(
            quantization_scheme(asymmetric=True, per_channel=False, bits=8)
        )
        .with_weight_quantization(quantization_scheme(asymmetric=False, per_channel=True, bits=8))
        .with_requantization_mode(RequantizationMode.sima)
        .with_calibration(CalibrationMethod.from_str("min_max"))
        .with_bias_correction(BiasCorrectionType.NONE)
        .with_channel_equalization(False)
        .with_smooth_quant(False)
    )


def _mla_tessellate_params(quantized: Any) -> dict[str, Any]:
    """The pinned compiler's compile-your-first-model direct-MLA layout recipe."""
    from afe.ir.node import node_is_tuple

    mla = quantized._net.nodes["MLA_0"]
    input_layout = TensorTessellateParameters(
        tile_shape=(0, 0, 0, 0), enable_mla=True, dram_layout=TensorDRAMLayout.HWC
    )
    output_layout = TensorTessellateParameters(
        tile_shape=(0, 0, 0, 0), enable_mla=True, dram_layout=TensorDRAMLayout.HWC16
    )
    result = dict.fromkeys(mla.input_names, input_layout)
    output = mla.ir.nodes[mla.ir.output_node_name]
    names = output.input_node_names if node_is_tuple(output) else [output.name]
    result.update({f"{name}_output": output_layout for name in names})
    return result


def audit_mpk(archive: Path, contract: DeploymentContract = DEFAULT_CONTRACT) -> dict[str, Any]:
    """Add model semantics to shared archive validation, not compiler node counts.

    Public FP32 tensor roles and physical INT8 carriers are separate contracts.
    Packing/aliases may change across compiler releases; reject host fallback,
    external layout conversion, or reordered equal-sized roles, not new helper
    names. Source operator counts belong to the ONNX audit, before fusion.
    """
    verify_mpk(archive.parent)
    with tarfile.open(archive, "r:gz") as stream:
        names = stream.getnames()
        manifests = [name for name in names if name.endswith("_mpk.json")]
        if len(manifests) != 1:
            raise RuntimeError("expected exactly one MPK manifest")
        source = stream.extractfile(manifests[0])
        if source is None:
            raise RuntimeError("cannot read MPK manifest")
        manifest = json.load(source)
    plugins = sorted(manifest.get("plugins", []), key=lambda plugin: plugin.get("sequence", 0))
    mla = [plugin for plugin in plugins if str(plugin.get("processor", "")).upper() == "MLA"]
    host = [plugin for plugin in plugins if str(plugin.get("processor", "")).upper() in {"APU", "A65", "CPU"}]
    if len(mla) != 1 or sum(name.endswith(".elf") for name in names) != 1 or host:
        raise RuntimeError("MPK must contain one MLA segment and no host compute")
    if any(name.endswith(".so") or ".so." in name for name in names):
        raise RuntimeError("MPK contains host shared objects")
    if str(manifest.get("model_sdk_version")) != COMPILER_VERSION:
        raise RuntimeError(f"MPK compiler mismatch: expected {COMPILER_VERSION}")
    inputs = [(node.get("name"), node.get("size")) for node in manifest.get("input_nodes", [])]
    if inputs != [(INPUT_NAME, math.prod(contract.input_shape) * 4)]:
        raise RuntimeError(f"MPK public input mismatch: {inputs}")
    expected = [(name, math.prod(shape) * 4) for name, shape in contract.outputs]
    terminal = plugins[-1]
    outputs = terminal.get("output_nodes", [])
    public_nodes = outputs
    # Older manifests alias pass-through outputs; their input names carry the
    # public roles. Do not identify roles by byte sizes: bbox_1 and logit_0 tie.
    roles = [(str(node.get("name", "")).rsplit("/", 1)[-1], node.get("size")) for node in outputs]
    if terminal.get("config_params", {}).get("kernel") == "pass_through":
        # Canonical output labels still alias their corresponding inputs. If
        # the compiler supplies public labels, verify them as well as the edges.
        if any(name in contract.output_names() for name, _ in roles) and roles != expected:
            raise RuntimeError(f"MPK public output labels/order mismatch: {roles}")
        public_nodes = terminal.get("input_nodes", [])
        roles = [
            (str(node.get("name", "")).rsplit("/", 1)[-1], node.get("size")) for node in terminal.get("input_nodes", [])
        ]
    if roles != expected or [node.get("size") for node in outputs] != [size for _, size in expected]:
        raise RuntimeError(f"MPK public output roles/order mismatch: {roles}")
    configs = [plugin.get("config_params", {}) for plugin in plugins]
    if any(config.get(key, 1) != 1 for config in configs for key in ("desired_batch_size", "actual_batch_size")):
        raise RuntimeError("MPK must preserve batch 1")
    if any(config.get("kernel") in {"tessellation_transform", "detessellation_transform"} for config in configs):
        raise RuntimeError("MPK must keep layout conversion on MLA")
    # Conversion boundary types are transport evidence, not a substitute for
    # current-compiler precision/qparam qualification of the internal graph.
    quantizers = [
        plugin for plugin in plugins if plugin.get("config_params", {}).get("kernel") == "quantization_transform"
    ]
    if len(quantizers) != 1 or quantizers[0]["config_params"].get("params", {}).get("output_data_type") != "int8":
        raise RuntimeError("MPK requires signed INT8 input transport")
    if [node.get("name") for node in quantizers[0].get("output_nodes", [])] != [
        node.get("name") for node in mla[0].get("input_nodes", [])
    ] or not mla[0].get("input_nodes"):
        raise RuntimeError("MPK quantizer must feed MLA directly")
    producers = {node.get("name"): plugin for plugin in plugins for node in plugin.get("output_nodes", [])}
    for output in public_nodes:
        config = producers.get(output.get("name"), {}).get("config_params", {})
        if (
            config.get("kernel") != "dequantization_transform"
            or config.get("params", {}).get("input_data_type") != "int8"
        ):
            raise RuntimeError(f"MPK output has no signed INT8 dequantizer: {output.get('name')}")
    for plugin in plugins:
        config = plugin.get("config_params", {})
        if config.get("kernel") == "unpack_transform":
            count = len(plugin.get("output_nodes", []))
            if not count or config.get("params", {}).get("tensor_types") != ["int8"] * count:
                raise RuntimeError("MPK packed output types do not match its INT8 tensors")
    return manifest


def compile_model(source: Path, output_dir: Path, profile_id: str, dataset: Path) -> Path:
    parameters = _parameters(profile_id)
    if parameters["precision"] != "w8a8" or parameters["model_size"] != 1600:
        raise ValueError("the qualified QAT source supports only the 1600px w8a8 build")
    if parameters["calibration_samples"] != 128:
        raise ValueError("the QAT calibration contract requires exactly 128 ordered samples")
    contract = DeploymentContract(parameters["model_size"])
    source_audit = audit_onnx(source, contract)
    if source_audit["sha256"] != load_metadata(MODEL_DIR)["components"][0]["source"]["sha256"]:
        raise ValueError("the ONNX does not match the pinned QAT source")
    samples = CalibrationSamples(dataset, contract, parameters["calibration_tensor_sha256"])
    # All image bytes and the concatenated tensor digest have been checked
    # before importing the model or letting the compiler consume any sample.
    with tempfile.TemporaryDirectory(prefix=f"{MODEL_ID}-", dir=os.environ.get("RUNNER_TEMP")) as temporary:
        build_dir = Path(temporary)
        importer = onnx_source(str(source), shape_dict={INPUT_NAME: contract.input_shape}, layout="NCHW")
        loaded = load_model(importer, target=gen2_target)
        quantized = loaded.quantize(
            calibration_data=samples,
            quantization_config=_quantization(),
            model_name=f"{MODEL_ID}_{profile_id}",
            log_level=logging.INFO,
        )
        precision = audit_quantized_model(quantized, source, build_dir / "quantization")
        logging.info("Quantized compute admission: %s", json.dumps(precision, sort_keys=True))
        quantized.compile(
            output_path=str(build_dir / "mpk"),
            batch_size=1,
            compress=True,
            log_level=logging.INFO,
            tessellate_parameters=_mla_tessellate_params(quantized),
            l2_caching_mode=L2CachingMode.NONE,
            deployable=True,
        )
        archives = sorted((build_dir / "mpk").glob("*_mpk.tar.gz"))
        if len(archives) != 1:
            raise RuntimeError(f"expected one MPK, found {len(archives)}")
        audit_mpk(archives[0], contract)
        component = output_dir / "model"
        component.mkdir(parents=True, exist_ok=False)
        archive = component / archives[0].name
        shutil.copyfile(archives[0], archive)
        precision["compiler"] = resolve_model_compiler(MODEL_DIR.parents[1])
        precision["mpk_sha256"] = sha256_file(archive)[0]
        (output_dir / "quantization-audit.json").write_text(json.dumps(precision, indent=2) + "\n", encoding="utf-8")
    return archive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--build-profile", required=True)
    parser.add_argument("--compile-invocation", required=True)
    args = parser.parse_args()
    if args.compile_invocation != args.build_profile:
        parser.error("a build compiles in one invocation named after its profile")
    if args.model.suffix != ".onnx" or not args.model.is_file() or not args.dataset.is_file():
        parser.error("--model and --dataset must be the verified local ONNX and registered calibration archive")
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logging.info("Compiled MPK: %s", compile_model(args.model, args.output_dir, args.build_profile, args.dataset))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
