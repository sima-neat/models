"""Compile both independently bound frozen native QAT checkpoints locally.

The ONNX is the numerical oracle, NOT a claim that a diagnostic ONNX can be
imported as a deployable graph. Native materialization must already have taken
place with the matching supported compiler. Missing sealing support is an
explicit prerequisite failure; this script neither overlays AFE nor recalibrates.
Only load your own trusted .sima files; they may contain serialized Python data.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import math
import tarfile
from pathlib import Path, PurePosixPath

from contract import COMPONENTS, load_contract, parse_components, require_sha, sha256_file

PROFILE = "w8a8_mixed"


def audit_package(path: Path, contract: dict, component: str) -> dict:
    with tarfile.open(path, "r:gz") as stream:
        members = stream.getmembers()
        names = []
        for member in members:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or member.issym() or member.islnk() or member.isdev():
                raise ValueError("unsafe compiled package member")
            names.append(member.name)
        if len(names) != len(set(names)):
            raise ValueError("duplicate compiled package member")
        elfs = [name for name in names if name.endswith(".elf")]
        manifests = [name for name in names if name.endswith("_mpk.json")]
        if len(elfs) != 1 or len(manifests) != 1 or any(name.endswith(".so") or ".so." in name for name in names):
            raise ValueError("each component must contain exactly one MLA ELF and no fallback libraries")
        file = stream.extractfile(manifests[0])
        if file is None:
            raise ValueError("missing package manifest payload")
        manifest = json.load(file)
    plugins = manifest.get("plugins", [])
    if sum(str(plugin.get("processor", "")).upper() == "MLA" for plugin in plugins) != 1:
        raise ValueError("unexpected MLA stage count")
    if any(str(plugin.get("processor", "")).upper() in {"A65", "APU"} for plugin in plugins):
        raise ValueError("unexpected host model fallback")
    expected = {port["name"]: math.prod(port["shape"]) for port in contract["components"][component]["inputs"]}
    actual = {port.get("name"): port.get("size") for port in manifest.get("input_nodes", [])}
    if actual != expected:
        raise ValueError(f"{component} compiled public inputs differ from the captured raw INT8 ABI: {actual}")
    terminal = max(plugins, key=lambda plugin: plugin.get("sequence", 0)) if plugins else {}
    expected_outputs = [math.prod(port["shape"]) for port in contract["components"][component]["outputs"]]
    if (
        terminal.get("config_params", {}).get("kernel") != "pass_through"
        or [port.get("size") for port in terminal.get("output_nodes", [])] != expected_outputs
    ):
        raise ValueError(f"{component} compiled public output ABI is not the declared dense INT8 tuple")
    return {"elf": elfs[0], "sha256": sha256_file(path)}


def compile_models(
    reference: Path, native_paths: dict[str, Path], native_hashes: dict[str, str], contract: dict, output: Path
) -> dict:
    if sha256_file(reference) != contract["reference_sha256"]:
        raise ValueError("source ONNX identity differs from the frozen handoff")
    if set(native_paths) != set(COMPONENTS) or set(native_hashes) != set(COMPONENTS):
        raise ValueError("both frozen native component paths and independent checksums are required")
    for component, path in native_paths.items():
        if path.suffix != ".sima" or sha256_file(path) != require_sha(native_hashes[component]):
            raise ValueError(f"{component}: native checkpoint checksum or extension mismatch")
    if output.exists():
        raise FileExistsError("compile output directory must be new")
    from afe.apis.model import L2CachingMode, Model

    if not callable(getattr(Model, "verify_numerical_realization", None)):
        raise RuntimeError(
            "installed AFE cannot verify the frozen numerical program; obtain a supported compiler "
            "with the required native-program ingestion/sealing support. Do not import the diagnostic "
            "ONNX, recalibrate it, or patch this runner as a fallback."
        )
    models, seals = {}, {}
    for component, path in native_paths.items():
        models[component] = Model.load(path.stem, str(path.parent), include_unquantized_net=False)
        seals[component] = models[component].verify_numerical_realization()
        expected = contract["native_seals"][component]
        for field in ("native_graph_sha256", "source_contract_sha256", "numerical_plan_sha256"):
            if require_sha(seals[component][field]) != require_sha(expected[field]):
                raise ValueError(f"{component}: frozen {field} differs from the qualified source lineage")
    output.mkdir()
    records = []
    for component in COMPONENTS:
        directory = output / component
        directory.mkdir()
        models[component].compile(
            output_path=str(directory),
            batch_size=1,
            compress=True,
            log_level=logging.INFO,
            tessellate_parameters=None,
            l2_caching_mode=L2CachingMode.NONE,
            preserve=True,
            deployable=True,
            retained_temporary_directory_name=None,
            use_power_limits=False,
            max_mla_power=None,
            layer_norm_use_fp32_intermediates=False,
            rms_norm_use_fp32_intermediates=False,
            enable_filter_sharing=False,
            layout_search_effort_level=0,
        )
        if models[component].verify_numerical_realization() != seals[component]:
            raise ValueError(f"{component}: compilation changed its frozen numerical program")
        archives = list(directory.glob("*_mpk.tar.gz"))
        if len(archives) != 1:
            raise ValueError(f"{component}: expected exactly one compiled package")
        audit = audit_package(archives[0], contract, component)
        records.append(
            {"component_id": component, "mpk_path": str(archives[0].relative_to(output)), "mpk_sha256": audit["sha256"]}
        )
    result = {
        "model_id": "rf_detr_nano_qat",
        "build_profile_id": PROFILE,
        "compile_invocation_id": PROFILE,
        "outputs": records,
        "derived_graphs": [
            {"component_id": component, "sha256": seals[component]["native_graph_sha256"]} for component in COMPONENTS
        ],
    }
    (output / "compile-outputs.json").write_text(json.dumps(result, indent=2) + "\n")
    versions = {
        name: importlib.metadata.version(name)
        for name in ("sima-frontend", "sima-mlc", "sima-ml-kernels", "sima-utils")
    }
    (output / "local-compile-lineage.json").write_text(
        json.dumps(
            {
                "checkpoint_sha256": contract["checkpoint_sha256"],
                "reference_sha256": contract["reference_sha256"],
                "native_checkpoint_sha256": native_hashes,
                "native_seals": seals,
                "installed_modules": versions,
                "production_compiler_pin_verified": False,
                "source_ONNX_parity": "NOT_RUN",
                "packaged_ELF_source_parity": "NOT_RUN",
                "precision_audit": "NOT_RUN",
                "board_accuracy": "NOT_RUN",
                "board_performance": "NOT_RUN",
                "note": (
                    "Fresh output hashes need a fresh verified runtime binding; "
                    "do not reuse the old package hash contract."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="unchanged QAT numerical-reference ONNX")
    parser.add_argument("--native-model", action="append", required=True, help="backbone=path.sima, decoder=path.sima")
    parser.add_argument("--native-sha256", action="append", required=True, help="component=independent SHA-256")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--build-profile", choices=[PROFILE], required=True)
    parser.add_argument("--compile-invocation", choices=[PROFILE], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    hashes = {name: str(value) for name, value in parse_components(args.native_sha256).items()}
    compile_models(
        args.model,
        parse_components(args.native_model),
        hashes,
        load_contract(args.contract, args.contract_sha256),
        args.output_dir,
    )


if __name__ == "__main__":
    main()
