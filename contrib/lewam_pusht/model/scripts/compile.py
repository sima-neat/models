"""Build four standard MPKs with the registry compiler; never wrap a raw ELF.

This is a candidate producer. A compile is not publishable until its exact MPKs
have independent final-ELF precision qualification. `candidate-outputs.json`
is intentionally NOT `compile-outputs.json`; the registry cannot publish it.
No old campaign precision certificate is transferable to these new binaries.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

from calibration import read_calibration
from model_registry.dependencies import resolve_model_compiler
from model_registry.model import load_model as load_metadata
from validation import COMPONENTS, digest, validate_sources

MODEL_ID = "lewam_pusht"
MODEL_DIR = Path(__file__).resolve().parents[1]


def component_name(profile: str, component: str) -> str:
    if profile != "w16a16" or component not in COMPONENTS:
        raise ValueError("Only the original four-component W16A16 profile is admitted")
    return f"{MODEL_ID}_{profile}_{component}"


def compile_options() -> dict:
    return {
        "batch_size": 1,
        "compress": True,
        "preserve": True,
        "deployable": True,
        "layer_norm_use_fp32_intermediates": False,
        "rms_norm_use_fp32_intermediates": False,
        "enable_filter_sharing": False,
        "layout_search_effort_level": 0,
    }


def unpack_dataset(source: Path, destination: Path) -> Path:
    if source.is_dir():
        return source
    with tarfile.open(source) as archive:
        members = archive.getmembers()
        expected = {"calibration.json", "selection.json", *(f"{name}.npz" for name in COMPONENTS)}
        if {member.name for member in members} != expected or len(members) != len(expected):
            raise ValueError("Calibration tar must contain exactly its six top-level files")
        if any(not member.isfile() for member in members) or sum(x.size for x in members) > 2 * 1024**3:
            raise ValueError("Unsafe or oversized calibration archive")
        for member in members:
            with archive.extractfile(member) as src, (destination / member.name).open("wb") as dst:
                shutil.copyfileobj(src, dst)
    return destination


def audit_archive(path: Path, name: str, compiler_version: str, graph: dict) -> dict:
    """Check RF-DETR's standard MPK contract, not numerical/precision parity."""
    if path.name != f"{name}_mpk.tar.gz":
        raise ValueError("Wrong MPK name; set the compiler model name, do not rename archives")
    with tarfile.open(path) as archive:
        members = archive.getmembers()
        names = [str(PurePosixPath(x.name)) for x in members]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate MPK members")
        for member in members:
            parsed = PurePosixPath(member.name)
            if parsed.is_absolute() or ".." in parsed.parts or not (member.isfile() or member.isdir()):
                raise ValueError("Unsafe MPK member")
        manifests = [x for x in members if x.isfile() and x.name.endswith("_mpk.json")]
        if len(manifests) != 1:
            raise ValueError("Exactly one standard MPK manifest required")
        if manifests[0].size > 1024 * 1024:
            raise ValueError("Oversized MPK manifest")
        manifest = json.load(archive.extractfile(manifests[0]))
        if str(manifest.get("model_sdk_version")) != compiler_version:
            raise ValueError("MPK compiler version differs from the central registry pin")
        elfs = [x for x in members if x.isfile() and x.name.endswith(".elf")]
        if len(elfs) != 1:
            raise ValueError("Expected one MLA executable per component")
        shared_objects = [name for name in names if name.endswith(".so") or ".so." in name]
        plugins = manifest.get("plugins", [])
        if (
            not isinstance(plugins, list)
            or not plugins
            or any(not isinstance(p, dict) or type(p.get("sequence")) is not int for p in plugins)
            or len({p["sequence"] for p in plugins}) != len(plugins)
        ):
            raise ValueError("MPK plugins require explicit unique sequence numbers")
        plugins = sorted(plugins, key=lambda p: p["sequence"])
        mla = [p for p in plugins if str(p.get("processor", "")).upper() == "MLA"]
        host = [p for p in plugins if str(p.get("processor", "")).upper() in {"A65", "APU", "CPU"}]
        if len(mla) != 1 or host or shared_objects:
            raise ValueError("Expected one MLA plugin and no host fallback or private runtime library")
        # Public tensors are logical FP32; the separate final-ELF gate owns MLA precision.
        inputs = manifest.get("input_nodes", [])
        expected_input = {graph["input"]: math.prod(graph["input_shape"]) * 4}
        if len(inputs) != 1 or {n.get("name"): n.get("size") for n in inputs} != expected_input:
            raise ValueError("MPK public input interface differs from the source graph")
        terminal = plugins[-1]
        output_sizes = [n.get("size") for n in terminal.get("output_nodes", [])]
        if terminal.get("config_params", {}).get("kernel") != "pass_through" or output_sizes != [
            math.prod(graph["output_shape"]) * 4
        ]:
            raise ValueError("MPK terminal output interface differs from the source graph")
        import hashlib

        elf_digest = hashlib.sha256()
        with archive.extractfile(elfs[0]) as elf:
            first = elf.read(4)
            if first != b"\x7fELF":
                raise ValueError("MLA payload is not an ELF")
            elf_digest.update(first)
            for block in iter(lambda: elf.read(1024 * 1024), b""):
                elf_digest.update(block)
    return {
        "mpk_sha256": digest(path),
        "elf_sha256": elf_digest.hexdigest(),
        "elf_member": elfs[0].name,
        "public_inputs": expected_input,
        "public_output_sizes": output_sizes,
    }


def compile_component(graph: Path, info: dict, calibration: Path, output: Path, name: str, params: dict) -> None:
    # Import only in the selected compiler subprocess, never during host contract tests.
    from afe.apis.defines import (
        BiasCorrectionType,
        HistogramMSEMethod,
        InputName,
        RequantizationMode,
        bfloat16_scheme,
        default_quantization,
        gen2_target,
    )
    from afe.apis.loaded_net import load_model
    from afe.core.utils import convert_data_generator_to_iterable
    from afe.load.importers.general_importer import onnx_source
    from sima_utils.data.data_generator import DataGenerator

    component = name.removeprefix(f"{MODEL_ID}_w16a16_")
    if params["precision"] != "w16a16":
        raise ValueError("No precision change is authorized in this initial registration")
    values = read_calibration(calibration, component, params["calibration_samples"])
    config = (
        default_quantization.with_activation_quantization(bfloat16_scheme())
        .with_weight_quantization(bfloat16_scheme())
        .with_requantization_mode(RequantizationMode.sima)
        .with_calibration(HistogramMSEMethod(params["calibration_bins"]))
        .with_bias_correction(BiasCorrectionType.NONE)
        .with_channel_equalization(False)
        .with_smooth_quant(False)
        .with_per_token_quant(False)
    )
    loaded = load_model(
        onnx_source(
            str(graph), shape_dict={info["input"]: tuple(info["input_shape"])}, layout="NCHW", flexible_batch_size=True
        ),
        target=gen2_target,
    )
    try:
        quantized = loaded.quantize(
            calibration_data=convert_data_generator_to_iterable(DataGenerator({InputName(info["input"]): values})),
            quantization_config=config,
            model_name=name,
            log_level=logging.INFO,
        )
        quantized.compile(output_path=str(output), **compile_options())
    except SystemExit as error:
        # Some compiler failures raise SystemExit(0). That is never a successful build.
        raise RuntimeError("Compiler terminated before candidate qualification") from error


def build(source: Path, dataset: Path, profile: str, output: Path) -> None:
    metadata = load_metadata(MODEL_DIR)
    profiles = [x for x in metadata["build_profiles"] if x["id"] == profile]
    if len(profiles) != 1:
        raise ValueError("Unknown build profile")
    params = {x["id"]: x["parameters"] for x in profiles[0]["components"]}
    if tuple(params) != COMPONENTS:
        raise ValueError("Component order changed")
    compiler = resolve_model_compiler(MODEL_DIR.parents[1])
    graphs = validate_sources(source.parent)
    # Source bytes must match the shared registry loader's pinned file set.
    declared = metadata["components"][0]["source"]["files"]
    pins = {row["path"]: row for row in declared}
    for component in COMPONENTS:
        pin = pins[f"{component}.onnx"]
        if (graphs[component]["sha256"], graphs[component]["size_bytes"]) != (pin["sha256"], pin["size_bytes"]):
            raise ValueError("Source graph differs from model.yaml")
    output.mkdir(parents=True, exist_ok=False)
    receipts = {}
    with tempfile.TemporaryDirectory(dir=output, prefix="calibration-") as temp:
        calibration = unpack_dataset(dataset, Path(temp))
        for component in COMPONENTS:
            name = component_name(profile, component)
            directory = output / component
            directory.mkdir()
            process = multiprocessing.get_context("spawn").Process(
                target=compile_component,
                args=(
                    source.parent / f"{component}.onnx",
                    graphs[component],
                    calibration,
                    directory,
                    name,
                    params[component],
                ),
            )
            process.start()
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(f"{component} compiler subprocess failed: {process.exitcode}")
            archive = directory / f"{name}_mpk.tar.gz"
            receipts[component] = audit_archive(archive, name, compiler["version"], graphs[component])
    candidate = {
        "model_id": MODEL_ID,
        "build_profile_id": profile,
        "compile_invocation_id": profile,
        "outputs": [
            {
                "component_id": c,
                "mpk_path": f"{c}/{component_name(profile, c)}_mpk.tar.gz",
                "mpk_sha256": receipts[c]["mpk_sha256"],
            }
            for c in COMPONENTS
        ],
        "derived_graphs": [{"component_id": c, "sha256": graphs[c]["sha256"]} for c in COMPONENTS],
    }
    (output / "candidate-outputs.json").write_text(json.dumps(candidate, indent=2) + "\n")
    (output / "qualification-required.json").write_text(
        json.dumps(
            {
                "status": "UNQUALIFIED",
                "expected_compiler": compiler,
                "components": receipts,
                "required": ["final_ELF_no_FP32", "standard_MPK_interface", "source_equivalence", "native_parity"],
            },
            indent=2,
        )
        + "\n"
    )
    raise RuntimeError("Four candidate MPKs built; publication withheld pending final-ELF and native qualification")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--build-profile", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compile-invocation", required=True)
    args = parser.parse_args()
    if args.compile_invocation != args.build_profile:
        parser.error("The compile invocation must equal the build profile")
    build(args.model.resolve(), args.dataset.resolve(), args.build_profile, args.output_dir.resolve())


if __name__ == "__main__":
    main()
