"""Compile the encoder and include the CPU frontend and CTC vocabulary."""

import argparse
import json
import logging
import shutil
import tarfile
import tempfile
from importlib.metadata import version
from pathlib import Path

from model_registry.fetch import sha256_file
from model_registry.model import load_model as load_metadata
from surgery import BOUNDARY, split

MODEL_DIR = Path(__file__).resolve().parents[1]


def audit_mpk(path: Path):
    """Check placement and the public float32 tensor contract."""
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        names = [item.name for item in members]
        manifests = [item for item in members if item.name.endswith("_mpk.json")]
        if len(manifests) != 1 or any(item.issym() or item.islnk() for item in members):
            raise ValueError("Expected one safe MPK manifest")
        manifest = json.load(archive.extractfile(manifests[0]))
    plugins = sorted(manifest.get("plugins", []), key=lambda item: item.get("sequence", 0))
    mla = [item for item in plugins if str(item.get("processor", "")).upper() == "MLA"]
    host = [item for item in plugins if str(item.get("processor", "")).upper() in {"A65", "APU"}]
    if len(mla) != 1 or len([name for name in names if name.endswith(".elf")]) != 1 or host:
        raise ValueError("Expected one MLA region and no host compute regions")
    dsp = [item for item in plugins if str(item.get("processor", "")).upper() == "EV74"]
    if len(dsp) != 2 or any(item.get("config_params", {}).get("kernel") != "cast_transform" for item in dsp):
        raise ValueError("Expected only the two BF16 boundary casts on EV74")
    if len(plugins) != 3:
        raise ValueError("Unexpected extra device stages")
    inputs = {item["name"]: item["size"] for item in manifest.get("input_nodes", [])}
    if inputs != {BOUNDARY: 1 * 249 * 768 * 4}:
        raise ValueError(f"Unexpected MPK input contract: {inputs}")
    if not plugins or [item["size"] for item in plugins[-1].get("output_nodes", [])] != [1 * 249 * 32 * 4]:
        raise ValueError("Unexpected MPK output contract")


def compile_model(source: Path, output: Path, profile: str):
    from afe.apis.defines import (
        RequantizationMode,
        SkipCalibration,
        bfloat16_scheme,
        default_quantization,
        gen2_target,
        quantization_scheme,
    )
    from afe.apis.loaded_net import load_model
    from afe.load.importers.general_importer import onnx_source

    for package in ("sima-frontend", "sima-mlc"):
        print(f"Compiler package: {package}={version(package)}", flush=True)
    metadata = load_metadata(MODEL_DIR)
    selected = next(item for item in metadata["build_profiles"] if item["id"] == profile)
    params = selected["components"][0]["parameters"]
    if params != {"precision": "w8a16"}:
        raise ValueError("Unexpected build profile parameters")
    component = output / "model"
    component.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="wav2vec2-", dir=output) as temporary:
        temporary = Path(temporary)
        frontend, encoder = split(source, temporary)
        shutil.copyfile(frontend, component / "frontend.onnx")
        shutil.copyfile(MODEL_DIR / "scripts/vocab.json", component / "vocab.json")
        net = load_model(onnx_source(str(encoder), layout="NCHW", flexible_batch_size=False), target=gen2_target)
        config = (
            default_quantization.with_activation_quantization(bfloat16_scheme())
            .with_weight_quantization(quantization_scheme(asymmetric=False, per_channel=True, bits=8))
            .with_requantization_mode(RequantizationMode.sima)
            .with_calibration(SkipCalibration())
        )
        name = metadata["id"] + "_" + profile
        quantized = net.quantize(
            calibration_data=None, quantization_config=config, model_name=name, log_level=logging.INFO
        )
        quantized.compile(output_path=str(temporary / "compiled"), batch_size=1, log_level=logging.INFO, preserve=False)
        archives = list((temporary / "compiled").rglob("*_mpk.tar.gz"))
        if len(archives) != 1:
            raise ValueError("Expected exactly one compiled archive")
        audit_mpk(archives[0])
        archive = component / archives[0].name
        shutil.copyfile(archives[0], archive)
    manifest = {
        "model_id": metadata["id"],
        "build_profile_id": profile,
        "compile_invocation_id": profile,
        "outputs": [
            {"component_id": "model", "mpk_path": f"model/{archive.name}", "mpk_sha256": sha256_file(archive)[0]}
        ],
        "runtime_files": [
            {"path": f"model/{name}", "sha256": sha256_file(component / name)[0]}
            for name in ("frontend.onnx", "vocab.json")
        ],
    }
    (output / "compile-outputs.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-profile", required=True)
    parser.add_argument("--compile-invocation", required=True)
    args = parser.parse_args()
    if args.build_profile != args.compile_invocation:
        parser.error("Compile invocation must match build profile")
    compile_model(
        args.model.resolve(strict=True),
        args.output_dir.resolve(),
        args.build_profile,
    )


if __name__ == "__main__":
    main()
