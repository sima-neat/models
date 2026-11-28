"""Classical Model Compiler build: BF16 CLIP and a mostly INT8 image detector.

CLIP uses SkipCalibration and no calibration samples. Only detector INT8
activation ranges are calibrated, on images from the pinned Dataset Registry
COCO calibration archive with all 80 COCO prompts. The build profile declares
the detector's calibration sample count, calibration method, and BF16 regions.
"""
import argparse
import json
import logging
from pathlib import Path
import shutil
import tarfile
import tempfile

from afe.apis.defines import (HistogramMSEMethod, MinMaxMethod, QuantizationParams,
                             RequantizationMode, SkipCalibration, bfloat16_scheme,
                             gen2_target, quantization_scheme)
from afe.apis.loaded_net import load_model
from afe.load.importers.general_importer import onnx_source
from calibration import calibration_inputs
from model_registry.fetch import sha256_file
from model_registry.model import build_profile_for, load_model as load_metadata
from surgery import COMPONENTS, prepare_model
from model_registry.compile_script import compiler_log_level

MODEL_DIR = Path(__file__).resolve().parents[1]


def quantization(component, parameters):
    if component == "text_encoder":
        if parameters.get("precision", "w16a16") != "w16a16" or parameters.get("context_tokens", 77) != 77:
            raise ValueError("the text encoder uses BF16 and a 77-token context")
        return QuantizationParams(activation_quantization_scheme=bfloat16_scheme(),
                                  weight_quantization_scheme=bfloat16_scheme(),
                                  calibration_method=SkipCalibration(), requantization_mode=RequantizationMode.sima,
                                  per_token_quant=False)
    method = parameters["calibration_method"]
    if method not in {"mse", "minmax"}:
        raise ValueError(f"unsupported calibration method: {method}")
    return QuantizationParams(activation_quantization_scheme=quantization_scheme(asymmetric=True, per_channel=False, bits=8),
                              weight_quantization_scheme=quantization_scheme(asymmetric=False, per_channel=True, bits=8),
                              calibration_method=HistogramMSEMethod(num_bins=2048) if method == "mse" else MinMaxMethod(),
                              requantization_mode=RequantizationMode.sima, per_token_quant=False)


def audit_mpk(path):
    with tarfile.open(path, "r:gz") as archive:
        names = archive.getnames()
        manifests = [n for n in names if n.endswith("_mpk.json")]
        if len(manifests) != 1:
            raise ValueError("expected one model manifest")
        manifest = json.load(archive.extractfile(manifests[0]))
    processors = [str(item.get("processor", "")).upper() for item in manifest["plugins"]]
    if processors.count("MLA") != 1 or any(p in {"A65", "APU", "CPU"} for p in processors):
        raise ValueError(f"expected one MLA segment without CPU fallback, got {processors}")
    if len([n for n in names if n.endswith(".elf")]) != 1 or any(n.endswith(".so") for n in names):
        raise ValueError("unexpected compiled placement")
    return manifest


def _bf16_regions(profile_id, component_parameters):
    """Return the detector regions selected by the mixed build profile."""
    if "bf16_regions" in component_parameters["text_encoder"]:
        raise ValueError("only the detector declares bf16_regions")
    parameters = component_parameters["detector"]
    selected = parameters.get("bf16_regions", [])
    if selected and (parameters["precision"] != "w8a8" or len(set(selected)) != len(selected)):
        raise ValueError(f"bf16_regions must name distinct regions of a W8A8 detector: {selected}")
    unknown = sorted(set(selected) - {"projection_attention"})
    if unknown:
        raise ValueError(f"unknown BF16 regions: {unknown}; choose projection_attention")
    if bool(selected) != ("mixed" in profile_id.split("_")):
        raise ValueError(f"build profile {profile_id!r} must have the mixed token exactly when it has bf16_regions")
    return selected


def compile_model(source, output, profile_id, dataset=None):
    profile = build_profile_for(load_metadata(MODEL_DIR), profile_id)
    parameters = {component["id"]: component["parameters"] for component in profile["components"]}
    selected = _bf16_regions(profile_id, parameters)
    with tempfile.TemporaryDirectory(prefix=MODEL_DIR.name + "-") as temporary:
        directory = Path(temporary)
        prepare_model(source, directory, bf16_regions=selected)
        compile_prepared(directory, output, profile_id, dataset)


def compile_prepared(directory, output, profile_id, dataset=None, quantized_output=None):
    """Compile the exported graphs; usable separately to move compilation to another host."""
    import numpy as np

    metadata = load_metadata(MODEL_DIR)
    profile = build_profile_for(metadata, profile_id)
    parameters = {component["id"]: component["parameters"] for component in profile["components"]}
    detector = parameters["detector"]
    if (detector["precision"], detector["prompt_slots"]) != ("w8a8", 80):
        raise ValueError("this recipe requires a W8A8 detector with 80 prompt slots")
    _bf16_regions(profile_id, parameters)
    if dataset is None:
        raise ValueError("the detector requires --dataset from dataset-registry")
    output.mkdir(parents=True, exist_ok=False)
    table = np.load(directory / "token_embedding_table.npy", allow_pickle=False)
    manifest = {
        "model_id": metadata["id"],
        "build_profile_id": profile_id,
        "compile_invocation_id": profile_id,
        "outputs": [],
        "derived_graphs": [],
    }

    def create_text_calibrator():
        loaded = load_model(onnx_source(str(directory / "text_calibration.onnx"), layout="NCHW", flexible_batch_size=False),
                            target=gen2_target)
        return loaded.quantize(calibration_data=None, quantization_config=quantization("text_encoder", parameters["text_encoder"]),
                               model_name="yolo_world_text_calibration", log_level=compiler_log_level())

    with tempfile.TemporaryDirectory(prefix=metadata["id"] + "-compile-") as temporary:
        build = Path(temporary)
        for component in COMPONENTS:
            name = f"{metadata['id']}_{profile_id}_{component}"
            graph = directory / (component + ".onnx")
            manifest["derived_graphs"].append({"component_id": component, "sha256": sha256_file(graph)[0]})
            loaded = load_model(onnx_source(str(graph), layout="NCHW", flexible_batch_size=False), target=gen2_target)
            data = (calibration_inputs(dataset, directory / "text_calibration.onnx", directory / "tokenizer.json", table,
                                       create_text_calibrator, detector["calibration_samples"])
                    if component == "detector" else None)
            quantized = loaded.quantize(calibration_data=data, quantization_config=quantization(component, parameters[component]),
                                        model_name=name, log_level=compiler_log_level())
            if quantized_output is not None:
                quantized.save(model_name=name, output_directory=str(quantized_output / component),
                               include_unquantized_net=False)
            quantized.compile(output_path=str(build / component), batch_size=1,
                              log_level=compiler_log_level(), layer_norm_use_fp32_intermediates=True)
            (archive,) = (build / component).glob("*_mpk.tar.gz")
            audit_mpk(archive)
            destination = output / component
            destination.mkdir()
            shutil.copyfile(archive, destination / archive.name)
            manifest["outputs"].append({
                "component_id": component,
                "mpk_path": f"{component}/{archive.name}",
                "mpk_sha256": sha256_file(destination / archive.name)[0],
            })
    assets = output / "prompt_assets"
    assets.mkdir()
    manifest["runtime_files"] = []
    for name in ("token_embedding_table.npy", "tokenizer.json"):
        destination = assets / name
        shutil.copyfile(directory / name, destination)
        manifest["runtime_files"].append({
            "path": destination.relative_to(output).as_posix(),
            "sha256": sha256_file(destination)[0],
        })
    (output / "compile-outputs.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-profile", required=True)
    parser.add_argument("--compile-invocation", required=True)
    parser.add_argument("--dataset", type=Path, required=True, help="verified Dataset Registry calibration archive")
    args = parser.parse_args()
    if args.compile_invocation != args.build_profile:
        parser.error("one compile invocation per build profile")
    logging.basicConfig(level=logging.INFO)
    import torch
    torch.set_num_threads(4)
    compile_model(args.model.resolve(), args.output_dir.resolve(), args.build_profile, args.dataset)


if __name__ == "__main__":
    main()
