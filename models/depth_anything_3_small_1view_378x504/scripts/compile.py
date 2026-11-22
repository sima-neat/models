"""Compile one DA3-Small view with BF16 activations, per-channel INT8 weights and per-token dynamic quantization."""

from __future__ import annotations

import json
import logging
import math
import tarfile
from pathlib import Path

from afe.apis.defines import (
    QuantizationParams,
    RequantizationMode,
    SkipCalibration,
    bfloat16_scheme,
    gen2_target,
    quantization_scheme,
)
from afe.apis.loaded_net import load_model
from afe.ir.tensor_type import ScalarType
from afe.load.importers.general_importer import onnx_source
from model_registry.compile_script import component_parameters, parse_arguments
from model_registry.model import load_model as load_metadata
from surgery import INPUT_SHAPES, prepare_model

MODEL_DIR = Path(__file__).resolve().parents[1]
COMPONENT_ID = "model"


def quantization() -> QuantizationParams:
    return QuantizationParams(
        calibration_method=SkipCalibration(),
        activation_quantization_scheme=bfloat16_scheme(),
        weight_quantization_scheme=quantization_scheme(asymmetric=False, per_channel=True, bits=8),
        requantization_mode=RequantizationMode.sima,
        per_token_quant=True,
    )


def audit_mpk(archive: Path) -> None:
    """Require one MLA segment, no host-compute library, and both full-resolution output maps."""
    with tarfile.open(archive, "r:gz") as stream:
        names = stream.getnames()
        manifests = [name for name in names if name.endswith("_mpk.json")]
        if len(manifests) != 1:
            raise RuntimeError("expected one MPK manifest")
        manifest = json.load(stream.extractfile(manifests[0]))
    plugins = sorted(manifest["plugins"], key=lambda plugin: plugin["sequence"])
    processors = [str(plugin.get("processor", "")).upper() for plugin in plugins]
    if processors.count("MLA") != 1 or set(processors) - {"MLA", "EV74"}:
        raise RuntimeError(f"expected one MLA segment and I/O transforms, found {processors}")
    if len([name for name in names if name.endswith(".elf")]) != 1 or any(".so" in name for name in names):
        raise RuntimeError("expected one ELF without host-compute libraries")
    expected_input = math.prod(INPUT_SHAPES["image"]) * 4
    if [(value["name"], value["size"]) for value in manifest["input_nodes"]] != [("image", expected_input)]:
        raise RuntimeError("unexpected MPK image input")
    output_size = expected_input // 3
    if [value["size"] for value in plugins[-1]["output_nodes"]] != [output_size, output_size]:
        raise RuntimeError("expected two float32 output maps")


def compile_model(source: Path, output_dir: Path, profile_id: str) -> Path:
    metadata = load_metadata(MODEL_DIR)
    parameters = component_parameters(MODEL_DIR, profile_id, COMPONENT_ID)
    if parameters["precision"] != "w8a16" or parameters["input_shape"] != INPUT_SHAPES["image"]:
        raise ValueError("expected W8A16 at the model's single-view input shape")

    component_dir = output_dir / COMPONENT_ID
    component_dir.mkdir(parents=True, exist_ok=False)
    prepared = prepare_model(source, output_dir / "prepared")
    importer = onnx_source(str(prepared), {"image": tuple(INPUT_SHAPES["image"])}, {"image": ScalarType.float32})
    name = f"{metadata['id']}_{profile_id}"
    try:
        model = load_model(importer, target=gen2_target).quantize(
            calibration_data=None, quantization_config=quantization(), model_name=name
        )
        model.compile(
            output_path=str(component_dir), batch_size=1, layer_norm_use_fp32_intermediates=True
        )
    except SystemExit as error:
        # The compiler can report a fatal error through SystemExit(0).
        raise RuntimeError(f"Model Compiler failed for {name}") from error
    archives = sorted(component_dir.glob("*_mpk.tar.gz"))
    expected = component_dir / f"{name}_mpk.tar.gz"
    if archives != [expected]:
        raise RuntimeError(f"expected exactly {expected}, found {archives}")
    audit_mpk(expected)
    return expected


def main() -> None:
    arguments = parse_arguments(__doc__, MODEL_DIR)
    logging.basicConfig(level=logging.INFO, force=True)
    print(compile_model(arguments.model, arguments.output_dir, arguments.build_profile))


if __name__ == "__main__":
    main()
