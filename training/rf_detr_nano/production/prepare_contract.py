"""Prepare a local handoff from a verified frozen artifact directory.

Usage: python prepare_contract.py --artifacts DIR --manifest-sha256 SHA --output FILE

No network, source registration, compilation, model execution or promotion.
The manifest checksum must come from the caller's independent frozen receipt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from contract import require_sha, sha256_file


def prepare(directory: Path, manifest_sha256: str) -> dict:
    manifest_path = directory / "MANIFEST.json"
    if sha256_file(manifest_path) != require_sha(manifest_sha256):
        raise ValueError("frozen manifest checksum mismatch")
    manifest = json.loads(manifest_path.read_text())
    # Verify the bytes of every artifact, not the historical absolute paths.
    checksums = {}
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.strip()
        if Path(name).name != name or name in {".", ".."} or name in checksums:
            raise ValueError("unsafe or duplicate artifact checksum entry")
        path = directory / name
        if path.is_symlink() or sha256_file(path) != require_sha(digest):
            raise ValueError(f"artifact checksum mismatch: {name}")
        checksums[name] = digest
    names = (
        "MANIFEST.json",
        "SPLIT_CONTRACT.json",
        "INPUT_QUANTIZATION.json",
        "OUTPUT_QUANTIZATION.json",
        "recovered-reference.onnx",
        "rf_detr_nano_part_a_mpk.tar.gz",
        "rf_detr_nano_part_b_mpk.tar.gz",
    )
    if not set(names) <= checksums.keys() or checksums["MANIFEST.json"] != manifest_sha256:
        raise ValueError("incomplete or inconsistent frozen artifact checksums")
    split = json.loads((directory / "SPLIT_CONTRACT.json").read_text())
    ingress = json.loads((directory / "INPUT_QUANTIZATION.json").read_text())
    egress = json.loads((directory / "OUTPUT_QUANTIZATION.json").read_text())
    if ingress["arithmetic"] != "Float32Divide_RoundToEven_AddZeroPoint_ClipInt8":
        raise ValueError("unsupported source input arithmetic")
    if egress["source_onnx_sha256"] != checksums["recovered-reference.onnx"]:
        raise ValueError("output grids do not bind the supplied source reference")
    if [entry["slot"] for entry in egress["outputs"]] != [0, 1, 2]:
        raise ValueError("output grid order is not deltas, logits, references")
    if any(entry["arithmetic"] != "Float32_SubtractZero_MultiplyStep" for entry in egress["outputs"]):
        raise ValueError("unsupported source output arithmetic")
    kernel = split["kernel"]
    components = {}
    for component, part in (("backbone", "a"), ("decoder", "b")):
        ports = split["parts_public_abi"][part]
        components[component] = {
            "sha256": checksums[f"rf_detr_nano_part_{part}_mpk.tar.gz"],
            "inputs": ports["inputs"],
            "outputs": ports["outputs"],
        }
    return {
        "schema": "rfdetr.nano_qat_handoff/v1",
        "checkpoint_sha256": require_sha(manifest["checkpoint_sha256"]),
        "reference_sha256": checksums["recovered-reference.onnx"],
        "components": components,
        "native_seals": {component: split["parts"][part] for component, part in (("backbone", "a"), ("decoder", "b"))},
        "input_grid": {"step_bits": ingress["step_float32_bits"], "zero_point": ingress["zero_point"]},
        "output_grids": [
            {"name": entry["name"], "step_bits": entry["source_step_float32_bits"], "zero_point": entry["zero_point"]}
            for entry in egress["outputs"]
        ],
        "topk": {key: kernel[key] for key in ("k", "ties", "exhaustion")},
        # The reviewed source processor rescales FLOAT32 before AA resize.
        # This is a requested contract, NOT new hardware/AP qualification.
        "preprocessing": {
            "stages": ["Rescale", "Resize", "Normalize", "Quantize"],
            "resize": "BILINEAR_ANTIALIASED",
            "rescale_float32_bits": 998277249,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = prepare(args.artifacts, args.manifest_sha256)
    with args.output.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"contract_sha256": sha256_file(args.output), "execution": "NOT_RUN"}))


if __name__ == "__main__":
    main()
