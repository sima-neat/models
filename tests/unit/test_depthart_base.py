from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "depthart_base"
TRAINING_DIR = ROOT / "training" / "depthart_base"


def load_compile_module():
    spec = importlib.util.spec_from_file_location(
        "depthart_base_compile", MODEL_DIR / "compile.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_metadata_declares_strict_int8_dynamic_recurrence() -> None:
    metadata = yaml.safe_load((MODEL_DIR / "model.yaml").read_text(encoding="utf-8"))
    assert metadata["taxonomy_version"] == 3
    variant = metadata["compile_variants"][0]
    parameters = variant["parameters"]
    assert variant["id"] == "modalix_strict_int8_dynamic_p64"
    assert parameters["public_input_dtype"] == "int8"
    assert parameters["public_output_dtype"] == "int8"
    assert parameters["recurrence"] == "dynamic_q8_p64"
    assert parameters["expected_bindings"] == 918
    assert parameters["require_zero_float_compute"] is True
    assert "compact_aux_dense_row_partition_v1" in parameters["required_compiler_features"]


def test_compile_profile_is_hash_valid_and_source_bound() -> None:
    module = load_compile_module()
    document = json.loads(module.PROFILE.read_text(encoding="utf-8"))
    assert sha256(module.PROFILE) == module.EXPECTED_PROFILE_SHA256
    unsigned = dict(document)
    manifest = unsigned.pop("manifest_sha256")
    assert module._canonical_sha256(unsigned) == manifest == module.EXPECTED_PROFILE_MANIFEST
    assert document["onnx_sha256"] == module.EXPECTED_SOURCE_SHA256
    assert len(document["bindings"]) == module.EXPECTED_BINDINGS


def test_profile_bindings_are_unique_tree_compositions() -> None:
    document = json.loads((MODEL_DIR / "depthart_dynamic_p64.json").read_text())
    names = []
    for binding in document["bindings"]:
        assert binding["scan_primitive"] == "tree_compose"
        assert binding["publication"]["zero_point"] == 0
        names.extend(
            [binding["product"]["source_node"], binding["state_add"]["source_node"]]
        )
    assert len(names) == 1836
    assert len(set(names)) == len(names)


def test_architecture_patch_applies_to_pinned_upstream_shape() -> None:
    config = yaml.safe_load((TRAINING_DIR / "qat.yaml").read_text(encoding="utf-8"))
    patch = TRAINING_DIR / "depthart_dynamic_p64_source.patch"
    assert sha256(patch) == config["source"]["architecture_patch"]["sha256"]
    assert config["source"]["revision"] == "0384521b3bcb4c64adf03eeb5d55ebdb1cbdd84c"
    assert config["architecture"]["equivalence_contract"]["changes_model_function"] is False


def test_qat_recipe_pins_exact_runner_and_acceptance() -> None:
    config = yaml.safe_load((TRAINING_DIR / "qat.yaml").read_text(encoding="utf-8"))
    assert config["runner_contract"]["revision"] == "621f1cffe3c1d47be5b626bee74cce96b4a15621"
    assert config["quantization"]["forbidden_compute_dtypes"] == [
        "bfloat16",
        "float16",
        "float32",
    ]
    assert config["acceptance"]["minimum_delta1"] == 0.90
    assert config["acceptance"]["maximum_abs_rel"] == 0.09
    assert config["acceptance"]["require_modalix_full_split"] is True
