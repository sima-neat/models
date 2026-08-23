from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OST = ROOT / "training" / "ostrack_vitb_noce_int8_qat"
EDGE = ROOT / "training" / "edgetam_p1_int8_qat"


def _load_reference(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["status"] == "qualified_pytorch_reference"
    return payload


def test_ostrack_reference_is_within_the_declared_two_point_gate() -> None:
    reference = _load_reference(OST / "qualification_reference.json")

    assert reference["selected_state"] == "update0"
    assert reference["acceptance"]["passed"] is True
    assert reference["acceptance"]["observed_iou_drop"] <= reference["acceptance"]["maximum_iou_drop"]
    assert reference["w8a8"]["peak"] == reference["bf16"]["peak"]
    assert reference["w8a8"]["success50"] == reference["bf16"]["success50"]


def test_edgetam_reference_is_within_the_declared_two_point_gate() -> None:
    reference = _load_reference(EDGE / "qualification_reference.json")

    assert reference["selected_state"] == "stage_a"
    assert reference["acceptance"]["passed"] is True
    assert reference["acceptance"]["observed_j_and_f_drop"] <= reference["acceptance"]["maximum_j_and_f_drop"]
    assert reference["deployment_contract"].endswith("plus_memory_encoder_p1")


def test_promotion_tools_are_syntax_valid_and_relocatable() -> None:
    for directory in (OST, EDGE):
        for path in directory.glob("*.py"):
            source = path.read_text(encoding="utf-8")
            ast.parse(source)
            assert "/workspace/" not in source
            assert "/home/manuel/" not in source
            assert "192.168." not in source


def test_edgetam_readme_names_all_three_p1_components() -> None:
    readme = (EDGE / "README.md").read_text(encoding="utf-8")

    assert "image-prompt/frame-zero" in readme
    assert "recurrent `s1_p1`" in readme
    assert "memory encoder" in readme
    assert "must not be described as" in readme


def test_qualified_catalogues_are_workspace_inputs_not_compile_evidence() -> None:
    for path in (OST / "README.md", EDGE / "README.md"):
        readme = path.read_text(encoding="utf-8")
        assert "Quantization Workspace" in readme
        assert "does not" in readme or "not itself" in readme
        assert "recalibration" in readme.lower()
