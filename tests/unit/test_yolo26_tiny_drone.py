from __future__ import annotations

import ast
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "training" / "yolo26_tiny_drone"
sys.path.insert(0, str(RECIPE))

import provenance  # noqa: E402
from paths import ultralytics_label_path  # noqa: E402


def test_int8_architecture_keeps_four_heads_and_removes_attention() -> None:
    architecture = yaml.safe_load((RECIPE / "yolo26n-p2-int8.yaml").read_text(encoding="utf-8"))
    assert architecture["nc"] == 1
    assert architecture["end2end"] is True
    assert architecture["reg_max"] == 1
    serialized = (RECIPE / "yolo26n-p2-int8.yaml").read_text(encoding="utf-8")
    assert "C2PSA" not in serialized
    assert "[1024, true, 0.5, true]" not in serialized
    assert architecture["head"][-1][0] == [19, 22, 25, 28]


def test_contract_declares_grouped_p2_through_p5_outputs() -> None:
    namespace: dict[str, object] = {}
    exec((RECIPE / "contract.py").read_text(encoding="utf-8"), namespace)
    outputs = namespace["OUTPUTS"]
    assert [name for name, _shape in outputs] == [
        "bbox_0",
        "bbox_1",
        "bbox_2",
        "bbox_3",
        "class_logit_0",
        "class_logit_1",
        "class_logit_2",
        "class_logit_3",
    ]
    assert [shape[-1] for _name, shape in outputs[:4]] == [160, 80, 40, 20]


def test_qat_script_enforces_representative_floor_and_batch_one() -> None:
    source = (RECIPE / "train_qat.py").read_text(encoding="utf-8")
    ast.parse(source)
    assert "observer_warmup_images < 128" in source
    assert "args.batch_size != 1" in source
    assert "requires_grad_(True)" in source
    assert "random.Random(seed).sample(train_records, required)" in source
    assert '"method": "seeded_random_without_replacement"' in source
    assert "scalar_loss / window_size" in source
    assert "evaluate_without_observer_updates" in source
    assert "model.apply(disable_observer)" in source
    assert "model.apply(enable_observer)" in source
    assert "validate_resume_checkpoint" in source
    assert 'qat_model.load_state_dict(resume["qat_state_dict"], strict=True)' in source
    assert '"resume_checkpoint_sha256": resume_digest' in source
    assert "initial_qat_metrics" in source
    assert "default=0.2" in source
    assert 'args.output_dir / "checkpoint-best-frozen.pt"' in source
    assert '"maximum_frozen_ap50_then_tiny_recall50"' in source


def test_export_audits_full_int8_operator_boundary() -> None:
    source = (RECIPE / "onnx_audit.py").read_text(encoding="utf-8")
    ast.parse(source)
    for operator in ("BatchNormalization", "MatMul", "Einsum", "Softmax"):
        assert f'"{operator}"' in source
    assert "signed_int8_symmetric_axis0_per_channel" in source
    evaluator = (RECIPE / "evaluate.py").read_text(encoding="utf-8")
    assert "head.anchors = head.anchors.to(device)" in evaluator
    assert "head.strides = head.strides.to(device)" in evaluator


def test_compile_recipe_requires_representative_data_and_audits_the_mpk() -> None:
    source = (RECIPE / "compile.py").read_text(encoding="utf-8")
    ast.parse(source)
    assert "MINIMUM_CALIBRATION_IMAGES = 128" in source
    assert "random.Random(seed).sample" in source
    assert '"qualification_eligible": not args.allow_smoke' in source
    assert "EXPECTED_CONVOLUTIONS = 131" in source
    assert 'tensor_types != ["int8"] * len(OUTPUTS)' in source
    assert 'str(plugin.get("processor", "")).upper() == "A65"' in source


def test_ultralytics_label_path_matches_training_loader_convention() -> None:
    image = Path("/dataset/images/sequence/frame.jpg")
    assert ultralytics_label_path(image) == Path("/dataset/labels/sequence/frame.txt")


def test_ultralytics_revision_prefers_distribution_provenance(monkeypatch) -> None:
    monkeypatch.setattr(provenance, "_distribution_revision", lambda: provenance.ULTRALYTICS_REVISION)
    monkeypatch.setattr(provenance, "_git_revision", lambda _package_root: "unrelated-project-revision")

    assert provenance.require_ultralytics_revision(RECIPE) == provenance.ULTRALYTICS_REVISION
