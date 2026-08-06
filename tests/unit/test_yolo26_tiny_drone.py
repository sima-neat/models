from __future__ import annotations

import ast
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "training" / "yolo26_tiny_drone"
sys.path.insert(0, str(RECIPE))

import provenance  # noqa: E402
from contract import DeploymentContract  # noqa: E402
from dataset import FrameRecord, read_yolo_label  # noqa: E402
from paths import ultralytics_label_path  # noqa: E402
from prepare_multiuav import (  # noqa: E402
    SequenceStats,
    _canonical_label_text,
    _validate_label,
    split_sequences,
)
from train_qat import select_qat_training_records  # noqa: E402


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


def test_contract_scales_every_public_tensor_for_high_resolution() -> None:
    contract = DeploymentContract(1280)

    assert contract.input_shape == (1, 3, 1280, 1280)
    assert contract.feature_sizes == (320, 160, 80, 40)
    assert contract.output_shapes() == [
        (1, 4, 320, 320),
        (1, 4, 160, 160),
        (1, 4, 80, 80),
        (1, 4, 40, 40),
        (1, 1, 320, 320),
        (1, 1, 160, 160),
        (1, 1, 80, 80),
        (1, 1, 40, 40),
    ]
    assert DeploymentContract.from_dict(contract.to_dict()) == contract


def test_multiuav_split_keeps_sequences_whole_and_balances_objects() -> None:
    sequences = [SequenceStats(f"MultiUAV-{index:03d}", 750 + index, 2_000 + index * 97) for index in range(30)]

    first = split_sequences(sequences, seed=20260806, fixed={"MultiUAV-003": "test"})
    second = split_sequences(sequences, seed=20260806, fixed={"MultiUAV-003": "test"})

    assert first == second
    assert first["MultiUAV-003"] == "test"
    assert set(first) == {item.sequence for item in sequences}
    assert set(first.values()) == {"train", "val", "test"}
    split_boxes = {
        split: sum(item.boxes for item in sequences if first[item.sequence] == split)
        for split in ("train", "val", "test")
    }
    total_boxes = sum(split_boxes.values())
    assert abs(split_boxes["train"] / total_boxes - 0.8) < 0.08
    assert abs(split_boxes["val"] / total_boxes - 0.1) < 0.05
    assert abs(split_boxes["test"] / total_boxes - 0.1) < 0.05


def test_qat_subset_covers_every_sequence_and_temporal_span(tmp_path: Path) -> None:
    records = [
        FrameRecord(
            image=tmp_path / f"{sequence}_{frame:04d}.jpg",
            label=tmp_path / f"{sequence}_{frame:04d}.txt",
            sequence=sequence,
            split="train",
        )
        for sequence in ("a", "b", "c")
        for frame in range(100)
    ]

    selected, evidence = select_qat_training_records(records, 30)

    assert len(selected) == 30
    assert evidence["selected_sequences"] == 3
    assert evidence["per_sequence_images"] == {"a": 10, "b": 10, "c": 10}
    for sequence in ("a", "b", "c"):
        frame_numbers = [int(record.image.stem.rsplit("_", 1)[1]) for record in selected if record.sequence == sequence]
        assert min(frame_numbers) < 10
        assert max(frame_numbers) > 90


def test_label_reader_canonicalizes_six_decimal_edge_rounding(tmp_path: Path) -> None:
    label = tmp_path / "edge.txt"
    label.write_text("0 0.502883 0.992637 0.038172 0.014727\n", encoding="utf-8")

    boxes = read_yolo_label(label)

    assert float(boxes[0, 2] + boxes[0, 4] / 2) <= 1.0


def test_multiuav_preparation_clips_out_of_frame_annotations(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("0 1.002094 0.398008 0.019937 0.039023\n", encoding="utf-8")

    boxes, clipped, dropped, duplicate, maximum_overflow = _validate_label(source)
    canonical = tmp_path / "canonical.txt"
    canonical.write_text(_canonical_label_text(source), encoding="utf-8")

    assert boxes == clipped == 1
    assert dropped == 0
    assert duplicate == 0
    assert 0.012 < maximum_overflow < 0.013
    assert _validate_label(canonical) == (1, 0, 0, 0, 0.0)


def test_multiuav_preparation_drops_non_intersecting_boxes(tmp_path: Path) -> None:
    source = tmp_path / "outside.txt"
    source.write_text("0 1.006049 0.437954 0.012097 0.014560\n", encoding="utf-8")

    boxes, clipped, dropped, duplicate, _maximum_overflow = _validate_label(source)

    assert (boxes, clipped, dropped, duplicate) == (0, 0, 1, 0)
    assert _canonical_label_text(source) == ""


def test_multiuav_preparation_removes_exact_duplicate_boxes(tmp_path: Path) -> None:
    source = tmp_path / "duplicates.txt"
    line = "0 0.5 0.5 0.1 0.1\n"
    source.write_text(line + line, encoding="utf-8")

    boxes, clipped, dropped, duplicate, _maximum_overflow = _validate_label(source)

    assert (boxes, clipped, dropped, duplicate) == (1, 0, 0, 1)
    assert _canonical_label_text(source).count("\n") == 1


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
    assert "shutil.copy2" in source
    assert "select_trainable_parameters" in source
    assert 'name.startswith("detector_model_29_")' in source
    assert '"_cv3_" in name' in source
    assert "F.smooth_l1_loss" in source
    assert 'for branch in ("one2many", "one2one")' in source
    assert '"class_distillation_loss": distillation_total / len(train_data)' in source
    assert '"trainable_parameter_count": trainable_parameter_count' in source
    assert "initial_qat_metrics" in source
    assert "default=0.2" in source
    assert 'args.output_dir / "checkpoint-best-frozen.pt"' in source
    assert '"maximum_frozen_ap50_then_tiny_recall50"' in source


def test_fp32_training_records_optional_teacher_initialization_lineage() -> None:
    source = (RECIPE / "train_fp32.py").read_text(encoding="utf-8")
    ast.parse(source)
    assert 'parser.add_argument("--teacher-init-weights", type=Path)' in source
    assert "teacher_initialization = args.teacher_init_weights or args.base_weights" in source
    assert "teacher = YOLO(str(teacher_initialization))" in source
    assert 'feature_heads = len(getattr(head, "cv2", ()))' in source
    assert '"teacher_initialization": {' in source
    assert '"warm_started": args.teacher_init_weights is not None' in source


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
    assert 'tensor_types != ["int8"] * len(contract.outputs)' in source
    assert 'str(plugin.get("processor", "")).upper() == "A65"' in source


def test_ultralytics_label_path_matches_training_loader_convention() -> None:
    image = Path("/dataset/images/sequence/frame.jpg")
    assert ultralytics_label_path(image) == Path("/dataset/labels/sequence/frame.txt")


def test_ultralytics_revision_prefers_distribution_provenance(monkeypatch) -> None:
    monkeypatch.setattr(provenance, "_distribution_revision", lambda: provenance.ULTRALYTICS_REVISION)
    monkeypatch.setattr(provenance, "_git_revision", lambda _package_root: "unrelated-project-revision")

    assert provenance.require_ultralytics_revision(RECIPE) == provenance.ULTRALYTICS_REVISION
