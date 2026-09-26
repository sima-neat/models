from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "training" / "yolo26_tiny_drone"
sys.path.insert(0, str(RECIPE))

import prepare_multiuav  # noqa: E402
import provenance  # noqa: E402
from contract import DeploymentContract  # noqa: E402
from dataset import (  # noqa: E402
    FrameRecord,
    dataset_content_sha256,
    read_yolo_label,
    require_unchanged_dataset,
)
from evaluate import AP_CANDIDATE_SCORE_THRESHOLD, _match_truth_indices  # noqa: E402
from paths import natural_path_key, ultralytics_label_path  # noqa: E402
from prepare_multiuav import (  # noqa: E402
    SequenceStats,
    _canonical_label_text,
    _validate_label,
    prepare_verified_archive,
    split_sequences,
    verify_source_root,
)
from train_qat import select_qat_training_records, validate_fp32_lineage  # noqa: E402


def test_deployment_architecture_keeps_four_heads_and_original_attention() -> None:
    architecture_path = RECIPE / "yolo26n-p2-deploy.yaml"
    architecture = yaml.safe_load(architecture_path.read_text(encoding="utf-8"))
    assert architecture["nc"] == 1
    assert architecture["end2end"] is True
    assert architecture["reg_max"] == 1
    serialized = architecture_path.read_text(encoding="utf-8")
    assert "C2PSA" in serialized
    assert "[1024, true, 0.5, true]" in serialized
    assert architecture["backbone"][8] == [-1, 2, "C3k2", [1024, True]]
    assert architecture["backbone"][9] == [-1, 1, "SPPF", [1024, 5, 3, True]]
    assert architecture["backbone"][10] == [-1, 2, "C2PSA", [1024]]
    assert architecture["head"][-1][0] == [19, 22, 25, 28]


def test_per_head_mla_attention_preserves_original_math_and_state_dict() -> None:
    import copy

    import torch
    from model import lower_attention_for_mla
    from ultralytics.nn.modules.block import Attention

    original = Attention(128, num_heads=2).eval()
    lowered, count = lower_attention_for_mla(copy.deepcopy(original))
    assert count == 1
    assert lowered.state_dict().keys() == original.state_dict().keys()
    inputs = torch.randn(1, 128, 8, 8)
    torch.testing.assert_close(lowered(inputs), original(inputs), rtol=1e-6, atol=1e-6)


def test_export_canonicalizes_rank_four_matmul_to_modelsdk_einsum(tmp_path: Path) -> None:
    import onnx
    from onnx import TensorProto, helper
    from onnx_audit import ATTENTION_EINSUM_EQUATION, canonicalize_attention_for_modelsdk

    path = tmp_path / "attention.onnx"
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["lhs", "rhs"], ["result"], name="attention/MatMul")],
        "attention",
        [
            helper.make_tensor_value_info("lhs", TensorProto.FLOAT, [1, 1, 2, 3]),
            helper.make_tensor_value_info("rhs", TensorProto.FLOAT, [1, 1, 3, 4]),
        ],
        [helper.make_tensor_value_info("result", TensorProto.FLOAT, [1, 1, 2, 4])],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)

    evidence = canonicalize_attention_for_modelsdk(path, expected_matmuls=1)
    rewritten = onnx.load(path)

    assert evidence["converted_matmuls"] == 1
    assert [node.op_type for node in rewritten.graph.node] == ["Einsum"]
    attributes = {
        attribute.name: helper.get_attribute_value(attribute) for attribute in rewritten.graph.node[0].attribute
    }
    assert attributes["equation"].decode() == ATTENTION_EINSUM_EQUATION


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


def test_multiuav_source_root_must_match_the_verified_archive(tmp_path: Path) -> None:
    from zipfile import ZipFile

    source_root = tmp_path / "fixture"
    for split in ("train", "val"):
        directory = source_root / split
        directory.mkdir(parents=True)
        (directory / "frame.jpg").write_bytes(f"{split}-image".encode())
        (directory / "frame.txt").write_text(f"{split}-label", encoding="utf-8")
    archive = tmp_path / "fixture.zip"
    with ZipFile(archive, "w") as compressed:
        for path in sorted(source_root.rglob("*")):
            if path.is_file():
                compressed.write(path, path.relative_to(tmp_path).as_posix())

    evidence = verify_source_root(archive, source_root)
    assert evidence["member_count"] == 4
    assert evidence["method"] == "byte_for_byte_verified_archive_members"

    (source_root / "val" / "frame.txt").write_text("val-labeL", encoding="utf-8")
    with pytest.raises(ValueError, match="bytes differ from archive"):
        verify_source_root(archive, source_root)


def test_multiuav_materializes_from_one_unaliased_private_archive_snapshot(tmp_path: Path, monkeypatch) -> None:
    from zipfile import ZipFile

    archive = tmp_path / "fixture.zip"
    expected_images = {}
    with ZipFile(archive, "w") as compressed:
        compressed.writestr("fixture/val/", b"")
        for index in range(1, 4):
            name = f"MultiUAV-{index:03d}_1"
            image = f"image-{index}".encode()
            expected_images[name] = image
            compressed.writestr(f"fixture/train/{name}.jpg", image)
            compressed.writestr(f"fixture/train/{name}.txt", "0 0.5 0.5 0.1 0.1\n")

    source_parent = tmp_path / "publisher-extraction"
    with ZipFile(archive) as compressed:
        compressed.extractall(source_parent)
    original_verify = prepare_multiuav.verify_source_root

    def verify_then_replace_original(snapshot: Path, source_root: Path) -> dict[str, int | str]:
        evidence = original_verify(snapshot, source_root)
        archive.write_bytes(b"replacement-after-verification")
        return evidence

    monkeypatch.setattr(prepare_multiuav, "verify_source_root", verify_then_replace_original)
    destination = tmp_path / "prepared"
    manifest, statistics = prepare_verified_archive(
        archive,
        source_parent / "fixture",
        destination,
        expected_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        seed=20260806,
        fixed_assignments={
            "MultiUAV-001": "train",
            "MultiUAV-002": "val",
            "MultiUAV-003": "test",
        },
    )

    assert manifest.is_file()
    assert statistics == {"frames": 3, "boxes": 3, "sequences": 3}
    assert archive.read_bytes() == b"replacement-after-verification"
    images = sorted(destination.glob("images/*/*.jpg"))
    assert {image.stem: image.read_bytes() for image in images} == expected_images
    assert all(image.stat().st_nlink == 1 for image in images)


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


def test_qat_temporal_bins_sort_non_padded_frame_numbers_naturally(tmp_path: Path) -> None:
    records = [
        FrameRecord(
            image=tmp_path / f"frame_{frame}.jpg",
            label=tmp_path / f"frame_{frame}.txt",
            sequence="one",
            split="train",
        )
        for frame in (1, 2, 3, 10)
    ]

    selected, _evidence = select_qat_training_records(records, 2)

    assert [record.image.name for record in selected] == ["frame_2.jpg", "frame_10.jpg"]
    assert natural_path_key(tmp_path / "frame_2.jpg") < natural_path_key(tmp_path / "frame_10.jpg")


def test_matching_tries_next_available_overlapping_truth() -> None:
    import torch

    predicted = torch.tensor([[0.0, 0.0, 20.0, 20.0], [0.0, 0.0, 20.0, 20.0]])
    truth = torch.tensor([[0.0, 0.0, 20.0, 20.0], [0.0, 0.0, 15.0, 15.0]])

    assert _match_truth_indices(predicted, truth, 0.50) == [0, 1]


def test_qat_propagates_fp32_lineage_eligibility(tmp_path: Path) -> None:
    import json

    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"fp32")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    contract = DeploymentContract(640)
    lineage_path = tmp_path / "lineage.json"
    lineage = {
        "schema_version": 2,
        "architecture": "yolo26n-p2-attention-one-class",
        "architecture_sha256": hashlib.sha256((RECIPE / "yolo26n-p2-deploy.yaml").read_bytes()).hexdigest(),
        "deployment": contract.to_dict(),
        "best_weights_sha256": hashlib.sha256(b"fp32").hexdigest(),
        "manifest_sha256": hashlib.sha256(b"{}").hexdigest(),
        "dataset_sha256": "dataset-digest",
        "qualification_eligible": False,
        "ultralytics_revision": provenance.ULTRALYTICS_REVISION,
        "ultralytics_package_sha256": provenance.ULTRALYTICS_PACKAGE_SHA256,
    }
    lineage_path.write_text(json.dumps(lineage), encoding="utf-8")

    assert (
        validate_fp32_lineage(lineage_path, checkpoint, manifest, contract, "dataset-digest")["qualification_eligible"]
        is False
    )

    with pytest.raises(RuntimeError, match="dataset contents do not match"):
        validate_fp32_lineage(lineage_path, checkpoint, manifest, contract, "changed-dataset")

    lineage["architecture_sha256"] = "0" * 64
    lineage_path.write_text(json.dumps(lineage), encoding="utf-8")
    with pytest.raises(RuntimeError, match="architecture digest"):
        validate_fp32_lineage(lineage_path, checkpoint, manifest, contract, "dataset-digest")


def test_dataset_digest_binds_every_referenced_file(tmp_path: Path) -> None:
    image = tmp_path / "images" / "frame.jpg"
    label = tmp_path / "labels" / "frame.txt"
    image.parent.mkdir()
    label.parent.mkdir()
    image.write_bytes(b"image-v1")
    label.write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    records = [FrameRecord(image=image, label=label, sequence="one", split="train")]

    original = dataset_content_sha256(manifest, records)
    manifest_digest = hashlib.sha256(b"{}").hexdigest()
    require_unchanged_dataset(
        manifest,
        records,
        manifest_sha256=manifest_digest,
        dataset_sha256=original,
        context="QAT",
    )
    image.write_bytes(b"image-v2")
    changed_image = dataset_content_sha256(manifest, records)
    label.write_text("", encoding="utf-8")
    changed_label = dataset_content_sha256(manifest, records)

    assert len({original, changed_image, changed_label}) == 3
    with pytest.raises(RuntimeError, match="QAT dataset contents changed"):
        require_unchanged_dataset(
            manifest,
            records,
            manifest_sha256=manifest_digest,
            dataset_sha256=original,
            context="QAT",
        )


def test_qat_eligibility_rejects_manifest_mutation(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    digest = hashlib.sha256(b"{}").hexdigest()
    manifest.write_text('{"changed": true}', encoding="utf-8")

    with pytest.raises(RuntimeError, match="QAT manifest changed"):
        require_unchanged_dataset(
            manifest,
            [],
            manifest_sha256=digest,
            dataset_sha256=dataset_content_sha256(manifest, []),
            context="QAT",
        )


def test_ap_uses_a_near_zero_candidate_threshold() -> None:
    assert AP_CANDIDATE_SCORE_THRESHOLD == 0.001


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
    assert 'parser.add_argument("--fp32-lineage", type=Path, required=True)' in source
    assert "validate_fp32_lineage" in source
    assert '"dataset_sha256": dataset_digest' in source
    assert "require_unchanged_dataset" in source
    assert '"qualification_eligible": qualification_eligible' in source
    assert 'fp32_lineage["qualification_eligible"] and not args.smoke and observers_frozen' in source
    assert 'qat_model.load_state_dict(resume["qat_state_dict"], strict=True)' in source
    assert '"resume_checkpoint_sha256": resume_digest' in source
    assert "load_verified_qat_sources" in source
    assert "swml_qat_runtime = activate_verified_swml_qat_runtime()" in source
    loader = source[source.index("def load_verified_qat_sources") : source.index("def select_trainable_parameters")]
    assert loader.index("activate_verified_ultralytics_runtime()") < loader.index("import ultralytics")
    assert '"swml_qat_revision": SWML_QAT_REVISION' in source
    assert '"swml_qat_package_sha256": swml_qat_runtime.package_sha256' in source
    assert '"ultralytics_package_sha256": sources.ultralytics_runtime.package_sha256' in source
    assert 'snapshot_file(args.fp32_checkpoint, snapshot_dir / "fp32.pt", fp32_digest)' in source
    assert "YOLO(str(fp32_snapshot))" in source
    assert "sha256_file(args.fp32_checkpoint)" not in source[source.index("def main()") :]
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


def test_fp32_training_uses_one_canonical_graph_and_records_transfer_lineage() -> None:
    source = (RECIPE / "train_fp32.py").read_text(encoding="utf-8")
    ast.parse(source)
    assert 'parser.add_argument("--initial-weights", type=Path)' in source
    assert 'Path(__file__).with_name("yolo26n-p2-deploy.yaml")' in source
    assert "initialization = args.initial_weights or args.base_weights" in source
    main = source[source.index("def main()") :]
    assert main.index("activate_verified_ultralytics_runtime()") < main.index("from ultralytics import YOLO")
    assert "model = YOLO(str(architecture_snapshot))" in source
    assert "model.load(str(initialization_snapshot))" in source
    assert "snapshot_file(" in source
    assert 'with tempfile.TemporaryDirectory(prefix="yolo26-fp32-inputs-")' in source
    assert source.count(".train(") == 1
    assert "teacher_results" not in source
    assert "student_results" not in source
    assert 'feature_heads = len(getattr(head, "cv2", ()))' in source
    assert '"initialization": {' in source
    assert '"compatible_tensors": len(compatible)' in source
    assert '"architecture_sha256": architecture_digest' in source
    assert '"sha256": initialization_digest' in source
    assert '"manifest_sha256": manifest_digest' in source
    assert '"dataset_sha256": dataset_digest' in source
    assert "not args.smoke and args.initial_weights is None" in source
    assert '"task_specific_initialization_without_lineage"' in source
    assert '"ultralytics_package_sha256": ultralytics_runtime.package_sha256' in source
    assert 'context="FP32 training"' in source


def test_fp32_loads_only_the_verified_read_only_snapshot(tmp_path: Path) -> None:
    from provenance import snapshot_file

    source = tmp_path / "weights.pt"
    destination = tmp_path / "snapshot.pt"
    source.write_bytes(b"verified-weights")
    digest = hashlib.sha256(b"verified-weights").hexdigest()

    snapshot_file(source, destination, digest)
    source.write_bytes(b"replacement-weights")

    assert destination.read_bytes() == b"verified-weights"
    assert destination.stat().st_mode & 0o777 == 0o400


def test_qat_requires_complete_attention_scaffolding() -> None:
    model_source = (RECIPE / "model.py").read_text(encoding="utf-8")
    train_source = (RECIPE / "train_qat.py").read_text(encoding="utf-8")
    bootstrap_source = (RECIPE / "bootstrap.sh").read_text(encoding="utf-8")
    assert "EXPECTED_ATTENTION_MATMULS = 8" in model_source
    assert "EXPECTED_ATTENTION_SOFTMAXES = 4" in model_source
    assert "lower_attention_for_mla(detector)" in train_source
    assert "validate_attention_qat_scaffolding(qat_model)" in train_source
    assert '"attention_scaffolding": attention_scaffolding' in train_source
    assert "Attention(128, num_heads=2)" in bootstrap_source
    assert "expected_matmuls=4" in bootstrap_source
    assert "expected_softmaxes=2" in bootstrap_source


def test_export_audits_full_int8_operator_boundary() -> None:
    source = (RECIPE / "onnx_audit.py").read_text(encoding="utf-8")
    ast.parse(source)
    for operator in ("BatchNormalization", "MatMul"):
        assert f'"{operator}"' in source
    assert '"Einsum": EXPECTED_ATTENTION_MATMULS' in source
    assert 'ATTENTION_EINSUM_EQUATION = "abij,abjk->abik"' in source
    assert "canonicalize_attention_for_modelsdk" in source
    assert '"Softmax": EXPECTED_ATTENTION_SOFTMAXES' in source
    assert 'producers[name].op_type != "DequantizeLinear"' in source
    assert 'consumer.op_type == "QuantizeLinear"' in source
    assert "signed_int8_symmetric_axis0_per_channel" in source
    evaluator = (RECIPE / "evaluate.py").read_text(encoding="utf-8")
    assert "head.anchors = head.anchors.to(device)" in evaluator
    assert "head.strides = head.strides.to(device)" in evaluator


def test_export_requires_dataset_content_lineage() -> None:
    import argparse

    from export import validate_checkpoint
    from model import EXPECTED_ATTENTION_MATMULS, EXPECTED_ATTENTION_SOFTMAXES

    checkpoint = {
        "schema_version": 1,
        "architecture": "yolo26n-p2-attention-one-class",
        "attention_scaffolding": {
            "matmul_count": EXPECTED_ATTENTION_MATMULS,
            "softmax_count": EXPECTED_ATTENTION_SOFTMAXES,
        },
        "deployment": DeploymentContract().to_dict(),
        "observer_images": 128,
        "observers_frozen": True,
        "qualification_eligible": True,
        "fp32_qualification_eligible": True,
    }

    with pytest.raises(RuntimeError, match="no valid dataset content digest"):
        validate_checkpoint(checkpoint, argparse.Namespace(allow_smoke=False))


def test_export_consumes_only_verified_input_snapshots() -> None:
    source = (RECIPE / "export.py").read_text(encoding="utf-8")
    ast.parse(source)
    loader = source[source.index("def load_verified_export_sources") : source.index("def main()")]
    main = source[source.index("def main()") :]
    assert 'with tempfile.TemporaryDirectory(prefix="yolo26-export-inputs-")' in loader
    assert "torch.load(checkpoint_snapshot" in loader
    assert "YOLO(str(fp32_snapshot))" in loader
    assert loader.index("activate_verified_ultralytics_runtime()") < loader.index("from ultralytics import YOLO")
    assert loader.index("activate_verified_swml_qat_runtime()") < loader.index("torch.load(checkpoint_snapshot")
    assert 'get("ultralytics_package_sha256")' in loader
    assert "ultralytics_runtime.package_sha256" in loader
    assert 'get("swml_qat_package_sha256") != swml_qat_runtime.package_sha256' in loader
    assert "torch.load(args.checkpoint" not in main
    assert "YOLO(str(args.fp32_checkpoint))" not in main
    assert '"checkpoint_sha256": sources.checkpoint_sha256' in main


def test_compile_recipe_requires_representative_data_and_audits_the_mpk() -> None:
    source = (RECIPE / "compile.py").read_text(encoding="utf-8")
    ast.parse(source)
    assert "MINIMUM_CALIBRATION_IMAGES = 128" in source
    assert "random.Random(seed).sample" in source
    assert 'export_report["qualification_eligible"] and not args.allow_smoke' in source
    assert 'if not report["qualification_eligible"] and not allow_smoke' in source
    assert "expected_convolutions: int" in source
    assert "EXPECTED_ATTENTION_MATMULS = 8" in source
    assert "EXPECTED_ATTENTION_SOFTMAXES = 4" in source
    assert '"BatchMatmul": EXPECTED_ATTENTION_MATMULS' in source
    assert "attention_types != {expected_io}" in source
    assert '"compute_scalar_type": "int8"' in source
    assert "any_shape_on_mla=" not in source
    assert "automatic_layout_conversion=" not in source
    assert "tessellate_parameters" not in source
    assert "len(mla_outputs) != len(contract.outputs)" in source
    assert '"tessellation": "mla"' in source
    assert "role_input_abi != expected_output_abi" in source
    assert 'str(plugin.get("processor", "")).upper() == "A65"' in source
    main = source[source.index("def main()") :]
    assert main.index("build_calibration(selected, contract)") < main.index("require_unchanged_dataset(")
    assert "with tempfile.TemporaryDirectory" in main
    assert main.count("validate_export(") == 2
    assert main.index("compile_model(") < main.rindex("validate_export(")


def _mpk_fixture(contract: DeploymentContract) -> tuple[dict[str, object], list[dict[str, object]]]:
    import numpy as np
    from contract import INPUT_NAME

    expected_outputs = [
        {"name": name, "size": int(np.prod(shape)) * np.dtype(np.float32).itemsize} for name, shape in contract.outputs
    ]
    terminal_inputs = [
        {"name": f"dequantize_{index}/{node['name']}", "size": node["size"]}
        for index, node in enumerate(expected_outputs)
    ]
    terminal_outputs = [
        {"name": f"pass_through_out_{index}", "size": node["size"]} for index, node in enumerate(expected_outputs)
    ]
    plugins = [
        {
            "sequence": 0,
            "processor": "EV74",
            "input_nodes": [{"name": INPUT_NAME}],
            "output_nodes": [{"name": "quantize_0"}],
            "config_params": {"kernel": "quantization_transform", "params": {"output_data_type": "int8"}},
        },
        {
            "sequence": 1,
            "processor": "MLA",
            "input_nodes": [{"name": "quantize_0"}],
            "output_nodes": [{"name": f"MLA_{i}"} for i in range(len(expected_outputs))],
            "config_params": {},
        },
        *[
            {
                "sequence": index + 3,
                "processor": "EV74",
                "config_params": {"kernel": "dequantization_transform", "params": {"input_data_type": "int8"}},
            }
            for index in range(len(expected_outputs))
        ],
        {
            "sequence": len(expected_outputs) + 3,
            "processor": "EV74",
            "input_nodes": terminal_inputs,
            "output_nodes": terminal_outputs,
            "config_params": {"kernel": "pass_through"},
        },
    ]
    return (
        {
            "model_sdk_version": json.loads((ROOT / "deps" / "manifest.json").read_text())["model-compiler"]["version"],
            "input_nodes": [
                {
                    "name": INPUT_NAME,
                    "size": int(np.prod(contract.input_shape)) * np.dtype(np.float32).itemsize,
                }
            ],
            "plugins": plugins,
        },
        terminal_inputs,
    )


def _write_mpk_fixture(path: Path, manifest: dict[str, object]) -> None:
    import io
    import json
    import tarfile

    document = json.dumps(manifest).encode()
    with tarfile.open(path, "w:gz") as stream:
        for name, content in (("model/model.elf", b"elf"), ("model/model_mpk.json", document)):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            stream.addfile(info, io.BytesIO(content))


def test_mpk_audit_rejects_equal_size_output_reordering(tmp_path: Path) -> None:
    from compile import audit_mpk

    contract = DeploymentContract()
    manifest, terminal_inputs = _mpk_fixture(contract)
    archive = tmp_path / "model.mpk"

    _write_mpk_fixture(archive, manifest)
    assert audit_mpk(archive, contract)["public_output_names"] == contract.output_names()

    terminal_inputs[1], terminal_inputs[4] = terminal_inputs[4], terminal_inputs[1]
    assert terminal_inputs[1]["size"] == terminal_inputs[4]["size"]
    _write_mpk_fixture(archive, manifest)
    with pytest.raises(RuntimeError, match="public output ABI mismatch"):
        audit_mpk(archive, contract)


@pytest.mark.parametrize("kernel", ["tessellation_transform", "detessellation_transform"])
def test_mpk_audit_rejects_external_tensor_layout(tmp_path: Path, kernel: str) -> None:
    from compile import audit_mpk

    contract = DeploymentContract()
    manifest, _terminal_inputs = _mpk_fixture(contract)
    plugins = manifest["plugins"]
    assert isinstance(plugins, list)
    plugins.insert(
        1,
        {
            "sequence": 1,
            "processor": "EV74",
            "config_params": {"kernel": kernel, "params": {}},
        },
    )
    archive = tmp_path / "external-layout.mpk"
    _write_mpk_fixture(archive, manifest)

    with pytest.raises(RuntimeError, match="does not implement tensor layout on MLA"):
        audit_mpk(archive, contract)


def test_mpk_audit_rejects_indirect_quantizer_to_mla_route(tmp_path: Path) -> None:
    from compile import audit_mpk

    contract = DeploymentContract()
    manifest, _terminal_inputs = _mpk_fixture(contract)
    plugins = manifest["plugins"]
    assert isinstance(plugins, list)
    mla = next(plugin for plugin in plugins if plugin.get("processor") == "MLA")
    mla["input_nodes"] = [{"name": "external_tessellated_input"}]
    archive = tmp_path / "indirect-mla.mpk"
    _write_mpk_fixture(archive, manifest)

    with pytest.raises(RuntimeError, match="quantizer does not feed MLA directly"):
        audit_mpk(archive, contract)


def test_compile_uses_a_private_export_snapshot(tmp_path: Path) -> None:
    import json

    from compile import snapshot_export

    export_dir = tmp_path / "export"
    snapshot_dir = tmp_path / "snapshot"
    export_dir.mkdir()
    snapshot_dir.mkdir()
    report = {"onnx": {"file": "model.onnx"}}
    (export_dir / "export-report.json").write_text(json.dumps(report), encoding="utf-8")
    (export_dir / "model.onnx").write_bytes(b"validated-model")

    report_snapshot, source_snapshot = snapshot_export(export_dir, snapshot_dir)
    (export_dir / "export-report.json").write_text("{}", encoding="utf-8")
    (export_dir / "model.onnx").write_bytes(b"mutated-model")

    assert json.loads(report_snapshot.read_text(encoding="utf-8")) == report
    assert source_snapshot.read_bytes() == b"validated-model"


def test_quantized_model_audit_reads_internal_int8_tensor_types(tmp_path: Path) -> None:
    import json

    from compile import audit_quantized_model

    def scalar(value: int) -> dict[str, object]:
        return {"__decode_key__": "ScalarType", "value": value}

    def layer(kind: str, inputs: int, **config: object) -> dict[str, object]:
        return {
            "class_name": f"MLA_0/{kind}",
            "config": {
                "_status": "SIMA_QUANTIZED",
                "_input_types": [scalar(1) for _ in range(inputs)],
                "_output_types": [scalar(1)],
                **config,
            },
        }

    layers = [layer("ConvAddActivation_0", 1, per_channel=True, weight_quant_data=[1])]
    layers.extend(layer(f"BatchMatmul_{index}", 2) for index in range(8))
    layers.extend(layer(f"Softmax_{index}", 1) for index in range(4))
    path = tmp_path / "model.sima.json"
    document = {"modelTopology": {"model_config": {"config": {"layers": layers}}}}
    path.write_text(json.dumps(document), encoding="utf-8")

    assert audit_quantized_model(path, 1)["compute_scalar_type"] == "int8"

    layers[1]["config"]["_input_types"][0] = scalar(6)  # type: ignore[index]
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(RuntimeError, match="BatchMatmul layers are not fully INT8"):
        audit_quantized_model(path, 1)


def test_compile_rejects_ineligible_export_without_smoke_override(tmp_path: Path) -> None:
    import hashlib
    import json

    from compile import ARCHITECTURE, validate_export

    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    report = {
        "schema_version": 1,
        "architecture": ARCHITECTURE,
        "deployment": DeploymentContract().to_dict(),
        "manifest_sha256": hashlib.sha256(b"{}").hexdigest(),
        "dataset_sha256": "dataset-digest",
        "qualification_eligible": False,
    }
    (export_dir / "export-report.json").write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="ineligible diagnostic export"):
        validate_export(export_dir, manifest, "dataset-digest", allow_smoke=False)

    with pytest.raises(ValueError, match="dataset contents do not match"):
        validate_export(export_dir, manifest, "changed-dataset", allow_smoke=True)


def test_ultralytics_label_path_matches_training_loader_convention() -> None:
    image = Path("/dataset/images/sequence/frame.jpg")
    assert ultralytics_label_path(image) == Path("/dataset/labels/sequence/frame.txt")

    nested = Path("/datasets/images/project/images/sequence/frame.jpg")
    assert ultralytics_label_path(nested) == Path("/datasets/images/project/labels/sequence/frame.txt")


def test_ultralytics_revision_prefers_distribution_provenance(monkeypatch) -> None:
    monkeypatch.setattr(provenance, "_distribution_revision", lambda: provenance.ULTRALYTICS_REVISION)
    monkeypatch.setattr(provenance, "_git_revision", lambda _package_root: "unrelated-project-revision")
    monkeypatch.setattr(
        provenance,
        "ultralytics_package_sha256",
        lambda _package_root: provenance.ULTRALYTICS_PACKAGE_SHA256,
    )

    assert provenance.require_ultralytics_revision(RECIPE) == provenance.ULTRALYTICS_REVISION


def test_ultralytics_package_digest_detects_runtime_source_changes(tmp_path: Path) -> None:
    package = tmp_path / "ultralytics"
    package.mkdir()
    source = package / "model.py"
    source.write_text("canonical = True\n", encoding="utf-8")
    expected = provenance.ultralytics_package_sha256(package)

    (package / "README.md").write_text("documentation does not execute", encoding="utf-8")
    cache = package / "__pycache__"
    cache.mkdir()
    (cache / "model.pyc").write_bytes(b"runtime-cache")
    assert provenance.ultralytics_package_sha256(package) == expected

    source.write_text("canonical = None\n", encoding="utf-8")
    assert provenance.ultralytics_package_sha256(package) != expected


def test_ultralytics_revision_rejects_modified_installed_contents(monkeypatch) -> None:
    monkeypatch.setattr(provenance, "_distribution_revision", lambda: provenance.ULTRALYTICS_REVISION)
    monkeypatch.setattr(provenance, "ultralytics_package_sha256", lambda _package_root: "modified")

    with pytest.raises(RuntimeError, match="package content mismatch"):
        provenance.require_ultralytics_revision(RECIPE)


def test_ultralytics_import_uses_the_verified_private_snapshot(tmp_path: Path, monkeypatch) -> None:
    package = tmp_path / "ultralytics"
    package.mkdir()
    source = package / "__init__.py"
    source.write_text('VALUE = "verified"\n', encoding="utf-8")

    class Specification:
        submodule_search_locations = [str(package)]

    loaded_modules = {
        name: module for name, module in sys.modules.items() if name == "ultralytics" or name.startswith("ultralytics.")
    }
    for name in loaded_modules:
        sys.modules.pop(name, None)
    monkeypatch.setattr(provenance.importlib.util, "find_spec", lambda _name: Specification())
    monkeypatch.setattr(provenance, "_distribution_revision", lambda: provenance.ULTRALYTICS_REVISION)
    monkeypatch.setattr(
        provenance,
        "runtime_package_sha256",
        lambda _package_root: provenance.ULTRALYTICS_PACKAGE_SHA256,
    )
    runtime = provenance.activate_verified_ultralytics_runtime()
    try:
        source.write_text('VALUE = "replacement"\n', encoding="utf-8")
        module = sys.modules["ultralytics"]
        assert module.VALUE == "verified"
        assert Path(module.__file__).resolve().parent == runtime.package_root
        assert runtime.package_root != package
        assert runtime.package_root.joinpath("__init__.py").stat().st_mode & 0o777 == 0o400
        assert runtime.revision == provenance.ULTRALYTICS_REVISION
    finally:
        for name in tuple(sys.modules):
            if name == "ultralytics" or name.startswith("ultralytics."):
                sys.modules.pop(name, None)
        sys.modules.update(loaded_modules)
        sys.path = [entry for entry in sys.path if entry != str(runtime.package_root.parent)]
        runtime.temporary.cleanup()


def test_swml_qat_runtime_requires_the_pinned_package_contents(tmp_path: Path, monkeypatch) -> None:
    package = tmp_path / "sima_qat"
    package.mkdir()

    class Specification:
        submodule_search_locations = [str(package)]

    monkeypatch.setattr(provenance.importlib.util, "find_spec", lambda _name: Specification())
    monkeypatch.setattr(
        provenance,
        "runtime_package_sha256",
        lambda _package_root: provenance.SWML_QAT_PACKAGE_SHA256,
    )
    assert provenance.require_swml_qat_runtime() == provenance.SWML_QAT_PACKAGE_SHA256

    monkeypatch.setattr(provenance, "runtime_package_sha256", lambda _package_root: "modified")
    with pytest.raises(RuntimeError, match="swml-qat package content mismatch"):
        provenance.require_swml_qat_runtime()


def test_swml_qat_import_uses_the_verified_private_snapshot(tmp_path: Path, monkeypatch) -> None:
    package = tmp_path / "sima_qat"
    package.mkdir()
    source = package / "__init__.py"
    source.write_text('VALUE = "verified"\n', encoding="utf-8")

    class Specification:
        submodule_search_locations = [str(package)]

    monkeypatch.setattr(provenance.importlib.util, "find_spec", lambda _name: Specification())
    monkeypatch.setattr(
        provenance,
        "runtime_package_sha256",
        lambda _package_root: provenance.SWML_QAT_PACKAGE_SHA256,
    )
    runtime = provenance.activate_verified_swml_qat_runtime()
    try:
        source.write_text('VALUE = "replacement"\n', encoding="utf-8")
        module = sys.modules["sima_qat"]
        assert module.VALUE == "verified"
        assert Path(module.__file__).resolve().parent == runtime.package_root
        assert runtime.package_root != package
        assert runtime.package_root.joinpath("__init__.py").stat().st_mode & 0o777 == 0o400
    finally:
        sys.modules.pop("sima_qat", None)
        sys.path = [entry for entry in sys.path if entry != str(runtime.package_root.parent)]
        runtime.temporary.cleanup()
