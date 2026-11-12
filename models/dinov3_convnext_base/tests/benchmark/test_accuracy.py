from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from model_registry.model_tests import measured_shortfalls


def test_mpk_argument_names_exactly_the_model_component(load_script: Callable[[str], ModuleType]) -> None:
    script = load_script("accuracy")

    assert script.parse_mpks(["model=/data/model_mpk.tar.gz"]) == Path("/data/model_mpk.tar.gz")
    with pytest.raises(SystemExit, match="exactly model=<path>"):
        script.parse_mpks(["backbone=/data/model_mpk.tar.gz"])


def test_prefetched_images_keep_their_order(load_script: Callable[[str], ModuleType]) -> None:
    script = load_script("accuracy")
    items = list(range(script.PREFETCH * 3 + 1))

    assert list(script.prefetched(lambda item: item * 2, items)) == [item * 2 for item in items]


def test_knn_scores_each_image_by_its_nearest_other_images(
    load_script: Callable[[str], ModuleType], monkeypatch: pytest.MonkeyPatch
) -> None:
    script = load_script("accuracy")
    monkeypatch.setattr(script, "NEIGHBOURS", 2)
    # Blocks and tiles smaller than the set, so the scoring crosses their edges.
    monkeypatch.setattr(script, "BLOCK", 3)
    monkeypatch.setattr(script, "TILE", 2)
    # Three images near each of two directions, and one image of class 1 whose nearest images are of class 0.
    embeddings = np.asarray(
        [[1, 0], [1, 0.02], [1, -0.02], [0, 1], [0.02, 1], [-0.02, 1], [1, 0.3]], dtype=np.float32
    ) * np.asarray([[1], [5], [0.2], [1], [3], [1], [2]], dtype=np.float32)
    classes = np.asarray([0, 0, 0, 1, 1, 1, 1])

    # Vector lengths do not matter, and the misplaced image is the only wrong one.
    assert script.knn_top1(embeddings, classes) == pytest.approx(6 / 7)


def test_knn_never_counts_an_image_as_its_own_neighbour(
    load_script: Callable[[str], ModuleType], monkeypatch: pytest.MonkeyPatch
) -> None:
    script = load_script("accuracy")
    monkeypatch.setattr(script, "NEIGHBOURS", 1)
    # Each image's only other neighbours belong to the other class.
    embeddings = np.asarray([[1, 0], [0.9, 0.1], [0, 1]], dtype=np.float32)

    assert script.knn_top1(embeddings, np.asarray([0, 1, 0])) == 0.0
    with pytest.raises(ValueError, match="more than 1 images"):
        script.knn_top1(embeddings[:1], np.asarray([0]))


def test_evaluation_embeds_the_evaluated_classes_and_scores_them(
    tmp_path: Path, load_script: Callable[[str], ModuleType], monkeypatch: pytest.MonkeyPatch
) -> None:
    script = load_script("accuracy")
    monkeypatch.setattr(script, "CLASSES", 4)
    monkeypatch.setattr(script, "EVALUATED_CLASSES", 2)
    monkeypatch.setattr(script, "NEIGHBOURS", 1)
    # Three images of each of four classes; an image's embedding points along its class.
    images = [{"id": index, "file_name": f"{index}.JPEG"} for index in range(12)]
    labels = {
        "categories": [{"id": index} for index in range(4)],
        "images": images,
        "annotations": [{"image_id": image["id"], "category_id": image["id"] % 4} for image in images],
    }
    (tmp_path / "annotations").mkdir()
    (tmp_path / "annotations" / "labels_val2012.json").write_text(json.dumps(labels), encoding="utf-8")
    (tmp_path / "val2012").mkdir()
    for image in images:
        (tmp_path / "val2012" / image["file_name"]).write_bytes(bytes([image["id"]]))
    monkeypatch.setattr(script, "preprocess_image", lambda content: np.eye(4, dtype=np.float32)[content[0] % 4])

    result = script.evaluate(tmp_path, lambda image: image + 0.01)

    assert (result["evaluated_samples"], result["available_samples"], result["split"]) == (6, 12, "val2012")
    assert result["metrics"] == {"knn_top1_accuracy": 1.0}
    assert (result["details"]["evaluated_classes"], result["details"]["neighbours"]) == (2, 1)


def test_fp32_baseline_feeds_the_onnx_the_nchw_image_and_reads_the_class_embedding(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    pytest.importorskip("onnxruntime")
    onnx = pytest.importorskip("onnx")
    baseline = load_script("fp32_baseline_accuracy")
    helper = onnx.helper
    # The class embedding echoes the NCHW input, so it shows the layout the model received.
    nodes = [
        helper.make_node("Identity", ["pixel_values"], ["class_embedding"]),
        helper.make_node("Transpose", ["pixel_values"], ["patch_embeddings"], perm=[0, 2, 3, 1]),
    ]
    graph = helper.make_graph(
        nodes,
        "layout",
        [helper.make_tensor_value_info("pixel_values", onnx.TensorProto.FLOAT, [1, 3, 2, 4])],
        [
            helper.make_tensor_value_info("class_embedding", onnx.TensorProto.FLOAT, [1, 3, 2, 4]),
            helper.make_tensor_value_info("patch_embeddings", onnx.TensorProto.FLOAT, [1, 2, 4, 3]),
        ],
    )
    model = tmp_path / "reference.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10), model)
    image = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)

    embedding = baseline.onnx_embedding(model)(image)

    np.testing.assert_array_equal(embedding, image.transpose(2, 0, 1).reshape(-1))


def test_fp32_baseline_scores_with_the_accuracy_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    baseline = load_script("fp32_baseline_accuracy")
    calls = []

    def evaluate(dataset: Path, embed: Callable, limit: int | None) -> dict[str, Any]:
        calls.append((dataset, embed, limit))
        return {"metrics": {"knn_top1_accuracy": 0.912345}, "evaluated_samples": 5000}

    monkeypatch.setattr(baseline.accuracy, "evaluate", evaluate)
    monkeypatch.setattr(baseline, "onnx_embedding", lambda onnx: f"session for {onnx.name}")
    output = tmp_path / "fp32-baseline.json"

    baseline.main(["--onnx", "reference.onnx", "--dataset", "imagenet-2012-val=/data/in", "--output", str(output)])

    assert calls == [(Path("/data/in"), "session for reference.onnx", None)]
    assert json.loads(output.read_text()) == {
        "dataset": "imagenet-2012-val",
        "images": 5000,
        "metrics": {"knn_top1_accuracy": 0.9123},
    }


def test_accuracy_meets_its_expected_values(accuracy: dict[str, Any], model_dir: Path, build_profile: str) -> None:
    assert measured_shortfalls(model_dir, "accuracy", build_profile, accuracy) == []
