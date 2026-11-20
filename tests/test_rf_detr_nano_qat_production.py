"""Focused regressions for the staged local QAT handoff, not target execution."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
SCRIPTS = ROOT / "training/rf_detr_nano/production"


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    loaded = {}
    for name in ("numerics", "contract", "prepare_contract", "acceptance", "runtime", "compile"):
        spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        loaded[name] = module
    return loaded


@pytest.mark.parametrize("minimum", [-128, -127, 0, 127])
def test_topk_matches_independent_repeated_max_including_exhaustion(modules, minimum):
    scores = np.full(576, minimum, dtype=np.int8)
    scores[10:16] = 127
    original = scores.copy()
    expected = []
    for _ in range(300):
        index = int(np.argmax(scores))
        expected.append(index)
        scores[index] = -128
    boxes = np.arange(576 * 4).reshape(1, 576, 4).astype(np.int8)
    selected, actual = modules["numerics"].top_proposals(original.reshape(1, 1, 1, 576), boxes)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(selected, boxes[:, expected, :])


def test_all_minimum_scores_repeat_zero_not_unique_argsort(modules):
    scores = np.full(576, -128, dtype=np.int8)
    boxes = np.zeros((1, 576, 4), dtype=np.int8)
    _, indices = modules["numerics"].top_proposals(scores, boxes)
    assert not indices.any()


def test_source_qdq_uses_divide_then_rne_then_zero_and_saturates(modules):
    grid = modules["numerics"].Int8Grid(1016526588, -14)
    values = np.array([-1e30, -0.5, -0.0, 0.5, 1e30], dtype=np.float32)
    expected = np.clip(np.rint(values / grid.step) + np.float32(-14), -128, 127).astype(np.int8)
    np.testing.assert_array_equal(grid.quantize(values), expected)
    with pytest.raises(ValueError, match="INT8 codes"):
        grid.dequantize(values)
    with pytest.raises(ValueError):
        modules["numerics"].Int8Grid(0, -14)


def test_cpu_box_formula_keeps_deltas_separate_from_final_absolute_grid(modules):
    deltas = np.zeros((1, 300, 4), dtype=np.float32)
    references = np.full_like(deltas, 0.25)
    np.testing.assert_array_equal(modules["numerics"].decode_boxes(deltas, references), references)
    deltas[..., :2] = 1
    expected = references.copy()
    expected[..., :2] += references[..., 2:]
    np.testing.assert_array_equal(modules["numerics"].decode_boxes(deltas, references), expected)


def test_class_filter_runs_after_pair_topk_and_preserves_raw_coco_ids(modules):
    numerics = modules["numerics"]
    grid = numerics.Int8Grid(1065353216, 0)
    logits = np.full((1, 300, 91), -128, dtype=np.int8)
    logits[:, :, 0] = 2  # All first 300 pairs are the unused COCO slot zero.
    logits[:, :, 1] = 1
    zeros = np.zeros((1, 300, 4), dtype=np.int8)
    result = numerics.detections([zeros, logits, zeros], (grid, grid, grid), width=640, height=480, image_id=1)
    assert result.shape == (0, 7)


def contract_document(modules):
    c = modules["contract"]
    return {
        "schema": "rfdetr.nano_qat_handoff/v1",
        "checkpoint_sha256": "1" * 64,
        "reference_sha256": "2" * 64,
        "input_grid": {"step_bits": 1016526588, "zero_point": -14},
        "output_grids": [
            {"name": name, "step_bits": 1065353216, "zero_point": 0}
            for name in ("box_deltas", "labels", "box_references")
        ],
        "components": {
            component: {
                "sha256": "3" * 64,
                **{
                    role: [
                        {"name": str(index), "shape": list(shape), "dtype": "int8"}
                        for index, shape in enumerate(shapes)
                    ]
                    for role, shapes in c.SHAPES[component].items()
                },
            }
            for component in c.COMPONENTS
        },
        "native_seals": {
            component: {
                field: "5" * 64 for field in ("native_graph_sha256", "source_contract_sha256", "numerical_plan_sha256")
            }
            for component in c.COMPONENTS
        },
        "topk": {"k": 300, "ties": "lower_original_index_first", "exhaustion": "MaskMinRepeatZero"},
        "preprocessing": {
            "stages": ["Rescale", "Resize", "Normalize", "Quantize"],
            "resize": "BILINEAR_ANTIALIASED",
            "rescale_float32_bits": 998277249,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    }


@pytest.mark.parametrize("mutation", ["step", "dtype", "topk", "stage_order", "hash"])
def test_contract_rejects_silent_substitutions(modules, tmp_path, mutation):
    doc = contract_document(modules)
    if mutation == "step":
        doc["input_grid"]["step_bits"] = 0
    elif mutation == "dtype":
        doc["components"]["decoder"]["inputs"][0]["dtype"] = "float32"
    elif mutation == "topk":
        doc["topk"]["exhaustion"] = "Unique"
    elif mutation == "stage_order":
        doc["preprocessing"]["stages"][:2] = ["Resize", "Rescale"]
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(doc))
    checksum = "4" * 64 if mutation == "hash" else modules["contract"].sha256_file(path)
    with pytest.raises(ValueError):
        modules["contract"].load_contract(path, checksum)


def test_runtime_refuses_legacy_preprocessing_and_float_handoff(modules):
    from types import SimpleNamespace

    pyneat = SimpleNamespace(ModelOptions=lambda: SimpleNamespace(preprocess=SimpleNamespace()))
    with pytest.raises(RuntimeError, match="ordered PreprocessProgram"):
        modules["runtime"].image_options(pyneat, contract_document(modules), 1280, 720)

    class Tensor:
        def __init__(self, shape):
            self.shape, self.dtype = shape, "float32"

        def to_numpy(self):
            return np.zeros(self.shape, dtype=np.float32)

    tensors = [Tensor(shape) for shape in modules["contract"].SHAPES["backbone"]["outputs"]]
    with pytest.raises(RuntimeError, match="raw INT8 handoff"):
        modules["runtime"].proposal_inputs(pyneat, tensors)


def test_absolute_gate_rejects_our_actual_full5000_results(modules):
    report = {
        "datasets": {
            "coco-2017-val": {
                "evaluated_samples": 5000,
                "available_samples": 5000,
                "metrics": {"coco_map": 0.4665476568626716},
                "details": {"official_full5000": True, "checkpoint_sha256": "1" * 64, "reference_sha256": "2" * 64},
            }
        }
    }
    failures = modules["acceptance"].accuracy_shortfalls(
        report, source_map=0.4698192104281438, expected_checkpoint_sha256="1" * 64, expected_reference_sha256="2" * 64
    )
    assert len(failures) == 3  # Original FP32 gap, source gap, and source/compiled gap.
    report["datasets"]["coco-2017-val"]["evaluated_samples"] = 100
    assert "accuracy requires all 5,000 official validation images" in modules["acceptance"].accuracy_shortfalls(
        report, source_map=0.479, expected_checkpoint_sha256="1" * 64, expected_reference_sha256="2" * 64
    )


def test_duplicate_or_missing_components_fail(modules):
    for values in (["backbone=a"], ["backbone=a", "backbone=b", "decoder=c"], ["wrong=a", "decoder=b"]):
        with pytest.raises(ValueError):
            modules["contract"].parse_components(values)


def test_stock_model_scripts_are_not_imported_or_modified():
    for path in SCRIPTS.glob("*.py"):
        assert "models/rf_detr_nano/scripts" not in path.read_text()
    assert not (ROOT / "models/rf_detr_nano_qat").exists()
