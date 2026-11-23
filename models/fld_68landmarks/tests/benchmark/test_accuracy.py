from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from model_registry.model_tests import measured_shortfalls


def test_mpk_argument_names_exactly_the_model_component(load_script: Callable[[str], ModuleType]) -> None:
    script = load_script("accuracy")

    assert script.parse_mpks(["model=/data/model_mpk.tar.gz"]) == Path("/data/model_mpk.tar.gz")
    with pytest.raises(SystemExit, match="exactly model=<path>"):
        script.parse_mpks(["backbone=/data/model_mpk.tar.gz"])


def test_accuracy_meets_its_expected_values(accuracy: dict[str, Any], model_dir: Path, build_profile: str) -> None:
    assert measured_shortfalls(model_dir, "accuracy", build_profile, accuracy) == []


@pytest.mark.parametrize("values", [["model="], ["model=/a", "model=/b"], ["missing-equals"]])
def test_mpk_argument_rejects_empty_duplicate_and_malformed_paths(load_script, values):
    with pytest.raises(SystemExit, match="exactly model"):
        load_script("accuracy").parse_mpks(values)


@pytest.mark.parametrize("value", ["other=/data", "afw-faces-default=", "afw-faces-default", "afw-faces-default= "])
def test_accuracy_requires_its_pinned_dataset_id(load_script, value):
    with pytest.raises(ValueError, match="--dataset"):
        load_script("landmarks").dataset_path(value)


def test_fp32_baseline_reuses_the_same_evaluator_without_neat(load_script, monkeypatch, tmp_path):
    import json
    import sys
    from types import SimpleNamespace

    import numpy as np

    script = load_script("fp32_baseline_accuracy")
    observed = []

    class Session:
        def __init__(self, path, providers):
            assert path == "source.onnx" and providers == ["CPUExecutionProvider"]

        def get_inputs(self):
            return [SimpleNamespace(name="image", shape=[1, 3, 256, 256])]

        def get_outputs(self):
            return [SimpleNamespace(name="heatmaps", shape=[1, 68, 64, 64])]

        def run(self, names, feeds):
            assert names == ["heatmaps"] and feeds["image"].shape == (1, 3, 256, 256)
            assert feeds["image"].flags.c_contiguous
            observed.append(feeds["image"])
            return [np.zeros((1, 68, 64, 64), np.float32)]

    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(InferenceSession=Session))
    monkeypatch.setattr(script, "test_faces", lambda path: ["real face"])

    def evaluate(faces, infer):
        assert faces == ["real face"]
        assert infer(np.zeros((256, 256, 3), np.float32)).shape == (68, 64, 64)
        return {"nme": 0.03}, {"median_nme": 0.03}

    monkeypatch.setattr(script, "evaluate", evaluate)
    destination = tmp_path / "baseline.json"
    script.main(["--onnx", "source.onnx", "--dataset", "afw-faces-default=/data", "--output", str(destination)])
    report = json.loads(destination.read_text())
    assert len(observed) == 1 and report["metrics"] == {"nme": 0.03} and report["samples"] == 1


@pytest.mark.parametrize("failure", [None, "inference", "decode"])
def test_accuracy_closes_its_runner_on_success_and_failure(load_script, monkeypatch, tmp_path, failure):
    from types import SimpleNamespace

    import numpy as np

    script = load_script("accuracy")
    closed = []

    def run(inputs, timeout_ms):
        if failure == "inference":
            raise RuntimeError("inference fault")
        return ["heatmap"]

    runner = SimpleNamespace(run=run, close=lambda: closed.append(True))
    monkeypatch.setattr(script, "test_faces", lambda path: ["face"])
    monkeypatch.setattr(script.pyneat, "Model", lambda *args: SimpleNamespace(build=lambda inputs: runner))
    monkeypatch.setattr(script, "heatmaps_of", lambda tensor: np.zeros((68, 64, 64), np.float32))

    def evaluate(faces, infer):
        infer(np.zeros((256, 256, 3), np.float32))
        if failure == "decode":
            raise ValueError("decode fault")
        return {"nme": 0.03}, {"median_nme": 0.03}

    monkeypatch.setattr(script, "evaluate", evaluate)
    monkeypatch.setattr(
        script.model_report,
        "from_accuracy",
        lambda document: SimpleNamespace(
            save=lambda output: None,
            document={"datasets": document},
        ),
    )
    args = ["--mpk", "model=/model", "--dataset", "afw-faces-default=/data", "--output", str(tmp_path)]
    if failure:
        with pytest.raises((ValueError, RuntimeError), match=f"{failure} fault"):
            script.main(args)
    else:
        assert script.main(args) == 0
    assert closed == [True]


@pytest.mark.parametrize("shape", [(64, 64, 68), (1, 64, 64, 68)])
def test_heatmap_adapter_accepts_only_one_logical_face(load_script, shape):
    from types import SimpleNamespace

    import numpy as np

    script = load_script("accuracy")
    source = np.ones(shape, np.float32)
    tensor = SimpleNamespace(to_numpy=lambda *, copy: source.copy() if copy else source)
    heatmaps = script.heatmaps_of(tensor)
    assert heatmaps.shape == (68, 64, 64)
    assert not np.shares_memory(source, heatmaps)


def test_heatmap_adapter_rejects_multiple_faces_instead_of_silently_taking_first(load_script):
    from types import SimpleNamespace

    import numpy as np

    tensor = SimpleNamespace(to_numpy=lambda **kwargs: np.zeros((2, 64, 64, 68), np.float32))
    with pytest.raises(ValueError, match="one face"):
        load_script("accuracy").heatmaps_of(tensor)


@pytest.mark.parametrize("copy", [False, True])
@pytest.mark.parametrize("shape", [(64, 64, 68), (1, 64, 64, 68)])
def test_heatmap_adapter_explicit_borrow_is_read_only_and_retains_array(load_script, copy, shape):
    import gc
    import weakref
    from types import SimpleNamespace

    import numpy as np

    source = np.ones(shape, np.float32)
    source.flags.writeable = False
    owner = weakref.ref(source)
    tensor = SimpleNamespace(to_numpy=lambda *, copy, owner=source: owner.copy() if copy else owner)
    heatmaps = load_script("accuracy").heatmaps_of(tensor, copy=copy)
    assert np.shares_memory(source, heatmaps) == (not copy)
    del source, tensor
    gc.collect()
    assert (owner() is not None) == (not copy)
    assert heatmaps.shape == (68, 64, 64) and (heatmaps == 1).all()
    if not copy:
        assert not heatmaps.flags.writeable
    del heatmaps
    gc.collect()
    assert owner() is None


@pytest.mark.parametrize("copy", [False, True])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_heatmap_adapter_rejects_nonfinite_in_both_ownership_modes(load_script, copy, value):
    from types import SimpleNamespace

    import numpy as np

    source = np.full((64, 64, 68), value, np.float32)
    tensor = SimpleNamespace(to_numpy=lambda **kwargs: source)
    with pytest.raises(ValueError, match="unexpected heatmaps"):
        load_script("accuracy").heatmaps_of(tensor, copy=copy)
