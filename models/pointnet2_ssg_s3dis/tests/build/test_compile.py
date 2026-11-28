"""Checks of the CPU steps between the components, and of compile.py's split, gate and MPK audit."""

from __future__ import annotations

import io
import json
import math
import tarfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import onnx
import pytest
from model_registry.model import load_model as load_metadata
from onnx import TensorProto, helper

ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = ROOT / "models" / "pointnet2_ssg_s3dis"


@pytest.fixture
def pointcloud(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("pointcloud")


def test_farthest_point_sampling_starts_at_zero_and_spreads(pointcloud: ModuleType) -> None:
    pos = np.array([[0, 0, 0], [0.1, 0, 0], [1, 0, 0], [0.5, 0, 0], [0.9, 0, 0]], np.float32)
    assert pointcloud.farthest_point_sample(pos, 3).tolist() == [0, 2, 3]


def test_ball_query_keeps_the_lowest_indices_inside_the_radius(pointcloud: ModuleType) -> None:
    pos = np.zeros((40, 3), np.float32)
    pos[:, 0] = np.arange(40) * 0.01  # 1 cm apart along x
    pos[5, 0] = 5.0  # far away
    index = pointcloud.ball_query(pos, pos[[0, 39]], radius=0.105)
    # Within 10.5 cm of point 0: points 0-4 and 6-10; of point 39: points 29-39. Pads repeat the first.
    assert index[0].tolist() == [0, 1, 2, 3, 4, 6, 7, 8, 9, 10] + [0] * 22
    assert index[1].tolist() == list(range(29, 40)) + [29] * 21
    many = pointcloud.ball_query(pos, pos[[20]], radius=1.0)
    assert many[0].tolist() == [i for i in range(40) if i != 5][:32]


def test_three_nearest_weights_by_inverse_squared_distance(pointcloud: ModuleType) -> None:
    coarse = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [5, 5, 5]], np.float32)
    fine = np.array([[0.5, 0, 0]], np.float32)
    nearest, weight = pointcloud.three_nearest(coarse, fine)
    assert nearest.tolist() == [[0, 1, 2]]
    raw = 1 / (np.array([0.25, 0.25, 4.25]) + 1e-8)
    np.testing.assert_allclose(weight[0], raw / raw.sum(), rtol=1e-6)


def _reference_ball_query(pos, centres, radius, neighbors):
    """PointNet++'s query: the lowest indices within the radius, padded with the first."""
    index = []
    for centre in centres:
        inside = [i for i in range(len(pos)) if ((pos[i] - centre) ** 2).sum() <= radius * radius][:neighbors]
        index.append(inside + [inside[0]] * (neighbors - len(inside)))
    return np.array(index)


def test_ball_query_matches_the_full_query_across_chunks(pointcloud: ModuleType) -> None:
    rng = np.random.RandomState(1)
    pos = rng.uniform(0, 1, (600, 3)).astype(np.float32)
    centres = pos[rng.choice(600, 150, replace=False)]
    expected = _reference_ball_query(pos.astype(np.float64), centres.astype(np.float64), 0.2, pointcloud.NEIGHBORS)
    # Three x-ordered chunks of centres, each measuring only its x window.
    np.testing.assert_array_equal(pointcloud.ball_query(pos, centres, 0.2, chunk=64), expected)


def test_farthest_point_sampling_matches_the_direct_loop(pointcloud: ModuleType) -> None:
    pos = np.random.RandomState(2).uniform(0, 1, (500, 3)).astype(np.float32)
    picked, distance, current = [], np.full(500, np.inf, np.float32), 0
    for _ in range(64):
        picked.append(current)
        np.minimum(distance, ((pos - pos[current]) ** 2).sum(1), out=distance)
        current = int(distance.argmax())
    assert pointcloud.farthest_point_sample(pos, 64).tolist() == picked


def test_three_nearest_matches_a_full_sort(pointcloud: ModuleType) -> None:
    rng = np.random.RandomState(3)
    coarse, fine = rng.uniform(0, 1, (64, 3)).astype(np.float32), rng.uniform(0, 1, (300, 3)).astype(np.float32)
    distances = ((fine[:, None, :] - coarse[None]) ** 2).sum(-1)
    nearest, weight = pointcloud.three_nearest(coarse, fine)
    np.testing.assert_array_equal(nearest, np.argsort(distances, axis=1, kind="stable")[:, :3])
    np.testing.assert_allclose(weight.sum(1), 1.0, rtol=1e-6)


def test_channels_last_inputs_hold_the_nchw_values(pointcloud: ModuleType) -> None:
    block = pointcloud.synthetic_block()
    indices = pointcloud.plan(block[:, :3])
    nchw, hwc = pointcloud.group(block, 0, indices), pointcloud.group(block, 0, indices, channels_last=True)
    np.testing.assert_array_equal(hwc, nchw[0].transpose(1, 2, 0))
    coarse, skip = np.random.RandomState(4).rand(1024, 64).astype(np.float32), block[:, :5]
    nchw = pointcloud.interpolate(coarse, skip, 0, indices)
    np.testing.assert_array_equal(pointcloud.interpolate(coarse, skip, 0, indices, channels_last=True),
                                  nchw[0].transpose(1, 2, 0))


def test_group_puts_features_then_relative_xyz(pointcloud: ModuleType) -> None:
    pos = np.array([[0, 0, 0], [0.05, 0, 0], [3, 3, 3]], np.float32)
    features = np.arange(3 * 9, dtype=np.float32).reshape(3, 9)
    indices = {"levels": [pos], "centres": [np.array([0])], "groups": [np.array([[0, 1] + [0] * 30])]}
    grouped = pointcloud.group(features, 0, indices)
    assert grouped.shape == (1, 12, 1, 32)
    assert grouped[0, :9, 0, 1].tolist() == features[1].tolist()
    np.testing.assert_allclose(grouped[0, 9:, 0, 1], [0.05, 0, 0])


def test_interpolate_mixes_neighbours_then_appends_the_skip(pointcloud: ModuleType) -> None:
    coarse = np.array([[1.0, 0.0], [0.0, 1.0]], np.float32)
    skip = np.array([[7.0]], np.float32)
    indices = {"interpolation": [(np.array([[0, 1, 1]]), np.array([[0.5, 0.25, 0.25]], np.float32))]}
    stacked = pointcloud.interpolate(coarse, skip, 0, indices)
    assert stacked.shape == (1, 3, 1, 1)
    np.testing.assert_allclose(stacked[0, :, 0, 0], [0.5, 0.5, 7.0])


@pytest.mark.parametrize("layout", ["nchw", "channels_last"])
def test_segment_runs_every_component_in_order(layout: str, pointcloud: ModuleType) -> None:
    calls = []

    def run(component, value):
        calls.append((component, value.shape))
        shape = pointcloud.OUTPUT_SHAPES[component]
        output = np.zeros(shape, np.float32)
        if component == "fp1":
            output[0, 2] = 1.0  # every point is a wall
        return output.transpose(0, 2, 3, 1) if layout == "channels_last" else output

    channels_last = layout == "channels_last"
    logits = pointcloud.segment(pointcloud.synthetic_block(), run, channels_last=channels_last)
    assert [name for name, _ in calls] == list(pointcloud.COMPONENTS)
    # NCHW, or the [H, W, C] tensor each MPK takes.
    shapes = [pointcloud.INPUT_SHAPES[name] for name in pointcloud.COMPONENTS]
    assert [shape for _, shape in calls] == [(h, w, c) if channels_last else (n, c, h, w) for n, c, h, w in shapes]
    assert logits.shape == (pointcloud.POINTS, pointcloud.CLASSES)
    assert (logits.argmax(1) == 2).all()


def test_synthetic_block_is_seeded(pointcloud: ModuleType) -> None:
    block = pointcloud.synthetic_block()
    assert block.shape == (pointcloud.POINTS, pointcloud.FEATURES) and block.dtype == np.float32
    np.testing.assert_array_equal(block, pointcloud.synthetic_block())


def _mpk_manifest(compile_module: ModuleType, component: str) -> dict:
    return {
        "input_nodes": [{"name": compile_module.input_name(component),
                         "size": math.prod(compile_module.INPUT_SHAPES[component]) * 4}],
        "plugins": [
            {"sequence": 0, "processor": "EV74", "config_params": {"kernel": "cast_transform"}},
            {"sequence": 1, "processor": "MLA", "config_params": {}},
            {
                "sequence": 2,
                "processor": "EV74",
                "config_params": {"kernel": "cast_transform"},
                "output_nodes": [{"size": math.prod(compile_module.OUTPUT_SHAPES[component]) * 4}],
            },
        ],
    }


def _write_mpk(path: Path, manifest: dict) -> None:
    with tarfile.open(path, "w:gz") as stream:
        for name, content in (("model.elf", b"elf"), ("model_mpk.json", json.dumps(manifest).encode())):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            stream.addfile(info, io.BytesIO(content))


@pytest.mark.parametrize("case", ["valid", "inputs", "outputs", "kernel", "two_mla", "host"])
def test_mpk_audit_requires_one_mla_segment_and_the_declared_interface(
    tmp_path: Path, case: str, load_script: Callable[[str], ModuleType]
) -> None:
    compile_module = load_script("compile")
    manifest = _mpk_manifest(compile_module, "sa2")
    plugins = manifest["plugins"]
    if case == "inputs":
        manifest["input_nodes"][0]["size"] -= 4
    elif case == "outputs":
        plugins[-1]["output_nodes"][0]["size"] = 4
    elif case == "kernel":
        plugins[-1]["config_params"]["kernel"] = "boxdecode"
    elif case == "two_mla":
        plugins.insert(2, {"sequence": 1, "processor": "MLA", "config_params": {}})
    elif case == "host":
        plugins.insert(2, {"sequence": 1, "processor": "A65", "config_params": {}})
    archive = tmp_path / "sa2_mpk.tar.gz"
    _write_mpk(archive, manifest)
    if case == "valid":
        compile_module.audit_mpk(archive, "sa2")
    else:
        with pytest.raises(RuntimeError):
            compile_module.audit_mpk(archive, "sa2")


def _stages_onnx(compile_module: ModuleType, path: Path, wrong: bool = False) -> None:
    """The side-by-side interface, each output a ReduceMax/Slice-free copy of a constant."""
    inputs, outputs, nodes = [], [], []
    for name in compile_module.COMPONENTS:
        out_shape = list(compile_module.OUTPUT_SHAPES[name])
        if wrong and name == "sa1":
            out_shape[1] = 32
        inputs.append(helper.make_tensor_value_info(compile_module.input_name(name), TensorProto.FLOAT,
                                                    list(compile_module.INPUT_SHAPES[name])))
        outputs.append(helper.make_tensor_value_info(compile_module.output_name(name), TensorProto.FLOAT, out_shape))
        nodes.append(helper.make_node(
            "Constant", [], [compile_module.output_name(name)],
            value=helper.make_tensor("v", TensorProto.FLOAT, out_shape, np.zeros(out_shape, np.float32).ravel()),
        ))
    graph = helper.make_graph(nodes, "stages", inputs, outputs)
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)


def test_compile_requires_the_converted_interface(tmp_path: Path, load_script: Callable[[str], ModuleType]) -> None:
    compile_module = load_script("compile")
    _stages_onnx(compile_module, tmp_path / "good.onnx")
    compile_module.validate_source(tmp_path / "good.onnx")
    _stages_onnx(compile_module, tmp_path / "bad.onnx", wrong=True)
    with pytest.raises(ValueError, match="Unsupported PointNet"):
        compile_module.validate_source(tmp_path / "bad.onnx")


def test_split_extracts_one_graph_per_component(tmp_path: Path, load_script: Callable[[str], ModuleType]) -> None:
    compile_module = load_script("compile")
    _stages_onnx(compile_module, tmp_path / "model.onnx")
    graphs = compile_module.split(tmp_path / "model.onnx", tmp_path / "graphs")
    assert list(graphs) == list(compile_module.COMPONENTS)
    for name, path in graphs.items():
        model = onnx.load(path)
        assert [v.name for v in model.graph.output] == [compile_module.output_name(name)]


def test_compile_reads_the_declared_build_parameters(load_script: Callable[[str], ModuleType]) -> None:
    compile_module = load_script("compile")
    parameters = compile_module._component_parameters("w16a16")
    assert list(parameters) == list(compile_module.COMPONENTS)
    assert {item["precision"] for item in parameters.values()} == {"w16a16"}
    with pytest.raises(ValueError, match="no build profile"):
        compile_module._component_parameters("w8a8")


def test_model_declares_the_components_in_run_order(pointcloud: ModuleType) -> None:
    model = load_metadata(MODEL_DIR)
    assert [component["id"] for component in model["components"]] == list(pointcloud.COMPONENTS)
    assert [step["component"] for step in model["pipeline"] if "component" in step] == list(pointcloud.COMPONENTS)
    assert (model["license"]["spdx"], model["license"]["redistribution"]) == ("MIT", "allowed")


def test_compiled_build_holds_every_component(compiled: Path) -> None:
    outputs = json.loads((compiled / "compile-outputs.json").read_text(encoding="utf-8"))
    assert [item["component_id"] for item in outputs["outputs"]] == [
        "sa1", "sa2", "sa3", "sa4", "fp4", "fp3", "fp2", "fp1"
    ]
    for item in outputs["outputs"]:
        assert (compiled / item["mpk_path"]).is_file()
