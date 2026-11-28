"""Checks of convert.py, and of the converted ONNX when the conversion stage supplies it."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper


@pytest.fixture
def converter(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("convert")


def _config(converter: ModuleType, **changes) -> dict:
    hparams = {**converter.HPARAMS, **changes}
    return {"task": "semantic-segmentation", "hparams": hparams}


@pytest.mark.parametrize("case", ["valid", "classes", "radii", "pos_first", "task"])
def test_config_must_be_the_implemented_model(tmp_path: Path, case: str, converter: ModuleType) -> None:
    config = _config(converter)
    if case == "classes":
        config = _config(converter, num_classes=20)
    elif case == "radii":
        config = _config(converter, radii=[0.1, 0.2, 0.4, 0.4])
    elif case == "pos_first":
        config = _config(converter, normalize_pos=True)
    elif case == "task":
        config["task"] = "classification"
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if case == "valid":
        converter.validate_config(tmp_path)
    else:
        with pytest.raises(ValueError):
            converter.validate_config(tmp_path)


def _weights(converter: ModuleType, extra: str | None = None) -> dict[str, np.ndarray]:
    """Random tensors under every checkpoint name the export reads, plus an optional unused one."""
    rng = np.random.RandomState(0)
    weights = {}

    def mlp(prefix, channels, plain_last=False):
        for i, (cin, cout) in enumerate(zip(channels, channels[1:])):
            weights[f"{prefix}.lins.{i}.weight"] = rng.normal(0, 0.1, (cout, cin)).astype(np.float32)
            weights[f"{prefix}.lins.{i}.bias"] = np.zeros(cout, np.float32)
            if plain_last and i == len(channels) - 2:
                break
            weights[f"{prefix}.norms.{i}.module.weight"] = np.ones(cout, np.float32)
            weights[f"{prefix}.norms.{i}.module.bias"] = np.zeros(cout, np.float32)
            weights[f"{prefix}.norms.{i}.module.running_mean"] = np.zeros(cout, np.float32)
            weights[f"{prefix}.norms.{i}.module.running_var"] = np.ones(cout, np.float32)
            weights[f"{prefix}.norms.{i}.module.num_batches_tracked"] = np.zeros((), np.int64)

    for level, (cin, channels) in enumerate(converter.SA_CHANNELS):
        mlp(f"encoder.sa_blocks.{level}.convs.0.local_nn", [cin + 3, *channels])
    for block, (cin, channels) in enumerate(converter.FP_CHANNELS):
        mlp(f"decoder.fp_blocks.{block}.mlp", [cin, *channels])
    mlp("head", [128, 128, converter.CLASSES], plain_last=True)
    if extra:
        weights[extra] = np.zeros(1, np.float32)
    return weights


def test_stages_consume_every_checkpoint_tensor(converter: ModuleType) -> None:
    pytest.importorskip("torch")
    stages = converter.build_stages(_weights(converter))
    assert list(stages) == list(converter.COMPONENTS)
    with pytest.raises(ValueError, match="does not use"):
        converter.build_stages(_weights(converter, extra="decoder.fp_blocks.9.mlp.lins.0.weight"))


def test_stages_have_the_component_interface(converter: ModuleType) -> None:
    torch = pytest.importorskip("torch")
    stages = converter.build_stages(_weights(converter))
    for name, stage in stages.items():
        with torch.inference_mode():
            output = stage(torch.zeros(converter.INPUT_SHAPES[name]))
        assert tuple(output.shape) == converter.OUTPUT_SHAPES[name], name


def _side_by_side(converter: ModuleType, case: str) -> onnx.ModelProto:
    """Identity-like graph with the declared inputs and outputs, broken as `case` asks."""
    inputs, outputs, nodes = [], [], []
    for name in converter.COMPONENTS:
        in_shape, out_shape = list(converter.INPUT_SHAPES[name]), list(converter.OUTPUT_SHAPES[name])
        if case == "shape" and name == "fp1":
            out_shape[1] = 20
        inputs.append(helper.make_tensor_value_info(converter.input_name(name), TensorProto.FLOAT, in_shape))
        outputs.append(helper.make_tensor_value_info(converter.output_name(name), TensorProto.FLOAT, out_shape))
        value = np.zeros(out_shape, np.float32)
        nodes.append(helper.make_node("Constant", [], [converter.output_name(name)],
                                      value=helper.make_tensor("v", TensorProto.FLOAT, out_shape, value.ravel())))
    if case == "order":
        outputs[0], outputs[1] = outputs[1], outputs[0]
    graph = helper.make_graph(nodes, "stages", inputs, outputs)
    opset = 13 if case == "opset" else converter.OPSET
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])


@pytest.mark.parametrize("case", ["valid", "opset", "order", "shape"])
def test_export_must_hold_every_component_in_order(tmp_path: Path, case: str, converter: ModuleType) -> None:
    path = tmp_path / "model.onnx"
    onnx.save(_side_by_side(converter, case), path)
    if case == "valid":
        converter.validate_export(path)
    else:
        with pytest.raises(ValueError):
            converter.validate_export(path)


@pytest.mark.parametrize("case", ["valid", "drift", "nan", "shape"])
def test_logits_must_agree(case: str, converter: ModuleType) -> None:
    rng = np.random.RandomState(0)
    expected = rng.normal(0, 3, (converter.POINTS, converter.CLASSES)).astype(np.float32)
    actual = expected + 1e-5
    if case == "drift":
        actual = expected + rng.normal(0, 0.5, expected.shape).astype(np.float32)
    elif case == "nan":
        actual[0, 0] = np.nan
    elif case == "shape":
        actual = actual[:-1]
    if case == "valid":
        assert converter.compare(expected, actual) == 1.0
    else:
        with pytest.raises((ValueError, AssertionError)):
            converter.compare(expected, actual)


def test_source_must_be_the_pinned_file_set(converter: ModuleType) -> None:
    source = {"type": "file_set", "entrypoint": "model.safetensors",
              "files": [{"path": "model.safetensors"}, {"path": "config.json"}]}
    converter.validate_source(source)
    with pytest.raises(ValueError):
        converter.validate_source({**source, "entrypoint": "config.json"})


def test_converted_model_has_the_declared_interface(converted: Path, converter: ModuleType) -> None:
    converter.validate_export(converted)
