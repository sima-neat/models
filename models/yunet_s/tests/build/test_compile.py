from __future__ import annotations

import io
import json
import random
import sys
import tarfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import onnx
import pytest
from onnx import helper, numpy_helper


def _yunet_like(path: Path, heads: dict[str, int], strides: tuple[int, ...], broken: str | None = None) -> Path:
    """A small graph with YuNet's input and head tails: Conv, Transpose(0, 2, 3, 1), Reshape, then Sigmoid on scores."""
    rng = np.random.RandomState(0)
    nodes, initializers, outputs = [], [], []
    for head, channels in heads.items():
        for stride in strides:
            name, side = f"{head}_{stride}", 640 // stride
            initializers.append(
                numpy_helper.from_array(
                    rng.normal(0, 0.01, (channels, 3, stride, stride)).astype(np.float32), f"{name}_w"
                )
            )
            initializers.append(numpy_helper.from_array(np.array([1, -1, channels], np.int64), f"{name}_shape"))
            nodes.append(helper.make_node("Conv", ["input", f"{name}_w"], [f"{name}_conv"], strides=[stride, stride],
                                          kernel_shape=[stride, stride]))
            perm = [0, 3, 2, 1] if name == broken else [0, 2, 3, 1]
            nodes.append(helper.make_node("Transpose", [f"{name}_conv"], [f"{name}_t"], perm=perm))
            scored = head in ("cls", "obj")
            nodes.append(helper.make_node("Reshape", [f"{name}_t", f"{name}_shape"], [f"{name}_r" if scored else name]))
            if scored:
                nodes.append(helper.make_node("Sigmoid", [f"{name}_r"], [name]))
            outputs.append(helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [1, side * side, channels]))
    graph = helper.make_graph(
        nodes, "yunet_like", [helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 3, 640, 640])],
        outputs, initializers,
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)], ir_version=10), path)
    return path


def test_surgery_exposes_the_head_maps_and_keeps_upstream_outputs_exactly(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    ort = pytest.importorskip("onnxruntime")
    surgery = load_script("surgery")
    source = _yunet_like(tmp_path / "yunet.onnx", surgery.HEADS, surgery.STRIDES)

    cut = surgery.prepare_model(source, tmp_path / "graphs")

    graph = onnx.load(cut).graph
    assert {node.op_type for node in graph.node} == {"Conv", "Sigmoid"}
    assert {value.name: [d.dim_value for d in value.type.tensor_type.shape.dim] for value in graph.output} == (
        surgery.OUTPUT_SHAPES
    )
    image = np.random.RandomState(1).uniform(0, 255, (1, 3, 640, 640)).astype(np.float32)
    upstream = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
    heads = ort.InferenceSession(str(cut), providers=["CPUExecutionProvider"])
    expected = dict(zip([value.name for value in upstream.get_outputs()], upstream.run(None, {"input": image})))
    actual = dict(zip([value.name for value in heads.get_outputs()], heads.run(None, {"input": image})))
    for name in surgery.OUTPUTS:
        channels = surgery.HEADS[name.split("_")[0]]
        np.testing.assert_array_equal(actual[name].transpose(0, 2, 3, 1).reshape(1, -1, channels), expected[name])


def test_surgery_rejects_a_head_that_does_not_flatten_channels_last(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    surgery = load_script("surgery")
    source = _yunet_like(tmp_path / "yunet.onnx", surgery.HEADS, surgery.STRIDES, broken="kps_32")

    with pytest.raises(ValueError, match="kps_32's Transpose permutes"):
        surgery.prepare_model(source, tmp_path / "graphs")


def _png(width: int, height: int, colour: tuple[int, int, int] = (40, 80, 120)) -> bytes:
    import cv2

    ok, encoded = cv2.imencode(".png", np.full((height, width, 3), colour, np.uint8))
    assert ok
    return encoded.tobytes()


@pytest.mark.parametrize(
    ("width", "height", "expected"), [(1024, 768, (640, 480, 0, 80)), (300, 401, (479, 640, 80, 0))]
)
def test_letterbox_centres_the_image_and_pads_with_zeros(
    load_script: Callable[[str], ModuleType], width: int, height: int, expected: tuple[int, int, int, int]
) -> None:
    calibration = load_script("calibration")
    resized_width, resized_height, left, top = expected

    image = calibration.preprocess_image(_png(width, height))

    assert (image.shape, image.dtype) == ((640, 640, 3), np.float32)
    # BGR pixel values in [0, 255], unnormalized: the PNG's (40, 80, 120) colour comes back as stored.
    np.testing.assert_array_equal(image[top + resized_height // 2, left + resized_width // 2], [40, 80, 120])
    assert (image[:top] == 0).all() and (image[:, :left] == 0).all()
    assert (image[top + resized_height :] == 0).all() and (image[:, left + resized_width :] == 0).all()


def _write_archive(path: Path, files: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w:gz") as archive:
        for name, content in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    return path


def test_calibration_and_fidelity_draw_disjoint_training_images(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    calibration = load_script("calibration")
    names = [f"wider-face-cal/WIDER_train/images/0--Parade/0_Parade_{index}.jpg" for index in range(10)]
    files = {name: _png(64, 48) for name in names}
    files["wider-face-cal/WIDER_val/images/0--Parade/0_Parade_val.jpg"] = _png(64, 48)
    archive = _write_archive(tmp_path / "wider.tgz", files)
    expected = sorted(names)
    random.Random(42).shuffle(expected)

    assert calibration.calibration_names(archive, 4) == expected[:4]
    assert calibration.fidelity_names(archive, 42, 4, 3) == expected[4:7]
    assert calibration.calibration_images(archive, 4).shape == (4, 640, 640, 3)
    with pytest.raises(ValueError, match="cannot select 11 calibration images from 10"):
        calibration.calibration_names(archive, 11)
    with pytest.raises(ValueError, match="needs 11 images"):
        calibration.fidelity_names(archive, 42, 8, 3)


@pytest.mark.parametrize("profile_id", ["w16a16", "w8a8"])
def test_compile_reads_each_declared_build_profile(load_script: Callable[[str], ModuleType], profile_id: str) -> None:
    recipe = load_script("compile")

    parameters = recipe._parameters(profile_id)

    assert parameters["precision"] == profile_id
    if profile_id == "w8a8":
        assert (parameters["calibration_bins"], parameters["calibration_samples"]) == (2048, 100)


def test_compile_rejects_an_undeclared_build_profile(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")

    with pytest.raises(ValueError, match="no build profile 'w8a16'"):
        recipe._parameters("w8a16")


def test_fidelity_images_must_follow_every_calibration_image(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")

    recipe.check_fidelity_selection({"seed": 42, "offset": 100})
    with pytest.raises(ValueError, match="seed 7, but calibration uses 42"):
        recipe.check_fidelity_selection({"seed": 7, "offset": 100})
    with pytest.raises(ValueError, match="inside the first 100 calibration images"):
        recipe.check_fidelity_selection({"seed": 42, "offset": 99})


@pytest.mark.parametrize(("precision", "calibrates"), [("w8a8", True), ("w16a16", False)])
def test_quantization_calibrates_only_int8_activations(
    load_script: Callable[[str], ModuleType], precision: str, calibrates: bool
) -> None:
    recipe = load_script("compile")

    recipe._quantization(precision, 2048 if calibrates else None)

    arguments = recipe.QuantizationParams.call_args.kwargs
    calibration = recipe.HistogramMSEMethod.return_value if calibrates else recipe.SkipCalibration.return_value
    assert arguments["calibration_method"] is calibration
    if calibrates:
        recipe.HistogramMSEMethod.assert_called_once_with(2048)
        assert arguments["biascorr_type"] is recipe.BiasCorrectionType.REGULAR


@pytest.mark.parametrize("precision", ["w16a16", "w8a8"])
def test_only_the_int8_build_quantizes_with_calibration_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType], precision: str
) -> None:
    recipe = load_script("compile")
    parameters = {"precision": precision}
    if precision == "w8a8":
        parameters |= {"calibration_bins": 2048, "calibration_samples": 2}
    drawn = []

    def images(dataset: Path, count: int) -> np.ndarray:
        drawn.append(count)
        return np.zeros((count, 4, 4, 3), np.float32)

    monkeypatch.setattr(recipe, "_parameters", lambda profile_id: parameters)
    monkeypatch.setattr(recipe, "calibration_images", images)
    monkeypatch.setattr(recipe, "prepare_model", lambda source, output_dir: output_dir / "heads.onnx")

    with pytest.raises(RuntimeError, match="expected one MPK"):
        recipe.compile_model(tmp_path / "model.onnx", tmp_path / "out", precision, tmp_path / "wider.tgz")

    data = recipe.load_model.return_value.quantize.call_args.kwargs["calibration_data"]
    if precision == "w8a8":
        assert drawn == [2]
        assert data is recipe.convert_data_generator_to_iterable.return_value
    else:
        assert drawn == []
        assert data is None


def _mpk_manifest(recipe: ModuleType) -> dict:
    (input_size,) = recipe.INPUT_BYTES.values()
    return {
        "input_nodes": [{"name": recipe.INPUT, "size": input_size}],
        "plugins": [
            {"sequence": 1, "processor": "EV74"},
            {"sequence": 2, "processor": "MLA"},
            {"sequence": 3, "processor": "EV74", "config_params": {"kernel": "pass_through"},
             "output_nodes": [{"size": size} for size in recipe.OUTPUT_BYTES]},
        ],
    }


def _write_mpk(path: Path, manifest: dict, extra: tuple[str, ...] = ()) -> None:
    members = {"model.elf": b"elf", "model_mpk.json": json.dumps(manifest).encode(), **{name: b"" for name in extra}}
    _write_archive(path, members)


@pytest.mark.parametrize("case", ["valid", "two_mla", "host", "shared_object", "reordered", "input"])
def test_mpk_audit_requires_one_mla_segment_and_the_declared_interface(
    tmp_path: Path, case: str, load_script: Callable[[str], ModuleType]
) -> None:
    recipe = load_script("compile")
    manifest = _mpk_manifest(recipe)
    plugins = manifest["plugins"]
    if case == "two_mla":
        plugins.insert(2, {"sequence": 2, "processor": "MLA"})
    elif case == "host":
        plugins.insert(2, {"sequence": 2, "processor": "A65"})
    elif case == "reordered":
        # cls_8 and kps_8 swapped: the sizes no longer follow the 12 heads' order.
        outputs = plugins[-1]["output_nodes"]
        outputs[0], outputs[9] = outputs[9], outputs[0]
    elif case == "input":
        manifest["input_nodes"][0]["name"] = "images"
    archive = tmp_path / "model_mpk.tar.gz"
    _write_mpk(archive, manifest, ("libhost.so",) if case == "shared_object" else ())

    if case == "valid":
        recipe.audit_mpk(archive)
    else:
        with pytest.raises(RuntimeError):
            recipe.audit_mpk(archive)


@pytest.mark.parametrize("layout", ["channels_last", "channels_first"])
def test_head_maps_return_to_the_upstream_layout(load_script: Callable[[str], ModuleType], layout: str) -> None:
    recipe = load_script("compile")
    upstream = np.arange(1600 * 4, dtype=np.float64).reshape(1, 1600, 4)
    nhwc = upstream.reshape(1, 40, 40, 4)
    value = nhwc if layout == "channels_last" else nhwc.transpose(0, 3, 1, 2)

    np.testing.assert_array_equal(recipe._flatten("bbox_16", value), upstream)
    with pytest.raises(RuntimeError, match="unexpected bbox_16 shape"):
        recipe._flatten("bbox_16", value[..., :2])


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--compile-invocation", "model"], "a build profile compiles in one invocation named after it"),
        (["--compile-invocation", "w16a16"], "pass --fidelity-dataset"),
    ],
)
def test_compile_main_rejects_an_unknown_invocation_or_missing_fidelity_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    load_script: Callable[[str], ModuleType],
    arguments: list[str],
    message: str,
) -> None:
    recipe = load_script("compile")
    source = tmp_path / "model.onnx"
    source.touch()
    monkeypatch.setattr(
        sys,
        "argv",
        ["compile.py", "--model", str(source), "--output-dir", str(tmp_path / "out"), "--build-profile", "w16a16"]
        + arguments,
    )

    with pytest.raises(SystemExit) as stopped:
        recipe.main()

    assert stopped.value.code == 2
    assert message in capsys.readouterr().err
