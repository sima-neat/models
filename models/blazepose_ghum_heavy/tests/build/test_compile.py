from __future__ import annotations

import io
import json
import tarfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import onnx
import pytest
from onnx import helper, numpy_helper


def _source_graph(case: str = "valid") -> onnx.ModelProto:
    """The converted interface: three channel Pads and a spatial Pad on the kept outputs, two outputs to drop."""
    rng = np.random.RandomState(0)
    nodes = [helper.make_node("Transpose", ["input_1"], ["nchw"], perm=[0, 3, 1, 2])]
    initializers = []

    def constant(name, value):
        initializers.append(numpy_helper.from_array(np.asarray(value), name))
        return name

    def pad(name, pads, **attributes):
        nodes.append(
            helper.make_node("Pad", ["nchw", constant(f"{name}_pads", np.array(pads, np.int64))], [name], **attributes)
        )

    def head(name, value, channels, size):
        nodes.append(helper.make_node("GlobalAveragePool", [value], [f"{name}_pool"]))
        nodes.append(helper.make_node("Flatten", [f"{name}_pool"], [f"{name}_flat"]))
        weight = constant(f"{name}_weight", rng.uniform(-1, 1, (channels, size)).astype(np.float32))
        nodes.append(helper.make_node("MatMul", [f"{name}_flat", weight], [f"{name}_out"]))
        return f"{name}_out"

    added = [1, 2, 0 if case == "two_channel_pads" else 3]
    for index, count in enumerate(added):
        pad(
            f"channel_{index}",
            [0, 0, 0, 0, 0, count, 0, 0],
            **({"mode": "reflect"} if case == "reflect" and index == 0 else {}),
        )
    pad("spatial", [0, 0, 1, 1, 0, 0, 1, 1])
    landmarks = head("landmarks", "channel_0", 4, 195)
    nodes.append(helper.make_node("Identity", [landmarks], ["Identity"]))
    nodes.append(helper.make_node("Sigmoid", [head("presence", "channel_1", 5, 1)], ["Identity_1"]))
    world = [head("world", "channel_2", 3 + added[2], 117), head("spatial_world", "spatial", 3, 117)]
    nodes.append(helper.make_node("Add", world, ["Identity_4"]))
    nodes.append(helper.make_node("ReduceMean", ["input_1"], ["Identity_2"], axes=[3]))
    nodes.append(
        helper.make_node(
            "Constant", [], ["Identity_3"], value=numpy_helper.from_array(np.zeros((1, 64, 64, 39), np.float32))
        )
    )
    input_shape = [1, 256, 256, 4] if case == "input_shape" else [1, 256, 256, 3]
    outputs = {
        "Identity": [1, 195],
        "Identity_1": [1, 1],
        "Identity_2": [1, 256, 256, 1],
        "Identity_3": [1, 64, 64, 39],
        "Identity_4": [1, 117],
    }
    graph = helper.make_graph(
        nodes,
        "fixture",
        [helper.make_tensor_value_info("input_1", onnx.TensorProto.FLOAT, input_shape)],
        [helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape) for name, shape in outputs.items()],
        initializers,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)], ir_version=8)


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "model.onnx"
    onnx.save(_source_graph(), path)
    return path


def test_crop_square_scales_the_longer_box_side_by_1_25_around_its_centre(
    load_script: Callable[[str], ModuleType],
) -> None:
    calibration = load_script("calibration")

    # A 40x80 box at (100, 50) centres on (120, 90); its square is 100 px, from (70, 40).
    assert calibration.crop_square([100.0, 50.0, 40.0, 80.0]) == (70.0, 40.0, 100.0)
    assert calibration.crop_square([0.0, 0.0, 200.0, 20.0]) == (-25.0, -115.0, 250.0)
    with pytest.raises(ValueError, match="empty person box"):
        calibration.crop_square([5.0, 5.0, 0.0, 0.0])


def test_crop_puts_the_square_corners_on_the_input_corners(load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (300, 400, 3), dtype=np.uint8)

    pixels = calibration.crop(image, 10.0, 20.0, 256.0)

    assert (pixels.shape, pixels.dtype) == ((256, 256, 3), np.float32)
    np.testing.assert_allclose(pixels, image[20:276, 10:266] / np.float32(255), atol=1e-6)


def test_crop_scales_bilinearly_and_repeats_the_image_edge_outside_it(load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")
    # Each pixel's value is its column.
    image = np.repeat(np.arange(200, dtype=np.uint8)[None, :, None], 3, axis=2).repeat(300, axis=0)

    # A 512 px square halves the image: input column u reads image column 2u, and columns past 199 repeat it.
    pixels = calibration.crop(image, 0.0, 0.0, 512.0) * 255

    np.testing.assert_allclose(pixels[0, [0, 50, 99], 0], [0, 100, 198], atol=1e-3)
    np.testing.assert_allclose(pixels[:, 100:, :], 199, atol=1e-3)


def _coco_archive(path: Path) -> None:
    """cal2017 and val2017 splits of four 64x64 images each, image i filled with 10 * i in cal2017 and 200 in
    val2017. In each split images 0-2 hold one labelled person, image 2 also a crowd and image 3 only an unlabelled
    person. Every box is 32x32 inside its image, so a person's crop is its image's fill."""
    import cv2

    files = {}
    for split in ("cal2017", "val2017"):
        annotations = []
        for index in range(4):
            fill = 10 * index if split == "cal2017" else 200
            files[f"coco/{split}/{index:012d}.png"] = cv2.imencode(".png", np.full((64, 64, 3), fill, np.uint8))[1]
            person = {"id": 10 * index, "image_id": index, "bbox": [16.0, 16.0, 32.0, 32.0], "iscrowd": 0}
            annotations.append({**person, "num_keypoints": 0 if index == 3 else 5})
            if index == 2:
                annotations.append({**person, "id": 21, "iscrowd": 1, "num_keypoints": 5})
        files[f"coco/annotations/person_keypoints_{split}.json"] = json.dumps({
            "images": [{"id": index, "file_name": f"{index:012d}.png"} for index in range(4)],
            "annotations": annotations,
        }).encode()
    with tarfile.open(path, "w:gz") as stream:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            stream.addfile(info, io.BytesIO(bytes(content)))


def test_calibration_draws_a_seeded_sample_of_labelled_cal2017_people(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    calibration = load_script("calibration")
    archive = tmp_path / "coco.tgz"
    _coco_archive(archive)

    people = calibration.calibration_people(archive, 2)

    assert people == calibration.calibration_people(archive, 2)
    assert len({person["id"] for person in people}) == 2
    assert {person["id"] for person in calibration.calibration_people(archive, 3)} == {0, 10, 20}
    with pytest.raises(ValueError, match="cannot select 4 calibration people from 3 labelled cal2017 people"):
        calibration.calibration_people(archive, 4)


def test_calibration_crops_each_person_from_its_cal2017_image(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    calibration = load_script("calibration")
    archive = tmp_path / "coco.tgz"
    _coco_archive(archive)
    people = calibration.calibration_people(archive, 3)

    images = calibration.calibration_images(archive, 3)

    assert (images.shape, images.dtype) == ((3, 256, 256, 3), np.float32)
    # Each crop is its person's cal2017 image, never the val2017 image of the same name.
    for person, pixels in zip(people, images):
        np.testing.assert_allclose(pixels, person["image_id"] * 10 / 255, atol=1e-6)


def test_surgery_keeps_three_outputs_and_turns_channel_pads_into_identity_convolutions(
    tmp_path: Path, source: Path, load_script: Callable[[str], ModuleType]
) -> None:
    surgery = load_script("surgery")

    prepared = onnx.load(surgery.prepare_model(source, tmp_path / "out"))

    outputs = {value.name: [d.dim_value for d in value.type.tensor_type.shape.dim] for value in prepared.graph.output}
    assert list(outputs) == surgery.OUTPUTS == ["Identity", "Identity_1", "Identity_4"]
    assert outputs == {"Identity": [1, 195], "Identity_1": [1, 1], "Identity_4": [1, 117]}
    assert not {"ReduceMean", "Constant"} & {node.op_type for node in prepared.graph.node}
    weights = {value.name: numpy_helper.to_array(value) for value in prepared.graph.initializer}
    convolutions = [node for node in prepared.graph.node if node.op_type == "Conv"]
    assert [weights[node.input[1]].shape for node in convolutions] == [(4, 3, 1, 1), (5, 3, 1, 1), (6, 3, 1, 1)]
    for node in convolutions:
        weight = weights[node.input[1]][:, :, 0, 0]
        np.testing.assert_array_equal(weight, np.eye(*weight.shape, dtype=np.float32))
    (spatial,) = [node for node in prepared.graph.node if node.op_type == "Pad"]
    assert weights[spatial.input[1]].tolist() == [0, 0, 1, 1, 0, 0, 1, 1]


@pytest.mark.parametrize(
    ("case", "match"),
    [("two_channel_pads", "found 2 channel Pads"), ("reflect", "does not pad with zeros"), ("input_shape", "inputs")],
)
def test_surgery_rejects_an_unexpected_graph(
    tmp_path: Path, case: str, match: str, load_script: Callable[[str], ModuleType]
) -> None:
    path = tmp_path / "model.onnx"
    onnx.save(_source_graph(case), path)

    with pytest.raises(ValueError, match=match):
        load_script("surgery").prepare_model(path, tmp_path / "out")


def test_rewrite_check_rejects_a_graph_that_changes_an_output(
    tmp_path: Path, source: Path, load_script: Callable[[str], ModuleType]
) -> None:
    surgery = load_script("surgery")
    prepared = onnx.load(surgery.prepare_model(source, tmp_path / "out"))
    weight = next(value for value in prepared.graph.initializer if value.name.endswith("/identity_weight"))
    weight.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(weight) * 1.01, weight.name))
    changed = tmp_path / "changed.onnx"
    onnx.save(prepared, changed)

    with pytest.raises(RuntimeError, match="prepared output Identity differs"):
        surgery.validate_rewrite(source, changed)


class Quantized:
    """Stands in for the compiler's quantized model: execute returns the source's outputs plus an offset each."""

    def __init__(self, source: Path, offsets: dict[str, float]) -> None:
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
        self.offsets = offsets

    def execute(self, inputs, use_jax):
        (sample,) = inputs.values()
        values = self.session.run(list(self.offsets), {"input_1": sample})
        return [value + offset for value, offset in zip(values, self.offsets.values())]


@pytest.mark.parametrize("precision", ["w16a16", "w8a8"])
@pytest.mark.parametrize("case", ["valid", "world_landmarks", "nan"])
def test_quantized_check_bounds_each_output_against_the_converted_graph(
    case: str, precision: str, source: Path, load_script: Callable[[str], ModuleType]
) -> None:
    recipe = load_script("compile")
    limits = recipe.MAX_MEAN_ERROR[precision]
    offsets = {name: bound / 2 for name, bound in limits.items()}
    if case == "valid":
        errors = recipe.check_quantized_outputs(source, Quantized(source, offsets), limits)
        assert errors == pytest.approx(offsets, rel=1e-3)
        return
    offsets["Identity_4"] = 1.5 * limits["Identity_4"] if case == "world_landmarks" else np.nan
    with pytest.raises(RuntimeError, match="Identity_4"):
        recipe.check_quantized_outputs(source, Quantized(source, offsets), limits)


def _mpk_manifest(recipe: ModuleType) -> dict:
    """The plugin chain of the compiled MPK: a cast of the frame, the MLA, then slices and casts per output."""
    return {
        "input_nodes": [{"name": "input_1", "size": 1 * 256 * 256 * 3 * 4}],
        "plugins": [
            {"sequence": 1, "processor": "EV74", "config_params": {"kernel": "cast_transform"}},
            {"sequence": 2, "processor": "MLA", "config_params": {}},
            {"sequence": 3, "processor": "EV74", "config_params": {"kernel": "cast_transform"}},
            {
                "sequence": 4,
                "processor": "EV74",
                "config_params": {"kernel": "pass_through"},
                "output_nodes": [{"size": size} for size in recipe.OUTPUT_BYTES],
            },
        ],
    }


def _write_mpk(path: Path, manifest: dict) -> None:
    with tarfile.open(path, "w:gz") as stream:
        for name, content in (("model.elf", b"elf"), ("model_mpk.json", json.dumps(manifest).encode())):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            stream.addfile(info, io.BytesIO(content))


@pytest.mark.parametrize("case", ["valid", "two_mla", "host", "input", "reordered"])
def test_mpk_audit_requires_one_mla_segment_and_the_declared_interface(
    tmp_path: Path, case: str, load_script: Callable[[str], ModuleType]
) -> None:
    recipe = load_script("compile")
    manifest = _mpk_manifest(recipe)
    plugins = manifest["plugins"]
    if case == "two_mla":
        plugins.insert(2, {"sequence": 2, "processor": "MLA", "config_params": {}})
    elif case == "host":
        plugins.insert(2, {"sequence": 2, "processor": "A65", "config_params": {}})
    elif case == "input":
        manifest["input_nodes"][0]["size"] //= 2
    elif case == "reordered":
        outputs = plugins[-1]["output_nodes"]
        outputs[0], outputs[2] = outputs[2], outputs[0]
    archive = tmp_path / "model_mpk.tar.gz"
    _write_mpk(archive, manifest)

    if case == "valid":
        recipe.audit_mpk(archive)
    else:
        with pytest.raises(RuntimeError):
            recipe.audit_mpk(archive)


@pytest.mark.parametrize("profile_id", ["w16a16", "w8a8"])
def test_compile_reads_the_declared_build_parameters(load_script: Callable[[str], ModuleType], profile_id: str) -> None:
    expected = {"precision": profile_id}
    if profile_id == "w8a8":
        expected |= {"calibration_bins": 2048, "calibration_samples": 50}

    assert load_script("compile")._parameters(profile_id) == expected


def test_compile_rejects_an_undeclared_build_profile(load_script: Callable[[str], ModuleType]) -> None:
    with pytest.raises(ValueError, match="no build profile"):
        load_script("compile")._parameters("w32a32")


def test_int8_quantization_calibrates_by_histogram_mse(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")

    recipe._quantization("w8a8", 2048)

    assert recipe.QuantizationParams.call_args.kwargs == {
        "calibration_method": recipe.HistogramMSEMethod.return_value,
        "activation_quantization_scheme": recipe.quantization_scheme.return_value,
        "weight_quantization_scheme": recipe.quantization_scheme.return_value,
        "biascorr_type": recipe.BiasCorrectionType.REGULAR,
    }
    recipe.HistogramMSEMethod.assert_called_once_with(2048)
    assert [call.kwargs for call in recipe.quantization_scheme.call_args_list] == [
        {"asymmetric": True, "per_channel": False, "bits": 8},
        {"asymmetric": False, "per_channel": True, "bits": 8},
    ]


def test_bf16_quantization_skips_calibration(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")

    recipe._quantization("w16a16", None)

    assert recipe.QuantizationParams.call_args.kwargs == {
        "calibration_method": recipe.SkipCalibration.return_value,
        "activation_quantization_scheme": recipe.bfloat16_scheme.return_value,
        "weight_quantization_scheme": recipe.bfloat16_scheme.return_value,
    }
    with pytest.raises(ValueError, match="unsupported precision"):
        recipe._quantization("w8a16", None)


@pytest.mark.parametrize("precision", ["w16a16", "w8a8"])
def test_only_the_int8_build_quantizes_with_calibration_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType], precision: str
) -> None:
    recipe = load_script("compile")
    parameters = {"precision": precision}
    if precision == "w8a8":
        parameters |= {"calibration_bins": 2048, "calibration_samples": 2}
    monkeypatch.setattr(recipe, "_parameters", lambda profile_id: parameters)
    monkeypatch.setattr(recipe, "calibration_images", lambda dataset, count: np.zeros((count, 256, 256, 3), np.float32))
    monkeypatch.setattr(recipe, "prepare_model", lambda source, output_dir: output_dir / "prepared.onnx")
    limits = []
    monkeypatch.setattr(recipe, "check_quantized_outputs", lambda source, quantized, limit: limits.append(limit))

    with pytest.raises(RuntimeError, match="expected one MPK"):
        recipe.compile_model(tmp_path / "model.onnx", tmp_path / "out", precision, tmp_path / "coco.tgz")

    data = recipe.load_model.return_value.quantize.call_args.kwargs["calibration_data"]
    assert data is (recipe.convert_data_generator_to_iterable.return_value if precision == "w8a8" else None)
    assert limits == [recipe.MAX_MEAN_ERROR[precision]]


def test_int8_build_requires_a_calibration_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    recipe = load_script("compile")
    parameters = {"precision": "w8a8", "calibration_bins": 2048, "calibration_samples": 2}
    monkeypatch.setattr(recipe, "_parameters", lambda profile_id: parameters)

    with pytest.raises(ValueError, match="requires --dataset"):
        recipe.compile_model(tmp_path / "model.onnx", tmp_path / "out", "w8a8", None)


def test_compiled_build_holds_the_audited_mpk(
    compiled: Path, build_profile: str, load_script: Callable[[str], ModuleType]
) -> None:
    (archive,) = (compiled / "model").glob("*_mpk.tar.gz")
    assert archive.name == f"blazepose_ghum_heavy_{build_profile}_mpk.tar.gz"
    load_script("compile").audit_mpk(archive)
