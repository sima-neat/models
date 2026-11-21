from __future__ import annotations

import hashlib
import io
import json
import random
import sys
import tarfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from PIL import Image


def _jpeg(width: int, height: int) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (width, height), (200, 100, 50)).save(stream, format="JPEG", quality=100)
    return stream.getvalue()


def _write_archive(path: Path, files: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w:gz") as archive:
        for name, content in files.items():
            member = tarfile.TarInfo(f"./{name}")
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    return path


def test_preprocessing_resizes_the_longer_side_normalizes_and_pads(load_script: Callable[[str], ModuleType]) -> None:
    pipeline = load_script("pipeline")
    image = np.full((300, 400, 3), 128, np.uint8)

    result = pipeline.preprocess(image)

    assert (result.shape, result.dtype) == ((1024, 1024, 3), np.float32)
    assert pipeline.resized_size(300, 400) == (768, 1024)
    np.testing.assert_allclose(result[100, 100], (128 - pipeline.PIXEL_MEAN) / pipeline.PIXEL_STD, atol=1e-5)
    # Padding below the resized image is zero, the normalized mean.
    assert (result[768:] == 0).all()


def test_prompt_inputs_normalize_pixel_centres_and_pad_the_slots(load_script: Callable[[str], ModuleType]) -> None:
    pipeline = load_script("pipeline")

    inputs = pipeline.prompt_inputs([(0, 0), (399, 299)], ["foreground", "background"], (300, 400))

    coords, types = inputs["point_coords"][0, :, 0, :].T, inputs["point_types"][0, :, 0, :].T
    assert inputs["point_coords"].shape == (1, 2, 1, pipeline.POINTS)
    assert inputs["point_types"].shape == (1, len(pipeline.POINT_TYPES), 1, pipeline.POINTS)
    # Pixel (0, 0) is the centre of the resized image's first pixel; (399, 299) scales by 1024 / 400.
    np.testing.assert_allclose(coords[0], [0.5 / 1024 * 2 - 1] * 2)
    np.testing.assert_allclose(coords[1], [(399 * 2.56 + 0.5) / 512 - 1, (299 * 2.56 + 0.5) / 512 - 1], rtol=1e-6)
    assert (coords[2:] == 0).all()
    assert [pipeline.POINT_TYPES[index] for index in types[:3].argmax(1)] == [
        "foreground",
        "background",
        "padding",
    ]
    assert (types[3:] == 0).all()


@pytest.mark.parametrize(
    ("points", "types", "message"),
    [
        ([(1, 1)] * 6, ["foreground"] * 6, "1 to 5 points"),
        ([(1, 1)], ["foreground", "background"], "1 to 5 points"),
        ([(1, 1)], ["padding"], "unknown point types"),
    ],
)
def test_prompt_inputs_reject_bad_prompts(points, types, message, load_script: Callable[[str], ModuleType]) -> None:
    pipeline = load_script("pipeline")

    with pytest.raises(ValueError, match=message):
        pipeline.prompt_inputs(points, types, (300, 400))


def test_postprocessing_returns_the_original_size(load_script: Callable[[str], ModuleType]) -> None:
    pipeline = load_script("pipeline")
    masks = np.full((1, 4, 256, 256), -1, np.float32)
    # The top-left quarter of the 1024x1024 input is positive.
    masks[:, :, :128, :128] = 1

    result = pipeline.postprocess_masks(masks, (300, 400))

    assert result.shape == (4, 300, 400)
    # The resized image is 768x1024, so the positive 512x512 corner is 200x200 of the original.
    assert (result[:, :190, :190] > 0).all() and (result[:, 210:, 210:] < 0).all()


def test_fixed_prompts_cover_click_box_and_token_capacity(load_script):
    calibration = load_script("calibration")
    pipeline = load_script("pipeline")
    prompts = calibration.image_prompts((480, 640))
    active_counts = [int(item["point_types"].sum()) for item in prompts]
    assert active_counts == [2, 3, 6, 2, 5]
    box_types = prompts[3]["point_types"][0, :, 0, :].T
    assert [pipeline.POINT_TYPES[index] for index in box_types[:2].argmax(1)] == ["box_top_left", "box_bottom_right"]
    assert (box_types[2:] == 0).all()


def test_calibration_draws_a_seeded_sample_from_cal2017_only(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    calibration = load_script("calibration")
    files = {f"cal2017/{index:02d}.jpg": _jpeg(64, 48) for index in range(6)}
    files["val2017/x.jpg"] = _jpeg(64, 48)
    archive = _write_archive(tmp_path / "coco.tgz", files)

    images = calibration.calibration_images(archive, 4)

    assert [image.shape for image in images] == [(48, 64, 3)] * 4
    with pytest.raises(ValueError, match="cannot select 7 calibration images from 6"):
        calibration.calibration_images(archive, 7)


def test_fidelity_selection_is_the_declared_seeded_val2017_sample(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    calibration = load_script("calibration")
    names = [f"val2017/{index:02d}.jpg" for index in range(10)]
    for name in [*names, "cal2017/c.jpg"]:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(b"")
    expected = sorted(names)
    random.Random(42).shuffle(expected)

    assert calibration.fidelity_names(tmp_path, 42, 2, 5) == expected[2:7]
    with pytest.raises(ValueError, match="needs 11 images"):
        calibration.fidelity_names(tmp_path, 42, 1, 10)


@pytest.mark.parametrize("profile_id", ["w16a16", "w8a16"])
def test_compile_binds_every_component_in_each_profile(
    load_script: Callable[[str], ModuleType], profile_id: str
) -> None:
    recipe = load_script("compile")
    parameters = recipe._component_parameters(profile_id)

    assert list(parameters) == ["image_encoder", "prompt_decoder"]
    assert {item["precision"] for item in parameters.values()} == {profile_id}
    # Both qualified BF16 components explicitly fit their ranges.
    assert [("calibration_samples" in item) for item in parameters.values()] == [True, True]
    with pytest.raises(ValueError, match="no build profile 'w8a8'"):
        recipe._component_parameters("w8a8")


def test_outputs_come_back_in_onnx_order(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")
    embeddings = np.random.RandomState(0).normal(size=(1, 256, 64, 64)).astype(np.float32)

    np.testing.assert_array_equal(recipe.channels_first(embeddings.transpose(0, 2, 3, 1), (1, 256, 64, 64)), embeddings)
    np.testing.assert_array_equal(recipe.channels_first(np.ones((1, 1, 1, 4)), (1, 4)), np.ones((1, 4)))


def _mpk(path: Path, processors: list[str], version: str) -> Path:
    manifest = {"model_sdk_version": version, "plugins": [{"processor": p} for p in processors]}
    content = json.dumps(manifest).encode()
    with tarfile.open(path, "w:gz") as archive:
        member = tarfile.TarInfo("model_mpk.json")
        member.size = len(content)
        archive.addfile(member, io.BytesIO(content))
    return path


def test_audit_requires_one_mla_segment_and_no_host_code(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    recipe = load_script("compile")
    version = "3.0.0.1"

    recipe.audit_mpk(_mpk(tmp_path / "ok.tar.gz", ["EV74", "MLA", "EV74"], version), "image_encoder")
    for name, processors, compiler in [
        ("two", ["MLA", "MLA"], version),
        ("host", ["MLA", "A65"], version),
    ]:
        with pytest.raises(RuntimeError):
            recipe.audit_mpk(_mpk(tmp_path / f"{name}.tar.gz", processors, compiler), "image_encoder")


def test_compile_main_rejects_an_unknown_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    load_script: Callable[[str], ModuleType],
) -> None:
    recipe = load_script("compile")
    source = tmp_path / "model.onnx"
    source.touch()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compile.py",
            "--model",
            str(source),
            "--output-dir",
            str(tmp_path / "out"),
            "--build-profile",
            "w16a16",
            "--compile-invocation",
            "model",
        ],
    )

    with pytest.raises(SystemExit) as stopped:
        recipe.main()

    assert stopped.value.code == 2
    assert "a build profile compiles in one invocation named after it" in capsys.readouterr().err


@pytest.mark.parametrize(
    "points,types,size,message",
    [
        ([(float("nan"), 1)], ["foreground"], (300, 400), "finite"),
        ([(1, 1)], ["box_top_left"], (300, 400), "corner"),
        ([(1, 1)], ["foreground"], (0, 400), "positive"),
    ],
)
def test_prompt_validation_preserves_finite_geometry(points, types, size, message, load_script):
    with pytest.raises(ValueError, match=message):
        load_script("pipeline").prompt_inputs(points, types, size)


def test_fp32_input_preparation_accepts_out_of_image_prompts(load_script):
    # This helper check does not qualify the compiled Sin/Cos lookup-table domain.
    arrays = load_script("pipeline").prompt_inputs([(-10, 500)], ["foreground"], (300, 400))
    assert np.isfinite(arrays["point_coords"]).all()


# Independent public FP32 byte contracts, not a snapshot of a generated mpk.json.
COMPONENT_CONTRACTS = {
    "image_encoder": {"inputs": [("image_encoder_image", 12582912)], "outputs": [4194304]},
    "prompt_decoder": {
        "inputs": [
            ("prompt_decoder_image_embeddings", 4194304),
            ("prompt_decoder_point_encoding", 6144),
            ("prompt_decoder_point_types", 120),
        ],
        "outputs": [1048576, 16],
    },
}


def assert_component_contract(manifest, component):
    expected = COMPONENT_CONTRACTS[component]
    assert [(node["name"], node["size"]) for node in manifest["input_nodes"]] == expected["inputs"]
    assert [node["size"] for node in manifest["plugins"][-1]["output_nodes"]] == expected["outputs"]


def compiled_archive(compiled, component):
    archives = list((compiled / component).glob("*_mpk.tar.gz"))
    assert len(archives) == 1, (component, archives)
    return archives[0]


@pytest.mark.parametrize("component", COMPONENT_CONTRACTS)
def test_actual_compiled_components_preserve_public_interfaces(compiled, component, load_script):
    recipe = load_script("compile")
    assert_component_contract(recipe._read_mpk_manifest(compiled_archive(compiled, component)), component)


@pytest.mark.parametrize("component", COMPONENT_CONTRACTS)
def test_actual_compiled_components_use_declared_sdk(compiled, component, load_script):
    load_script("compile").audit_mpk(compiled_archive(compiled, component), component)


@pytest.mark.parametrize("component", COMPONENT_CONTRACTS)
@pytest.mark.parametrize("fault", ["none", "input-size", "input-order", "output-size"])
def test_public_interface_checks_reject_shape_and_order_changes(component, fault):
    expected = COMPONENT_CONTRACTS[component]
    inputs = [{"name": name, "size": size} for name, size in expected["inputs"]]
    outputs = [{"size": size} for size in expected["outputs"]]
    manifest = {"input_nodes": inputs, "plugins": [{"output_nodes": outputs}]}
    if fault == "input-size":
        inputs[0]["size"] -= 4
    elif fault == "input-order":
        inputs[0]["name"] = "wrong-input"
    elif fault == "output-size":
        outputs[0]["size"] -= 4
    else:
        assert_component_contract(manifest, component)
        return
    with pytest.raises(AssertionError):
        assert_component_contract(manifest, component)


@pytest.mark.parametrize("size", [(300, 600), (640, 480), (433, 301), (1024, 1024), (17, 13)])
def test_calibration_covers_fourier_extrema_and_full_frame_box(load_script, size):
    prompts = load_script("calibration").image_prompts(size)
    assert len(prompts) == 5
    height, width = size
    scale = 1024 / max(height, width)
    resized = np.array([int(width * scale + 0.5), int(height * scale + 0.5)], np.float32)
    lower = np.full(2, 0.5 / 1024 * 2 - 1, np.float32)
    upper = (resized + 0.5) / 1024 * 2 - 1
    real_points = []
    for prompt in prompts:
        active = prompt["point_types"][0, 1:, 0, :].sum(axis=0) == 1
        real_points.extend(prompt["point_coords"][0, :, 0, active])
    actual = np.asarray(real_points)
    # Independent signed Fourier directions attain their extrema at rectangle corners. These include both
    # off-diagonals; testing x/y ranges alone would miss positive/negative mixed projections.
    weights = np.array([[1, 0], [0, 1], [1, 1], [1, -1], [-1, 1], [-2, -0.5]], np.float64)
    expected_min = np.where(weights >= 0, weights * lower, weights * upper).sum(axis=1)
    expected_max = np.where(weights >= 0, weights * upper, weights * lower).sum(axis=1)
    projected = actual @ weights.T
    np.testing.assert_allclose(projected.min(axis=0), expected_min, rtol=0, atol=1e-6)
    np.testing.assert_allclose(projected.max(axis=0), expected_max, rtol=0, atol=1e-6)
    np.testing.assert_allclose(prompts[3]["point_coords"][0, :, 0, :2].T, [lower, upper], rtol=0, atol=1e-6)
    assert prompts[3]["point_types"][0, :, 0, :2].argmax(axis=0).tolist() == [3, 4]


@pytest.mark.parametrize(
    "expected,actual,cosine",
    [
        ([0, 0], [0, 0], 1),
        ([0, 0], [1, -1], 0),
        ([1, -1], [0, 0], 0),
        ([1, -1], [2, -2], 1),
        ([1, -1], [-1, 1], -1),
        ([1, 0], [0, 1], 0),
    ],
)
def test_fidelity_cosine_distinguishes_zero_and_nonzero_outputs(load_script, expected, actual, cosine):
    expected, actual = np.asarray(expected, np.float32), np.asarray(actual, np.float32)
    metrics = load_script("compile")._metrics("masks", [expected], [actual])
    assert metrics["cosine_similarity"] == pytest.approx(cosine)
    assert all(np.isfinite(value) for key, value in metrics.items() if key != "name")
    assert metrics["mean_absolute_error"] == pytest.approx(np.abs(expected - actual).mean())


def metadata(pipeline):
    gaussian = np.random.RandomState(4).normal(size=(2, 128)).astype(np.float32)
    value = {
        "version": 1,
        "source_sha256": "a" * 64,
        "gaussian": gaussian.tolist(),
        "gaussian_sha256": hashlib.sha256(gaussian.astype("<f4").tobytes()).hexdigest(),
        "phase_scale": float(np.float32(2 * np.pi)),
    }
    return value


def encoder(pipeline, value):
    return pipeline.PromptEncoding.from_metadata({pipeline.PROMPT_ENCODING_METADATA: json.dumps(value)})


def test_source_and_compiled_interfaces_are_explicit_and_distinct(load_script):
    pipeline = load_script("pipeline")
    assert list(pipeline.INPUT_SHAPES["prompt_decoder"]) == ["image_embeddings", "point_coords", "point_types"]
    assert pipeline.INPUT_SHAPES["prompt_decoder"]["point_coords"] == (1, 2, 1, 6)
    assert list(pipeline.COMPILED_INPUT_SHAPES["prompt_decoder"]) == [
        "image_embeddings",
        "point_encoding",
        "point_types",
    ]
    assert pipeline.COMPILED_INPUT_SHAPES["prompt_decoder"]["point_encoding"] == (1, 256, 1, 6)


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "invalid-json",
        "oversize",
        "version",
        "source",
        "hash",
        "shape",
        "string",
        "bool",
        "scale-string",
        "scale-bool",
        "scale",
        "nan",
    ],
)
def test_metadata_rejects_missing_malformed_and_incompatible_payloads(load_script, fault):
    pipeline = load_script("pipeline")
    value = metadata(pipeline)
    if fault == "version":
        value["version"] = True
    if fault == "source":
        value["source_sha256"] = "not-a-hash"
    if fault == "hash":
        value["gaussian_sha256"] = "b" * 64
    if fault == "shape":
        value["gaussian"][0].pop()
    if fault == "string":
        value["gaussian"][0][0] = "1.0"
    if fault == "bool":
        value["gaussian"][0][0] = True
    if fault == "scale-string":
        value["phase_scale"] = str(value["phase_scale"])
    if fault == "scale-bool":
        value["phase_scale"] = True
    if fault == "scale":
        value["phase_scale"] = 1.0
    if fault == "nan":
        value["gaussian"][0][0] = float("nan")
    payload = {pipeline.PROMPT_ENCODING_METADATA: json.dumps(value)}
    if fault == "missing":
        payload = {}
    if fault == "invalid-json":
        payload[pipeline.PROMPT_ENCODING_METADATA] = "{"
    if fault == "oversize":
        payload[pipeline.PROMPT_ENCODING_METADATA] = " " * 65537
    with pytest.raises(ValueError):
        pipeline.PromptEncoding.from_metadata(payload)


def test_encoding_owns_bounded_fp32_output_and_preserves_every_slot(load_script):
    pipeline = load_script("pipeline")
    table = encoder(pipeline, metadata(pipeline))
    coords = np.array([[-100, 200], [0, 0], [0, -0.0], [-100, 200], [1, 1], [-1, 1]], np.float32).T[None, :, None, :]
    before = coords.copy()
    actual = table(coords)
    assert actual.shape == (1, 256, 1, 6) and actual.dtype == np.float32 and actual.flags.c_contiguous
    assert np.isfinite(actual).all() and np.abs(actual).max() <= 1
    assert not table.gaussian.flags.writeable
    np.testing.assert_array_equal(coords, before)
    np.testing.assert_array_equal(actual[..., 0], actual[..., 3])
    # A separate true float32 fmaf per original slot is an independent reference for duplicate-row reuse.
    import ctypes

    fmaf = ctypes.CDLL("libm.so.6").fmaf
    fmaf.argtypes = (ctypes.c_float,) * 3
    fmaf.restype = ctypes.c_float
    expected = []
    for x, y in coords[0, :, 0, :].T:
        products = np.multiply(x, table.gaussian[0], dtype=np.float32)
        dots = np.array([fmaf(y, g, first) for g, first in zip(table.gaussian[1], products)], np.float32)
        phase = dots * table.phase_scale
        expected.append(np.concatenate((np.sin(phase), np.cos(phase))))
    np.testing.assert_array_equal(actual[0, :, 0, :].T.view(np.uint32), np.asarray(expected).view(np.uint32))


@pytest.mark.parametrize("fault", ["shape", "dtype", "nonfinite", "phase-overflow"])
def test_invalid_or_overflowing_coordinates_fail_clearly(load_script, fault):
    pipeline = load_script("pipeline")
    table = encoder(pipeline, metadata(pipeline))
    coords = np.zeros((1, 2, 1, 6), np.float32)
    if fault == "shape":
        coords = coords[..., :5]
    if fault == "dtype":
        coords = coords.astype(np.float64)
    if fault == "nonfinite":
        coords[0, 0, 0, 0] = np.inf
    if fault == "phase-overflow":
        coords.fill(np.finfo(np.float32).max)
    with pytest.raises(ValueError):
        table(coords)


def test_compiled_calibration_encodes_prompts_but_reference_retains_coordinates(load_script):
    pipeline = load_script("pipeline")
    recipe = load_script("compile")
    table = encoder(pipeline, metadata(pipeline))
    embeddings = np.zeros((1, 256, 64, 64), np.float32)
    source = recipe.decoder_feeds(embeddings, [(300, 400)])
    compiled = recipe.decoder_feeds(embeddings, [(300, 400)], table)
    assert len(source) == len(compiled) == 5
    for original, actual in zip(source, compiled):
        assert list(original) == ["image_embeddings", "point_coords", "point_types"]
        assert list(actual) == ["image_embeddings", "point_encoding", "point_types"]
        np.testing.assert_array_equal(actual["point_encoding"], table(original["point_coords"]))
        np.testing.assert_array_equal(actual["point_types"], original["point_types"])


def test_actual_decoder_carries_valid_fourier_table(compiled, load_script):
    import tarfile

    pipeline = load_script("pipeline")
    archives = list((compiled / "prompt_decoder").glob("*_mpk.tar.gz"))
    assert len(archives) == 1
    with tarfile.open(archives[0], "r:gz") as archive:
        value = json.load(archive.extractfile("metadata.json"))
    table = pipeline.PromptEncoding.from_metadata({key: json.dumps(item) for key, item in value.items()})
    assert table.gaussian.shape == (2, 128)


def test_reachable_large_phase_keeps_single_float32_fma_rounding(load_script):
    pipeline = load_script("pipeline")
    value = metadata(pipeline)
    # Regression coefficient pair from the source prefix: double-rounded emulation changes this mask's phase.
    gaussian = np.zeros((2, 128), np.float32)
    gaussian[:, 0] = [-0.48265618085861206, 1.1196969747543335]
    value["gaussian"] = gaussian.tolist()
    value["gaussian_sha256"] = hashlib.sha256(gaussian.astype("<f4").tobytes()).hexdigest()
    prompt = pipeline.prompt_inputs([[511.499969482421875, 844424930131968.0]], ["foreground"], (1024, 1024))
    actual = encoder(pipeline, value)(prompt["point_coords"])
    # Independently extracted ONNX Runtime FP32 source-prefix output, not calculated by the helper under test.
    assert actual[0, 0, 0, 0] == pytest.approx(-0.8333171, abs=1e-7)
    first = np.float32(prompt["point_coords"][0, 0, 0, 0] * gaussian[0, 0])
    second = np.float64(prompt["point_coords"][0, 1, 0, 0]) * np.float64(gaussian[1, 0])
    double_rounded = np.float32(np.float64(first) + second) * np.float32(2 * np.pi)
    assert abs(float(np.sin(double_rounded)) - float(actual[0, 0, 0, 0])) > 0.1


def fourier_source(pipeline):
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    value = metadata(pipeline)
    names = ["gather0", "gather1", "transpose", "matmul", "phase", "sin", "cos", "/prompt_decoder/Concat_output_0"]
    nodes = [
        helper.make_node("Gather", ["prompt_decoder_point_coords", "zero"], [names[0]], axis=0),
        helper.make_node("Gather", [names[0], "zero"], [names[1]], axis=1),
        helper.make_node("Transpose", [names[1]], [names[2]], perm=[1, 0]),
        helper.make_node("MatMul", [names[2], "gaussian"], [names[3]]),
        helper.make_node("Mul", [names[3], "scale"], [names[4]]),
        helper.make_node("Sin", [names[4]], [names[5]]),
        helper.make_node("Cos", [names[4]], [names[6]]),
        helper.make_node("Concat", names[5:7], [names[7]], axis=-1),
        helper.make_node("Identity", [names[7]], ["prompt_decoder_masks"]),
        helper.make_node("Identity", ["prompt_decoder_image_embeddings"], ["prompt_decoder_iou_predictions"]),
    ]
    tensors = [
        numpy_helper.from_array(np.asarray(0, np.int64), "zero"),
        numpy_helper.from_array(np.asarray(value["gaussian"], np.float32), "gaussian"),
        numpy_helper.from_array(np.asarray(value["phase_scale"], np.float32), "scale"),
    ]
    inputs = [
        helper.make_tensor_value_info("prompt_decoder_" + name, TensorProto.FLOAT, shape)
        for name, shape in pipeline.INPUT_SHAPES["prompt_decoder"].items()
    ]
    outputs = [
        helper.make_tensor_value_info("prompt_decoder_masks", TensorProto.FLOAT, [6, 256]),
        helper.make_tensor_value_info("prompt_decoder_iou_predictions", TensorProto.FLOAT, [1, 256, 64, 64]),
    ]
    return onnx.shape_inference.infer_shapes(
        helper.make_model(
            helper.make_graph(nodes, "source", inputs, outputs, tensors), opset_imports=[helper.make_opsetid("", 17)]
        )
    )


@pytest.mark.parametrize("fault", [None, "operation", "order", "scale", "shape"])
def test_surgery_cuts_only_verified_fourier_prefix_and_preserves_source(tmp_path, load_script, fault):
    import onnx
    from onnx import numpy_helper

    pipeline, surgery = load_script("pipeline"), load_script("surgery")
    model = fourier_source(pipeline)
    if fault == "operation":
        model.graph.node[5].op_type = "Cos"
    if fault == "order":
        model.graph.node[7].input.reverse()
    if fault == "scale":
        model.graph.initializer[2].CopyFrom(numpy_helper.from_array(np.float32(1), "scale"))
    if fault == "shape":
        model.graph.initializer[1].CopyFrom(numpy_helper.from_array(np.zeros((2, 127), np.float32), "gaussian"))
    source, output = tmp_path / "source.onnx", tmp_path / "compiled.onnx"
    onnx.save(model, source)
    original = source.read_bytes()
    if fault:
        with pytest.raises(ValueError):
            surgery.compile_decoder(source, output, "a" * 64)
        assert not output.exists()
    else:
        result = surgery.compile_decoder(source, output, "a" * 64)
        table = pipeline.PromptEncoding.from_metadata({key: json.dumps(value) for key, value in result.items()})
        assert table.source_sha256 == "a" * 64
        compiled_model = onnx.load(output)
        assert [item.name for item in compiled_model.graph.input] == [
            "prompt_decoder_image_embeddings",
            "prompt_decoder_point_encoding",
            "prompt_decoder_point_types",
        ]
        assert [item.dim_value for item in compiled_model.graph.input[1].type.tensor_type.shape.dim] == [1, 256, 1, 6]
        assert not any(node.op_type in ("Sin", "Cos") for node in compiled_model.graph.node)
        assert [item.name for item in compiled_model.graph.output] == [
            "prompt_decoder_masks",
            "prompt_decoder_iou_predictions",
        ]
        onnx.checker.check_model(compiled_model)
    assert source.read_bytes() == original


@pytest.mark.parametrize("retain", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_build_retention_is_optional_and_survives_compile_failure(load_script, monkeypatch, tmp_path, retain, fails):
    recipe = load_script("compile")
    calls = []
    retained = tmp_path / "retained" if retain else None

    def compile_component(**kwargs):
        calls.append(kwargs)
        if retain:
            Path(kwargs["retained_temporary_directory_name"], "compiler-diagnostic.txt").write_text("retained")
        if fails:
            raise SystemExit(1)
        Path(kwargs["output_path"], "test_mpk.tar.gz").write_bytes(b"mock archive")

    quantized = SimpleNamespace(compile=compile_component)
    monkeypatch.setattr(recipe, "onnx_source", lambda *args, **kwargs: None)
    monkeypatch.setattr(recipe, "load_model", lambda *args, **kwargs: SimpleNamespace(quantize=lambda **kw: quantized))
    monkeypatch.setattr(recipe, "_quantization", lambda parameters: None)
    monkeypatch.setattr(recipe, "calibration_data", lambda component, samples: None)
    monkeypatch.setattr(recipe, "audit_mpk", lambda archive, component: None)
    args = (tmp_path / "source.onnx", tmp_path / "build", "test", "image_encoder", {"precision": "w16a16"}, [], [])
    if fails:
        with pytest.raises(RuntimeError, match="compiling image_encoder failed"):
            recipe._build(*args, retain_build_dir=retained)
    else:
        archive, outputs = recipe._build(*args, retain_build_dir=retained)
        assert archive.read_bytes() == b"mock archive" and outputs == []
    expected = dict(output_path=str(tmp_path / "build"), batch_size=1, compress=True, log_level=recipe.logging.INFO)
    if retain:
        expected["retained_temporary_directory_name"] = str(retained)
        assert (retained / "compiler-diagnostic.txt").read_text() == "retained"
    assert calls == [expected]  # No changed verification, precision, compression or numerical options.


@pytest.mark.parametrize("kind", ["existing", "file", "symlink", "same", "child", "parent"])
def test_retention_paths_fail_before_component_work(load_script, monkeypatch, tmp_path, kind):
    recipe = load_script("compile")
    monkeypatch.setattr(recipe, "validate_source", lambda source: None)
    monkeypatch.setattr(recipe, "export_components", lambda *args: pytest.fail("component work must not start"))
    output = tmp_path / "out"
    retained = tmp_path / "retained"
    if kind == "existing":
        retained.mkdir()
    elif kind == "file":
        retained.write_text("do not replace")
    elif kind == "symlink":
        retained.symlink_to(tmp_path / "not-created", target_is_directory=True)
    elif kind == "same":
        retained = output
    elif kind == "child":
        retained = output / "diagnostics"
    elif kind == "parent":
        output = retained / "out"
    with pytest.raises((FileExistsError, ValueError)):
        recipe.compile_model(tmp_path / "model.onnx", output, "w16a16", None, None, retained)
    assert not output.exists()


@pytest.mark.parametrize("retain", [False, True])
def test_all_components_receive_distinct_persistent_directories(load_script, monkeypatch, tmp_path, retain):
    recipe = load_script("compile")
    monkeypatch.setattr(recipe, "validate_source", lambda source: None)
    monkeypatch.setattr(
        recipe, "_component_parameters", lambda _: {c: {"precision": "w16a16"} for c in recipe.COMPONENTS}
    )
    monkeypatch.setattr(recipe, "load_metadata", lambda _: {})

    def split(image_encoder, weights, directory):
        assert weights == image_encoder.with_name("weights.pth")
        directory.mkdir()
        graphs = {c: directory / f"{c}.onnx" for c in recipe.COMPONENTS}
        for graph in graphs.values():
            graph.write_bytes(b"mock source")
        return graphs

    def decoder(source, destination, source_sha):
        destination.write_bytes(b"mock compiled boundary")
        return {}

    monkeypatch.setattr(recipe, "export_components", split)
    monkeypatch.setattr(recipe, "compile_decoder", decoder)
    monkeypatch.setattr(
        recipe,
        "PromptEncoding",
        SimpleNamespace(from_metadata=lambda _: lambda coords: np.zeros((1, 256, 1, 6), np.float32)),
    )
    calls = []

    def build(*args):
        calls.append(args)
        directory = args[1]
        directory.mkdir()
        archive = directory / "test_mpk.tar.gz"
        archive.write_bytes(b"mock archive")
        if args[7] is not None:
            args[7].mkdir()
            (args[7] / "compiler-diagnostic.txt").write_text(args[3])
        return archive, []

    monkeypatch.setattr(recipe, "_in_process", build)
    source = tmp_path / "source.onnx"
    source.write_bytes(b"mock source")
    source.with_name("weights.pth").write_bytes(b"mock weights")
    retained = tmp_path / "retained" if retain else None
    outputs = recipe.compile_model(source, tmp_path / "out", "w16a16", None, None, retained)
    assert list(outputs) == list(recipe.COMPONENTS)
    import hashlib

    from model_registry.runtime_files import read_runtime_files

    runtime = read_runtime_files(tmp_path / "out")
    assert runtime == [{"path": recipe.PROMPT_ENCODING_FILE,
                        "sha256": hashlib.sha256(b"{}\n").hexdigest()}]
    assert [args[3] for args in calls] == list(recipe.COMPONENTS)
    assert [args[7] for args in calls] == [retained / c if retain else None for c in recipe.COMPONENTS]
    assert all(not args[1].exists() for args in calls)  # Ordinary temporary builds still close.
    if retain:
        assert {p.name for p in retained.iterdir()} == set(recipe.COMPONENTS)
        for component in recipe.COMPONENTS:
            assert (retained / component / "compiler-diagnostic.txt").read_text() == component


@pytest.mark.parametrize("retain", [False, True])
def test_retention_cli_forwarding(load_script, monkeypatch, tmp_path, retain):
    recipe = load_script("compile")
    source = tmp_path / "source.onnx"
    source.touch()
    source.with_name("weights.pth").touch()
    argv = [
        "compile.py",
        "--model",
        str(source),
        "--output-dir",
        str(tmp_path / "out"),
        "--build-profile",
        "w16a16",
        "--compile-invocation",
        "w16a16",
    ]
    if retain:
        argv += ["--retain-build-dir", str(tmp_path / "retained")]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(recipe, "load_metadata", lambda _: {})
    calls = []
    monkeypatch.setattr(recipe, "compile_model", lambda *args: calls.append(args))
    assert recipe.main() == 0
    assert len(calls) == 1 and calls[0][-1] == (tmp_path / "retained" if retain else None)


@pytest.mark.parametrize(
    "precision,bins,requires,per_token",
    [
        ("w16a16", None, False, False),
        ("w16a16", 2048, True, False),
        ("w8a8", 2048, True, False),
        ("w8a16", None, False, True),
        ("w8a16", 2048, True, True),
    ],
)
def test_explicit_precision_calibration_policy(load_script, monkeypatch, precision, bins, requires, per_token):
    recipe = load_script("compile")
    skip, histogram = object(), object()

    class Config:
        def __init__(self, **kwargs):
            self.values = kwargs

        def with_requantization_mode(self, mode):
            return self

    monkeypatch.setattr(recipe, "QuantizationParams", Config)
    monkeypatch.setattr(recipe, "SkipCalibration", lambda: skip)
    monkeypatch.setattr(recipe, "HistogramMSEMethod", lambda count: histogram)
    parameters = {"precision": precision, **({"calibration_bins": bins} if bins else {})}
    actual = recipe._quantization(parameters)
    assert recipe._requires_calibration(parameters) is requires
    assert actual.values["calibration_method"] is (histogram if requires else skip)
    assert actual.values["per_token_quant"] is per_token
    assert actual.values["node_names"] == {""}


@pytest.mark.parametrize("component", ["image_encoder", "prompt_decoder"])
def test_uncalibrated_bf16_passes_none_for_all_components(load_script, monkeypatch, tmp_path, component):
    recipe = load_script("compile")
    calls = []

    class Loaded:
        def quantize(self, **kwargs):
            calls.append(kwargs)
            raise RuntimeError("observed quantize boundary")

    monkeypatch.setattr(recipe, "load_model", lambda *args, **kwargs: Loaded())
    monkeypatch.setattr(recipe, "_quantization", lambda parameters: "BF16_SKIP_CONFIG")
    monkeypatch.setattr(recipe, "calibration_data", lambda *args: pytest.fail("BF16 must not construct data"))
    with pytest.raises(RuntimeError, match="observed quantize boundary"):
        recipe._build(tmp_path / "source.onnx", tmp_path / "build", "test", component, {"precision": "w16a16"}, [], [])
    assert calls[0]["calibration_data"] is None
    assert calls[0]["quantization_config"] == "BF16_SKIP_CONFIG"


def test_compiler_identity_is_checked_by_shared_artifact_verification_not_the_model_recipe(load_script, tmp_path):
    recipe = load_script("compile")
    recipe.audit_mpk(_mpk(tmp_path / "model.tar.gz", ["MLA"], "different-recorded-version"), "image_encoder")


@pytest.mark.parametrize("layout", ["flat", "components", "explicit"])
def test_prompt_encoding_uses_existing_package_supporting_files(load_script, tmp_path, layout):
    from model_registry.errors import ValidationError
    from model_registry.fetch import sha256_file
    from model_registry.runtime_files import read_runtime_files

    pipeline = load_script("pipeline")
    data = metadata(pipeline)
    root = tmp_path / "package"
    root.mkdir()
    path = root / pipeline.PROMPT_ENCODING_FILE
    path.write_text(json.dumps({pipeline.PROMPT_ENCODING_METADATA: data}))
    (root / "compile-outputs.json").write_text(json.dumps({
        "runtime_files": [{"path": path.name, "sha256": sha256_file(path)[0]}],
    }))
    assert read_runtime_files(root)[0]["path"] == path.name
    decoder = root / "decoder.tar.gz" if layout == "flat" else root / "prompt_decoder/decoder.tar.gz"
    resources = root if layout == "explicit" else None
    actual = pipeline.PromptEncoding.from_package(decoder, resources)
    expected = encoder(pipeline, data)
    np.testing.assert_array_equal(actual.gaussian, expected.gaussian)
    coords = np.random.RandomState(42).uniform(-1, 2, (1, 2, 1, 6)).astype(np.float32)
    np.testing.assert_array_equal(actual(coords), expected(coords))
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValidationError, match="SHA-256"):
        read_runtime_files(root)


@pytest.mark.parametrize("failure", ["missing", "ambiguous", "oversized", "json", "fields", "coefficients"])
def test_prompt_supporting_file_rejected_before_native_work(load_script, tmp_path, failure):
    pipeline = load_script("pipeline")
    root = tmp_path / "package"
    root.mkdir()
    decoder = root / "component/decoder.tar.gz"
    path = root / pipeline.PROMPT_ENCODING_FILE
    value = {pipeline.PROMPT_ENCODING_METADATA: metadata(pipeline)}
    path.write_text(json.dumps(value))
    if failure == "missing":
        path.unlink()
    elif failure == "ambiguous":
        other = decoder.parent / pipeline.PROMPT_ENCODING_FILE
        other.parent.mkdir()
        other.write_bytes(path.read_bytes())
    elif failure == "oversized":
        path.write_bytes(b" " * 65537)
    elif failure == "json":
        path.write_text("not JSON")
    elif failure == "fields":
        path.write_text('{"unexpected": 1}')
    else:
        value[pipeline.PROMPT_ENCODING_METADATA]["gaussian"][0][0] += 1
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        pipeline.PromptEncoding.from_package(decoder)


def test_decoder_calibration_uses_the_whole_quantized_encoders_own_embeddings(load_script, monkeypatch, tmp_path):
    """Do not silently substitute the FP32 encoder while fitting the second component."""
    from types import SimpleNamespace

    recipe = load_script("compile")
    monkeypatch.setattr(recipe, "validate_source", lambda _: None)
    parameters = {name: {"precision": "w16a16", "calibration_bins": 2048, "calibration_samples": 1}
                  for name in recipe.COMPONENTS}
    monkeypatch.setattr(recipe, "_component_parameters", lambda _: parameters)
    monkeypatch.setattr(recipe, "load_metadata", lambda _: {})
    monkeypatch.setattr(recipe, "calibration_images", lambda *args: [np.zeros((20, 40, 3), np.uint8)])
    monkeypatch.setattr(recipe, "encoder_inputs", lambda images: np.zeros((1, 8, 8, 3), np.float32))

    def split(image_encoder, weights, directory):
        assert weights == image_encoder.with_name("weights.pth")
        directory.mkdir()
        graphs = {name: directory / f"{name}.onnx" for name in recipe.COMPONENTS}
        for path in graphs.values():
            path.write_bytes(b"fixture")
        return graphs

    def cut(source, output, source_sha):
        output.write_bytes(b"fixture decoder")
        return {}

    monkeypatch.setattr(recipe, "export_components", split)
    monkeypatch.setattr(recipe, "compile_decoder", cut)
    monkeypatch.setattr(recipe, "PromptEncoding", SimpleNamespace(from_metadata=lambda _: "encoding"))
    monkeypatch.setattr(recipe, "float_embeddings", lambda *args: pytest.fail("FP32 must not supply calibration"))
    own = np.full((1, 256, 64, 64), 0.375, np.float32)
    feeds = []

    def decode_feeds(embeddings, sizes, encoding):
        np.testing.assert_array_equal(embeddings, own)
        assert sizes == [(20, 40)] and encoding == "encoding"
        return [{"image_embeddings": embeddings}]

    monkeypatch.setattr(recipe, "decoder_feeds", decode_feeds)

    def build(*args):
        component, samples, executed = args[3], args[5], args[6]
        feeds.append(component)
        args[1].mkdir()
        path = args[1] / "fixture_mpk.tar.gz"
        path.write_bytes(b"fixture archive")
        if component == "image_encoder":
            assert len(samples) == len(executed) == 1
            return path, [[own]]
        assert len(samples) == 1 and not executed
        np.testing.assert_array_equal(samples[0]["image_embeddings"], own)
        return path, []

    monkeypatch.setattr(recipe, "_in_process", build)
    source = tmp_path / "source.onnx"
    source.write_bytes(b"fixture source")
    source.with_name("weights.pth").write_bytes(b"fixture weights")
    archives = recipe.compile_model(source, tmp_path / "output", "w16a16", tmp_path / "calibration", None)
    assert list(archives) == feeds == ["image_encoder", "prompt_decoder"]


def test_model_has_exactly_two_components_without_a_split_encoder_fallback(load_script):
    recipe = load_script("compile")
    model = recipe.load_metadata(recipe.MODEL_DIR)
    assert recipe.COMPONENTS == ("image_encoder", "prompt_decoder")
    assert [component["id"] for component in model["components"]] == list(recipe.COMPONENTS)
    assert set(recipe.INPUT_SHAPES["image_encoder"]) == {"image"}
    assert set(recipe.OUTPUT_SHAPES["image_encoder"]) == {"image_embeddings"}
