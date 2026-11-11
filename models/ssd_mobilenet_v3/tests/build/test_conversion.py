"""Independent export contracts; heavy source/ORT parity is run by convert.py."""

import numpy as np
import onnx
import pytest


@pytest.fixture
def converter(load_script):
    return load_script("convert")


def test_weight_digest_is_full_and_fail_closed(tmp_path, converter):
    path = tmp_path / "wrong.pth"
    path.write_bytes(b"not the official weights")
    assert len(converter.WEIGHTS_SHA256) == 64
    with pytest.raises(ValueError, match="digest mismatch"):
        converter.verified_weights(path)


def test_raw_heads_have_3234_anchors_and_91_logits(load_script):
    converter = load_script("surgery")
    assert len(converter.OUTPUTS) == 12
    assert [name for name, _shape in converter.OUTPUTS] == [
        *[f"ssd_loc_{index}" for index in range(6)],
        *[f"ssd_confidence_{index}" for index in range(6)],
    ]
    assert [shape[1] for _name, shape in converter.OUTPUTS] == [24] * 6 + [546] * 6
    assert sum(shape[2] * shape[3] * 6 for _name, shape in converter.OUTPUTS[:6]) == 3234


def test_flatten_is_cell_anchor_coordinate_order_not_channel_major(converter):
    # Independent coordinate construction: each scalar labels (cell,anchor,field).
    head = np.zeros((1, 24, 2, 2), np.float32)
    expected = []
    for row in range(2):
        for column in range(2):
            for anchor in range(6):
                values = []
                for coordinate in range(4):
                    value = 1000 * row + 100 * column + 10 * anchor + coordinate
                    head[0, anchor * 4 + coordinate, row, column] = value
                    values.append(value)
                expected.append(values)
    np.testing.assert_array_equal(converter.flatten_heads([head], 4), np.array([expected], np.float32))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_parity_rejects_nonfinite(bad, converter):
    with pytest.raises(ValueError, match="nonfinite"):
        converter.parity(np.array([1.0]), np.array([bad]))


def test_parity_rejects_changed_shape_or_large_error(converter):
    with pytest.raises(ValueError, match="shape"):
        converter.parity(np.ones(1), np.ones(2))
    assert converter.parity(np.ones(1), np.array([1.01]))["max_abs"] > converter.MAX_ABSOLUTE_ERROR
    assert converter.MAX_ABSOLUTE_ERROR == 5e-4


def test_actual_export_keeps_full_upstream_fp32_contract(converted, converter):
    report = converter.audit(converted)
    assert report["operators"]["Conv"] == 98
    assert report["operators"]["BatchNormalization"] == 70
    assert report["operators"]["Softmax"] == 1
    model = onnx.load(converted)
    properties = {item.key: item.value for item in model.metadata_props}
    assert properties["sima.precision.source"] == "fp32_pretrained_not_qat"
    assert properties["sima.input.normalization"] == "rgb_float32_0_to_1; normalization inside graph"
    assert properties["sima.torchvision.weights_sha256"] == converter.WEIGHTS_SHA256


@pytest.mark.parametrize("case", ["names", "order", "dtype", "shape", "opset"])
def test_actual_export_corruption_rejected(case, tmp_path, converted, converter):
    model = onnx.load(converted)
    if case == "names":
        # Rename both producer and public binding to retain ONNX validity.
        old = model.graph.output[0].name
        model.graph.output[0].name = "wrong_identity"
        for node in model.graph.node:
            for index, output in enumerate(node.output):
                if output == old:
                    node.output[index] = "wrong_identity"
    elif case == "order":
        first, second = onnx.ValueInfoProto(), onnx.ValueInfoProto()
        first.CopyFrom(model.graph.output[0])
        second.CopyFrom(model.graph.output[1])
        model.graph.output[0].CopyFrom(second)
        model.graph.output[1].CopyFrom(first)
    elif case == "dtype":
        model.graph.output[0].type.tensor_type.elem_type = onnx.TensorProto.DOUBLE
    elif case == "shape":
        model.graph.output[0].type.tensor_type.shape.dim[1].dim_value = 3
    else:
        model.opset_import[0].version = 18
    path = tmp_path / "corrupted.onnx"
    onnx.save(model, path)
    with pytest.raises((ValueError, onnx.checker.ValidationError, onnx.shape_inference.InferenceError)):
        converter.audit(path)


def test_official_weights_use_the_canonical_verified_fetcher(monkeypatch, converter, tmp_path):
    from model_registry.source import SourceSpec

    path = tmp_path / "weights.pth"
    seen = []

    class FakeFetcher:
        def fetch(self, source):
            assert isinstance(source, SourceSpec)
            seen.append(source)
            return path, True

    monkeypatch.setattr(
        converter,
        "load_and_validate_model",
        lambda _: {"conversion": {"default": {"source": {"type": "file", **converter.WEIGHTS_SOURCE}}}},
    )
    monkeypatch.setattr(converter, "Fetcher", FakeFetcher)
    monkeypatch.setattr(converter, "verified_weights", lambda value: value)
    assert converter.resolve_weights(None) == path
    assert seen[0].uri == converter.WEIGHTS_URL
    assert seen[0].sha256 == converter.WEIGHTS_SHA256
    assert seen[0].size_bytes == 14069355


def test_explicit_local_weights_never_trigger_network(monkeypatch, converter, tmp_path):
    class NoFetcher:
        def __init__(self):
            raise AssertionError("an explicit local file must not fetch a fallback")

    monkeypatch.setattr(converter, "Fetcher", NoFetcher)
    path = tmp_path / "wrong.pth"
    path.write_bytes(b"wrong weights")
    with pytest.raises(ValueError, match="digest mismatch"):
        converter.resolve_weights(path)


@pytest.mark.parametrize(("machine", "vision"), [("x86_64", "0.23.0+cpu"), ("aarch64", "0.23.0")])
def test_conversion_architectures_pin_official_cpu_wheel_versions(converter, machine, vision):
    assert converter.expected_versions(machine) == {**converter.EXPECTED_VERSIONS, "torchvision": vision}


def test_conversion_rejects_unpinned_architecture(converter):
    with pytest.raises(ValueError, match="unsupported conversion architecture"):
        converter.expected_versions("unknown")


@pytest.fixture
def reference():
    # Reuse the test-only source oracle, never the production accuracy entry point.
    from models.ssd_mobilenet_v3.tests.benchmark.test_accuracy import SourceReference

    return SourceReference()


@pytest.fixture
def calibration(load_script):
    return load_script("calibration")


def _png(pixels):
    import io

    from PIL import Image

    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format="PNG")
    return stream.getvalue()


def test_source_priors_and_postprocessing_against_torchvision(converted, reference):
    import torch
    import torchvision
    from torchvision.models.detection.image_list import ImageList

    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None).eval()
    features = [torch.empty((1, 1, size, size)) for size in reference.GRIDS]
    anchors = model.anchor_generator(ImageList(torch.empty((1, 3, 320, 320)), [(320, 320)]), features)[0]
    np.testing.assert_array_equal(reference.ANCHORS, anchors.numpy())
    reg = np.random.default_rng(1).normal(0, 0.05, (3234, 4)).astype(np.float32)
    logits = np.full((3234, 91), -100, np.float32)
    logits[:, 0] = 20
    for index, label, score in [(0, 1, 7), (300, 2, 8), (1100, 90, 9)]:
        logits[index, 0] = 0
        logits[index, label] = score
    actual = reference.post_numpy(reg, logits)
    expected = model.postprocess_detections(
        {"bbox_regression": torch.from_numpy(reg)[None], "cls_logits": torch.from_numpy(logits)[None]},
        [anchors],
        [(320, 320)],
    )[0]
    np.testing.assert_array_equal(actual["labels"], expected["labels"].numpy())
    np.testing.assert_allclose(actual["scores"], expected["scores"].numpy(), atol=2e-7, rtol=0)
    np.testing.assert_allclose(actual["boxes"], expected["boxes"].numpy(), atol=4e-5, rtol=0)


def test_rgb_and_half_pixel_normalization_before_resize(converted, calibration):
    import torch
    from torchvision.models.detection.transform import GeneralizedRCNNTransform

    y, x = np.indices((450, 600))
    pixels = np.stack([x % 256, y % 256, (x + y) % 256], axis=-1).astype(np.uint8)
    actual = calibration.preprocess_image(_png(pixels))
    transform = GeneralizedRCNNTransform(320, 320, [0.5] * 3, [0.5] * 3, size_divisible=1, fixed_size=(320, 320)).eval()
    image = torch.from_numpy(pixels).permute(2, 0, 1).float() / 255
    expected = transform([image])[0].tensors[0].permute(1, 2, 0).numpy()
    assert actual.shape == (320, 320, 3) and actual.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)
    red = np.zeros((17, 29, 3), np.uint8)
    red[:, :, 0] = 255
    np.testing.assert_array_equal(calibration.preprocess_image(_png(red))[160, 160], [1, -1, -1])


@pytest.mark.parametrize("field", ["sha256", "size_bytes", "uri"])
def test_checkpoint_rejects_changed_identity_before_fetch(monkeypatch, converter, field):
    source = {"type": "file", **converter.WEIGHTS_SOURCE}
    source[field] = {"sha256": "0" * 64, "size_bytes": 1, "uri": "https://example.invalid/weights.pth"}[field]
    monkeypatch.setattr(converter, "load_and_validate_model", lambda _: {"conversion": {"default": {"source": source}}})

    class NoFetcher:
        def __init__(self):
            raise AssertionError("a rejected checkpoint identity must not fetch")

    monkeypatch.setattr(converter, "Fetcher", NoFetcher)
    with pytest.raises(ValueError):
        converter.resolve_weights(None)


def test_actual_surgery_cut_keeps_twelve_heads_and_leaves_folding_to_compile(converted, load_script):
    from collections import Counter

    surgery = load_script("surgery")
    surgery.audit_source(converted)
    raw = surgery.cut_raw_heads(onnx.load(converted))
    onnx.checker.check_model(raw, full_check=True)
    assert [value.name for value in raw.graph.input] == ["images"]
    assert [value.name for value in raw.graph.output] == [name for name, _ in surgery.OUTPUTS]
    assert [tuple(d.dim_value for d in value.type.tensor_type.shape.dim) for value in raw.graph.output] == [
        shape for _, shape in surgery.OUTPUTS
    ]
    operators = Counter(node.op_type for node in raw.graph.node)
    assert operators["Conv"] == 98 and operators["BatchNormalization"] == 70
    assert not any(operators[name] for name in ("Softmax", "TopK", "Resize", "NonMaxSuppression"))


@pytest.mark.parametrize("boundary", ["input", "head"])
def test_actual_surgery_rejects_missing_exact_boundary(converted, load_script, boundary):
    graph = onnx.load(converted)
    convolutions = [node for node in graph.graph.node if node.op_type == "Conv"]
    if boundary == "input":
        convolutions[0].name = "/unexpected/first/Conv"
    else:
        next(
            node for node in convolutions if node.name.endswith("regression_head/module_list.0/module_list.0.1/Conv")
        ).output[0] = "/unexpected/head/Conv_output_0"
    with pytest.raises(ValueError, match="missing unique"):
        load_script("surgery").cut_raw_heads(graph)


@pytest.mark.parametrize("case", ["count", "nonfinite", "error", "shape"])
def test_kept_head_parity_fails_closed(converter, case):
    heads = [np.zeros((1, channels, 1, 1), np.float32) for channels in ([24] * 6 + [546] * 6)]
    expected = {"bbox_regression": np.zeros((1, 36, 4), np.float32), "cls_logits": np.zeros((1, 36, 91), np.float32)}
    if case == "count":
        heads.pop()
    elif case == "nonfinite":
        heads[0][0, 0, 0, 0] = np.nan
    elif case == "error":
        heads[0][0, 0, 0, 0] = 0.01
    else:
        expected["bbox_regression"] = np.zeros((1, 35, 4), np.float32)
    with pytest.raises(ValueError):
        converter.compare_heads(expected, heads)


@pytest.fixture
def real_source_outputs(converted, converter, coco_source_images):
    """Run the published full source and upstream forward on the same real images."""
    import onnxruntime as ort
    import torch
    from PIL import Image

    torch.set_num_threads(1)
    detector = converter.load_detector(converter.resolve_weights(None))
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    # Source conformance keeps the original operations; optional backend fusions
    # can perturb near-tied NMS ordering without changing the saved ONNX.
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    probe = onnx.load(converted)
    probe.graph.output.extend(onnx.helper.make_empty_tensor_value_info(name) for name in converter.HEADS)
    session = ort.InferenceSession(probe.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])
    results = []
    for path in coco_source_images:
        with Image.open(path) as image:
            rgb = np.array(image.convert("RGB"), copy=True)
        pixels = torch.from_numpy(rgb).permute(2, 0, 1).float()[None] / 255
        image = torch.nn.functional.interpolate(pixels, size=(320, 320), mode="bilinear", align_corners=False)
        with torch.inference_mode():
            expected = {key: value.numpy() for key, value in detector([image[0]])[0].items()}
        actual = session.run(None, {converter.SOURCE_INPUT: image.numpy()})
        results.append((path, expected, actual))
    return results


def test_actual_export_final_detections_match_upstream_on_real_coco(converted, converter, real_source_outputs):
    # Final ordered labels/counts are exact; one absolute FP32 tolerance applies
    # to final boxes and scores. A real NMS-selection change fails, not warns.
    for path, expected, actual in real_source_outputs:
        np.testing.assert_array_equal(actual[2], expected["labels"], err_msg=path.name)
        for key, observed in zip(("boxes", "scores"), actual[:2], strict=True):
            np.testing.assert_allclose(
                observed, expected[key], atol=converter.MAX_ABSOLUTE_ERROR, rtol=0, err_msg=f"{path.name}: {key}"
            )


def test_actual_prepared_graph_matches_full_source_heads_on_real_coco(
    converted, converter, calibration, load_script, real_source_outputs, tmp_path
):
    import onnxruntime as ort

    surgery = load_script("surgery")
    prepared = surgery.prepare_model(converted, tmp_path / "prepared.onnx")
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    # Source conformance keeps the original operations; optional backend fusions
    # can perturb near-tied NMS ordering without changing the saved ONNX.
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(prepared), sess_options=options, providers=["CPUExecutionProvider"])
    for path, _expected, full in real_source_outputs:
        image = calibration.preprocess_image(path.read_bytes()).transpose(2, 0, 1)[None]
        outputs = session.run(None, {surgery.INPUT_NAME: image})
        assert len(outputs) == len(full[3:]) == 12
        for (name, _shape), expected, actual in zip(surgery.OUTPUTS, full[3:], outputs, strict=True):
            np.testing.assert_allclose(
                actual, expected, atol=converter.MAX_ABSOLUTE_ERROR, rtol=0, err_msg=f"{path.name}: {name}"
            )
