"""Checks of the conversion's interface and of the CPU decode the application and the DevKit stages share."""

from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import onnx
import pytest
from model_registry.dependencies import onnx_opset_version

OPSET = onnx_opset_version()


@pytest.fixture
def converter(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("convert")


@pytest.fixture
def keypoints(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("keypoints")


def _logits(keypoints: ModuleType, cells: dict[tuple[int, int], int], strength: float = 12.0) -> np.ndarray:
    """Keypoint logits that put every cell in the dustbin except the given (row, column) -> position-in-cell ones."""
    logits = np.zeros((keypoints.KEYPOINT_CHANNELS, keypoints.HEIGHT // 8, keypoints.WIDTH // 8), np.float32)
    logits[-1] = strength
    for (row, column), position in cells.items():
        logits[position, row, column] = 2 * strength
    return logits


def test_decode_places_keypoints_inside_their_cells(keypoints: ModuleType) -> None:
    # Cell (10, 20), position 3 * 8 + 5: pixel x = 20 * 8 + 5, y = 10 * 8 + 3.
    logits = _logits(keypoints, {(10, 20): 3 * 8 + 5, (30, 40): 0})
    descriptors = np.ones((keypoints.DESCRIPTOR_SIZE, 60, 80), np.float32)

    points, scores, unit = keypoints.decode(logits, descriptors)

    np.testing.assert_array_equal(points, [[165, 83], [320, 240]])
    assert (scores > 0.99).all()
    np.testing.assert_allclose(np.linalg.norm(unit, axis=1), 1, rtol=1e-6)


def test_decode_drops_the_border_and_keeps_the_best_first(keypoints: ModuleType) -> None:
    # Position 0 of cell (0, 0) is pixel (0, 0), inside the 4-pixel border.
    logits = _logits(keypoints, {(0, 0): 0, (10, 10): 0, (20, 20): 0})
    logits[0, 20, 20] += 1.0
    descriptors = np.ones((keypoints.DESCRIPTOR_SIZE, 60, 80), np.float32)

    points, scores, _ = keypoints.decode(logits, descriptors, limit=1)

    np.testing.assert_array_equal(points, [[160, 160]])
    assert len(keypoints.decode(logits, descriptors)[0]) == 2


def test_nms_keeps_one_keypoint_per_radius(keypoints: ModuleType) -> None:
    scores = np.zeros((32, 32), np.float32)
    scores[10, 10], scores[10, 13], scores[10, 20] = 0.9, 0.5, 0.4

    kept = keypoints.simple_nms(scores, 4)

    assert {tuple(point) for point in np.argwhere(kept > 0)} == {(10, 10), (10, 20)}


def test_descriptors_sample_bilinearly_between_cells(keypoints: ModuleType) -> None:
    dense = np.zeros((2, 60, 80), np.float32)
    dense[0, :, :40], dense[1, :, 40:] = 1.0, 1.0
    # transformers maps x = 3.5 to cell 0 and x = 635.5 + 3.5 to cell 79, so the middle lies halfway between 39 and 40.
    halfway = np.array([[3.5 + 635.5 / 2, 100.0]], np.float32)

    sampled = keypoints.sample_descriptors(halfway, dense)

    np.testing.assert_allclose(sampled, [[2 ** -0.5, 2 ** -0.5]], atol=1e-3)


def test_preprocess_makes_a_640x480_grayscale_input(keypoints: ModuleType) -> None:
    image = np.zeros((100, 200, 3), np.uint8)
    image[..., 0] = 255

    model_input = keypoints.preprocess(image)

    assert model_input.shape == (480, 640) and model_input.dtype == np.float32
    np.testing.assert_allclose(model_input, 0.2989, rtol=1e-4)


def test_synthetic_scenes_are_seeded(keypoints: ModuleType) -> None:
    np.testing.assert_array_equal(keypoints.synthetic_image(3), keypoints.synthetic_image(3))
    assert not np.array_equal(keypoints.synthetic_image(3), keypoints.synthetic_image(4))


@pytest.mark.parametrize("case", ["valid", "names", "shape", "opset"])
def test_fixed_onnx_interface(tmp_path: Path, case: str, converter: ModuleType) -> None:
    helper = onnx.helper
    interface = dict(converter.INTERFACE)
    if case == "shape":
        interface["descriptors"] = [1, 128, 60, 80]
    names = list(interface)
    if case == "names":
        names[1] = "scores"
    values = [helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)
              for name, shape in zip(names, interface.values())]
    nodes = [helper.make_node("Identity", [names[0]], [name]) for name in names[1:]]
    opset = OPSET + 1 if case == "opset" else OPSET
    model = helper.make_model(
        helper.make_graph(nodes, "fixture", values[:1], values[1:]), opset_imports=[helper.make_opsetid("", opset)]
    )
    path = tmp_path / "model.onnx"
    onnx.save(model, path)

    if case == "valid":
        converter.validate_export(path)
    else:
        with pytest.raises(ValueError):
            converter.validate_export(path)


def test_source_must_be_the_pinned_checkpoint(converter: ModuleType) -> None:
    converter.validate_source({"uri": "hf://models/magic-leap-community/superpoint@rev/model.safetensors"})
    with pytest.raises(ValueError, match="Expected the checkpoint file"):
        converter.validate_source({"uri": "hf://models/magic-leap-community/superpoint@rev/pytorch_model.bin"})


def test_exported_heads_decode_to_the_transformers_keypoints(converter: ModuleType) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    model = transformers.SuperPointForKeypointDetection(transformers.SuperPointConfig(**converter.CONFIG)).eval()
    # transformers' initialization scores nearly every pixel 1/65, where float rounding breaks NMS ties differently
    # in torch and NumPy. Larger weights give a confident head, like a trained one; compare keypoints above 1/65.
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer, torch.nn.Conv2d):
                torch.nn.init.normal_(layer.weight, 0, 3 / layer.weight[0].numel() ** 0.5)
    model.keypoint_decoder.keypoint_threshold = 0.02
    heads = converter.build_heads(model)
    image = converter.synthetic_image(converter.SEEDS.start)

    with torch.inference_mode():
        logits, descriptors = (value.numpy() for value in heads(torch.from_numpy(image[None, None])))
    count, error = converter.compare_with_reference(model, image, logits, descriptors)

    assert count > 0 and error <= converter.DESCRIPTOR_TOLERANCE


def test_converted_model_has_the_declared_interface(converted: Path, converter: ModuleType) -> None:
    converter.validate_export(converted)
