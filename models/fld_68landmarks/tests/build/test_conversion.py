"""Checks of convert.py's source and export checks, and of the converted ONNX when the conversion stage supplies it."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def convert(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("convert")


def test_source_must_be_the_pinned_2dfan2_alt_weights(convert: ModuleType) -> None:
    uri = "hf://models/public-data/ibug-face-alignment@8aef0e928499d334364f4c98e33dc6c4ffc8111a/fan/2dfan2_alt.pth"
    convert.validate_source({"type": "file", "uri": uri})
    with pytest.raises(ValueError, match="must pin exactly fan/2dfan2_alt.pth"):
        convert.validate_source({"type": "file", "uri": uri.replace("2dfan2_alt", "2dfan4")})


def test_network_has_ibug_2dfan2_alt_layout(convert: ModuleType) -> None:
    pytest.importorskip("torch")
    network = convert.build_network()
    names = set(network.state_dict())
    # Two hourglass modules of depth 4, each followed by its heatmap head; only the first feeds the second.
    assert {"conv1.weight", "m0.b2_plus_1.conv1.weight", "m1.b1_4.bn1.running_mean", "l1.weight"} <= names
    assert {"bl0.weight", "al0.weight"} <= names and "bl1.weight" not in names
    assert round(sum(p.numel() for p in network.parameters()) / 1e6, 2) == 12.11


def test_converted_model_has_the_declared_interface(converted: Path, convert: ModuleType) -> None:
    convert.validate_export(converted)


def test_real_images_match_pinned_source(converted, coco_source_images, convert, load_script):
    """Export parity on real pixels; these generic crops are not landmark-accuracy cases."""
    import numpy as np
    import onnxruntime as ort
    import torch
    from model_registry.fetch import Fetcher
    from model_registry.model import load_and_validate_model
    from PIL import Image

    source = load_and_validate_model(convert.MODEL_DIR)["conversion"]["default"]["source"]
    convert.validate_source(source)
    checkpoint, *_ = Fetcher().fetch_source(source)
    network = convert.load(checkpoint)
    landmarks = load_script("landmarks")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(converted), sess_options=options, providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        for path in coco_source_images:
            pixels = np.asarray(Image.open(path).convert("RGB"))
            height, width = pixels.shape[:2]
            cropped, _ = landmarks.crop(pixels, np.array([0, 0, width, height]))
            inputs = np.ascontiguousarray(cropped.transpose(2, 0, 1)[None])
            expected = network(torch.from_numpy(inputs)).numpy()
            actual = session.run([convert.OUTPUT_NAME], {convert.INPUT_NAME: inputs})[0]
            assert np.isfinite(actual).all()
            np.testing.assert_allclose(actual, expected, rtol=0, atol=convert.MAX_HEATMAP_ERROR, err_msg=path.name)
