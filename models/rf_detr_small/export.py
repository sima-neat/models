"""Export the pinned RF-DETR Small checkpoint to ONNX."""

from __future__ import annotations

from copy import deepcopy
from importlib.metadata import version
from pathlib import Path

import numpy as np
import onnx
from PIL import Image

PACKAGE_VERSION = "1.9.4"
INTERFACE = {
    "input": [1, 3, 512, 512],
    "dets": [1, 300, 4],
    "labels": [1, 300, 91],
}


def _shape(value: onnx.ValueInfoProto) -> list[int]:
    return [dimension.dim_value for dimension in value.type.tensor_type.shape.dim]


def validate_export(path: Path) -> None:
    model = onnx.load(path)
    onnx.checker.check_model(model)
    if [item.version for item in model.opset_import if not item.domain] != [17]:
        raise RuntimeError("RF-DETR export must use ONNX opset 17")
    values = {value.name: value for value in (*model.graph.input, *model.graph.output)}
    if set(values) != set(INTERFACE):
        raise RuntimeError(f"unexpected RF-DETR interface: {sorted(values)}")
    for name, expected_shape in INTERFACE.items():
        value = values[name]
        if value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT or _shape(value) != expected_shape:
            raise RuntimeError(f"unexpected RF-DETR tensor {name}: dtype or shape does not match {expected_shape}")


def _validation_image() -> np.ndarray:
    return np.random.RandomState(123).randint(0, 256, size=(512, 512, 3), dtype=np.uint8)


def validate_numerical_export(detector: object, path: Path, validation_image: Path) -> None:
    import onnxruntime as ort
    import torch
    from rfdetr.export.main import make_infer_image

    source_model = detector.model.model
    if source_model is None:
        raise RuntimeError("RF-DETR exporter did not retain its framework model")
    sample = make_infer_image(str(validation_image), (512, 512), 1, "cpu").numpy()
    model = deepcopy(source_model).cpu().eval()
    with torch.no_grad():
        output = model(torch.from_numpy(sample))
    expected = [output["pred_boxes"].numpy(), output["pred_logits"].numpy()]
    actual = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"input": sample})
    for name, expected_value, actual_value in zip(("dets", "labels"), expected, actual, strict=True):
        if not np.allclose(actual_value, expected_value, rtol=1e-3, atol=1e-3):
            maximum_error = float(np.max(np.abs(actual_value - expected_value)))
            raise RuntimeError(f"RF-DETR ONNX export does not match the framework output {name}: {maximum_error}")


def export_model(checkpoint: Path, output_dir: Path) -> Path:
    from rfdetr import RFDETRSmall

    if version("rfdetr") != PACKAGE_VERSION:
        raise RuntimeError(f"RF-DETR exporter must be version {PACKAGE_VERSION}")
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_image = output_dir / "validation.png"
    Image.fromarray(_validation_image()).save(validation_image)
    detector = RFDETRSmall(pretrain_weights=str(checkpoint), resolution=512)
    exported = detector.export(
        output_dir=str(output_dir),
        infer_dir=str(validation_image),
        shape=(512, 512),
        batch_size=1,
        dynamic_batch=False,
        opset_version=17,
        format="onnx",
        fp16=False,
        verbose=False,
    )
    path = Path(exported).resolve()
    if not path.is_file():
        raise RuntimeError(f"RF-DETR exporter did not create an ONNX model: {path}")
    validate_export(path)
    validate_numerical_export(detector, path, validation_image)
    return path
