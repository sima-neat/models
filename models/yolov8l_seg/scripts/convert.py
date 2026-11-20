"""Export YOLOv8l-seg from its Ultralytics checkpoint and check ONNX against PyTorch."""

import argparse
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath

import numpy as np
import onnx
from model_registry.fetch import Fetcher
from model_registry.model import load_and_validate_model

MODEL_DIR = Path(__file__).resolve().parents[1]

CHECKPOINT = "yolov8l-seg.pt"
EXPORTER = "8.4.0"
IMAGE_SIZE = 640
CLASS_COUNT = 80
MASK_CHANNELS = 32
ANCHOR_COUNT = sum((IMAGE_SIZE // stride) ** 2 for stride in (8, 16, 32))
INTERFACE = {
    "images": [1, 3, IMAGE_SIZE, IMAGE_SIZE],
    "output0": [1, 4 + CLASS_COUNT + MASK_CHANNELS, ANCHOR_COUNT],
    "output1": [1, MASK_CHANNELS, IMAGE_SIZE // 4, IMAGE_SIZE // 4],
}
SEEDS = range(123, 133)


def validate_source(source):
    declared = (source.get("type"), PurePosixPath(str(source.get("uri", ""))).name)
    if declared != ("file", CHECKPOINT):
        raise ValueError(f"Expected the checkpoint file {CHECKPOINT}, model.yaml declares {declared}")


def validate_export(path):
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    if [item.version for item in model.opset_import if not item.domain] != [12]:
        raise ValueError("YOLOv8 export requires opset 12")
    values = [*model.graph.input, *model.graph.output]
    if [value.name for value in values] != list(INTERFACE):
        raise ValueError("Expected input images and outputs output0 and output1")
    for value in values:
        tensor = value.type.tensor_type
        shape = [d.dim_value for d in tensor.shape.dim]
        if tensor.elem_type != onnx.TensorProto.FLOAT or shape != INTERFACE[value.name]:
            raise ValueError(f"Unexpected dtype or shape for {value.name}")
    if any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in model.graph.initializer):
        raise ValueError("This conversion workflow requires one self-contained ONNX file")


def compare_outputs(expected, actual):
    """Check the complete upstream predictions and prototypes, before compiler-specific surgery."""
    for name, reference, value in zip(("output0", "output1"), expected, actual):
        if value.shape != reference.shape or not np.isfinite(value).all():
            raise ValueError(f"Unexpected or non-finite {name}: {value.shape} vs {reference.shape}")
        np.testing.assert_allclose(value, reference, rtol=1e-3, atol=1e-3, err_msg=name)


def convert(source, output_dir):
    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "model.onnx"
    with tempfile.TemporaryDirectory(prefix="yolov8l-seg-") as temporary:
        # Ultralytics must not install packages or reach the network, and keeps its settings here.
        os.environ.update({"YOLO_AUTOINSTALL": "False", "YOLO_OFFLINE": "True", "YOLO_CONFIG_DIR": temporary})
        import onnxruntime as ort
        import torch
        import ultralytics
        from ultralytics import YOLO

        if ultralytics.__version__ != EXPORTER:
            raise RuntimeError(f"Expected Ultralytics {EXPORTER}, found {ultralytics.__version__}")
        torch.set_num_threads(4)
        # The exporter writes next to the checkpoint, so it gets a private copy.
        checkpoint = Path(temporary) / CHECKPOINT
        shutil.copyfile(source, checkpoint)
        segmenter = YOLO(str(checkpoint))
        exported = segmenter.export(
            format="onnx",
            imgsz=IMAGE_SIZE,
            batch=1,
            device="cpu",
            opset=12,
            half=False,
            dynamic=False,
            simplify=False,
            nms=False,
        )
        shutil.move(exported, output)
        validate_export(output)

        reference = segmenter.model.cpu().eval()
        # Compare the upstream decoded predictions and prototypes; raw-head extraction belongs to surgery.py.
        reference.model[-1].export = False
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        session = ort.InferenceSession(str(output), sess_options=options, providers=["CPUExecutionProvider"])
        for seed in SEEDS:
            sample = np.random.RandomState(seed).uniform(0, 1, INTERFACE["images"]).astype(np.float32)
            with torch.inference_mode():
                (predictions, prototypes), _ = reference(torch.from_numpy(sample))
            expected = (predictions.numpy(), prototypes.numpy())
            compare_outputs(expected, session.run(["output0", "output1"], {"images": sample}))
    print(f"Synthetic-input PyTorch/ONNX output check passed with seeds {SEEDS.start}-{SEEDS.stop - 1}: {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversion", default="default", help="conversion entry in model.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    conversions = load_and_validate_model(MODEL_DIR).get("conversion", {})
    if args.conversion not in conversions:
        parser.error(f"unknown conversion {args.conversion!r}; model.yaml declares: {', '.join(sorted(conversions))}")
    source = conversions[args.conversion]["source"]
    validate_source(source)
    path, *_ = Fetcher().fetch_source(source)
    convert(path, args.output_dir.resolve())


if __name__ == "__main__":
    main()
