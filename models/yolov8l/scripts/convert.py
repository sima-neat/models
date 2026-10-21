"""Export YOLOv8l from its Ultralytics checkpoint and check ONNX against PyTorch."""

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

CHECKPOINT = "yolov8l.pt"
EXPORTER = "8.4.0"
IMAGE_SIZE = 640
CLASS_COUNT = 80
ANCHOR_COUNT = sum((IMAGE_SIZE // stride) ** 2 for stride in (8, 16, 32))
INTERFACE = {"images": [1, 3, IMAGE_SIZE, IMAGE_SIZE], "output0": [1, 4 + CLASS_COUNT, ANCHOR_COUNT]}
# Last convolution of each Detect level: box distributions (cv2) and class logits (cv3). The surgery
# cuts the graph at these tensors, so they are what the comparison checks.
HEADS = {
    key: [f"/model.22/{branch}.{level}/{branch}.{level}.2/Conv_output_0" for level in range(3)]
    for key, branch in (("boxes", "cv2"), ("scores", "cv3"))
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
        raise ValueError("Expected input images and output output0")
    for value in values:
        tensor = value.type.tensor_type
        shape = [d.dim_value for d in tensor.shape.dim]
        if tensor.elem_type != onnx.TensorProto.FLOAT or shape != INTERFACE[value.name]:
            raise ValueError(f"Unexpected dtype or shape for {value.name}")
    if any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in model.graph.initializer):
        raise ValueError("This conversion workflow requires one self-contained ONNX file")


def fold_constants(path):
    from onnxsim import simplify

    model, valid = simplify(
        onnx.load(path),
        perform_optimization=False,
        overwrite_input_shapes={"images": INTERFACE["images"]},
        dynamic_input_shape=False,
    )
    if not valid:
        raise RuntimeError("onnxsim could not validate the folded YOLOv8 export")
    onnx.save(model, path)


def with_head_outputs(path):
    """Also return the raw Detect head tensors, located by tensor name."""
    model = onnx.load(path)
    names = [*HEADS["boxes"], *HEADS["scores"]]
    produced = {name for node in model.graph.node for name in node.output}
    missing = [name for name in names if name not in produced]
    if missing:
        raise ValueError(f"Export lacks the Detect head tensors: {missing}")
    model.graph.output.extend(onnx.helper.make_empty_tensor_value_info(name) for name in names)
    return model.SerializeToString(), names


def compare_heads(expected, actual):
    """Compare Ultralytics' raw boxes and scores with the three ONNX levels joined per role."""
    for key, levels in (("boxes", actual[:3]), ("scores", actual[3:])):
        joined = np.concatenate([value.reshape(1, value.shape[1], -1) for value in levels], axis=2)
        reference = expected[key].numpy()
        if joined.shape != reference.shape or not np.isfinite(joined).all():
            raise ValueError(f"Unexpected or non-finite {key}: {joined.shape} vs {reference.shape}")
        np.testing.assert_allclose(joined, reference, rtol=1e-3, atol=1e-3, err_msg=key)


def convert(source, output_dir):
    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "model.onnx"
    with tempfile.TemporaryDirectory(prefix="yolov8l-") as temporary:
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
        detector = YOLO(str(checkpoint))
        exported = detector.export(
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
        fold_constants(output)
        validate_export(output)

        reference = detector.model.cpu().eval()
        # Detect then returns its raw box distributions and class logits beside the decoded output.
        reference.model[-1].export = False
        model, heads = with_head_outputs(output)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        session = ort.InferenceSession(model, sess_options=options, providers=["CPUExecutionProvider"])
        for seed in SEEDS:
            sample = np.random.RandomState(seed).uniform(0, 1, INTERFACE["images"]).astype(np.float32)
            with torch.inference_mode():
                expected = reference(torch.from_numpy(sample))[1]
            compare_heads(expected, session.run(heads, {"images": sample}))
    print(f"Synthetic-input PyTorch/ONNX head check passed with seeds {SEEDS.start}-{SEEDS.stop - 1}: {output}")
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
