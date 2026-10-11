"""Export YOLO26s-pose from its Ultralytics checkpoint and check ONNX against PyTorch."""

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

CHECKPOINT = "yolo26s-pose.pt"
EXPORTER = "8.4.0"
IMAGE_SIZE = 640
QUERY_COUNT = 300
KEYPOINT_CHANNELS = 17 * 3
INTERFACE = {"images": [1, 3, IMAGE_SIZE, IMAGE_SIZE], "output0": [1, QUERY_COUNT, 6 + KEYPOINT_CHANNELS]}
# Last convolution of each one-to-one Pose level: box distances (cv2), person logits (cv3) and raw keypoints
# (cv4_kpts). The surgery cuts the graph at these tensors, so they are what the comparison checks.
HEADS = {
    key: [f"/model.23/{branch.format(level=level)}/Conv_output_0" for level in range(3)]
    for key, branch in (
        ("boxes", "one2one_cv2.{level}/one2one_cv2.{level}.2"),
        ("scores", "one2one_cv3.{level}/one2one_cv3.{level}.2"),
        ("kpts", "one2one_cv4_kpts.{level}"),
    )
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
        raise ValueError("YOLO26 export requires opset 12")
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
        raise RuntimeError("onnxsim could not validate the folded YOLO26 export")
    onnx.save(model, path)


def with_head_outputs(path):
    """Also return the raw one-to-one head tensors, located by tensor name."""
    model = onnx.load(path)
    names = [name for levels in HEADS.values() for name in levels]
    produced = {name for node in model.graph.node for name in node.output}
    missing = [name for name in names if name not in produced]
    if missing:
        raise ValueError(f"Export lacks the one-to-one head tensors: {missing}")
    model.graph.output.extend(onnx.helper.make_empty_tensor_value_info(name) for name in names)
    return model.SerializeToString(), names


def compare_heads(expected, actual):
    """Compare Ultralytics' one-to-one boxes, scores and raw keypoints with the three ONNX levels joined per role."""
    for key, levels in zip(HEADS, (actual[0:3], actual[3:6], actual[6:9])):
        joined = np.concatenate([value.reshape(1, value.shape[1], -1) for value in levels], axis=2)
        reference = expected[key].numpy()
        if joined.shape != reference.shape or not np.isfinite(joined).all():
            raise ValueError(f"Unexpected or non-finite {key}: {joined.shape} vs {reference.shape}")
        np.testing.assert_allclose(joined, reference, rtol=1e-3, atol=1e-3, err_msg=key)


def convert(source, output_dir):
    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "model.onnx"
    with tempfile.TemporaryDirectory(prefix="yolo26s-pose-") as temporary:
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
        estimator = YOLO(str(checkpoint))
        exported = estimator.export(
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

        reference = estimator.model.cpu().eval()
        # Pose then returns its raw one-to-one boxes, scores and keypoints instead of the selected detections.
        reference.model[-1].export = False
        model, heads = with_head_outputs(output)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        session = ort.InferenceSession(model, sess_options=options, providers=["CPUExecutionProvider"])
        for seed in SEEDS:
            sample = np.random.RandomState(seed).uniform(0, 1, INTERFACE["images"]).astype(np.float32)
            with torch.inference_mode():
                expected = reference(torch.from_numpy(sample))[1]["one2one"]
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
