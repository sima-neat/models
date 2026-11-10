"""Export DETR ResNet-101 from its Hugging Face checkpoint and check ONNX against PyTorch."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from model_registry.fetch import Fetcher
from model_registry.model import load_and_validate_model
from PIL import Image

MODEL_DIR = Path(__file__).resolve().parents[1]

# The input is a 720p camera frame, so Neat feeds frames without resizing them. Its 720-pixel short side is
# within DETR's training scales of 480 to 800.
INPUT_HEIGHT, INPUT_WIDTH = 720, 1280
QUERY_COUNT = 100
# 91 COCO category slots plus the "no object" class.
CLASS_COUNT = 92
BACKBONE = "resnet101"
DILATION = False
INTERFACE = {
    "input": [1, 3, INPUT_HEIGHT, INPUT_WIDTH],
    "logits": [1, QUERY_COUNT, CLASS_COUNT],
    "pred_boxes": [1, QUERY_COUNT, 4],
}
OUTPUTS = ("logits", "pred_boxes")
SOURCE = ("file_set", "model.safetensors", ["model.safetensors", "config.json", "preprocessor_config.json"])
# Frozen batch norms keep no running counters, so the checkpoint's four downsample counters stay unused.
UNUSED_WEIGHTS = {f"model.backbone.model.layer{stage}.0.downsample.1.num_batches_tracked" for stage in range(1, 5)}
SEEDS = range(123, 126)


def validate_source(source):
    declared = (source.get("type"), source.get("entrypoint"), [item.get("path") for item in source.get("files", [])])
    if declared != SOURCE:
        raise ValueError(f"Expected the Hugging Face file set {SOURCE}, model.yaml declares {declared}")


def validate_config(directory):
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    declared = (config.get("model_type"), config.get("backbone"), config.get("dilation"), config.get("num_queries"))
    if declared != ("detr", BACKBONE, DILATION, QUERY_COUNT):
        raise ValueError(f"config.json declares {declared}, expected ('detr', {BACKBONE!r}, {DILATION}, {QUERY_COUNT})")


def validate_export(path):
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    if [item.version for item in model.opset_import if not item.domain] != [17]:
        raise ValueError("DETR export requires opset 17")
    names = ([value.name for value in model.graph.input], [value.name for value in model.graph.output])
    if names != (["input"], list(OUTPUTS)):
        raise ValueError(f"Expected input, logits and pred_boxes in that order, found {names}")
    for value in [*model.graph.input, *model.graph.output]:
        tensor = value.type.tensor_type
        shape = [d.dim_value for d in tensor.shape.dim]
        if tensor.elem_type != onnx.TensorProto.FLOAT or shape != INTERFACE[value.name]:
            raise ValueError(f"Unexpected dtype or shape for {value.name}")
    if any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in model.graph.initializer):
        raise ValueError("This conversion workflow requires one self-contained ONNX file")


def compare_outputs(expected, actual):
    for name, reference, candidate in zip(OUTPUTS, expected, actual, strict=True):
        if list(reference.shape) != INTERFACE[name] or reference.shape != candidate.shape:
            raise ValueError(f"Unexpected shape for {name}")
        if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
            raise ValueError(f"Non-finite output: {name}")
        np.testing.assert_allclose(candidate, reference, rtol=1e-3, atol=1e-3, err_msg=name)


def fold_constants(path):
    from onnxsim import simplify

    model, valid = simplify(
        onnx.load(path),
        perform_optimization=False,
        overwrite_input_shapes={"input": INTERFACE["input"]},
        dynamic_input_shape=False,
    )
    if not valid:
        raise RuntimeError("onnxsim could not validate the folded DETR export")
    onnx.save(model, path)


def convert(source, output_dir):
    import onnxruntime as ort
    import torch
    from transformers import AutoImageProcessor, DetrForObjectDetection

    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "model.onnx"
    torch.set_num_threads(4)
    validate_config(source.parent)
    processor = AutoImageProcessor.from_pretrained(source.parent, local_files_only=True)
    detector, loading = DetrForObjectDetection.from_pretrained(
        source.parent,
        local_files_only=True,
        use_safetensors=True,
        attn_implementation="eager",
        output_loading_info=True,
    )
    if set(loading["unexpected_keys"]) != UNUSED_WEIGHTS or any(
        loading.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")
    ):
        raise ValueError(f"Checkpoint does not match the model: {loading}")

    class Wrapper(torch.nn.Module):
        """The upstream forward pass on one unpadded image, so the pixel mask covers every pixel."""

        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, value):
            result = self.model(pixel_values=value, return_dict=True)
            return result.logits, result.pred_boxes

    def synthetic(seed):
        image = Image.fromarray(
            np.random.RandomState(seed).randint(0, 256, (INPUT_HEIGHT, INPUT_WIDTH, 3), dtype=np.uint8)
        )
        size = {"height": INPUT_HEIGHT, "width": INPUT_WIDTH}
        return processor(images=image, size=size, return_tensors="pt")["pixel_values"]

    detector = detector.cpu().float().eval()
    samples = [synthetic(seed) for seed in SEEDS]
    # Tracing leaves the eager model returning different outputs, so the references come first. The reference
    # passes the all-ones mask explicitly; the export derives the same mask from the input shape.
    mask = torch.ones(1, INPUT_HEIGHT, INPUT_WIDTH, dtype=torch.long)
    with torch.inference_mode():
        references = [detector(pixel_values=sample, pixel_mask=mask) for sample in samples]
    expected = [[result.logits.numpy(), result.pred_boxes.numpy()] for result in references]
    with torch.inference_mode():
        torch.onnx.export(
            Wrapper(detector),
            (samples[0],),
            str(output),
            input_names=["input"],
            output_names=list(OUTPUTS),
            opset_version=17,
            dynamo=False,
        )
    fold_constants(output)
    validate_export(output)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(output), sess_options=options, providers=["CPUExecutionProvider"])
    for sample, reference in zip(samples, expected, strict=True):
        compare_outputs(reference, session.run(list(OUTPUTS), {"input": sample.numpy()}))
    print(f"Synthetic-input PyTorch/ONNX check passed for seeds {list(SEEDS)}: {output}")
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
