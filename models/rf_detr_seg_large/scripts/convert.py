"""Export RF-DETR Seg Large from its Hugging Face checkpoint and check ONNX against PyTorch."""

import argparse
import json
import types
from copy import deepcopy
from pathlib import Path

import numpy as np
import onnx
from model_registry.fetch import Fetcher
from model_registry.model import load_and_validate_model
from PIL import Image

MODEL_DIR = Path(__file__).resolve().parents[1]

IMAGE_SIZE = 504
# The backbone emits one feature-map cell per 12x12 patch, and the encoder scores one proposal per cell.
FEATURE_SIZE = IMAGE_SIZE // 12
PROPOSAL_COUNT = FEATURE_SIZE * FEATURE_SIZE
QUERY_COUNT = 300
CLASS_COUNT = 91
# The segmentation head predicts each query's mask at a quarter of the input resolution.
MASK_SIZE = IMAGE_SIZE // 4
INTERFACE = {
    "input": [1, 3, IMAGE_SIZE, IMAGE_SIZE],
    "dets": [1, QUERY_COUNT, 4],
    "labels": [1, QUERY_COUNT, CLASS_COUNT],
    "masks": [1, QUERY_COUNT, MASK_SIZE, MASK_SIZE],
}
OUTPUTS = ("dets", "labels", "masks")
SOURCE = ("file_set", "model.safetensors", ["model.safetensors", "config.json", "preprocessor_config.json"])
# Synthetic inputs tried until PyTorch and ONNX Runtime select the same proposals in the same order.
SEEDS = range(123, 133)


def validate_source(source):
    declared = (source.get("type"), source.get("entrypoint"), [item.get("path") for item in source.get("files", [])])
    if declared != SOURCE:
        raise ValueError(f"Expected the Hugging Face file set {SOURCE}, model.yaml declares {declared}")


def validate_config(directory):
    model_type = json.loads((directory / "config.json").read_text(encoding="utf-8")).get("model_type")
    if model_type != "rf_detr":
        raise ValueError(f"config.json names model_type {model_type!r}, expected 'rf_detr'")


def validate_export(path):
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    if [item.version for item in model.opset_import if not item.domain] != [17]:
        raise ValueError("RF-DETR export requires opset 17")
    values = [*model.graph.input, *model.graph.output]
    if [value.name for value in model.graph.input] != ["input"] or [value.name for value in model.graph.output] != list(
        OUTPUTS
    ):
        raise ValueError("Expected input, dets, labels and masks in that order")
    for value in values:
        tensor = value.type.tensor_type
        shape = [d.dim_value for d in tensor.shape.dim]
        if tensor.elem_type != onnx.TensorProto.FLOAT or shape != INTERFACE[value.name]:
            raise ValueError(f"Unexpected dtype or shape for {value.name}")
    if any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in model.graph.initializer):
        raise ValueError("This conversion workflow requires one self-contained ONNX file")


def compare_outputs(expected, actual):
    for name, reference, candidate in zip(OUTPUTS, expected, actual, strict=True):
        if reference.shape != candidate.shape or list(reference.shape) != INTERFACE[name]:
            raise ValueError(f"Unexpected shape for {name}")
        if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
            raise ValueError(f"Non-finite output: {name}")
        # Mask logits reach about 100, so their absolute tolerance scales with their RMS.
        scale = float(np.sqrt(np.mean(np.square(reference)))) if name == "masks" else 1.0
        np.testing.assert_allclose(candidate, reference, rtol=1e-3, atol=1e-3 * scale, err_msg=name)


def compare_selection(expected_scores, actual_scores, actual_indices):
    """Compare the proposal scores and report whether TopK picks the same proposals in the same order.

    Rounding can swap two proposals whose scores tie, which pairs decoder queries with different
    boxes. Only such swaps are allowed; the caller then retries with another input.
    """
    np.testing.assert_allclose(actual_scores, expected_scores, rtol=1e-3, atol=1e-3, err_msg="proposal scores")
    expected_indices = np.argsort(-expected_scores, axis=1, kind="stable")[:, :QUERY_COUNT]
    swapped = expected_indices != actual_indices
    gaps = np.abs(
        np.take_along_axis(actual_scores, expected_indices, 1) - np.take_along_axis(actual_scores, actual_indices, 1)
    )[swapped]
    if gaps.size and gaps.max() > 1e-5:
        raise ValueError(f"TopK selects different proposals; largest score gap {gaps.max():.2e}")
    return not swapped.any()


def fold_constants(path):
    from onnxsim import simplify

    model, valid = simplify(
        onnx.load(path),
        perform_optimization=False,
        overwrite_input_shapes={"input": INTERFACE["input"]},
        dynamic_input_shape=False,
    )
    if not valid:
        raise RuntimeError("onnxsim could not validate the folded RF-DETR export")
    onnx.save(model, path)


def with_selection_outputs(path):
    """Also return TopK's proposal scores and selected indices, located by operator type."""
    model = onnx.load(path)
    (topk,) = [node for node in model.graph.node if node.op_type == "TopK"]
    names = [topk.input[0], topk.output[1]]
    model.graph.output.extend(onnx.helper.make_empty_tensor_value_info(name) for name in names)
    return model.SerializeToString(), names


def convert(source, output_dir):
    import onnxruntime as ort
    import torch
    from transformers import AutoImageProcessor, RfDetrForInstanceSegmentation

    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "model.onnx"
    torch.set_num_threads(4)
    torch.manual_seed(123)
    validate_config(source.parent)
    processor = AutoImageProcessor.from_pretrained(source.parent, local_files_only=True)
    segmenter, loading = RfDetrForInstanceSegmentation.from_pretrained(
        source.parent,
        local_files_only=True,
        use_safetensors=True,
        attn_implementation="eager",
        output_loading_info=True,
    )
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Checkpoint does not match the model: {loading}")

    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, value):
            result = self.model(pixel_values=value, return_dict=True)
            return result.pred_boxes, result.logits, result.pred_masks

    def synthetic(seed):
        image = Image.fromarray(
            np.random.RandomState(seed).randint(0, 256, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        )
        return processor(images=image, return_tensors="pt")["pixel_values"]

    reference = Wrapper(segmenter).cpu().float().eval()
    prepared = deepcopy(reference)
    embeddings = [m for m in prepared.modules() if m.__class__.__name__ == "RfDetrDinov2Embeddings"]
    if len(embeddings) != 1:
        raise ValueError("Expected one positional-encoding module")
    module = embeddings[0]
    with torch.no_grad():
        # One token per patch plus the class token.
        dummy = torch.zeros(1, FEATURE_SIZE * FEATURE_SIZE + 1, module.position_embeddings.shape[-1])
        fixed = module.interpolate_pos_encoding(dummy, IMAGE_SIZE, IMAGE_SIZE)
    module.register_buffer("fixed_position_encoding", fixed.detach().clone())
    module.interpolate_pos_encoding = types.MethodType(lambda self, *args: self.fixed_position_encoding, module)
    with torch.inference_mode():
        torch.onnx.export(
            prepared,
            (synthetic(SEEDS[0]),),
            str(output),
            input_names=["input"],
            output_names=list(OUTPUTS),
            opset_version=17,
            dynamo=False,
        )
    fold_constants(output)
    validate_export(output)

    scores = []

    def capture_scores(_module, _inputs, value):
        if value.shape[1] == PROPOSAL_COUNT:
            scores.append(value.detach().amax(-1).numpy())

    # The segmenter wraps the detector, whose model scores the proposals.
    segmenter.model.model.enc_out_class_embed[0].register_forward_hook(capture_scores)
    model, selection = with_selection_outputs(output)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(model, sess_options=options, providers=["CPUExecutionProvider"])
    for seed in SEEDS:
        sample = synthetic(seed)
        scores.clear()
        with torch.inference_mode():
            expected = reference(sample)
        *actual, proposal_scores, indices = session.run([*OUTPUTS, *selection], {"input": sample.numpy()})
        if compare_selection(scores[0], proposal_scores, indices):
            compare_outputs([value.numpy() for value in expected], actual)
            print(f"Synthetic-input PyTorch/ONNX smoke check passed with seed {seed}: {output}")
            return output
        print(f"Seed {seed}: tied proposal scores swapped order; trying the next input")
    raise RuntimeError(f"Every synthetic input in {SEEDS} swapped tied proposals; outputs were not compared")


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
