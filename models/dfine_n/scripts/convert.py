"""Export the pinned D-FINE Nano checkpoint to static, raw-box/raw-logit ONNX and check it against PyTorch."""

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import onnx
from model_registry.dependencies import onnx_opset_version, require_onnx_opset_version
from model_registry.fetch import Fetcher
from model_registry.model import load_and_validate_model

MODEL_DIR = Path(__file__).resolve().parents[1]
MODEL_ID = "dfine_n"
IMAGE_SIZE = 640
QUERY_COUNT = 300
CLASSES = 80
OUTPUTS = ("pred_boxes", "pred_logits")
INTERFACE = {
    "input": [1, 3, IMAGE_SIZE, IMAGE_SIZE],
    "pred_boxes": [1, QUERY_COUNT, 4],
    "pred_logits": [1, QUERY_COUNT, CLASSES],
}
SOURCE = ("file_set", "model.safetensors", ["model.safetensors", "config.json", "preprocessor_config.json"])
# What the Nano checkpoint's config must declare: encoder width, decoder width, feature levels,
# decoder layers, strides and sampling points, plus the options the export relies on.
CONFIG = {
    "model_type": "d_fine",
    "encoder_hidden_dim": 128,
    "d_model": 128,
    "num_feature_levels": 2,
    "decoder_layers": 3,
    "feat_strides": [16, 32],
    "decoder_n_points": [6, 6],
    "encode_proj_layers": [1],
    "num_queries": QUERY_COUNT,
    "max_num_bins": 32,
    "decoder_method": "default",
    "eval_size": None,
    "learn_initial_query": False,
}
# The checkpoint's distribution refinement parameters; JSON defaults do not govern them.
REFINEMENT = {"reg_scale": 4.0, "up": 0.5}
METADATA = {"model_id": MODEL_ID, "box_format": "normalized_cxcywh", "class_output": "logits"}
# Fixed synthetic inputs; every one must select the same proposals in PyTorch and ONNX Runtime.
SEEDS = (123, 456, 789)


def validate_source(source):
    actual = (
        source.get("type"),
        source.get("entrypoint"),
        [item.get("path") for item in source.get("files", [])],
    )
    if actual != SOURCE:
        raise ValueError(f"Expected the Hugging Face file set {SOURCE}, got {actual}")


def validate_config(directory):
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    for key, value in CONFIG.items():
        if config.get(key) != value:
            raise ValueError(f"{MODEL_ID}: config {key}={config.get(key)!r}, expected {value!r}")
    if set(config.get("id2label", {})) != {str(i) for i in range(CLASSES)}:
        raise ValueError("Expected the contiguous 80-class COCO label map")
    processor = json.loads((directory / "preprocessor_config.json").read_text(encoding="utf-8"))
    for key, value in {
        "do_normalize": False,
        "do_pad": False,
        "do_rescale": True,
        "do_resize": True,
        "size": {"height": IMAGE_SIZE, "width": IMAGE_SIZE},
        "rescale_factor": 1 / 255,
    }.items():
        if processor.get(key) != value:
            raise ValueError(f"Unexpected preprocessing {key}: {processor.get(key)!r}")


def validate_export(path):
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    metadata = {item.key: item.value for item in model.metadata_props}
    if any(metadata.get(key) != value for key, value in METADATA.items()):
        raise ValueError(f"Unexpected D-FINE semantic identity: {metadata}")
    require_onnx_opset_version(model)
    if [v.name for v in model.graph.input] != ["input"] or [v.name for v in model.graph.output] != list(OUTPUTS):
        raise ValueError("Expected input, pred_boxes and pred_logits in that order")
    for value in [*model.graph.input, *model.graph.output]:
        tensor = value.type.tensor_type
        if (
            tensor.elem_type != onnx.TensorProto.FLOAT
            or [d.dim_value for d in tensor.shape.dim] != INTERFACE[value.name]
        ):
            raise ValueError(f"Unexpected dtype or static shape for {value.name}")
    if any(t.data_location == onnx.TensorProto.EXTERNAL for t in model.graph.initializer):
        raise ValueError("Expected a self-contained ONNX file")


def compare_outputs(expected, actual):
    errors = {}
    for name, reference, candidate in zip(OUTPUTS, expected, actual, strict=True):
        if reference.shape != candidate.shape or list(reference.shape) != INTERFACE[name]:
            raise ValueError(f"Unexpected output shape for {name}")
        if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
            raise ValueError(f"Non-finite output: {name}")
        np.testing.assert_allclose(candidate, reference, rtol=1e-3, atol=1e-3, err_msg=name)
        errors[name] = float(np.max(np.abs(reference - candidate)))
    return errors


def proposal_alignment(expected_scores, actual_scores, expected_indices, actual_indices):
    """Match the same selected proposals, allowing only rounding-level rank ties."""
    if expected_scores.shape != actual_scores.shape or expected_scores.ndim != 2 or expected_scores.shape[0] != 1:
        raise ValueError("Unexpected proposal score shape")
    if not np.isfinite(expected_scores).all() or not np.isfinite(actual_scores).all():
        raise ValueError("Non-finite proposal scores")
    np.testing.assert_allclose(actual_scores, expected_scores, rtol=1e-3, atol=1e-3)
    for indices in (expected_indices, actual_indices):
        if indices.shape != (1, QUERY_COUNT) or indices.dtype.kind not in "iu":
            raise ValueError("Expected 300 integer proposal indices")
        if np.any(indices < 0) or np.any(indices >= expected_scores.shape[1]) or np.unique(indices).size != QUERY_COUNT:
            raise ValueError("Invalid or repeated proposal indices")
    left, right = expected_indices[0], actual_indices[0]
    if not np.array_equal(np.sort(left), np.sort(right)):
        raise ValueError("TopK selected different proposals")
    swapped = left != right
    for scores in (expected_scores[0], actual_scores[0]):
        if np.any(np.abs(scores[left[swapped]] - scores[right[swapped]]) > 1e-5):
            raise ValueError("TopK reordered proposals whose scores are not tied")
    positions = {index: row for row, index in enumerate(right)}
    return np.array([positions[index] for index in left])


def selection_model(path):
    """Add proposal diagnostics to an in-memory copy; the published ONNX keeps two outputs."""
    model = onnx.load(path)
    constants = {value.name: onnx.numpy_helper.to_array(value) for value in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type == "Constant":
            value = next((attribute.t for attribute in node.attribute if attribute.name == "value"), None)
            if value is not None:
                constants[node.output[0]] = onnx.numpy_helper.to_array(value)
    candidates = [
        node
        for node in model.graph.node
        if node.op_type == "TopK"
        and node.input[1] in constants
        and constants[node.input[1]].size == 1
        and constants[node.input[1]].item() == QUERY_COUNT
    ]
    if len(candidates) != 1:
        raise ValueError("Expected one 300-query proposal TopK")
    topk = candidates[0]
    names = [topk.input[0], topk.output[1]]
    model.graph.output.extend(onnx.helper.make_empty_tensor_value_info(name) for name in names)
    return model.SerializeToString(), names


def load_reference(source):
    """Strictly load local weights; checkpoint parameters, not JSON defaults, govern refinement."""
    from transformers import DFineForObjectDetection

    validate_config(source.parent)
    detector, loading = DFineForObjectDetection.from_pretrained(
        source.parent,
        local_files_only=True,
        use_safetensors=True,
        attn_implementation="eager",
        output_loading_info=True,
    )
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Checkpoint does not match the model: {loading}")
    decoder = detector.model.decoder
    effective = {"reg_scale": float(decoder.reg_scale.item()), "up": float(decoder.up.item())}
    if effective != REFINEMENT:
        raise ValueError(f"Unexpected checkpoint refinement parameters: {effective}")
    return detector.cpu().float().eval(), effective


def convert(source, output_dir):
    import onnxruntime as ort
    import torch
    import transformers

    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "model.onnx"
    torch.set_num_threads(4)
    torch.manual_seed(123)
    detector, effective = load_reference(source)

    class RawOutputs(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, value):
            result = self.model(pixel_values=value, return_dict=True)
            return (
                result.pred_boxes.reshape(INTERFACE["pred_boxes"]),
                result.logits.reshape(INTERFACE["pred_logits"]),
            )

    reference = RawOutputs(detector).eval()
    prepared = deepcopy(reference)

    class FixedPosition(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.register_buffer("value", value)

        def forward(self, width, height, device, dtype):
            return self.value

    # The input is static 640x640. Evaluate the encoder's positional constants before tracing, which
    # avoids runtime float64 Sin/Cos that ONNX Runtime cannot run, without changing their values.
    for layer, index in zip(prepared.model.model.encoder.aifi, detector.config.encode_proj_layers, strict=True):
        side = IMAGE_SIZE // detector.config.feat_strides[index]
        value = layer.position_embedding(side, side, torch.device("cpu"), torch.float32)
        layer.position_embedding = FixedPosition(value.detach())
    samples = [np.random.default_rng(seed).random(INTERFACE["input"], dtype=np.float32) for seed in SEEDS]
    with torch.inference_mode():
        torch.onnx.export(
            prepared,
            (torch.from_numpy(samples[0]),),
            str(output),
            input_names=["input"],
            output_names=list(OUTPUTS),
            opset_version=onnx_opset_version(),
            dynamo=False,
        )
    model = onnx.load(output)
    onnx.helper.set_model_props(model, METADATA)
    onnx.save(model, output)
    validate_export(output)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    diagnostic_model, selection = selection_model(output)
    session = ort.InferenceSession(diagnostic_model, sess_options=options, providers=["CPUExecutionProvider"])
    captured_scores = []
    detector.model.enc_score_head.register_forward_hook(
        lambda _module, _inputs, value: captured_scores.append(value.detach().max(-1).values)
    )
    checks = []
    alignment_checks = []
    for sample in samples:
        captured_scores.clear()
        with torch.inference_mode():
            expected = [v.numpy() for v in reference(torch.from_numpy(sample))]
        boxes, logits, scores, indices = session.run([*OUTPUTS, *selection], {"input": sample})
        if len(captured_scores) != 1:
            raise ValueError("Expected one proposal scoring call")
        expected_scores = captured_scores[0].numpy()
        expected_indices = captured_scores[0].topk(QUERY_COUNT, dim=1).indices.numpy()
        order = proposal_alignment(expected_scores, scores, expected_indices, indices)
        checks.append(compare_outputs(expected, [boxes[:, order], logits[:, order]]))
        alignment_checks.append({"reordered_queries": int(np.count_nonzero(order != np.arange(QUERY_COUNT)))})
    report = {
        "model_id": MODEL_ID,
        "effective_refinement": effective,
        "static_position_embeddings": len(detector.config.encode_proj_layers),
        "software": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "numpy": np.__version__,
        },
        "sources": {name: hashlib.sha256((source.parent / name).read_bytes()).hexdigest() for name in SOURCE[2]},
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "output_size_bytes": output.stat().st_size,
        "interface": INTERFACE,
        "smoke_seeds": list(SEEDS),
        "max_absolute_errors": checks,
        "proposal_alignment": alignment_checks,
        "qualification": "PyTorch-to-ONNX smoke only; original-checkpoint parity and COCO accuracy are separate gates",
    }
    (output_dir / "conversion.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversion", default="default", help="conversion entry in model.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    conversions = load_and_validate_model(MODEL_DIR).get("conversion", {})
    if args.conversion not in conversions:
        parser.error(f"unknown conversion {args.conversion!r}")
    source = conversions[args.conversion]["source"]
    validate_source(source)
    path, *_ = Fetcher().fetch_source(source)
    print(convert(path, args.output_dir))


if __name__ == "__main__":
    main()
