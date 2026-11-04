"""Export a pinned D-FINE checkpoint to static, raw-box/raw-logit ONNX."""

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import onnx
from model_registry.fetch import Fetcher
from model_registry.model import load_and_validate_model

MODEL_DIR = Path(__file__).resolve().parents[1]
OUTPUTS = ("pred_boxes", "pred_logits")
INTERFACE = {"input": [1, 3, 640, 640], "pred_boxes": [1, 300, 4], "pred_logits": [1, 300, 80]}
SOURCE = ("file_set", "model.safetensors", ["model.safetensors", "config.json", "preprocessor_config.json"])
# Encoder width, decoder width, feature levels, decoder layers, strides, sampling points.
SIZES = {
    "n": (128, 128, 2, 3, [16, 32], [6, 6]),
    "s": (256, 256, 3, 3, [8, 16, 32], [3, 6, 3]),
    "m": (256, 256, 3, 4, [8, 16, 32], [3, 6, 3]),
    "l": (256, 256, 3, 6, [8, 16, 32], [3, 6, 3]),
    "x": (384, 256, 3, 6, [8, 16, 32], [3, 6, 3]),
}


def identity(model_id):
    parts = model_id.split("_")
    if len(parts) not in (2, 3) or parts[0] != "dfine" or parts[1] not in SIZES:
        raise ValueError(f"Not a D-FINE family entry: {model_id}")
    if len(parts) == 3 and parts[2] != "discrete":
        raise ValueError(f"Unknown sampling variant: {model_id}")
    return parts[1], "discrete" if len(parts) == 3 else "default"


def validate_source(source):
    actual = (
        source.get("type"),
        source.get("entrypoint"),
        [item.get("path") for item in source.get("files", [])],
    )
    if actual != SOURCE:
        raise ValueError(f"Expected the Hugging Face file set {SOURCE}, got {actual}")


def validate_config(directory, model_id):
    size, _ = identity(model_id)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    expected = dict(
        zip(
            (
                "encoder_hidden_dim",
                "d_model",
                "num_feature_levels",
                "decoder_layers",
                "feat_strides",
                "decoder_n_points",
            ),
            SIZES[size],
            strict=True,
        )
    )
    expected.update(
        model_type="d_fine",
        num_queries=300,
        max_num_bins=32,
        decoder_method="default",
        eval_size=None,
        learn_initial_query=False,
        encode_proj_layers=[SIZES[size][2] - 1],
    )
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"{model_id}: config {key}={config.get(key)!r}, expected {value!r}")
    if set(config.get("id2label", {})) != {str(i) for i in range(80)}:
        raise ValueError("Expected the contiguous 80-class COCO label map")
    processor = json.loads((directory / "preprocessor_config.json").read_text(encoding="utf-8"))
    for key, value in {
        "do_normalize": False,
        "do_pad": False,
        "do_rescale": True,
        "do_resize": True,
        "size": {"height": 640, "width": 640},
        "rescale_factor": 1 / 255,
    }.items():
        if processor.get(key) != value:
            raise ValueError(f"Unexpected preprocessing {key}: {processor.get(key)!r}")


def validate_export(path):
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    _, sampling = identity(MODEL_DIR.name)
    expected = {
        "model_id": MODEL_DIR.name,
        "sampling": sampling,
        "box_format": "normalized_cxcywh",
        "class_output": "logits",
    }
    metadata = {item.key: item.value for item in model.metadata_props}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Unexpected D-FINE semantic identity: {metadata}")
    if [item.version for item in model.opset_import if not item.domain] != [17]:
        raise ValueError("D-FINE export requires opset 17")
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
        if indices.shape != (1, 300) or indices.dtype.kind not in "iu":
            raise ValueError("Expected 300 integer proposal indices")
        if np.any(indices < 0) or np.any(indices >= expected_scores.shape[1]) or np.unique(indices).size != 300:
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
        and constants[node.input[1]].item() == 300
    ]
    if len(candidates) != 1:
        raise ValueError("Expected one 300-query proposal TopK")
    topk = candidates[0]
    names = [topk.input[0], topk.output[1]]
    model.graph.output.extend(onnx.helper.make_empty_tensor_value_info(name) for name in names)
    return model.SerializeToString(), names


def load_reference(source, model_id):
    """Strictly load local weights; checkpoint parameters, not JSON defaults, govern refinement."""
    from transformers import DFineConfig, DFineForObjectDetection

    size, sampling = identity(model_id)
    validate_config(source.parent, model_id)
    config = DFineConfig.from_pretrained(source.parent, local_files_only=True)
    config.decoder_method = sampling
    detector, loading = DFineForObjectDetection.from_pretrained(
        source.parent,
        config=config,
        local_files_only=True,
        use_safetensors=True,
        attn_implementation="eager",
        output_loading_info=True,
    )
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Checkpoint does not match the model: {loading}")
    decoder = detector.model.decoder
    effective = {"reg_scale": float(decoder.reg_scale.item()), "up": float(decoder.up.item())}
    if effective != {"reg_scale": 8.0 if size == "x" else 4.0, "up": 0.5}:
        raise ValueError(f"Unexpected checkpoint refinement parameters: {effective}")
    return detector.cpu().float().eval(), effective


def per_head_default_attention(
    value,
    value_spatial_shapes,
    sampling_locations,
    attention_weights,
    num_points_list,
    method="default",
):
    """Preserve source sampling/reduction order; only separate independent heads/channels."""
    import torch
    import torch.nn.functional as F

    if method != "default":
        raise ValueError("This compatibility function only supports default sampling")
    if value.ndim != 4 or sampling_locations.ndim != 5 or attention_weights.ndim != 4:
        raise ValueError("Unexpected D-FINE attention tensor ranks")
    batch, length, heads, channels = value.shape
    shapes = [(int(height), int(width)) for height, width in value_spatial_shapes]
    points = list(num_points_list)
    if batch != 1 or heads != 8 or channels not in (16, 32):
        raise ValueError("Expected batch-one D-FINE attention with eight 16/32-channel heads")
    if len(shapes) not in (2, 3) or len(shapes) != len(points):
        raise ValueError("D-FINE spatial levels and sampling points disagree")
    if any(height <= 0 or width <= 0 for height, width in shapes) or any(
        not isinstance(count, int) or count <= 0 for count in points
    ):
        raise ValueError("Expected positive static spatial dimensions and point counts")
    queries = sampling_locations.shape[1]
    if (
        sum(height * width for height, width in shapes) != length
        or tuple(sampling_locations.shape) != (1, queries, heads, sum(points), 2)
        or tuple(attention_weights.shape) != (1, queries, heads, sum(points))
    ):
        raise ValueError("D-FINE attention input shapes do not match the declared levels")

    # This is the original source normalization, not tile-local coordinate arithmetic.
    grids = 2 * sampling_locations - 1
    outputs = []
    for head in range(heads):
        head_value = value[:, :, head, :].transpose(1, 2)
        sampled_levels = []
        value_begin = point_begin = 0
        for (height, width), count in zip(shapes, points, strict=True):
            value_end = value_begin + height * width
            level = head_value[:, :, value_begin:value_end].reshape(1, channels, height, width)
            grid = grids[:, :, head, point_begin : point_begin + count, :]
            # Split only the demonstrated C32/40x40 capacity failure; keep other shapes native.
            chunk_size = 16 if channels == 32 and (height, width) == (40, 40) else channels
            chunks = [
                F.grid_sample(
                    level[:, channel : channel + chunk_size],
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )
                for channel in range(0, channels, chunk_size)
            ]
            sampled_levels.append(torch.cat(chunks, dim=1) if len(chunks) > 1 else chunks[0])
            value_begin, point_begin = value_end, point_begin + count
        # Keep the original level/point concatenation and single reduction, not per-level sums.
        weighted = torch.cat(sampled_levels, dim=-1) * attention_weights[:, :, head, :].unsqueeze(1)
        outputs.append(weighted.sum(-1).transpose(1, 2))
    return torch.cat(outputs, dim=-1).contiguous()


def prepare_default_attention(model):
    """Change only the prepared instances' existing function slot; leave discrete unchanged."""
    from transformers.models.d_fine.modeling_d_fine import (
        DFineMultiscaleDeformableAttention,
        multi_scale_deformable_attention_v2,
    )

    prepared = 0
    for module in model.modules():
        if not isinstance(module, DFineMultiscaleDeformableAttention):
            continue
        if module.decoder_method == "discrete":
            continue
        if (
            module.decoder_method != "default"
            or module.ms_deformable_attn_core is not multi_scale_deformable_attention_v2
        ):
            raise ValueError("Unexpected D-FINE attention implementation")
        module.ms_deformable_attn_core = per_head_default_attention
        prepared += 1
    return prepared


def batch_preserving_integral(module, corners, project):
    """Keep the real batch axis and original projection; do not fold queries into batch."""
    import torch
    import torch.nn.functional as F

    if corners.ndim != 3 or project.ndim != 1:
        raise ValueError("Expected Integral corners[B,Q,4K] and projection[K]")
    batch, queries, width = corners.shape
    bins = module.max_num_bins + 1
    if width != 4 * bins or project.numel() != bins:
        raise ValueError("Integral geometry does not match its source bin count")
    probabilities = F.softmax(corners.reshape(batch, queries * 4, bins), dim=-1)
    column = project.to(corners.device).reshape(bins, 1)
    return torch.matmul(probabilities, column).reshape(batch, queries, 4)


def prepare_integral(model):
    """Adapt only prepared instances, retaining the source's runtime projection argument."""
    from types import MethodType

    from transformers.models.d_fine.modeling_d_fine import DFineIntegral

    prepared = 0
    for module in model.modules():
        if not isinstance(module, DFineIntegral):
            continue
        if (
            type(module) is not DFineIntegral
            or module.max_num_bins != 32
            or getattr(module.forward, "__func__", None) is not DFineIntegral.forward
        ):
            raise ValueError("Unexpected D-FINE Integral implementation or bin count")
        module.forward = MethodType(batch_preserving_integral, module)
        prepared += 1
    return prepared


def convert(source, output_dir):
    import onnxruntime as ort
    import torch
    import transformers

    model_id = MODEL_DIR.name
    size, sampling = identity(model_id)
    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "model.onnx"
    torch.set_num_threads(4)
    torch.manual_seed(123)
    detector, effective = load_reference(source, model_id)

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

    # The input is static640. Compute source positional constants before tracing, avoiding
    # runtime float64 Sin/Cos unsupported by ONNX Runtime without changing their values.
    for layer, index in zip(prepared.model.model.encoder.aifi, detector.config.encode_proj_layers, strict=True):
        side = INTERFACE["input"][2] // detector.config.feat_strides[index]
        value = layer.position_embedding(side, side, torch.device("cpu"), torch.float32)
        layer.position_embedding = FixedPosition(value.detach())
    default_attention_layers = prepare_default_attention(prepared)
    expected_attention_layers = SIZES[size][3] if sampling == "default" else 0
    if default_attention_layers != expected_attention_layers:
        raise ValueError("Prepared attention count does not match the D-FINE source contract")
    integral_modules = prepare_integral(prepared)
    if integral_modules != 1:
        raise ValueError("Prepared Integral count does not match the D-FINE source contract")
    # All fixed cases must pass. Do not retry until a convenient TopK order happens to agree.
    samples = [np.random.default_rng(seed).random(INTERFACE["input"], dtype=np.float32) for seed in (123, 456, 789)]
    with torch.inference_mode():
        torch.onnx.export(
            prepared,
            (torch.from_numpy(samples[0]),),
            str(output),
            input_names=["input"],
            output_names=list(OUTPUTS),
            opset_version=17,
            dynamo=False,
        )
    model = onnx.load(output)
    onnx.helper.set_model_props(
        model,
        {
            "model_id": model_id,
            "sampling": sampling,
            "box_format": "normalized_cxcywh",
            "class_output": "logits",
        },
    )
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
    preparation_checks = []
    alignment_checks = []
    for sample in samples:
        captured_scores.clear()
        with torch.inference_mode():
            expected = [v.numpy() for v in reference(torch.from_numpy(sample))]
            preparation_checks.append(
                compare_outputs(expected, [v.numpy() for v in prepared(torch.from_numpy(sample))])
            )
        boxes, logits, scores, indices = session.run([*OUTPUTS, *selection], {"input": sample})
        if len(captured_scores) != 1:
            raise ValueError("Expected one proposal scoring call")
        expected_scores = captured_scores[0].numpy()
        expected_indices = captured_scores[0].topk(300, dim=1).indices.numpy()
        order = proposal_alignment(expected_scores, scores, expected_indices, indices)
        checks.append(compare_outputs(expected, [boxes[:, order], logits[:, order]]))
        alignment_checks.append({"reordered_queries": int(np.count_nonzero(order != np.arange(300)))})
    report = {
        "model_id": model_id,
        "sampling": sampling,
        "effective_refinement": effective,
        "preparation": {
            "static_position_embeddings": len(detector.config.encode_proj_layers),
            "per_head_default_attention_layers": default_attention_layers,
            "batch_preserving_integral_modules": integral_modules,
            "channel_chunking_policy": (
                "C32 at 40x40 only; 20x20 and 80x80 remain unchunked" if default_attention_layers else "none"
            ),
        },
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
        "smoke_seeds": [123, 456, 789],
        "max_absolute_errors": checks,
        "preparation_max_absolute_errors": preparation_checks,
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
