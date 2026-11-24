"""Adapt the upstream checkpoint in PyTorch and export fixed-shape graphs during compilation."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch
from onnx_export import frontend_metadata
from torch import nn
from torch.nn import functional as F

MEL, FRAMES = 1400, 175
MASK_OFFSETS = (0, 704, 1056)
MASK_LENGTHS = (700, 350, 175)


def controls(valid, device="cpu"):
    masks = torch.zeros(1, 1, 1232, 1, device=device)
    for stride, offset, size in zip((2, 4, 8), MASK_OFFSETS, MASK_LENGTHS):
        count = (valid + stride - 1) // stride
        masks[:, :, offset : offset + min(count, size)] = 1
    bias = torch.zeros(1, 1, 1, 176, device=device)
    bias[..., (valid + 7) // 8 :] = -10000
    return masks, bias


class Subsampling(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.sub = model.encoder.subsampling
        self.scale = model.encoder.input_scale

    def forward(self, features, padding_masks):
        x = features
        stage = -1
        for index, layer in enumerate(self.sub.layers):
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                if layer.stride != (1, 1):
                    stage += 1
                # The following pointwise Conv has its own identical mask.
                # Retain that later mask, including when the Conv has a bias.
                if index not in (2, 5):
                    start, size = MASK_OFFSETS[stage], MASK_LENGTHS[stage]
                    x = x * padding_masks[:, :, start : start + size]
        # Original Flatten(C,F)+Linear is exactly this frequency-wide Conv.
        x = F.conv2d(x, self.sub.linear.weight.reshape(1024, 256, 1, 16), self.sub.linear.bias)
        return x.permute(0, 2, 3, 1).reshape(1, FRAMES, 1024) * self.scale


class Block(nn.Module):
    def __init__(self, model, index):
        super().__init__()
        self.block = model.encoder.layers[index]
        self.register_buffer("positions", model.encoder.encode_positions(torch.zeros(1, FRAMES, 1024)))
        a = self.block.self_attn
        self.register_buffer("qkv_weight", torch.cat([p.weight for p in (a.q_proj, a.k_proj, a.v_proj)]))
        self.register_buffer(
            "qkv_bias",
            None if a.q_proj.bias is None else torch.cat([p.bias for p in (a.q_proj, a.k_proj, a.v_proj)]),
        )
        for number, ff in enumerate((self.block.feed_forward1, self.block.feed_forward2), 1):
            self.register_buffer(f"ff{number}_weight_half", ff.linear2.weight.detach() * 0.5)
            self.register_buffer(
                f"ff{number}_bias_half", None if ff.linear2.bias is None else ff.linear2.bias.detach() * 0.5
            )

    def _feed_forward(self, ff, x, number):
        x = ff.activation(ff.linear1(x))
        x = F.dropout(x, p=ff.activation_dropout, training=self.training)
        return F.linear(x, getattr(self, f"ff{number}_weight_half"), getattr(self, f"ff{number}_bias_half"))

    def _attention_nchw(self, norm, attention_bias):
        a = self.block.self_attn
        qkv = F.conv2d(norm.transpose(1, 2).unsqueeze(2), self.qkv_weight[:, :, None, None], self.qkv_bias)
        q, k, v = [torch.cat(x.split(128, dim=1), dim=2) for x in qkv.split(1024, dim=1)]
        bu = a.bias_u.reshape(1, 8, 128, 1).permute(0, 2, 1, 3)
        bv = a.bias_v.reshape(1, 8, 128, 1).permute(0, 2, 1, 3)
        rk = a.relative_k_proj(self.positions).reshape(1, 2 * FRAMES - 1, 8, 128).permute(0, 3, 2, 1)
        ac = torch.einsum("nchw,nchq->nqhw", q + bu, k) * a.scaling
        relative = torch.einsum("nchw,nchq->nqhw", q + bv, rk)
        shifted = F.pad(relative.permute(0, 3, 1, 2), (0, 0, 1, 0))
        shifted = shifted.reshape(1, 2 * FRAMES, FRAMES, 8)[:, 1:]
        shifted = shifted.reshape(1, FRAMES, 2 * FRAMES - 1, 8)[:, :, :FRAMES]
        bd = shifted.permute(0, 2, 3, 1) * a.scaling
        # Key positions occupy axis 1 throughout score normalization.
        weights = F.softmax(ac + bd + attention_bias[..., :FRAMES].reshape(1, FRAMES, 1, 1), dim=1)
        attended = torch.einsum("nchw,nqhc->nqhw", weights, v)
        attended = torch.cat(attended.split(1, dim=2), dim=1)
        projected = F.conv2d(attended, a.o_proj.weight[:, :, None, None], a.o_proj.bias)
        return projected.squeeze(2).transpose(1, 2)

    def forward(self, hidden, padding_masks, attention_bias):
        b = self.block
        x = hidden + self._feed_forward(b.feed_forward1, b.norm_feed_forward1(hidden), 1)
        norm = b.norm_self_att(x)
        projected = self._attention_nchw(norm, attention_bias)
        x = x + projected
        c = b.conv
        conv = c.pointwise_conv1(b.norm_conv(x).transpose(1, 2))
        conv = F.glu(conv, dim=1)
        # Padded queries are irrelevant, but must not leak through depthwise Conv.
        conv = conv * padding_masks[:, :, 1056:1231, 0]
        conv = c.pointwise_conv2(c.activation(c.norm(c.depthwise_conv(conv))))
        x = x + conv.transpose(1, 2)
        x = x + self._feed_forward(b.feed_forward2, b.norm_feed_forward2(x), 2)
        return b.norm_out(x)


class Projection(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.projection = model.encoder_projector

    def forward(self, hidden):
        return self.projection(hidden)


class Predictor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.decoder = model.decoder

    def forward(self, token, hidden, cell):
        x = self.decoder.embedding(token)
        x, (h, c) = self.decoder.lstm(x, (hidden, cell))
        return self.decoder.decoder_projector(x), h, c


class Joiner(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.joint = model.joint

    def forward(self, encoder_frame, predictor_output):
        logits = self.joint(predictor_output, encoder_frame)
        return logits[..., :8193], logits[..., 8193:]


def compose(model, directory):
    import gc

    from onnx import helper
    from onnx_export import append_graph, export_part

    result = helper.make_model(
        helper.make_graph([], "parakeet_buffered", [], []), opset_imports=[helper.make_opsetid("", 17)], ir_version=8
    )
    public_inputs = set()

    def add(module, arguments, inputs, outputs, prefix, replacements):
        part = export_part(module, arguments, inputs, outputs, directory / "part.onnx")
        graph = append_graph(result, part, prefix=prefix, replacements=replacements)
        produced = {name for node in result.graph.node for name in node.output}
        for value in graph.input:
            if value.name not in produced and value.name not in public_inputs:
                result.graph.input.append(value)
                public_inputs.add(value.name)
        result.graph.value_info.extend(graph.output)
        return graph

    masks, bias = controls(MEL)
    graph = add(
        Subsampling(model),
        (torch.zeros(1, 1, MEL, 128), masks),
        ["features", "padding_masks"],
        ["hidden"],
        "sub/",
        {"sub/features": "features", "sub/padding_masks": "padding_masks"},
    )
    hidden = graph.output[0].name
    for layer in range(24):
        prefix = f"b{layer:02d}/"
        graph = add(
            Block(model, layer),
            (torch.zeros(1, FRAMES, 1024), masks, bias),
            ["hidden", "padding_masks", "attention_bias"],
            ["next_hidden"],
            prefix,
            {
                prefix + "hidden": hidden,
                prefix + "padding_masks": "padding_masks",
                prefix + "attention_bias": "attention_bias",
            },
        )
        hidden = graph.output[0].name
        gc.collect()
    graph = add(
        Projection(model),
        (torch.zeros(1, FRAMES, 1024),),
        ["hidden"],
        ["encoded"],
        "projection/",
        {"projection/hidden": hidden, "projection/encoded": "encoded"},
    )
    result.graph.output.extend(graph.output)
    encoder_outputs = ["encoded"]
    return result, encoder_outputs


MODEL_DIR = Path(__file__).resolve().parents[1]
KIND = "parakeet"


def export_checkpoint(checkpoint: Path, output: Path) -> Path:
    import onnx
    import torch
    from transformers import AutoModelForTDT, AutoProcessor

    torch.set_num_threads(4)
    checkpoint = checkpoint.parent if checkpoint.is_file() else checkpoint
    output.mkdir(parents=True, exist_ok=False)
    model = AutoModelForTDT.from_pretrained(
        checkpoint, dtype=torch.float32, attn_implementation="sdpa", local_files_only=True
    ).eval()
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    with tempfile.TemporaryDirectory(dir=output, prefix="export-") as work:
        with torch.inference_mode():
            graph, encoder_outputs = compose(model, Path(work))
    from onnx_export import export_part

    for module, arguments, inputs, outputs, name in (
        (
            Predictor(model),
            (torch.tensor([[8192]]), torch.zeros(2, 1, 640), torch.zeros(2, 1, 640)),
            ["token", "hidden", "cell"],
            ["prediction", "next_hidden", "next_cell"],
            "predictor",
        ),
        (
            Joiner(model),
            (torch.zeros(1, 1, 640), torch.zeros(1, 1, 640)),
            ["encoder_frame", "predictor_output"],
            ["token_logits", "duration_logits"],
            "joiner",
        ),
    ):
        path = output / (name + ".onnx")
        cpu_graph = export_part(module, arguments, inputs, outputs, path)
        onnx.save_model(cpu_graph, path)
        onnx.checker.check_model(str(path))
    runtime = {
        "kind": KIND,
        "sample_rate": 16000,
        "blank_token_id": 8192,
        "max_symbols_per_step": int(model.config.max_symbols_per_step),
        "encoder_outputs": encoder_outputs,
        "frontend": frontend_metadata(processor),
        "assets": {
            name: json.loads((checkpoint / name).read_text()) for name in ("tokenizer.json", "processor_config.json")
        },
    }
    runtime.update(left_seconds=10, chunk_seconds=2, right_seconds=2, window_seconds=14)
    entry = graph.metadata_props.add()
    entry.key = "sima.asr"
    entry.value = json.dumps(runtime, sort_keys=True)
    for key, filename in (("license", "LICENSE.txt"), ("attribution", "NOTICE.txt")):
        item = graph.metadata_props.add()
        item.key, item.value = key, (MODEL_DIR / "scripts" / filename).read_text()
    path = output / "model.onnx"
    onnx.save_model(graph, path, save_as_external_data=True, location=path.name + ".ext")
    onnx.checker.check_model(str(path))
    validation = compare_checkpoint(path, model)
    (output / "source-validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    return path


def compare_checkpoint(converted: Path, model) -> dict:
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch

    converted = Path(converted)
    graph = onnx.load(converted, load_external_data=False)
    inputs = {
        value.name: np.zeros([dim.dim_value for dim in value.type.tensor_type.shape.dim], np.float32)
        for value in graph.graph.input
    }
    rng = np.random.default_rng(440)
    torch.set_num_threads(2)
    features = rng.normal(size=(1, 1400, 128)).astype(np.float32)
    inputs["features"][:, 0] = features
    inputs["padding_masks"].fill(1)
    inputs["attention_bias"][..., -1] = -10000
    with torch.inference_mode():
        reference = model.get_audio_features(
            torch.from_numpy(features), attention_mask=torch.ones(1, 1400, dtype=torch.long)
        ).pooler_output.numpy()
    tolerance = 0.003
    del model
    del graph
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(converted), options, providers=["CPUExecutionProvider"])
    actual = session.run(["encoded"], inputs)[0]
    np.testing.assert_allclose(actual, reference, atol=tolerance, rtol=2e-4)
    return {
        "checked_output": "encoded",
        "max_abs_error": float(np.max(np.abs(actual - reference))),
        "atol": tolerance,
        "rtol": 2e-4,
    }


def restore_checkpoint(source: Path, directory: Path) -> None:
    """Restore the pinned checkpoint locally, without fetching model data during compilation."""
    from safetensors.torch import save_file

    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    directory.mkdir(parents=True, exist_ok=False)
    for name, content in checkpoint["assets"].items():
        if Path(name).name != name or not name.endswith(".json"):
            raise ValueError(f"Unexpected checkpoint asset: {name}")
        (directory / name).write_bytes(content.encode("utf-8"))
    save_file(checkpoint["state_dict"], directory / "model.safetensors", metadata={"format": "pt"})


def export_model(source: Path, output: Path) -> Path:
    """Run PyTorch surgery and source-parity validation during compilation."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix="checkpoint-") as work:
        checkpoint = Path(work) / "checkpoint"
        restore_checkpoint(source, checkpoint)
        return export_checkpoint(checkpoint, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    export_model(args.weights.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
