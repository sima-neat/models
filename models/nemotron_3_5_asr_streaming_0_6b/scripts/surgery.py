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


class PaddingCache:
    def __init__(self, states):
        self.states = states
        self.next = {}

    def update(self, x, key, conv):
        old = self.states[key]
        axis = 2 if x.ndim == 4 else -1
        self.next[key] = torch.cat([old, x], dim=axis).narrow(
            axis, old.shape[axis] + x.shape[axis] - old.shape[axis], old.shape[axis]
        )
        return torch.cat([old, x], dim=axis)


class ProjectionStep(nn.Module):
    def __init__(self, model, frames=7):
        super().__init__()
        self.linear = model.prompt_projector.linear_1
        self.activation = model.prompt_projector.act
        tail = model.prompt_projector.linear_2
        projection = model.encoder_projector
        # Fold the consecutive affine layers before quantization, retaining the ReLU.
        self.register_buffer(
            "weight", (projection.weight.detach().double() @ tail.weight.detach().double()).to(projection.weight.dtype)
        )
        self.register_buffer(
            "bias",
            F.linear(
                tail.bias.detach().double(), projection.weight.detach().double(), projection.bias.detach().double()
            ).to(projection.weight.dtype),
        )
        self.frames = frames

    def forward(self, hidden, language_one_hot):
        prompt = language_one_hot.expand(1, self.frames, 128)
        hidden = self.activation(self.linear(torch.cat([hidden, prompt], dim=-1)))
        return F.linear(hidden, self.weight, self.bias)


class PredictorStep(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.decoder = model.decoder

    def forward(self, token, hidden, cell):
        x = self.decoder.embedding(token)
        x, (h, c) = self.decoder.lstm(x, (hidden, cell))
        return self.decoder.decoder_projector(x), h, c


class JoinerStep(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.joint = model.joint

    def forward(self, encoder_frame, predictor_output):
        return self.joint(predictor_output, encoder_frame)


class UnifiedSubsamplingConvProjectionStep(nn.Module):
    """One steady-shape graph; stage masks reproduce first-chunk zero padding."""

    def __init__(self, model, frames=7):
        super().__init__()
        self.subsampling = model.encoder.subsampling
        self.scale = model.encoder.input_scale
        self.frames = frames

    def forward(self, features, sub0, sub1, sub2, subsampling_mask):
        s = self.subsampling
        cache = PaddingCache({"subsampling.0": sub0[..., :131], "subsampling.1": sub1, "subsampling.2": sub2})
        # First chunk: prepend seven zero mel frames; zero the three artificial
        # stem rows and one artificial second-stage row, including their biases.
        x = s.act_fn(s.conv_in(features.unsqueeze(1), padding_cache=cache))
        x = x * subsampling_mask[:, :, : self.frames * 4, :]
        x, _ = s.layers[0](x, None, padding_cache=cache)
        x = s.act_fn(x) * subsampling_mask[:, :, 32 : 32 + self.frames * 2, :]
        x, _ = s.layers[1](x, None, padding_cache=cache)
        x = s.act_fn(x)
        x = F.conv2d(x, s.linear.weight.reshape(1024, 256, 1, 17), s.linear.bias)
        x = x.permute(0, 2, 3, 1).reshape(1, self.frames, 1024) * self.scale
        return x, cache.next["subsampling.1"], cache.next["subsampling.2"]


class OptimizedBlockStep(nn.Module):
    def __init__(self, model, index, frames=7):
        super().__init__()
        self.block = model.encoder.layers[index]
        self.index = index
        self.frames = frames
        positions = model.encoder.encode_positions(torch.zeros(1, frames, 1024), cached_frames=56)
        self.register_buffer("positions", positions)
        a = self.block.self_attn
        self.register_buffer(
            "qkv_weight", torch.cat([p.weight.detach() for p in (a.q_proj, a.k_proj, a.v_proj)])[:, :, None, None]
        )
        self.register_buffer(
            "qkv_bias",
            None if a.q_proj.bias is None else torch.cat([p.bias.detach() for p in (a.q_proj, a.k_proj, a.v_proj)]),
        )
        self.register_buffer("attention_output_weight", a.o_proj.weight.detach()[:, :, None, None])

    @staticmethod
    def pack(value):
        return torch.cat(value.split(128, dim=1), dim=2)

    def forward(self, hidden, keys, values, conv, attention_bias):
        b = self.block
        a = b.self_attn
        x = hidden + 0.5 * b.feed_forward1(b.norm_feed_forward1(hidden))
        normalized = b.norm_self_att(x)
        projections = F.conv2d(normalized.transpose(1, 2).unsqueeze(2), self.qkv_weight, self.qkv_bias)
        q, k, v = [self.pack(part) for part in projections.split(1024, dim=1)]
        k = torch.cat([keys, k], dim=3)
        v = torch.cat([values, v], dim=3)
        qu = q + a.bias_u.reshape(1, 8, 128, 1).permute(0, 2, 1, 3)
        qv = q + a.bias_v.reshape(1, 8, 128, 1).permute(0, 2, 1, 3)
        length = 56 + self.frames
        # The shift reads only Q-1-query+key: 69 of the original 125 positions.
        rk = a.relative_k_proj(self.positions)[:, : length + self.frames - 1]
        rk = rk.reshape(1, length + self.frames - 1, 8, 128).permute(0, 3, 2, 1)
        relative = torch.einsum("nchw,nchq->nqhw", qv, rk)
        rows = [
            relative[:, self.frames - 1 - i : self.frames - 1 - i + length, :, i : i + 1] for i in range(self.frames)
        ]
        bd = torch.cat(rows, dim=3) * a.scaling
        ac = torch.einsum("nchw,nchq->nqhw", qu, k) * a.scaling
        bias = attention_bias[..., :length].permute(0, 3, 1, 2)
        weights = F.softmax(ac + bd + bias, dim=1)
        attended = torch.einsum("nchw,nqhc->nqhw", weights, v)
        attended = torch.cat(attended.split(1, dim=2), dim=1)
        projected = F.conv2d(attended, self.attention_output_weight, a.o_proj.bias).squeeze(2).transpose(1, 2)
        x = x + projected
        cache = PaddingCache({f"conv.{self.index}": conv})
        c = b.conv
        z = F.glu(c.pointwise_conv1(b.norm_conv(x).transpose(1, 2)), dim=1)
        z = c.depthwise_conv(z, padding_cache=cache).transpose(1, 2)
        z = c.activation(c.norm(z))
        z = F.linear(z, c.pointwise_conv2.weight[:, :, 0], c.pointwise_conv2.bias)
        x = x + z
        x = x + 0.5 * b.feed_forward2(b.norm_feed_forward2(x))
        return b.norm_out(x), k[:, :, :, -56:], v[:, :, :, -56:], cache.next[f"conv.{self.index}"]


def compose(model, directory):
    import gc

    from onnx import helper
    from onnx_export import append_graph, export_part

    result = helper.make_model(
        helper.make_graph([], "nemotron_streaming", [], []), opset_imports=[helper.make_opsetid("", 17)], ir_version=8
    )
    public_inputs = set()
    state_outputs = []

    def add(module, arguments, inputs, outputs, prefix, replacements):
        part = export_part(module, arguments, inputs, outputs, directory / "part.onnx")
        graph = append_graph(result, part, prefix=prefix, replacements=replacements)
        produced = {value for node in result.graph.node for value in node.output}
        for value in graph.input:
            if value.name not in produced and value.name not in public_inputs:
                result.graph.input.append(value)
                public_inputs.add(value.name)
        result.graph.value_info.extend(graph.output)
        return graph

    sub_inputs = ["features", "sub0", "sub1", "sub2", "subsampling_mask"]
    graph = add(
        UnifiedSubsamplingConvProjectionStep(model),
        (
            torch.zeros(1, 56, 128),
            torch.zeros(1, 1, 1, 136),
            torch.zeros(1, 256, 1, 68),
            torch.zeros(1, 256, 1, 36),
            torch.ones(1, 1, 48, 1),
        ),
        sub_inputs,
        ["hidden", "next_sub1", "next_sub2"],
        "sub/",
        {"sub/" + name: name for name in sub_inputs},
    )
    hidden = graph.output[0].name
    state_outputs.extend(graph.output[1:])
    for layer in range(24):
        prefix = f"b{layer:02d}/"
        bias = torch.zeros(1, 1, 7, 64)
        graph = add(
            OptimizedBlockStep(model, layer, 7),
            (
                torch.zeros(1, 7, 1024),
                torch.zeros(1, 128, 8, 56),
                torch.zeros(1, 128, 8, 56),
                torch.zeros(1, 1024, 8),
                bias,
            ),
            ["hidden", "keys", "values", "conv", "attention_bias"],
            ["next_hidden", "next_keys", "next_values", "next_conv"],
            prefix,
            {prefix + "hidden": hidden, prefix + "attention_bias": "attention_bias"},
        )
        hidden = graph.output[0].name
        state_outputs.extend(graph.output[1:])
        gc.collect()
    graph = add(
        ProjectionStep(model, 7),
        (torch.zeros(1, 7, 1024), torch.zeros(1, 1, 128)),
        ["hidden", "language_one_hot"],
        ["encoded"],
        "projection/",
        {
            "projection/hidden": hidden,
            "projection/language_one_hot": "language_one_hot",
            "projection/encoded": "encoded",
        },
    )
    result.graph.output.extend([*graph.output, *state_outputs])
    encoder_outputs = [value.name for value in result.graph.output]
    # Independently exported blocks repeat this preparation. Hoist equal constants
    # and operations; no model values or mask domains change.
    from onnx import numpy_helper

    constants = {value.name: value for value in result.graph.initializer}
    producers = {name: node for node in result.graph.node for name in node.output}
    shared = {}
    aliases = {}
    removed = set()
    for node in result.graph.node:
        if node.op_type != "Transpose" or list(helper.get_attribute_value(node.attribute[0])) != [0, 3, 1, 2]:
            continue
        sliced = producers.get(node.input[0])
        if sliced is None or sliced.op_type != "Slice" or sliced.input[0] != "attention_bias":
            continue
        signature = tuple(
            (
                numpy_helper.to_array(constants[name]).dtype.str,
                tuple(constants[name].dims),
                numpy_helper.to_array(constants[name]).tobytes(),
            )
            for name in sliced.input[1:]
        )
        if signature in shared:
            aliases[node.output[0]] = shared[signature]
            removed.update([node.name, sliced.name])
        else:
            shared[signature] = node.output[0]
    kept = []
    for node in result.graph.node:
        if node.name in removed:
            continue
        for i, name in enumerate(node.input):
            node.input[i] = aliases.get(name, name)
        kept.append(node)
    result.graph.ClearField("node")
    result.graph.node.extend(kept)
    return result, encoder_outputs


MODEL_DIR = Path(__file__).resolve().parents[1]
KIND = "nemotron"


def export_checkpoint(checkpoint: Path, output: Path) -> Path:
    import onnx
    import torch
    from transformers import AutoModelForRNNT, AutoProcessor

    torch.set_num_threads(4)
    checkpoint = checkpoint.parent if checkpoint.is_file() else checkpoint
    output.mkdir(parents=True, exist_ok=False)
    model = AutoModelForRNNT.from_pretrained(
        checkpoint, dtype=torch.float32, attn_implementation="sdpa", local_files_only=True
    ).eval()
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    with tempfile.TemporaryDirectory(dir=output, prefix="export-") as work:
        with torch.inference_mode():
            graph, encoder_outputs = compose(model, Path(work))
    from onnx_export import export_part

    for module, arguments, inputs, outputs, name in (
        (
            PredictorStep(model),
            (torch.tensor([[13087]]), torch.zeros(2, 1, 640), torch.zeros(2, 1, 640)),
            ["token", "hidden", "cell"],
            ["prediction", "next_hidden", "next_cell"],
            "predictor",
        ),
        (
            JoinerStep(model),
            (torch.zeros(1, 1, 640), torch.zeros(1, 1, 640)),
            ["encoder_frame", "predictor_output"],
            ["logits"],
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
        "blank_token_id": 13087,
        "max_symbols_per_step": int(model.config.max_symbols_per_step),
        "encoder_outputs": encoder_outputs,
        "frontend": frontend_metadata(processor),
        "assets": {
            name: json.loads((checkpoint / name).read_text()) for name in ("tokenizer.json", "processor_config.json")
        },
    }
    runtime.update(chunk_ms=560, lookahead=6, history_frames=56)
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
    features = rng.normal(size=(1, 49, 128)).astype(np.float32)
    inputs["features"][:, 7:] = features
    inputs["subsampling_mask"].fill(1)
    inputs["subsampling_mask"][:, :, :3] = 0
    inputs["subsampling_mask"][:, :, 32] = 0
    inputs["attention_bias"][..., :56] = -10000
    inputs["language_one_hot"][..., 101] = 1
    with torch.inference_mode():
        hidden = torch.from_numpy(rng.normal(size=(1, 7, 1024)).astype(np.float32))
        prompt = torch.from_numpy(inputs["language_one_hot"])
        expected_projection = model.encoder_projector(
            model.prompt_projector(torch.cat([hidden, prompt.expand(1, 7, 128)], dim=-1))
        )
        projection = ProjectionStep(model, frames=7)
        torch.testing.assert_close(projection(hidden, prompt), expected_projection, atol=1e-3, rtol=1e-4)
        reference = model.get_audio_features(
            torch.from_numpy(features),
            prompt_ids=torch.tensor([101]),
            use_cache=True,
            num_lookahead_tokens=6,
            output_attention_mask=False,
        ).pooler_output.numpy()
    tolerance = 0.03
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
