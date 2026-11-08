"""Build SuperGlue outdoor's matcher for the MLA from the converted weights, export it and check it.

The rebuilt matcher runs the keypoint encoder, the attentional GNN and the final projection of one image pair as
4D, batch-one operations on [1, C, 1, N] tensors, and returns the Sinkhorn couplings: the descriptor scores bordered
by the learned dustbin score. The application runs the Sinkhorn iterations and match extraction on the CPU; the
export check below runs them as SuperGlue's log_optimal_transport does, on transformers' own layers and on the ONNX.

compile.py calls export_matcher. To get the FP32 ONNX on a host, run:

    python models/superglue_outdoor/scripts/surgery.py --weights weights.pth --output model.onnx
"""

import argparse
from pathlib import Path

import numpy as np
import onnx

EXPORTER = "4.57.1"
# transformers' SuperGlueConfig default, which the exported couplings are checked under.
SINKHORN_ITERATIONS = 100
KEYPOINTS = 512
DESCRIPTOR_SIZE = 256
# SuperGlueConfig of magic-leap-community/superglue_outdoor, written out so surgery needs only the weights.
CONFIG = {
    "hidden_size": DESCRIPTOR_SIZE,
    "keypoint_encoder_sizes": [32, 64, 128, 256],
    "gnn_layers_types": ["self", "cross"] * 9,
    "num_attention_heads": 4,
    "sinkhorn_iterations": SINKHORN_ITERATIONS,
    "matching_threshold": 0.0,
}
# Each image's keypoints run as the W axis of a [1, C, 1, N] tensor, so every layer is a 4D, batch-one MLA operation.
# A Neat tensor input is channels-last: the application passes [N, 2], [N] and [N, 256] arrays.
INPUTS = {
    "keypoints0": [1, 2, 1, KEYPOINTS],
    "scores0": [1, 1, 1, KEYPOINTS],
    "descriptors0": [1, DESCRIPTOR_SIZE, 1, KEYPOINTS],
    "keypoints1": [1, 2, 1, KEYPOINTS],
    "scores1": [1, 1, 1, KEYPOINTS],
    "descriptors1": [1, DESCRIPTOR_SIZE, 1, KEYPOINTS],
}
# couplings[0, j, 0, i] scores image-0 keypoint i against image-1 keypoint j; index KEYPOINTS is the dustbin.
OUTPUT = {"couplings": [1, KEYPOINTS + 1, 1, KEYPOINTS + 1]}
INTERFACE = {**INPUTS, **OUTPUT}
SEEDS = range(123, 129)
# Exported and PyTorch matches must agree on at least this share of keypoints.
MIN_ASSIGNMENT_AGREEMENT = 0.995


def validate_export(path):
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    if [item.version for item in model.opset_import if not item.domain] != [17]:
        raise ValueError("SuperGlue export requires opset 17")
    values = [*model.graph.input, *model.graph.output]
    if [value.name for value in values] != list(INTERFACE):
        raise ValueError(f"Expected inputs and outputs {list(INTERFACE)}")
    for value in values:
        tensor = value.type.tensor_type
        shape = [d.dim_value for d in tensor.shape.dim]
        if tensor.elem_type != onnx.TensorProto.FLOAT or shape != INTERFACE[value.name]:
            raise ValueError(f"Unexpected dtype or shape for {value.name}")
    if any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in model.graph.initializer):
        raise ValueError("This conversion workflow requires one self-contained ONNX file")


def load_superglue(weights):
    """Return transformers' SuperGlue with the converted weights."""
    import torch
    from transformers import SuperGlueConfig, SuperGlueForKeypointMatching

    model = SuperGlueForKeypointMatching(SuperGlueConfig(**CONFIG))
    model.load_state_dict(torch.load(weights, weights_only=True), strict=True)
    return model.eval()


def reference_matcher(model):
    """transformers' matcher layers run on one image pair, returning [1, N+1, N+1] couplings; the parity reference."""
    import torch

    class Reference(torch.nn.Module):
        def forward(self, keypoints0, scores0, descriptors0, keypoints1, scores1, descriptors1):
            hidden0 = descriptors0 + model.keypoint_encoder(keypoints0, scores0)[0]
            hidden1 = descriptors1 + model.keypoint_encoder(keypoints1, scores1)[0]
            for layer, kind in zip(model.gnn.layers, CONFIG["gnn_layers_types"]):
                other0, other1 = (hidden1, hidden0) if kind == "cross" else (None, None)
                delta0 = layer(hidden0, encoder_hidden_states=other0)[0]
                delta1 = layer(hidden1, encoder_hidden_states=other1)[0]
                hidden0, hidden1 = hidden0 + delta0, hidden1 + delta1
            projected0, projected1 = model.final_projection(hidden0), model.final_projection(hidden1)
            scores = projected0 @ projected1.transpose(1, 2) / DESCRIPTOR_SIZE**0.5
            right = model.bin_score.expand(1, KEYPOINTS, 1)
            bottom = model.bin_score.expand(1, 1, KEYPOINTS + 1)
            return torch.cat([torch.cat([scores, right], 2), bottom], 1)

    return Reference().eval()


def build_matcher(model):
    """Return the matcher with every layer as a 4D, batch-one operation on [1, C, 1, N] tensors.

    Linear layers become 1x1 convolutions with their BatchNorm folded in, and each attention runs per head as
    [1, 64, heads, N] Einsums with the softmax over keys on axis 1, the layout RF-DETR's decoder uses on the MLA.
    """
    import torch

    heads = CONFIG["num_attention_heads"]
    head_size = DESCRIPTOR_SIZE // heads

    def conv(linear, norm=None, scale=1.0):
        weight, bias = linear.weight.detach().clone(), linear.bias.detach().clone()
        if norm is not None:
            factor = norm.weight.detach() / torch.sqrt(norm.running_var.detach() + norm.eps)
            weight, bias = weight * factor[:, None], (bias - norm.running_mean.detach()) * factor + norm.bias.detach()
        layer = torch.nn.Conv2d(weight.shape[1], weight.shape[0], 1)
        layer.weight.data = (weight * scale)[:, :, None, None]
        layer.bias.data = bias * scale
        return layer

    def perceptron(layers):
        """An MLP of SuperGlueMultiLayerPerceptron blocks ending in a Linear, as 1x1 convolutions."""
        blocks = []
        for layer in layers[:-1]:
            blocks += [conv(layer.linear, layer.batch_norm), torch.nn.ReLU()]
        return torch.nn.Sequential(*blocks, conv(layers[-1]))

    class Attention(torch.nn.Module):
        def __init__(self, attention):
            super().__init__()
            self.query = conv(attention.self.query, scale=head_size**-0.5)
            self.key = conv(attention.self.key)
            self.value = conv(attention.self.value)
            self.output = conv(attention.output.dense)

        @staticmethod
        def split(tensor):
            # [1, heads*64, 1, N] -> [1, 64, heads, N]; channel h*64+d is head h, as transformers splits heads.
            return tensor.reshape(1, heads, head_size, KEYPOINTS).transpose(1, 2)

        def forward(self, hidden, source):
            query = self.split(self.query(hidden))
            key, value = self.split(self.key(source)), self.split(self.value(source))
            scores = torch.einsum("nchw,nchq->nqhw", query, key)
            probabilities = torch.softmax(scores, dim=1)
            context = torch.einsum("nchw,nqhc->nqhw", probabilities, value)
            merged = torch.cat(torch.split(context, 1, dim=2), dim=1)
            return self.output(merged)

    class Propagation(torch.nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.attention = Attention(layer.attention)
            self.mlp = perceptron(layer.mlp)

        def forward(self, hidden, source):
            return self.mlp(torch.cat([hidden, self.attention(hidden, source)], dim=1))

    class Matcher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = perceptron(model.keypoint_encoder.encoder)
            self.layers = torch.nn.ModuleList(Propagation(layer) for layer in model.gnn.layers)
            self.projection = conv(model.final_projection.final_proj, scale=DESCRIPTOR_SIZE**-0.25)
            self.register_buffer("bin_column", model.bin_score.detach().expand(1, KEYPOINTS, 1, 1).clone())
            self.register_buffer("bin_row", model.bin_score.detach().expand(1, 1, 1, KEYPOINTS + 1).clone())

        def forward(self, keypoints0, scores0, descriptors0, keypoints1, scores1, descriptors1):
            hidden0 = descriptors0 + self.encoder(torch.cat([keypoints0, scores0], dim=1))
            hidden1 = descriptors1 + self.encoder(torch.cat([keypoints1, scores1], dim=1))
            for layer, kind in zip(self.layers, CONFIG["gnn_layers_types"]):
                source0, source1 = (hidden1, hidden0) if kind == "cross" else (hidden0, hidden1)
                hidden0, hidden1 = hidden0 + layer(hidden0, source0), hidden1 + layer(hidden1, source1)
            # Both projections carry 256**-0.25, so their product carries SuperGlue's 1/sqrt(256).
            scores = torch.einsum("nchw,nchq->nqhw", self.projection(hidden0), self.projection(hidden1))
            # Image 0's dustbin is the last W column; image 1's is the last C row.
            return torch.cat([torch.cat([scores, self.bin_column], dim=3), self.bin_row], dim=1)

    return Matcher().eval()


def synthetic_pair(seed):
    """Two keypoint sets in which part of image 1 re-observes image 0, with SuperPoint-like scores and descriptors.

    Returns [N, ...] arrays: to_inputs shapes them for the export, and [None] for the reference.
    """
    rng = np.random.RandomState(seed)
    keypoints0 = rng.uniform(-0.7, 0.7, (KEYPOINTS, 2))
    descriptors0 = rng.normal(0, 1, (KEYPOINTS, DESCRIPTOR_SIZE))
    shared = rng.permutation(KEYPOINTS)[: KEYPOINTS // 2]
    keypoints1 = rng.uniform(-0.7, 0.7, (KEYPOINTS, 2))
    descriptors1 = rng.normal(0, 1, (KEYPOINTS, DESCRIPTOR_SIZE))
    target = rng.permutation(KEYPOINTS)[: len(shared)]
    keypoints1[target] = keypoints0[shared] * 0.95 + 0.02 + rng.normal(0, 0.003, (len(shared), 2))
    descriptors1[target] = descriptors0[shared] + rng.normal(0, 0.3, (len(shared), DESCRIPTOR_SIZE))
    unit = lambda values: values / np.linalg.norm(values, axis=-1, keepdims=True)  # noqa: E731
    sample = {
        "keypoints0": keypoints0, "scores0": rng.uniform(0.005, 0.5, KEYPOINTS), "descriptors0": unit(descriptors0),
        "keypoints1": keypoints1, "scores1": rng.uniform(0.005, 0.5, KEYPOINTS), "descriptors1": unit(descriptors1),
    }
    return {name: value.astype(np.float32) for name, value in sample.items()}


def to_inputs(sample):
    """Shape [N, ...] arrays as the exported graph's [1, C, 1, N] inputs."""
    return {
        name: np.ascontiguousarray(sample[name].reshape(KEYPOINTS, -1).T.reshape(shape))
        for name, shape in INPUTS.items()
    }


def couplings_matrix(couplings):
    """Return exported [1, N+1, 1, N+1] couplings as the [N+1, N+1] matrix with image-0 keypoints as rows."""
    return np.asarray(couplings).reshape(KEYPOINTS + 1, KEYPOINTS + 1).T


def log_sinkhorn(couplings, iterations):
    """Return the log optimal-transport matrix of [M+1, N+1] couplings, as SuperGlue's log_optimal_transport."""
    rows, columns = couplings.shape[-2] - 1, couplings.shape[-1] - 1
    normalization = -np.log(rows + columns)
    log_mu = np.concatenate([np.full(rows, normalization), [np.log(columns) + normalization]])
    log_nu = np.concatenate([np.full(columns, normalization), [np.log(rows) + normalization]])
    cost = np.asarray(couplings, dtype=np.float64).reshape(rows + 1, columns + 1)
    u, v = np.zeros_like(log_mu), np.zeros_like(log_nu)
    for _ in range(iterations):
        u = log_mu - _logsumexp(cost + v[None, :], axis=1)
        v = log_nu - _logsumexp(cost + u[:, None], axis=0)
    return cost + u[:, None] + v[None, :] - normalization


def _logsumexp(values, axis):
    top = values.max(axis=axis, keepdims=True)
    return (top + np.log(np.exp(values - top).sum(axis=axis, keepdims=True))).squeeze(axis)


def extract_matches(log_transport, threshold=0.0):
    """Return, for each image-0 keypoint, its mutual nearest image-1 keypoint (-1 when unmatched) and match score."""
    inner = log_transport[:-1, :-1]
    best0, best1 = inner.argmax(1), inner.argmax(0)
    mutual = best1[best0] == np.arange(len(best0))
    scores = np.where(mutual, np.exp(inner.max(1)), 0.0)
    valid = mutual & (scores > threshold)
    return np.where(valid, best0, -1), np.where(valid, scores, 0.0)


def compare_matches(expected, actual):
    """Require the exported couplings to give the reference's matches after Sinkhorn and match extraction."""
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError(f"Unexpected or non-finite couplings: {actual.shape} vs {expected.shape}")
    expected_matches, _ = extract_matches(log_sinkhorn(expected, SINKHORN_ITERATIONS))
    actual_matches, _ = extract_matches(log_sinkhorn(actual, SINKHORN_ITERATIONS))
    agreement = float((expected_matches == actual_matches).mean())
    if agreement < MIN_ASSIGNMENT_AGREEMENT or (expected_matches >= 0).sum() == 0:
        raise AssertionError(f"matches agree on only {agreement:.4f} of keypoints")
    return agreement


def fold_constants(path):
    from onnxsim import simplify

    model, valid = simplify(onnx.load(path), perform_optimization=False)
    if not valid:
        raise RuntimeError("onnxsim could not validate the folded SuperGlue export")
    onnx.save(share_initializers(model), path)


def share_initializers(model):
    """Point every use of identical weights at one initializer; folding copies the shared layers once per image."""
    first = {}
    renamed = {}
    kept = []
    for initializer in model.graph.initializer:
        array = onnx.numpy_helper.to_array(initializer)
        key = (array.dtype.str, array.shape, array.tobytes())
        if key in first:
            renamed[initializer.name] = first[key]
        else:
            first[key] = initializer.name
            kept.append(initializer)
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            node.input[index] = renamed.get(name, name)
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    return model


def export_matcher(weights, output):
    """Export the rebuilt matcher to output and require its matches to follow transformers' layers."""
    import onnxruntime as ort
    import torch
    import transformers

    if transformers.__version__ != EXPORTER:
        raise RuntimeError(f"Expected transformers {EXPORTER}, found {transformers.__version__}")
    torch.set_num_threads(4)
    model = load_superglue(weights)
    reference, matcher = reference_matcher(model), build_matcher(model)
    sample = to_inputs(synthetic_pair(SEEDS.start))
    arguments = tuple(torch.from_numpy(sample[name]) for name in INPUTS)
    torch.onnx.export(
        matcher, arguments, str(output), input_names=list(INPUTS), output_names=list(OUTPUT),
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
    fold_constants(output)
    validate_export(output)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(output), sess_options=options, providers=["CPUExecutionProvider"])
    for seed in SEEDS:
        pair = synthetic_pair(seed)
        with torch.inference_mode():
            expected = reference(*(torch.from_numpy(pair[name])[None] for name in INPUTS))[0].numpy()
        actual = couplings_matrix(session.run(None, to_inputs(pair))[0])
        agreement = compare_matches(expected, actual)
        print(
            f"seed {seed}: exported and reference matches agree on {agreement:.4f} of keypoints, "
            f"max coupling difference {np.abs(actual - expected).max():.2e}",
            flush=True,
        )
    print(f"Synthetic-input PyTorch/ONNX match check passed with seeds {SEEDS.start}-{SEEDS.stop - 1}: {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True, help="the converted weights.pth")
    parser.add_argument("--output", type=Path, required=True, help="the ONNX to write")
    args = parser.parse_args()
    export_matcher(args.weights.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
