"""Build LightGlue's matcher for the MLA from the converted weights, export it and check it.

The rebuilt matcher runs LightGlue's nine transformer layers and its final match assignment on two images' 512
keypoints as 4D, batch-one operations on [1, C, 1, 512] tensors, one per image, and returns the soft assignment: the
probability that keypoint i of image 0 matches keypoint j of image 1, the exponential of the log assignment LightGlue
extracts its matches from. Every layer runs, since early stopping and keypoint pruning are off, so the graph is
static. Each rewrite is exact and spares the compiler data movement:

- each Linear runs as a 1x1 convolution, and attention runs per head as [1, 64, heads, 512] Einsums with the softmax
  over keys on axis 1, as SuperGlue's matcher does;
- self attention's rotary encoding is folded into its query and key weights, and the power-of-two attention and
  similarity scales into the weights before them;
- the all-zero masks transformers adds are left out, since every keypoint is real.

The export check compares the ONNX's assignment with transformers' own matcher on seeded synthetic pairs.
compile.py calls export_matcher. To get the FP32 ONNX on a host, run:

    python models/lightglue_superpoint/scripts/surgery.py --weights weights.pth --output model.onnx
"""

import argparse
import math
from pathlib import Path

import numpy as np
import onnx
from calibration import (
    DESCRIPTOR_DIM,
    ENCODING_DIM,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    KEYPOINT_PROJECTION,
    KEYPOINTS,
    MATCH_THRESHOLD,
    encoding,
    matches,
    normalized,
)

# LightGlueConfig of ETH-CVG/lightglue_superpoint, which convert.py checks config.json against, with early stopping
# and keypoint pruning off. config validation rejects the integer -1, so they are floats.
CONFIG = {
    "descriptor_dim": DESCRIPTOR_DIM,
    "hidden_size": DESCRIPTOR_DIM,
    "intermediate_size": 2 * DESCRIPTOR_DIM,
    "num_hidden_layers": 9,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "hidden_act": "gelu",
    "attention_bias": True,
    "filter_threshold": MATCH_THRESHOLD,
    "depth_confidence": -1.0,
    "width_confidence": -1.0,
}
# Each image's keypoints run as the W axis of a [1, C, 1, N] tensor. A Neat tensor input is channels-last: the
# application passes [1, 1, 512, 128] and [1, 1, 512, 256] arrays.
INPUTS = {
    "encoding0": [1, ENCODING_DIM, 1, KEYPOINTS],
    "descriptors0": [1, DESCRIPTOR_DIM, 1, KEYPOINTS],
    "encoding1": [1, ENCODING_DIM, 1, KEYPOINTS],
    "descriptors1": [1, DESCRIPTOR_DIM, 1, KEYPOINTS],
}
# assignment[0, j, 0, i] scores image-0 keypoint i against image-1 keypoint j; channels-last, the application reads
# it as [i, j].
OUTPUT = {"assignment": [1, KEYPOINTS, 1, KEYPOINTS]}
INTERFACE = {**INPUTS, **OUTPUT}
OPSET = 17
SEEDS = range(123, 128)


def validate_export(path):
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    if [item.version for item in model.opset_import if not item.domain] != [OPSET]:
        raise ValueError(f"LightGlue export requires opset {OPSET}")
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
    dynamic = sorted({node.op_type for node in model.graph.node if node.op_type in {"Shape", "Size", "If", "Loop"}})
    if dynamic:
        raise ValueError(f"The folded export still computes shapes or control flow: {dynamic}")


def load_lightglue(weights):
    """transformers' LightGlue with the converted matcher weights. The converted weights leave out SuperPoint, so only
    its keypoint detector stays unloaded."""
    import torch
    from transformers import LightGlueConfig, LightGlueForKeypointMatching

    config = LightGlueConfig(**CONFIG)
    config._attn_implementation = "eager"
    model = LightGlueForKeypointMatching(config)
    missing, unexpected = model.load_state_dict(torch.load(weights, weights_only=True), strict=False)
    if unexpected or any(not name.startswith("keypoint_detector.") for name in missing):
        raise ValueError(f"weights.pth does not hold LightGlue's matcher: missing {missing}, unexpected {unexpected}")
    projection = model.positional_encoder.projector.weight.detach().numpy()
    # The CPU encoding uses calibration's copy of the positional encoder's projection; it must be the checkpoint's.
    if not np.array_equal(projection, KEYPOINT_PROJECTION):
        raise ValueError("calibration.KEYPOINT_PROJECTION differs from the checkpoint's positional encoder projection")
    return model.eval()


def rotation(rotate_half, head_dim):
    """The matrix R of one head's channels with rotate_half(x) = R x, taken from transformers' rotate_half itself: on
    the identity's rows e_i it returns R e_i, R's columns. R must be a signed permutation, and unique channel values
    prove it reproduces rotate_half, since any slip in the pairing or the signs shows up in them."""
    import torch

    matrix = rotate_half(torch.eye(head_dim, dtype=torch.float64)).T
    channels = torch.arange(1, head_dim + 1, dtype=torch.float64)
    ones = torch.ones(head_dim, dtype=torch.float64)
    if (not torch.equal(matrix.abs().sum(0), ones) or not torch.equal(matrix.abs().sum(1), ones)
            or not torch.equal(matrix @ channels, rotate_half(channels))):
        raise ValueError("transformers' rotate_half is not a fixed signed permutation of each head's channels")
    return matrix.float()


def build_matcher(lightglue):
    """LightGlue's matcher with every layer as a 4D, batch-one operation on [1, C, 1, N] tensors."""
    import torch
    from torch import nn
    from transformers.models.lightglue.modeling_lightglue import rotate_half

    if not isinstance(lightglue.input_projection, nn.Identity):
        raise ValueError("LightGlue projects the descriptors before its layers; the rebuilt matcher does not")
    heads = CONFIG["num_attention_heads"]
    head_dim = DESCRIPTOR_DIM // heads
    rotate = rotation(rotate_half, head_dim)

    def conv(linear, scale=1.0, rotated=False):
        """A Linear as a 1x1 convolution, its output times scale, and with rotated, times R in each head."""
        weight, bias = linear.weight.detach() * scale, linear.bias.detach() * scale
        if rotated:
            weight = (rotate @ weight.view(heads, head_dim, -1)).reshape(weight.shape)
            bias = (rotate @ bias.view(heads, head_dim, 1)).reshape(bias.shape)
        layer = nn.Conv2d(weight.shape[1], weight.shape[0], 1)
        layer.weight.data = weight[:, :, None, None].clone()
        layer.bias.data = bias.clone()
        return layer

    def split(tensor):
        # [1, heads*64, 1, N] -> [1, 64, heads, N]; channel h*64+d is head h, as transformers splits heads.
        return tensor.reshape(1, heads, head_dim, KEYPOINTS).transpose(1, 2)

    class Attention(nn.Module):
        """LightGlueAttention, with its 1/8 scale, a power of two, folded into the query weights."""

        def __init__(self, attention):
            super().__init__()
            if not math.log2(attention.scaling).is_integer():
                raise ValueError(f"The attention scale {attention.scaling} is no power of two; folding it is inexact")
            self.query = conv(attention.q_proj, attention.scaling)
            self.key = conv(attention.k_proj)
            self.value = conv(attention.v_proj)
            self.output = conv(attention.o_proj)

        def query_and_key(self, hidden, source, cos, sin):
            return split(self.query(hidden)), split(self.key(source))

        def forward(self, hidden, source, cos=None, sin=None):
            query, key = self.query_and_key(hidden, source, cos, sin)
            probabilities = torch.softmax(torch.einsum("nchw,nchq->nqhw", query, key), dim=1)
            context = torch.einsum("nchw,nqhc->nqhw", probabilities, split(self.value(source)))
            return self.output(torch.cat(torch.split(context, 1, dim=2), dim=1))

    class RotaryAttention(Attention):
        """Self attention. transformers rotates queries and keys as q * cos + rotate_half(q) * sin, and rotate_half is
        the fixed signed permutation R of each head's channels, so rotate_half(W x + b) = (R W) x + R b is a second
        convolution: the encoding needs no slices, negations or stacks."""

        def __init__(self, attention):
            super().__init__(attention)
            self.rotated_query = conv(attention.q_proj, attention.scaling, rotated=True)
            self.rotated_key = conv(attention.k_proj, rotated=True)

        def query_and_key(self, hidden, source, cos, sin):
            # cos and sin [1, 64, 1, N] apply alike to every head.
            query = split(self.query(hidden)) * cos + split(self.rotated_query(hidden)) * sin
            key = split(self.key(source)) * cos + split(self.rotated_key(source)) * sin
            return query, key

    class Perceptron(nn.Module):
        """LightGlueMLP: fc1, the layer norm over each keypoint's channels, GELU and fc2. GELU is written as
        0.5 x (1 + erf(x / sqrt(2))) with a multiply by 1 / sqrt(2), the form the compiler matches as one GELU; torch's
        own export divides by sqrt(2) instead."""

        def __init__(self, mlp):
            super().__init__()
            self.first, self.second = conv(mlp.fc1), conv(mlp.fc2)
            self.epsilon = mlp.layer_norm.eps
            self.register_buffer("scale", mlp.layer_norm.weight.detach().view(1, -1, 1, 1).clone())
            self.register_buffer("shift", mlp.layer_norm.bias.detach().view(1, -1, 1, 1).clone())

        def forward(self, hidden):
            hidden = self.first(hidden)
            centered = hidden - hidden.mean(1, keepdim=True)
            hidden = centered / torch.sqrt(centered.square().mean(1, keepdim=True) + self.epsilon)
            hidden = hidden * self.scale + self.shift
            return self.second(0.5 * hidden * (1 + torch.erf(hidden * (1 / math.sqrt(2)))))

    class Layer(nn.Module):
        """LightGlueTransformerLayer: self attention within each image, then cross attention between them."""

        def __init__(self, layer):
            super().__init__()
            self.self_attention, self.self_mlp = RotaryAttention(layer.self_attention), Perceptron(layer.self_mlp)
            self.cross_attention, self.cross_mlp = Attention(layer.cross_attention), Perceptron(layer.cross_mlp)

        def forward(self, hidden0, hidden1, rotary0, rotary1):
            hidden0 = hidden0 + self.self_mlp(torch.cat([hidden0, self.self_attention(hidden0, hidden0, *rotary0)], 1))
            hidden1 = hidden1 + self.self_mlp(torch.cat([hidden1, self.self_attention(hidden1, hidden1, *rotary1)], 1))
            attended0, attended1 = self.cross_attention(hidden0, hidden1), self.cross_attention(hidden1, hidden0)
            return (hidden0 + self.cross_mlp(torch.cat([hidden0, attended0], 1)),
                    hidden1 + self.cross_mlp(torch.cat([hidden1, attended1], 1)))

    class Matcher(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(Layer(layer) for layer in lightglue.transformer_layers)
            assignment = lightglue.match_assignment_layers[-1]
            # descriptor_dim ** -0.25 on each image's projection gives the similarity's 1 / sqrt(256).
            self.projection = conv(assignment.final_projection, assignment.descriptor_dim**-0.25)
            self.matchability = conv(assignment.matchability)

        def forward(self, encoding0, descriptors0, encoding1, descriptors1):
            # Each encoding holds the cosines, then the sines, of the keypoint's rotary phases.
            half = ENCODING_DIM // 2
            rotary0 = encoding0[:, :half], encoding0[:, half:]
            rotary1 = encoding1[:, :half], encoding1[:, half:]
            hidden0, hidden1 = descriptors0, descriptors1
            for layer in self.layers:
                hidden0, hidden1 = layer(hidden0, hidden1, rotary0, rotary1)
            # similarity[0, j, 0, i] of image-0 keypoint i and image-1 keypoint j.
            similarity = torch.einsum("nchw,nchq->nqhw", self.projection(hidden0), self.projection(hidden1))
            matchability0 = torch.sigmoid(self.matchability(hidden0))
            matchability1 = torch.sigmoid(self.matchability(hidden1)).reshape(1, KEYPOINTS, 1, 1)
            # LightGlue's log assignment is log_softmax over j + log_softmax over i + log sigmoid of each keypoint's
            # matchability; its exponential is this product, which stays in [0, 1].
            return torch.softmax(similarity, 1) * torch.softmax(similarity, 3) * matchability0 * matchability1

    return Matcher().eval()


def sample_pair(seed):
    """Two images' keypoints for one seed: image 1 sees most of image 0's keypoints, shifted and with noisy descriptors,
    in another order, so the pair has many true matches as well as unmatched keypoints."""
    rng = np.random.default_rng(seed)
    size = np.asarray([IMAGE_WIDTH, IMAGE_HEIGHT], np.float64)
    pixels0 = rng.uniform(0, 1, (KEYPOINTS, 2)) * size
    descriptors0 = rng.normal(0, 1, (KEYPOINTS, DESCRIPTOR_DIM))
    order = rng.permutation(KEYPOINTS)
    shared = order[: KEYPOINTS * 3 // 4]
    pixels1 = rng.uniform(0, 1, (KEYPOINTS, 2)) * size
    descriptors1 = rng.normal(0, 1, (KEYPOINTS, DESCRIPTOR_DIM))
    pixels1[: len(shared)] = np.clip(pixels0[shared] + rng.normal(0, 2, (len(shared), 2)) + [12.0, -7.0], 0, size - 1)
    descriptors1[: len(shared)] = descriptors0[shared] + rng.normal(0, 0.3, (len(shared), DESCRIPTOR_DIM))
    pair = []
    for pixels, descriptors in ((pixels0, descriptors0), (pixels1, descriptors1)):
        descriptors /= np.linalg.norm(descriptors, axis=1, keepdims=True)  # SuperPoint's descriptors are unit length
        pair.append((pixels.astype(np.float32), descriptors.astype(np.float32)))
    return pair


def onnx_inputs(pair):
    """A pair's ONNX inputs: each image's [512, C] encoding and descriptors as [1, C, 1, 512]."""
    arrays = []
    for pixels, descriptors in pair:
        arrays += [encoding(normalized(pixels)), descriptors]
    return {name: np.ascontiguousarray(array.T[None, :, None]) for name, array in zip(INPUTS, arrays)}


def assignment_matrix(output):
    """The ONNX's [1, j, 1, i] assignment as the matrix with image-0 keypoints as rows."""
    return np.asarray(output).reshape(KEYPOINTS, KEYPOINTS).T


def compare_assignment(expected, actual):
    """The exported assignment must equal transformers' to float32 rounding and give the same matches; a pair whose
    assignment sits at the threshold may fall either side, so 1% of the matches may differ."""
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError(f"Unexpected or non-finite assignment: {actual.shape} vs {expected.shape}")
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-3, err_msg="assignment")
    reference = {tuple(row) for row in matches(expected).tolist()}
    exported = {tuple(row) for row in matches(actual).tolist()}
    if len(reference) < KEYPOINTS // 4:
        raise ValueError(f"transformers found only {len(reference)} matches in a pair built to have hundreds")
    if len(reference ^ exported) > len(reference) // 100:
        raise ValueError(f"The export matches {len(exported)} pairs and transformers {len(reference)}; "
                         f"{len(reference ^ exported)} differ")


def fold_constants(path):
    from onnxsim import simplify

    model, valid = simplify(onnx.load(path), perform_optimization=False)
    if not valid:
        raise RuntimeError("onnxsim could not validate the folded LightGlue export")
    onnx.save(model, path)


def export_matcher(weights, output):
    """Export the rebuilt matcher to output and require its assignment to follow transformers' matcher."""
    import onnxruntime as ort
    import torch

    torch.set_num_threads(4)
    lightglue = load_lightglue(weights)
    matcher = build_matcher(lightglue)
    example = tuple(torch.from_numpy(value) for value in onnx_inputs(sample_pair(SEEDS.start)).values())
    torch.onnx.export(
        matcher, example, str(output), input_names=list(INPUTS), output_names=list(OUTPUT),
        opset_version=OPSET, do_constant_folding=True, dynamo=False,
    )
    fold_constants(output)
    validate_export(output)

    # transformers' own matcher keeps its log assignment in the last match assignment layer's output.
    captured = {}
    lightglue.match_assignment_layers[-1].register_forward_hook(
        lambda module, inputs, scores: captured.update(scores=scores))
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    for seed in SEEDS:
        pair = sample_pair(seed)
        (pixels0, descriptors0), (pixels1, descriptors1) = pair
        with torch.inference_mode():
            lightglue._match_image_pair(torch.from_numpy(np.stack([pixels0, pixels1]))[None],
                                        torch.from_numpy(np.stack([descriptors0, descriptors1]))[None],
                                        IMAGE_HEIGHT, IMAGE_WIDTH, mask=torch.ones((1, 2, KEYPOINTS)))
        expected = captured["scores"][0, :-1, :-1].exp().numpy()
        compare_assignment(expected, assignment_matrix(session.run(None, onnx_inputs(pair))[0]))
    print(f"Seeded-pair transformers/ONNX assignment and match check passed with seeds {SEEDS.start}-{SEEDS.stop - 1}: "
          f"{output}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True, help="the converted weights.pth")
    parser.add_argument("--output", type=Path, required=True, help="the ONNX to write")
    args = parser.parse_args()
    export_matcher(args.weights.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
