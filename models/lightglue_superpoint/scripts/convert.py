"""Export LightGlue's matcher for SuperPoint keypoints from its Hugging Face checkpoint and check ONNX against PyTorch.

The ONNX matches two images' 512 keypoints. Inputs: encoding0 and encoding1 [1, 512, 128], each keypoint's rotary
position encoding, which the CPU computes in float64 (calibration.encoding), and descriptors0 and descriptors1
[1, 512, 256], SuperPoint's descriptors. Output: assignment [1, 512, 512], LightGlue's soft assignment: the probability
that keypoint i of image 0 matches keypoint j of image 1, the exponential of the log assignment LightGlue extracts its
matches from. Every layer runs, since early stopping and keypoint pruning are off, so the graph is static. The
checkpoint also holds SuperPoint, which the export leaves out.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from calibration import ENCODING_DIM, KEYPOINT_PROJECTION, encoding
from model_registry.fetch import Fetcher
from model_registry.model import load_and_validate_model

MODEL_DIR = Path(__file__).resolve().parents[1]

SOURCE = ("file_set", "model.safetensors", ["model.safetensors", "config.json"])
KEYPOINTS = 512
DESCRIPTOR_DIM = 256
OPSET = 18
INTERFACE = {
    "encoding0": [1, KEYPOINTS, ENCODING_DIM],
    "descriptors0": [1, KEYPOINTS, DESCRIPTOR_DIM],
    "encoding1": [1, KEYPOINTS, ENCODING_DIM],
    "descriptors1": [1, KEYPOINTS, DESCRIPTOR_DIM],
    "assignment": [1, KEYPOINTS, KEYPOINTS],
}
# SuperPoint's 640x480 input, in whose pixels the keypoints are found.
IMAGE_WIDTH, IMAGE_HEIGHT = 640, 480
# LightGlue's filter_threshold: a mutual best pair matches when its assignment exceeds it.
MATCH_THRESHOLD = 0.1
SEEDS = range(123, 128)


def validate_source(source):
    declared = (source.get("type"), source.get("entrypoint"), [item.get("path") for item in source.get("files", [])])
    if declared != SOURCE:
        raise ValueError(f"Expected the Hugging Face file set {SOURCE}, model.yaml declares {declared}")


def validate_config(directory):
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    declared = (config.get("model_type"), config.get("descriptor_dim"))
    if declared != ("lightglue", DESCRIPTOR_DIM):
        raise ValueError(f"config.json declares model_type and descriptor_dim {declared}, expected ('lightglue', 256)")


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


def known_shapes(model):
    inferred = onnx.shape_inference.infer_shapes(model).graph
    shapes = {}
    for value in [*inferred.value_info, *inferred.input, *inferred.output]:
        dims = value.type.tensor_type.shape.dim
        if dims and all(dim.HasField("dim_value") for dim in dims):
            shapes[value.name] = [dim.dim_value for dim in dims]
    return shapes


def fold_constants(path):
    """Fold the export to a static graph. onnxsim leaves Shape nodes of tensors whose shapes are known; each becomes a
    constant and the graph is folded again, until none is left."""
    from onnxsim import simplify

    model, valid = simplify(onnx.load(path), perform_optimization=False)
    for _ in range(8):
        if not valid:
            raise RuntimeError("onnxsim could not validate the folded LightGlue export")
        shapes = known_shapes(model)
        replaced = 0
        for index, node in enumerate(model.graph.node):
            if node.op_type == "Shape" and node.input[0] in shapes:
                attributes = {item.name: onnx.helper.get_attribute_value(item) for item in node.attribute}
                value = shapes[node.input[0]][attributes.get("start", 0) : attributes.get("end")]
                constant = onnx.numpy_helper.from_array(np.asarray(value, np.int64))
                model.graph.node[index].CopyFrom(onnx.helper.make_node("Constant", [], list(node.output), value=constant))
                replaced += 1
        if not replaced:
            break
        model, valid = simplify(model, perform_optimization=False)
    onnx.save(model, path)


def matcher(lightglue):
    """LightGlue's matcher on two images' fixed keypoints, from their rotary encodings and descriptors, returning the
    soft assignment without the dustbins."""
    import torch
    from torch import nn

    class Matcher(nn.Module):
        def __init__(self):
            super().__init__()
            self.lightglue = lightglue

        @staticmethod
        def layer(layer, descriptors, encoding, mask):
            """LightGlueTransformerLayer.forward, with the swap of the two images for cross attention written as a
            concatenation: the exported flip(1) of the batch does not swap them."""
            attended, _ = layer.self_attention(descriptors, position_embeddings=encoding, attention_mask=mask)
            descriptors = descriptors + layer.self_mlp(torch.cat([descriptors, attended], -1))
            other = torch.cat([descriptors[1:2], descriptors[0:1]], 0)
            attended, _ = layer.cross_attention(descriptors, encoder_hidden_states=other,
                                                encoder_attention_mask=torch.cat([mask[1:2], mask[0:1]], 0))
            return descriptors + layer.cross_mlp(torch.cat([descriptors, attended], -1))

        def forward(self, encoding0, descriptors0, encoding1, descriptors1):
            # The positional encoder's (cosines, sines), computed on the CPU from the keypoints.
            encodings = torch.cat([encoding0, encoding1], 0)
            half = encodings.shape[-1] // 2
            rotary = (encodings[..., :half], encodings[..., half:])
            descriptors = self.lightglue.input_projection(torch.cat([descriptors0, descriptors1], 0))
            # Every keypoint is real: an all-zero additive mask lets each attend to all.
            mask = torch.zeros((2, 1, 1, encodings.shape[1]))
            for layer in self.lightglue.transformer_layers:
                descriptors = self.layer(layer, descriptors, rotary, mask)
            assignment = self.lightglue.match_assignment_layers[-1]
            projected = assignment.final_projection(descriptors) / assignment.descriptor_dim**0.25
            similarity = projected[0:1] @ projected[1:2].transpose(-1, -2)
            matchability = torch.sigmoid(assignment.matchability(descriptors))
            # LightGlue's log assignment is log_softmax over j + log_softmax over i + log sigmoid of each keypoint's
            # matchability; its exponential is this product, which stays in [0, 1].
            return (torch.softmax(similarity, 2) * torch.softmax(similarity, 1)
                    * matchability[0:1] * matchability[1:2].transpose(1, 2))

    return Matcher().eval()


def matches(assignment, threshold=MATCH_THRESHOLD):
    """LightGlue's match extraction: mutual best pairs whose assignment exceeds the threshold, as (i, j) rows."""
    best0, best1 = assignment.argmax(1), assignment.argmax(0)
    rows = np.arange(len(best0))
    keep = (best1[best0] == rows) & (assignment[rows, best0] > threshold)
    return np.stack([rows[keep], best0[keep]], 1)


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


def normalized(pixels):
    return ((pixels - [IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2]) / (max(IMAGE_WIDTH, IMAGE_HEIGHT) / 2)).astype(np.float32)


def check_projection(lightglue):
    """The CPU encoding uses calibration's copy of the positional encoder's projection; it must be the checkpoint's."""
    weight = lightglue.positional_encoder.projector.weight.detach().numpy()
    if weight.shape != KEYPOINT_PROJECTION.shape or not np.array_equal(weight, KEYPOINT_PROJECTION):
        raise ValueError("calibration.KEYPOINT_PROJECTION differs from the checkpoint's positional encoder projection")


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


def convert(source, output_dir):
    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "model.onnx"
    validate_config(source.parent)
    import onnxruntime as ort
    import torch
    from transformers import LightGlueForKeypointMatching

    torch.set_num_threads(4)
    # Early stopping and keypoint pruning off, as floats: config validation rejects the integer -1.
    lightglue = LightGlueForKeypointMatching.from_pretrained(source.parent, local_files_only=True,
                                                            depth_confidence=-1.0, width_confidence=-1.0,
                                                            attn_implementation="eager").eval()
    check_projection(lightglue)
    model = matcher(lightglue)
    (pixels0, descriptors0), (pixels1, descriptors1) = sample_pair(SEEDS.start)
    example = tuple(torch.from_numpy(value)[None] for value in
                    (encoding(normalized(pixels0)), descriptors0, encoding(normalized(pixels1)), descriptors1))
    # The TorchScript exporter records the attention heads' reshape with a fixed, wrong shape; torch.export traces it.
    program = torch.onnx.export(model, example, dynamo=True, opset_version=OPSET, input_names=list(INTERFACE)[:4],
                                output_names=["assignment"])
    program.save(str(output), external_data=False)
    fold_constants(output)
    validate_export(output)

    # transformers' own matcher keeps its log assignment in the last match assignment layer's output.
    captured = {}
    lightglue.match_assignment_layers[-1].register_forward_hook(lambda module, inputs, scores: captured.update(scores=scores))
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    for seed in SEEDS:
        (pixels0, descriptors0), (pixels1, descriptors1) = sample_pair(seed)
        with torch.inference_mode():
            lightglue._match_image_pair(torch.from_numpy(np.stack([pixels0, pixels1]))[None],
                                        torch.from_numpy(np.stack([descriptors0, descriptors1]))[None],
                                        IMAGE_HEIGHT, IMAGE_WIDTH, mask=torch.ones((1, 2, KEYPOINTS)))
        expected = captured["scores"][0, :-1, :-1].exp().numpy()
        feeds = dict(zip(list(INTERFACE)[:4], (encoding(normalized(pixels0))[None], descriptors0[None],
                                               encoding(normalized(pixels1))[None], descriptors1[None])))
        compare_assignment(expected, session.run(None, feeds)[0][0])
    print(f"Seeded-pair transformers/ONNX assignment and match check passed with seeds {SEEDS.start}-{SEEDS.stop - 1}: "
          f"{output}")
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
