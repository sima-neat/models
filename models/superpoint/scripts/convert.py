"""Export SuperPoint up to its raw heads and check ONNX plus the CPU decode against transformers.

The exported graph takes one 640x480 grayscale image in [0, 1] and returns the 65 keypoint logits and the 256
descriptor channels before L2 normalization, both on the 80x60 grid of 8x8 cells. The application decodes them on
the CPU, or with Neat's SuperPoint decoder; keypoints.py is the reference decode.
"""

import argparse
from pathlib import Path, PurePosixPath

import numpy as np
import onnx
from keypoints import (
    BORDER,
    CELL,
    DESCRIPTOR_SIZE,
    HEIGHT,
    KEYPOINT_CHANNELS,
    KEYPOINT_THRESHOLD,
    NMS_RADIUS,
    WIDTH,
    decode,
    synthetic_image,
)
from model_registry.fetch import Fetcher
from model_registry.model import load_model as load_metadata

MODEL_DIR = Path(__file__).resolve().parents[1]

CHECKPOINT = "model.safetensors"
EXPORTER = "4.57.1"
OPSET = 17
# SuperPointConfig of magic-leap-community/superpoint, written out so conversion needs only the weights.
CONFIG = {
    "encoder_hidden_sizes": [64, 64, 128, 128],
    "decoder_hidden_size": 256,
    "keypoint_decoder_dim": KEYPOINT_CHANNELS,
    "descriptor_decoder_dim": DESCRIPTOR_SIZE,
    "keypoint_threshold": KEYPOINT_THRESHOLD,
    "max_keypoints": -1,
    "nms_radius": NMS_RADIUS,
    "border_removal_distance": BORDER,
}
INPUT = {"image": [1, 1, HEIGHT, WIDTH]}
OUTPUTS = {
    "logits": [1, KEYPOINT_CHANNELS, HEIGHT // CELL, WIDTH // CELL],
    "descriptors": [1, DESCRIPTOR_SIZE, HEIGHT // CELL, WIDTH // CELL],
}
INTERFACE = {**INPUT, **OUTPUTS}
SEEDS = range(11, 17)
DESCRIPTOR_TOLERANCE = 1e-4


def validate_source(source):
    declared = PurePosixPath(str(source.get("uri", ""))).name
    if declared != CHECKPOINT:
        raise ValueError(f"Expected the checkpoint file {CHECKPOINT}, model.yaml declares {declared!r}")


def validate_export(path):
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    if [item.version for item in model.opset_import if not item.domain] != [OPSET]:
        raise ValueError(f"SuperPoint export requires opset {OPSET}")
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


def load_superpoint(checkpoint):
    """Return transformers' SuperPoint with the checkpoint's weights."""
    from safetensors.torch import load_file
    from transformers import SuperPointConfig, SuperPointForKeypointDetection

    model = SuperPointForKeypointDetection(SuperPointConfig(**CONFIG))
    model.load_state_dict(load_file(str(checkpoint)), strict=True)
    return model.eval()


def build_heads(model):
    """SuperPoint's encoder and its two heads up to their last convolutions, the graph the MPK runs."""
    import torch

    class Heads(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = model.encoder
            self.keypoints = model.keypoint_decoder
            self.descriptors = model.descriptor_decoder

        def forward(self, image):
            encoded = self.encoder(image, return_dict=False)[0]
            logits = self.keypoints.conv_score_b(torch.relu(self.keypoints.conv_score_a(encoded)))
            descriptors = self.descriptors.conv_descriptor_b(torch.relu(self.descriptors.conv_descriptor_a(encoded)))
            return logits, descriptors

    return Heads().eval()


def compare_with_reference(model, image, logits, descriptors):
    """Require the decode of the exported heads to give transformers' keypoints, scores and descriptors."""
    import torch

    with torch.inference_mode():
        pixels = torch.from_numpy(np.repeat(image[None, None], 3, axis=1))
        keypoints, scores, expected_descriptors, mask = model(pixels, return_dict=False)[:4]
    expected_keypoints = (keypoints[0, mask[0].bool()].numpy() * [WIDTH, HEIGHT]).round().astype(np.float32)
    # transformers removes the border at the top and left only: it checks the bottom and right edges against
    # the image size times 8 again. The decode removes all four sides, as Magic Leap's code does.
    kept = (expected_keypoints[:, 0] < WIDTH - BORDER) & (expected_keypoints[:, 1] < HEIGHT - BORDER)
    count = int(kept.sum())
    threshold = model.keypoint_decoder.keypoint_threshold
    actual_keypoints, actual_scores, actual_descriptors = decode(logits[0], descriptors[0], threshold)
    if count == 0 or not np.array_equal(actual_keypoints, expected_keypoints[kept]):
        raise AssertionError(f"keypoints differ: {len(actual_keypoints)} decoded, {count} from transformers")
    np.testing.assert_allclose(actual_scores, scores[0, mask[0].bool()].numpy()[kept], rtol=1e-4, atol=1e-6)
    expected_descriptors = expected_descriptors[0, mask[0].bool()].numpy()[kept]
    error = float(np.abs(actual_descriptors - expected_descriptors).max())
    if error > DESCRIPTOR_TOLERANCE:
        raise AssertionError(f"descriptors differ by up to {error:.2e}")
    return count, error


def convert(source, output_dir):
    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "model.onnx"
    import onnxruntime as ort
    import torch
    import transformers

    if transformers.__version__ != EXPORTER:
        raise RuntimeError(f"Expected transformers {EXPORTER}, found {transformers.__version__}")
    torch.set_num_threads(4)
    model = load_superpoint(source)
    with torch.inference_mode():
        torch.onnx.export(
            build_heads(model), torch.zeros(INPUT["image"]), str(output), input_names=list(INPUT),
            output_names=list(OUTPUTS), opset_version=OPSET, do_constant_folding=True, dynamo=False,
        )
    validate_export(output)

    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    for seed in SEEDS:
        image = synthetic_image(seed)
        logits, descriptors = session.run(None, {"image": image[None, None]})
        count, error = compare_with_reference(model, image, logits, descriptors)
        print(f"seed {seed}: {count} keypoints identical to transformers, descriptors within {error:.1e}")
    print(f"Synthetic-input PyTorch/ONNX keypoint check passed with seeds {SEEDS.start}-{SEEDS.stop - 1}: {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversion", default="default", help="conversion entry in model.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    conversions = load_metadata(MODEL_DIR).get("conversion", {})
    if args.conversion not in conversions:
        parser.error(f"unknown conversion {args.conversion!r}; model.yaml declares: {', '.join(sorted(conversions))}")
    source = conversions[args.conversion]["source"]
    validate_source(source)
    path, *_ = Fetcher().fetch_source(source)
    convert(path, args.output_dir.resolve())


if __name__ == "__main__":
    main()
