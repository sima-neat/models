"""Export NetVLAD (VGG16 conv5_3, NetVLAD pooling and WPCA) from its Patch-NetVLAD checkpoint and check the ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import onnx
from model_registry.fetch import Fetcher
from model_registry.model import load_model

MODEL_DIR = Path(__file__).resolve().parents[1]

CHECKPOINT = "pitts_WPCA4096.pth.tar"
IMAGE_HEIGHT, IMAGE_WIDTH = 480, 640
FEATURE_DIM = 512
CLUSTERS = 64
DESCRIPTOR_DIM = 4096
OPSET = 17
INTERFACE = {"image": [1, 3, IMAGE_HEIGHT, IMAGE_WIDTH], "descriptor": [1, DESCRIPTOR_DIM]}
# torchvision's VGG16 features up to conv5_3: Patch-NetVLAD drops the last ReLU and max pool.
ENCODER_LAYERS = 29
# Checkpoint tensors and shapes; vladv1 has no assignment bias, and WPCA is stored as float64.
STATE_SHAPES = {
    "pool.centroids": (CLUSTERS, FEATURE_DIM),
    "pool.conv.weight": (CLUSTERS, FEATURE_DIM, 1, 1),
    "WPCA.0.weight": (DESCRIPTOR_DIM, CLUSTERS * FEATURE_DIM, 1, 1),
    "WPCA.0.bias": (DESCRIPTOR_DIM,),
}
# torch.nn.functional.normalize's default epsilon.
NORM_EPS = 1e-12
SEEDS = range(123, 128)
# Tolerances of the export check: the ONNX holds the float64 WPCA cast to float32, as upstream's
# load_state_dict does, while the reference projects in float64.
MIN_COSINE = 0.9999
ATOL = 1e-4


def validate_source(source: dict[str, Any]) -> None:
    declared = (source.get("type"), PurePosixPath(str(source.get("uri", ""))).name)
    if declared != ("file", CHECKPOINT):
        raise ValueError(f"Expected the checkpoint file {CHECKPOINT}, model.yaml declares {declared}")


def validate_state(state: dict[str, Any]) -> None:
    """Require the vladv1 NetVLAD and 4096-D WPCA tensors this export rebuilds, and no others besides VGG16."""
    shapes = {name: tuple(value.shape) for name, value in state.items() if not name.startswith("encoder.")}
    if shapes != STATE_SHAPES:
        raise ValueError(f"Unexpected NetVLAD checkpoint tensors: {shapes}")


def _l2_normalize(value: np.ndarray, axis: int) -> np.ndarray:
    return value / np.maximum(np.linalg.norm(value, axis=axis, keepdims=True), NORM_EPS)


def reference_descriptor(
    features: np.ndarray, centroids: np.ndarray, assignment: np.ndarray, wpca_weight: np.ndarray, wpca_bias: np.ndarray
) -> np.ndarray:
    """Patch-NetVLAD's NetVLAD forward and WPCA projection for one conv5_3 map (C, H, W), cluster by cluster."""
    local = _l2_normalize(features.reshape(features.shape[0], -1).astype(np.float64), axis=0)
    logits = assignment.reshape(len(centroids), -1).astype(np.float64) @ local
    soft = np.exp(logits - logits.max(axis=0))
    soft /= soft.sum(axis=0)
    vlad = np.stack(
        [((local - centroids[cluster, :, None]) * soft[cluster]).sum(axis=1) for cluster in range(len(centroids))]
    )
    vlad = _l2_normalize(_l2_normalize(vlad, axis=1).ravel(), axis=0)
    return _l2_normalize(wpca_weight.reshape(len(wpca_bias), -1) @ vlad + wpca_bias, axis=0)


def compare_descriptors(expected: np.ndarray, actual: np.ndarray) -> float:
    """Return the cosine similarity of two descriptors after checking shape, finiteness and closeness."""
    if actual.shape != tuple(INTERFACE["descriptor"]) or not np.isfinite(actual).all():
        raise ValueError(f"Unexpected or non-finite descriptor: {actual.shape}")
    actual = actual[0].astype(np.float64)
    cosine = float(np.dot(expected, actual) / (np.linalg.norm(expected) * np.linalg.norm(actual)))
    if cosine < MIN_COSINE:
        raise ValueError(f"Descriptor cosine similarity {cosine:.6f} is below {MIN_COSINE}")
    np.testing.assert_allclose(actual, expected, rtol=0, atol=ATOL, err_msg="descriptor")
    return cosine


def validate_export(path: Path) -> None:
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    if [item.version for item in model.opset_import if not item.domain] != [OPSET]:
        raise ValueError(f"NetVLAD export requires opset {OPSET}")
    values = [*model.graph.input, *model.graph.output]
    if [value.name for value in values] != list(INTERFACE):
        raise ValueError("Expected input image and output descriptor")
    for value in values:
        tensor = value.type.tensor_type
        shape = [d.dim_value for d in tensor.shape.dim]
        if tensor.elem_type != onnx.TensorProto.FLOAT or shape != INTERFACE[value.name]:
            raise ValueError(f"Unexpected dtype or shape for {value.name}")
    if any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in model.graph.initializer):
        raise ValueError("This conversion workflow requires one self-contained ONNX file")
    unfolded = sorted({node.op_type for node in model.graph.node} & {"Constant", "Shape"})
    if unfolded:
        raise ValueError(f"The export still computes static values with {unfolded}; fold it with onnxsim")


def fold_constants(path: Path) -> None:
    """Fold the Shape/Expand chains torch exports for F.normalize, as the other converted models do."""
    from onnxsim import simplify

    model, valid = simplify(
        onnx.load(path),
        perform_optimization=False,
        overwrite_input_shapes={"image": INTERFACE["image"]},
        dynamic_input_shape=False,
    )
    if not valid:
        raise RuntimeError("onnxsim could not validate the folded NetVLAD export")
    onnx.save(model, path)


def _build_model(state: dict[str, Any]) -> Any:
    """Rebuild Patch-NetVLAD's get_model with append_pca_layer, aggregating residuals with one MatMul."""
    import torch
    import torch.nn.functional as F
    import torchvision

    class NetVLADDescriptor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torchvision.models.vgg16(weights=None).features[:ENCODER_LAYERS]
            self.encoder.load_state_dict(
                {name.removeprefix("encoder."): value for name, value in state.items() if name.startswith("encoder.")}
            )
            self.assignment = torch.nn.Conv2d(FEATURE_DIM, CLUSTERS, kernel_size=1, bias=False)
            self.assignment.weight.data.copy_(state["pool.conv.weight"])
            self.register_buffer("centroids", state["pool.centroids"].float())
            self.wpca = torch.nn.Conv2d(CLUSTERS * FEATURE_DIM, DESCRIPTOR_DIM, kernel_size=1, bias=False)
            self.wpca.weight.data.copy_(state["WPCA.0.weight"])
            self.register_buffer("wpca_bias", state["WPCA.0.bias"].float())

        def forward(self, image: torch.Tensor) -> torch.Tensor:
            features = F.normalize(self.encoder(image), dim=1)
            soft = F.softmax(self.assignment(features).flatten(2), dim=1)
            local = features.flatten(2)
            # sum_n a_kn (x_n - c_k) = A X^T - c_k sum_n a_kn, the upstream per-cluster loop in one MatMul.
            vlad = soft @ local.transpose(1, 2) - self.centroids * soft.sum(dim=2, keepdim=True)
            vlad = F.normalize(vlad, dim=2)
            # normalize(W v / |v| + b) == normalize(W v + |v| b): scaling the bias by the VLAD's norm replaces the
            # 32768-wide L2 normalization, which the Model Compiler cannot segment onto the MLA at INT8. The norm
            # is reduced per cluster first, so no reduction spans all 32768 values at once.
            norm = vlad.square().sum(dim=2).sum(dim=1, keepdim=True).sqrt()
            projected = self.wpca(vlad.flatten(1)[:, :, None, None]).flatten(1) + norm * self.wpca_bias
            return F.normalize(projected, dim=1)

    return NetVLADDescriptor().eval()


def convert(source: Path, output_dir: Path) -> Path:
    import onnxruntime as ort
    import torch

    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "model.onnx"
    torch.set_num_threads(4)
    state = torch.load(source, map_location="cpu", weights_only=True)["state_dict"]
    validate_state(state)
    model = _build_model(state)
    sample = torch.zeros(INTERFACE["image"])
    torch.onnx.export(
        model, (sample,), output, input_names=["image"], output_names=["descriptor"], opset_version=OPSET, dynamo=False
    )
    fold_constants(output)
    validate_export(output)

    reference = {name: state[name].numpy() for name in STATE_SHAPES}
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(output), sess_options=options, providers=["CPUExecutionProvider"])
    for seed in SEEDS:
        image = np.random.RandomState(seed).normal(0, 1, INTERFACE["image"]).astype(np.float32)
        with torch.inference_mode():
            features = model.encoder(torch.from_numpy(image))[0].numpy()
        expected = reference_descriptor(
            features,
            reference["pool.centroids"],
            reference["pool.conv.weight"],
            reference["WPCA.0.weight"],
            reference["WPCA.0.bias"],
        )
        cosine = compare_descriptors(expected, session.run(["descriptor"], {"image": image})[0])
        print(f"seed {seed}: descriptor cosine similarity {cosine:.8f}")
    print(f"NetVLAD/WPCA reference check passed with seeds {SEEDS.start}-{SEEDS.stop - 1}: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversion", default="default", help="conversion entry in model.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    # load_model, not load_and_validate_model: this runs before model.yaml pins the ONNX it produces.
    conversions = load_model(MODEL_DIR).get("conversion", {})
    if args.conversion not in conversions:
        parser.error(f"unknown conversion {args.conversion!r}; model.yaml declares: {', '.join(sorted(conversions))}")
    source = conversions[args.conversion]["source"]
    validate_source(source)
    path, *_ = Fetcher().fetch_source(source)
    convert(path, args.output_dir.resolve())


if __name__ == "__main__":
    main()
