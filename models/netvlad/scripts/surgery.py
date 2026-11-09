"""Build NetVLAD for the MLA from the converted weights, export it and check it against Patch-NetVLAD's forward.

The rebuilt network keeps Patch-NetVLAD's VGG16 conv5_3 encoder, NetVLAD pooling and WPCA projection, and makes two
exact rewrites the Model Compiler needs: it aggregates the soft-assigned residuals with one MatMul instead of a loop
over the 64 clusters, and it scales the WPCA bias by the VLAD's norm instead of normalizing all 32768 values.

compile.py compiles what export_netvlad writes and measures its fidelity against reference_netvlad. To get the FP32
ONNX on a host, run:

    python models/netvlad/scripts/surgery.py --weights weights.pth --output model.onnx
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import onnx

IMAGE_HEIGHT, IMAGE_WIDTH = 480, 640
FEATURE_DIM = 512
CLUSTERS = 64
DESCRIPTOR_DIM = 4096
OPSET = 17
INTERFACE = {"image": [1, 3, IMAGE_HEIGHT, IMAGE_WIDTH], "descriptor": [1, DESCRIPTOR_DIM]}
# torchvision's VGG16 configuration up to conv5_3: output channels of each 3x3 convolution, "M" for a 2x2 max pool.
ENCODER_CONFIG = (64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512)
# torch.onnx.export names each encoder operation after its module, encoder.<index>.
ENCODER_SCOPE = "/encoder/"
# torch.nn.functional.normalize's default epsilon.
NORM_EPS = 1e-12
SEEDS = range(123, 128)
# The export must match the reference, which runs the same float32 weights cluster by cluster in float64.
MIN_COSINE = 0.9999
ATOL = 1e-4


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
        raise ValueError("NetVLAD compiles from one self-contained ONNX file")
    unfolded = sorted({node.op_type for node in model.graph.node} & {"Constant", "Shape"})
    if unfolded:
        raise ValueError(f"The export still computes static values with {unfolded}; fold it with onnxsim")


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


def fold_constants(path: Path) -> None:
    """Fold the static values torch exports."""
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


def build_encoder() -> Any:
    """VGG16's features up to conv5_3, laid out as torchvision's so the checkpoint's encoder.N names load."""
    import torch

    layers = []
    channels = 3
    for item in ENCODER_CONFIG:
        if item == "M":
            layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers += [torch.nn.Conv2d(channels, item, kernel_size=3, padding=1), torch.nn.ReLU(inplace=True)]
            channels = item
    # Patch-NetVLAD takes vgg16().features[:29], which ends at conv5_3 before its ReLU.
    return torch.nn.Sequential(*layers[:-1])


def load_netvlad(weights: Path) -> tuple[Any, dict[str, np.ndarray]]:
    """The checkpoint's encoder as a module, and its NetVLAD and WPCA tensors as arrays."""
    import torch

    state = torch.load(weights, map_location="cpu", weights_only=True)
    encoder = build_encoder()
    encoder.load_state_dict(
        {name.removeprefix("encoder."): value for name, value in state.items() if name.startswith("encoder.")}
    )
    pooling = {name: value.numpy() for name, value in state.items() if not name.startswith("encoder.")}
    return encoder.eval(), pooling


def reference_netvlad(encoder: Any, pooling: dict[str, np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    """Patch-NetVLAD's descriptor of an NCHW image: its encoder in PyTorch, then its pooling in float64."""
    import torch

    def describe(image: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            features = encoder(torch.from_numpy(image))[0].numpy()
        return reference_descriptor(
            features,
            pooling["pool.centroids"],
            pooling["pool.conv.weight"],
            pooling["WPCA.0.weight"],
            pooling["WPCA.0.bias"],
        )

    return describe


def build_netvlad(encoder: Any, pooling: dict[str, np.ndarray]) -> Any:
    """Patch-NetVLAD's network with the residuals aggregated by one MatMul and the final normalization folded."""
    import torch
    import torch.nn.functional as F

    def normalize(value: torch.Tensor, dim: int) -> torch.Tensor:
        # F.normalize without its expand_as: the SDK's onnxsim 0.4 leaves the Shape nodes that exports unfolded, as it
        # does for flatten, so the network reshapes its batch of one explicitly. AFE lowers ReduceL2, Clip and Div to
        # its L2Norm operator.
        return value / value.norm(p=2, dim=dim, keepdim=True).clamp_min(NORM_EPS)

    class NetVLAD(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = encoder
            self.assignment = torch.nn.Conv2d(FEATURE_DIM, CLUSTERS, kernel_size=1, bias=False)
            self.assignment.weight.data.copy_(torch.from_numpy(pooling["pool.conv.weight"]))
            self.register_buffer("centroids", torch.from_numpy(pooling["pool.centroids"]))
            self.wpca = torch.nn.Conv2d(CLUSTERS * FEATURE_DIM, DESCRIPTOR_DIM, kernel_size=1, bias=False)
            self.wpca.weight.data.copy_(torch.from_numpy(pooling["WPCA.0.weight"]))
            self.register_buffer("wpca_bias", torch.from_numpy(pooling["WPCA.0.bias"]))

        def forward(self, image: torch.Tensor) -> torch.Tensor:
            features = normalize(self.encoder(image), dim=1)
            soft = F.softmax(self.assignment(features).reshape(1, CLUSTERS, -1), dim=1)
            local = features.reshape(1, FEATURE_DIM, -1)
            # sum_n a_kn (x_n - c_k) = A X^T - c_k sum_n a_kn, the upstream per-cluster loop in one MatMul.
            vlad = soft @ local.transpose(1, 2) - self.centroids * soft.sum(dim=2, keepdim=True)
            vlad = normalize(vlad, dim=2)
            # normalize(W v / |v| + b) == normalize(W v + |v| b): scaling the bias by the VLAD's norm replaces the
            # 32768-wide L2 normalization, which the Model Compiler cannot segment onto the MLA at INT8. The norm
            # is reduced per cluster first, so no reduction spans all 32768 values at once.
            norm = vlad.square().sum(dim=2).sum(dim=1, keepdim=True).sqrt()
            projected = self.wpca(vlad.reshape(1, -1, 1, 1)).reshape(1, -1) + norm * self.wpca_bias
            return normalize(projected, dim=1)

    return NetVLAD().eval()


def annotate_int8_encoder(path: Path) -> dict[str, int]:
    """Mark the VGG16 encoder's operations int8, except conv5_3, which returns to bfloat16.

    AFE gives an AnnotatePrecision's precision to the operation producing its input, and each later operation takes
    the precision its inputs share. So the per-cell normalization that reads conv5_3 inherits BF16 unmarked: a mark
    on its ReduceL2 would split the ReduceL2, Clip and Div that AFE fuses into one L2Norm, and leave a Clip the MLA
    compiler rejects. Returns the number of operations marked with each precision.
    """
    from onnx import helper

    model = onnx.load(path)
    encoder = [node for node in model.graph.node if node.name.startswith(ENCODER_SCOPE)]
    marks = {node.name: "int8" for node in encoder[:-1]}
    marks[encoder[-1].name] = "bfloat16"
    nodes = []
    for node in model.graph.node:
        nodes.append(node)
        if node.name in marks:
            value = node.output[0]
            node.output[0] = f"{value}/unannotated"
            nodes.append(
                helper.make_node(
                    "AnnotatePrecision",
                    [node.output[0]],
                    [value],
                    name=f"{node.name}/AnnotatePrecision",
                    domain="ai.sima",
                    precision=marks[node.name],
                )
            )
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    model.opset_import.append(helper.make_opsetid("ai.sima", 1))
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return {precision: list(marks.values()).count(precision) for precision in ("int8", "bfloat16")}


def export_netvlad(weights: Path, output: Path) -> Path:
    """Export the rebuilt network to output and require its descriptors to match Patch-NetVLAD's forward."""
    import onnxruntime as ort
    import torch

    torch.set_num_threads(4)
    encoder, pooling = load_netvlad(weights)
    reference = reference_netvlad(encoder, pooling)
    torch.onnx.export(
        build_netvlad(encoder, pooling),
        (torch.zeros(INTERFACE["image"]),),
        str(output),
        input_names=["image"],
        output_names=["descriptor"],
        opset_version=OPSET,
        dynamo=False,
    )
    fold_constants(output)
    validate_export(output)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(output), sess_options=options, providers=["CPUExecutionProvider"])
    for seed in SEEDS:
        image = np.random.RandomState(seed).normal(0, 1, INTERFACE["image"]).astype(np.float32)
        cosine = compare_descriptors(reference(image), session.run(["descriptor"], {"image": image})[0])
        print(f"seed {seed}: descriptor cosine similarity {cosine:.8f}", flush=True)
    print(f"NetVLAD reference check passed with seeds {SEEDS.start}-{SEEDS.stop - 1}: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True, help="the converted weights.pth")
    parser.add_argument("--output", type=Path, required=True, help="the ONNX to write")
    args = parser.parse_args()
    export_netvlad(args.weights.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
