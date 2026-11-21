"""Build MobileSAM's compiler graphs from the converted image encoder and weights, and check them.

Conversion exports upstream TinyViT unchanged as image_encoder.onnx and saves the checkpoint's state dict as
weights.pth. At compile time, where segment-anything is not installed, this module rebuilds MobileSAM from weights.pth
with the vendored TinyViT (tiny_vit_sam.py) and segment-anything modules (sam_modules.py), then:

- rewrites the image encoder in the 4D layout the compiler handles (tiny_vit_4d.py): every Linear is a 1x1
  convolution and attention windows stay on the feature map; it computes upstream's embedding;
- builds the prompt decoder from segment-anything's point embedding and mask prediction with fixed prompt slots:
  one-hot point types select the learned embeddings, padding slots never act as attention keys (sam_decoder.py), and
  no mask prompt is taken; it returns all four masks and their predicted quality;
- exports both side by side, splits them into one graph per component, and cuts the decoder's Fourier encoding of
  the points to the CPU (compile_decoder).

Each rewrite is checked on seeded inputs: the 4D encoder against upstream TinyViT and the converted image_encoder.onnx,
the prompt decoder against segment-anything's prompt encoder and mask decoder on every check prompt, and the exported
ONNX against PyTorch. compile.py calls export_components. To get the FP32 graphs on a host, run:

    python models/mobile_sam/scripts/surgery.py --image-encoder image_encoder.onnx --weights weights.pth \
        --output-dir graphs
"""

import argparse
import math
import tempfile
from pathlib import Path

import numpy as np
import onnx
from pipeline import (
    COMPILED_INPUT_SHAPES,
    COMPONENTS,
    EMBEDDING_SHAPE,
    IMAGE_SIZE,
    INPUT_SHAPES,
    MAX_POINTS,
    OUTPUT_SHAPES,
    PROMPT_ENCODING_METADATA,
    PromptEncoding,
    prompt_inputs,
    resized_size,
    tensor_name,
)

OPSET = 17
SEEDS = range(123, 128)
# Prompts the checks feed the decoder, in pixels of a 1024x768 image: (points, types).
PROMPTS = [
    (
        [(200 + 100 * index, 150 + 80 * index) for index in range(count)],
        ["foreground" if index % 2 == 0 else "background" for index in range(count)],
    )
    for count in range(1, MAX_POINTS + 1)
] + [
    ([(512, 384)], ["background"]),
    *[
        (
            [(100, 80), (900, 700), *[(300 + 100 * index, 200 + 100 * index) for index in range(count)]],
            [
                "box_top_left",
                "box_bottom_right",
                *["foreground" if index % 2 == 0 else "background" for index in range(count)],
            ],
        )
        for count in range(MAX_POINTS - 1)
    ],
]
ORIGINAL_SIZE = (768, 1024)


def build(weights):
    """MobileSAM as build_sam_vit_t builds it, from the vendored modules, with weights.pth loaded strictly."""
    import torch
    from sam_modules import MaskDecoder, PromptEncoder, TwoWayTransformer
    from tiny_vit_sam import TinyViT

    embedding = EMBEDDING_SHAPE[1]

    class Sam(torch.nn.Module):
        """segment-anything's Sam modules; Sam's pixel statistics are not part of its state dict."""

        def __init__(self):
            super().__init__()
            self.image_encoder = TinyViT(
                img_size=IMAGE_SIZE,
                in_chans=3,
                num_classes=1000,
                embed_dims=[64, 128, 160, 320],
                depths=[2, 2, 6, 2],
                num_heads=[2, 4, 5, 10],
                window_sizes=[7, 7, 14, 7],
                mlp_ratio=4.0,
                drop_rate=0.0,
                drop_path_rate=0.0,
                use_checkpoint=False,
                mbconv_expand_ratio=4.0,
                local_conv_size=3,
                layer_lr_decay=0.8,
            )
            self.prompt_encoder = PromptEncoder(
                embed_dim=embedding,
                image_embedding_size=EMBEDDING_SHAPE[2:],
                input_image_size=(IMAGE_SIZE, IMAGE_SIZE),
                mask_in_chans=16,
            )
            self.mask_decoder = MaskDecoder(
                num_multimask_outputs=3,
                transformer=TwoWayTransformer(depth=2, embedding_dim=embedding, mlp_dim=2048, num_heads=8),
                transformer_dim=embedding,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
            )

    sam = Sam()
    sam.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    return sam.eval()


def modules(sam):
    """The side-by-side export module: the 4D image encoder and the fixed-slot prompt decoder."""
    import torch
    from sam_decoder import predict_masks
    from tiny_vit_4d import ImageEncoder

    class PromptDecoder(torch.nn.Module):
        """segment-anything's point embedding and mask prediction, without a mask prompt."""

        def __init__(self):
            super().__init__()
            prompt = sam.prompt_encoder
            self.decoder = sam.mask_decoder
            self.register_buffer("gaussian", prompt.pe_layer.positional_encoding_gaussian_matrix.clone())
            # One row per point type, in POINT_TYPES order.
            table = torch.cat([prompt.not_a_point_embed.weight, *(item.weight for item in prompt.point_embeddings)])
            self.register_buffer("table", table.clone())
            dense = prompt.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(1, -1, *EMBEDDING_SHAPE[2:])
            self.register_buffer("dense", dense.clone())
            self.register_buffer("image_pe", prompt.get_dense_pe().clone())

        def forward(self, image_embeddings, point_coords, point_types):
            coords = point_coords[0, :, 0, :].transpose(0, 1)
            types = point_types[0, :, 0, :].transpose(0, 1)
            projected = 2 * math.pi * (coords @ self.gaussian)
            # A padding point gets no positional encoding, only its own embedding.
            encoding = torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1) * types[:, 1:].sum(
                dim=-1, keepdim=True
            )
            sparse = (encoding + types @ self.table)[None]
            return predict_masks(
                self.decoder, image_embeddings, self.image_pe, sparse, self.dense, types.sum(dim=-1)[None]
            )

    class SideBySide(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.image_encoder = ImageEncoder(sam.image_encoder)
            self.prompt_decoder = PromptDecoder()

        def forward(self, image, image_embeddings, point_coords, point_types):
            return (
                self.image_encoder(image),
                *self.prompt_decoder(image_embeddings, point_coords, point_types),
            )

    return SideBySide().eval()


def reference_decoder(sam, image_embeddings, points, types):
    """segment-anything's prompt encoder and mask decoder through their public calls; never pad or mask tokens."""
    import torch

    height, width = ORIGINAL_SIZE
    resized_height, resized_width = resized_size(height, width)
    scaled = np.asarray(points, np.float64) * (resized_width / width, resized_height / height)
    clicks = [index for index, kind in enumerate(types) if kind in ("foreground", "background")]
    point_input = None
    if clicks:
        coordinates = torch.as_tensor(scaled[clicks][None], dtype=torch.float32)
        labels = torch.tensor([[int(types[index] == "foreground") for index in clicks]])
        point_input = (coordinates, labels)
    boxes = None
    if "box_top_left" in types:
        corners = [types.index("box_top_left"), types.index("box_bottom_right")]
        boxes = torch.as_tensor(scaled[corners].reshape(1, 4), dtype=torch.float32)
    sparse, dense = sam.prompt_encoder(points=point_input, boxes=boxes, masks=None)
    inputs = dict(
        image_embeddings=image_embeddings,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
    )
    single = sam.mask_decoder(**inputs, multimask_output=False)
    multiple = sam.mask_decoder(**inputs, multimask_output=True)
    return tuple(torch.cat([one, three], dim=1) for one, three in zip(single, multiple))


def seeded_image(seed):
    """Normalized pixels lie in about [-2.1, 2.7]; the image's bottom rows are padding, as for a 4:3 image."""
    image = np.random.RandomState(seed).uniform(-2.1, 2.7, INPUT_SHAPES["image_encoder"]["image"]).astype(np.float32)
    image[:, :, 768:] = 0
    return image


def compare(expected, actual, name):
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError(f"Unexpected or non-finite {name}: {actual.shape} vs {expected.shape}")
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3, err_msg=name)


def _session(path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def check_encoder(sam, side_by_side, image_encoder):
    """The 4D encoder gives upstream TinyViT's embedding, and the converted encoder's, on seeded images."""
    import torch

    reference = _session(image_encoder)
    for seed in SEEDS[:2]:
        image = seeded_image(seed)
        with torch.inference_mode():
            expected = sam.image_encoder(torch.from_numpy(image)).numpy()
            actual = side_by_side.image_encoder(torch.from_numpy(image)).numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4, err_msg=f"4D encoder (seed {seed})")
        (converted,) = reference.run(None, {reference.get_inputs()[0].name: image})
        compare(converted, actual, f"4D encoder against image_encoder.onnx (seed {seed})")


def check_embedding(sam, side_by_side):
    """The decoder's own point embedding gives segment-anything's masks and scores for every check prompt."""
    import torch

    embeddings = torch.from_numpy(np.random.RandomState(SEEDS.start).standard_normal(EMBEDDING_SHAPE) * 0.3).float()
    for points, types in PROMPTS:
        inputs = {key: torch.from_numpy(value) for key, value in prompt_inputs(points, types, ORIGINAL_SIZE).items()}
        with torch.inference_mode():
            actual = side_by_side.prompt_decoder(embeddings, inputs["point_coords"], inputs["point_types"])
            expected = reference_decoder(sam, embeddings, points, types)
        # Mask logits reach about 16; float32 reordering on the CI runners moves a few by up to 2e-4.
        for got, want in zip(actual, expected):
            np.testing.assert_allclose(got.numpy(), want.numpy(), rtol=1e-4, atol=1e-3, err_msg=str(types))


def validate_export(path):
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    if [item.version for item in model.opset_import if not item.domain] != [OPSET]:
        raise ValueError(f"MobileSAM export requires opset {OPSET}")
    shape = lambda value: tuple(d.dim_value for d in value.type.tensor_type.shape.dim)  # noqa: E731
    expected = (
        {tensor_name(c, n): s for c in COMPONENTS for n, s in INPUT_SHAPES[c].items()},
        {tensor_name(c, n): s for c in COMPONENTS for n, s in OUTPUT_SHAPES[c].items()},
    )
    actual = ({v.name: shape(v) for v in model.graph.input}, {v.name: shape(v) for v in model.graph.output})
    if actual != expected:
        raise ValueError(f"Unexpected interface: {actual}")
    if any(
        value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT
        for value in [*model.graph.input, *model.graph.output]
    ):
        raise ValueError("Every input and output must be float32")
    if any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in model.graph.initializer):
        raise ValueError("The side-by-side export must be one self-contained ONNX file")


def fold_constants(path):
    from onnxsim import simplify

    # Folding only: onnxsim's optimizer would fuse token Linears into 2D Gemms, whose bias the compiler mis-types.
    model, valid = simplify(onnx.load(path), perform_optimization=False)
    if not valid:
        raise RuntimeError("onnxsim could not validate the folded MobileSAM export")
    onnx.save(model, path)


def check_onnx(side_by_side, path):
    """The side-by-side ONNX outputs match PyTorch on seeded images and the check prompts."""
    import torch

    session = _session(path)
    outputs = [tensor_name(c, n) for c in COMPONENTS for n in OUTPUT_SHAPES[c]]
    for index, (points, types) in enumerate(PROMPTS):
        seed = SEEDS[index % len(SEEDS)]
        image = seeded_image(seed)
        embeddings = (np.random.RandomState(seed).standard_normal(EMBEDDING_SHAPE) * 0.3).astype(np.float32)
        prompt = prompt_inputs(points, types, ORIGINAL_SIZE)
        feeds = {
            tensor_name("image_encoder", "image"): image,
            tensor_name("prompt_decoder", "image_embeddings"): embeddings,
            **{tensor_name("prompt_decoder", key): value for key, value in prompt.items()},
        }
        with torch.inference_mode():
            expected = side_by_side(*(torch.from_numpy(value) for value in feeds.values()))
        for name, want, got in zip(outputs, expected, session.run(outputs, feeds)):
            compare(want.numpy(), got, f"{name} (seed {seed})")


def export_side_by_side(image_encoder: Path, weights: Path, output: Path) -> Path:
    """Build, check and export both components side by side in one ONNX, written to output only after every check."""
    import torch

    torch.set_num_threads(4)
    sam = build(weights)
    side_by_side = modules(sam)
    check_encoder(sam, side_by_side, image_encoder)
    check_embedding(sam, side_by_side)
    inputs = [tensor_name(c, n) for c in COMPONENTS for n in INPUT_SHAPES[c]]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".mobile-sam-", dir=output.parent) as temporary:
        exported = Path(temporary) / "side_by_side.onnx"
        torch.onnx.export(
            side_by_side,
            tuple(torch.zeros(INPUT_SHAPES[c][n]) for c in COMPONENTS for n in INPUT_SHAPES[c]),
            str(exported),
            input_names=inputs,
            output_names=[tensor_name(c, n) for c in COMPONENTS for n in OUTPUT_SHAPES[c]],
            opset_version=OPSET,
            do_constant_folding=True,
            dynamo=False,
        )
        fold_constants(exported)
        validate_export(exported)
        check_onnx(side_by_side, exported)
        exported.replace(output)
    return output


def split_components(source: Path, directory: Path) -> dict[str, Path]:
    """Extract each component's graph from the side-by-side ONNX."""
    directory.mkdir(parents=True)
    graphs = {}
    for component in COMPONENTS:
        graphs[component] = directory / f"{component}.onnx"
        onnx.utils.extract_model(
            str(source),
            str(graphs[component]),
            [tensor_name(component, name) for name in INPUT_SHAPES[component]],
            [tensor_name(component, name) for name in OUTPUT_SHAPES[component]],
        )
    return graphs


def export_components(image_encoder: Path, weights: Path, directory: Path) -> dict[str, Path]:
    """Each component's checked FP32 graph from the conversion outputs, in a new directory."""
    with tempfile.TemporaryDirectory(prefix=".mobile-sam-") as temporary:
        source = export_side_by_side(image_encoder, weights, Path(temporary) / "side_by_side.onnx")
        return split_components(source, directory)


def compile_decoder(source: Path, output: Path, source_sha256: str) -> dict:
    """Cut only source Fourier encoding; retain learned prompt types, masking and attention on MLA.

    The exact FP32 table travels in a checksum-bound package supporting file, not a runtime checkpoint download.
    The source decoder remains untouched for reference inference; the compiled decoder has a different input.
    """
    import hashlib

    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load(source)
    encoded = "/prompt_decoder/Concat_output_0"
    prefix = onnx.utils.Extractor(model).extract_model(["prompt_decoder_point_coords"], [encoded])
    nodes = list(prefix.graph.node)
    if [node.op_type for node in nodes] != ["Gather", "Gather", "Transpose", "MatMul", "Mul", "Sin", "Cos", "Concat"]:
        raise ValueError("unsupported source Fourier encoding graph")
    constants = {item.name: numpy_helper.to_array(item) for item in prefix.graph.initializer}
    attributes = lambda node: {item.name: helper.get_attribute_value(item) for item in node.attribute}  # noqa: E731
    expected_inputs = [
        ["prompt_decoder_point_coords", nodes[0].input[1]],
        [nodes[0].output[0], nodes[1].input[1]],
        [nodes[1].output[0]],
        [nodes[2].output[0], nodes[3].input[1]],
        [nodes[3].output[0], nodes[4].input[1]],
        [nodes[4].output[0]],
        [nodes[4].output[0]],
        [nodes[5].output[0], nodes[6].output[0]],
    ]
    if (
        [list(node.input) for node in nodes] != expected_inputs
        or [attributes(node) for node in nodes]
        != [{"axis": 0}, {"axis": 1}, {"perm": [1, 0]}, {}, {}, {}, {}, {"axis": -1}]
        or any(
            constants[nodes[index].input[1]].shape != () or constants[nodes[index].input[1]] != 0 for index in (0, 1)
        )
    ):
        raise ValueError("unsupported source Fourier encoding operation order")
    gaussian, scale = constants[nodes[3].input[1]], constants[nodes[4].input[1]]
    if gaussian.dtype != np.float32 or scale.dtype != np.float32:
        raise ValueError("source Fourier coefficients must be float32")
    metadata = {
        PROMPT_ENCODING_METADATA: {
            "version": 1,
            "source_sha256": source_sha256,
            "gaussian": gaussian.tolist(),
            "gaussian_sha256": hashlib.sha256(gaussian.astype("<f4").tobytes()).hexdigest(),
            "phase_scale": float(scale),
        }
    }
    import json

    PromptEncoding.from_metadata({PROMPT_ENCODING_METADATA: json.dumps(metadata[PROMPT_ENCODING_METADATA])})
    cut = onnx.utils.Extractor(model).extract_model(
        ["prompt_decoder_image_embeddings", encoded, "prompt_decoder_point_types"],
        [tensor_name("prompt_decoder", name) for name in OUTPUT_SHAPES["prompt_decoder"]],
    )
    shape = COMPILED_INPUT_SHAPES["prompt_decoder"]["point_encoding"]
    cut.graph.input[1].CopyFrom(
        helper.make_tensor_value_info("prompt_decoder_point_encoding", TensorProto.FLOAT, shape)
    )
    shape_name, transposed = "mobile_sam_encoding_shape", "mobile_sam_encoding_transposed"
    occupied = {name for node in cut.graph.node for name in [*node.input, *node.output]}
    if {shape_name, transposed} & occupied:
        raise ValueError("compiled Fourier boundary tensor name collision")
    cut.graph.initializer.append(numpy_helper.from_array(np.array([shape[3], shape[1]], np.int64), shape_name))
    boundary = [
        helper.make_node("Transpose", ["prompt_decoder_point_encoding"], [transposed], perm=[0, 3, 2, 1]),
        helper.make_node("Reshape", [transposed, shape_name], [encoded]),
    ]
    original = list(cut.graph.node)
    del cut.graph.node[:]
    cut.graph.node.extend(boundary + original)
    if any(node.op_type in ("Sin", "Cos") for node in cut.graph.node):
        raise ValueError("compiled decoder still contains Fourier lookup operations")
    onnx.checker.check_model(cut)
    onnx.save(cut, output)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image-encoder", type=Path, required=True, help="the converted image_encoder.onnx")
    parser.add_argument("--weights", type=Path, required=True, help="the converted weights.pth")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    graphs = export_components(args.image_encoder, args.weights, args.output_dir)
    print("\n".join(str(path) for path in graphs.values()))


if __name__ == "__main__":
    main()
