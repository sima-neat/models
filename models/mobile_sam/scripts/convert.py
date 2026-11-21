"""Convert MobileSAM's pinned mobile_sam.pt into the image encoder ONNX and the weights.pth surgery.py builds from.

The checkpoint holds MobileSAM's state dict: the TinyViT image encoder, and segment-anything's prompt encoder and mask
decoder. Conversion builds the model as MobileSAM's build_sam_vit_t does, loads the checkpoint strictly, then:

- exports upstream TinyViT unchanged: a 1024x1024 normalized RGB image to its 256x64x64 image embedding;
- saves the whole state dict as weights.pth, from which surgery.py builds the compiler graphs of both components at
  compile time, where segment-anything is not installed.

The export check compares the ONNX with TinyViT on seeded images.
"""

import argparse
import tempfile
from pathlib import Path, PurePosixPath

import numpy as np
import onnx
from model_registry.conversion import conversion_outputs
from model_registry.fetch import Fetcher
from model_registry.model import load_and_validate_model
from pipeline import EMBEDDING_SHAPE, IMAGE_SIZE, INPUT_SHAPES, OUTPUT_SHAPES, tensor_name

MODEL_DIR = Path(__file__).resolve().parents[1]

CHECKPOINT = "mobile_sam.pt"
OPSET = 17
SEEDS = range(123, 126)
INPUT = tensor_name("image_encoder", "image")
OUTPUT = tensor_name("image_encoder", "image_embeddings")


def validate_source(source):
    declared = (source.get("type"), PurePosixPath(str(source.get("uri", ""))).name)
    if declared != ("file", CHECKPOINT):
        raise ValueError(f"Expected the checkpoint file {CHECKPOINT}, model.yaml declares {declared}")


def build(checkpoint):
    """MobileSAM as MobileSAM's build_sam_vit_t builds it, with the checkpoint loaded strictly."""
    import torch
    from segment_anything.modeling import MaskDecoder, PromptEncoder, Sam, TwoWayTransformer
    from tiny_vit_sam import TinyViT

    encoder = TinyViT(
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
    embedding = EMBEDDING_SHAPE[1]
    sam = Sam(
        image_encoder=encoder,
        prompt_encoder=PromptEncoder(
            embed_dim=embedding,
            image_embedding_size=EMBEDDING_SHAPE[2:],
            input_image_size=(IMAGE_SIZE, IMAGE_SIZE),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(depth=2, embedding_dim=embedding, mlp_dim=2048, num_heads=8),
            transformer_dim=embedding,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    return sam.eval()


def seeded_image(seed):
    """Normalized pixels lie in about [-2.1, 2.7]; the image's bottom rows are padding, as for a 4:3 image."""
    image = np.random.RandomState(seed).uniform(-2.1, 2.7, INPUT_SHAPES["image_encoder"]["image"]).astype(np.float32)
    image[:, :, 768:] = 0
    return image


def validate_export(path):
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    if [item.version for item in model.opset_import if not item.domain] != [OPSET]:
        raise ValueError(f"MobileSAM export requires opset {OPSET}")
    shape = lambda value: tuple(d.dim_value for d in value.type.tensor_type.shape.dim)  # noqa: E731
    expected = (
        {INPUT: INPUT_SHAPES["image_encoder"]["image"]},
        {OUTPUT: OUTPUT_SHAPES["image_encoder"]["image_embeddings"]},
    )
    actual = ({v.name: shape(v) for v in model.graph.input}, {v.name: shape(v) for v in model.graph.output})
    if actual != expected:
        raise ValueError(f"Unexpected interface: {actual}")
    if any(
        value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT
        for value in [*model.graph.input, *model.graph.output]
    ):
        raise ValueError("The image and its embedding must be float32")
    if any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in model.graph.initializer):
        raise ValueError("This conversion workflow requires one self-contained ONNX file")


def fold_constants(path):
    from onnxsim import simplify

    # Folding only, which keeps upstream's computation.
    model, valid = simplify(onnx.load(path), perform_optimization=False)
    if not valid:
        raise RuntimeError("onnxsim could not validate the folded MobileSAM image encoder")
    onnx.save(model, path)


def compare(expected, actual, name):
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError(f"Unexpected or non-finite {name}: {actual.shape} vs {expected.shape}")
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3, err_msg=name)


def check_onnx(encoder, path):
    """The image encoder ONNX gives TinyViT's embedding on seeded images."""
    import onnxruntime as ort
    import torch

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    for seed in SEEDS:
        image = seeded_image(seed)
        with torch.inference_mode():
            expected = encoder(torch.from_numpy(image)).numpy()
        compare(expected, session.run([OUTPUT], {INPUT: image})[0], f"image_embeddings (seed {seed})")


def convert(source, output_dir):
    import torch

    torch.set_num_threads(4)
    output_dir.mkdir(parents=True, exist_ok=False)
    encoder_name, weights_name = conversion_outputs(MODEL_DIR)
    sam = build(source)
    with tempfile.TemporaryDirectory(prefix=".mobile-sam-", dir=output_dir) as temporary:
        exported, weights = Path(temporary) / encoder_name, Path(temporary) / weights_name
        torch.onnx.export(
            sam.image_encoder,
            (torch.zeros(INPUT_SHAPES["image_encoder"]["image"]),),
            str(exported),
            input_names=[INPUT],
            output_names=[OUTPUT],
            opset_version=OPSET,
            do_constant_folding=True,
            dynamo=False,
        )
        fold_constants(exported)
        validate_export(exported)
        check_onnx(sam.image_encoder, exported)
        torch.save(sam.state_dict(), weights)
        weights.replace(output_dir / weights_name)
        exported.replace(output_dir / encoder_name)
    seeds = f"{SEEDS.start}-{SEEDS.stop - 1}"
    print(f"Image encoder PyTorch/ONNX check passed with seeds {seeds}: {output_dir / encoder_name}, {weights_name}")
    return output_dir / encoder_name


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
