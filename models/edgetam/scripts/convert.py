"""Convert EdgeTAM's pinned edgetam.pt into the image encoder ONNX and the weights.pth surgery.py builds from.

The checkpoint holds the original facebookresearch/EdgeTAM state dict. Conversion renames its tensors to
transformers' EdgeTAM video model with the key mapping of transformers' own converter, then:

- exports the image encoder, which runs the RepViT backbone through timm, as transformers' get_image_features
  computes it: the 64x64 vision features and the two high-resolution features after the mask decoder's 1x1
  projections, for a 1024x1024 normalized RGB image;
- saves the whole renamed state dict as weights.pth, from which surgery.py rebuilds the memory attention, mask
  decoder and memory encoder at compile time, where timm is not installed.

The export check compares the ONNX with transformers' vision model on seeded images.
"""

import argparse
import re
from pathlib import Path, PurePosixPath

import numpy as np
from model_registry.conversion import conversion_outputs
from model_registry.fetch import Fetcher
from model_registry.model import load_model as load_metadata

MODEL_DIR = Path(__file__).resolve().parents[1]

CHECKPOINT = "edgetam.pt"
IMAGE_SIZE = 1024
OPSET = 17
SEEDS = range(123, 126)
# The largest difference the export check accepts between the ONNX and transformers, relative to the output's range;
# it allows only for float32 accumulation order.
MAX_RELATIVE_ERROR = 1e-4
# convert.py runs before the compile scripts exist in its environment, so it keeps its own copy of the normalization
# calibration.py applies to frames.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)
# transformers' EdgeTAM vision model: the RepViT M1 backbone through timm, with an FPN neck.
BACKBONE = {
    "model_type": "timm_wrapper",
    "architecture": "repvit_m1",
    "model_args": {"in_chans": 3, "features_only": True, "out_indices": [0, 1, 2, 3]},
}
VISION_PREFIX = "vision_encoder."
OUTPUTS = {
    "vision_features": [1, 256, 64, 64],
    "high_res_features_0": [1, 32, 256, 256],
    "high_res_features_1": [1, 64, 128, 128],
}

# The key mapping of transformers' convert_edgetam_video_to_hf.py at commit f262c71703eb138381d65ced86b3aaa496be0196.
# Substrings are replaced in order, then the patterns below rename numbered layers.
RENAMES = {
    "iou_prediction_head.layers.0": "iou_prediction_head.proj_in",
    "iou_prediction_head.layers.1": "iou_prediction_head.layers.0",
    "iou_prediction_head.layers.2": "iou_prediction_head.proj_out",
    "mask_decoder.output_upscaling.0": "mask_decoder.upscale_conv1",
    "mask_decoder.output_upscaling.1": "mask_decoder.upscale_layer_norm",
    "mask_decoder.output_upscaling.3": "mask_decoder.upscale_conv2",
    "mask_downscaling.0": "mask_embed.conv1",
    "mask_downscaling.1": "mask_embed.layer_norm1",
    "mask_downscaling.3": "mask_embed.conv2",
    "mask_downscaling.4": "mask_embed.layer_norm2",
    "mask_downscaling.6": "mask_embed.conv3",
    "dwconv": "depthwise_conv",
    "pwconv": "pointwise_conv",
    "fuser": "memory_fuser",
    "point_embeddings": "point_embed",
    "pe_layer.positional_encoding_gaussian_matrix": "shared_embedding.positional_embedding",
    "obj_ptr_tpos_proj": "temporal_positional_encoding_projection_layer",
    "no_obj_embed_spatial": "occlusion_spatial_embedding_parameter",
    "sam_prompt_encoder": "prompt_encoder",
    "sam_mask_decoder": "mask_decoder",
    "maskmem_tpos_enc": "memory_temporal_positional_encoding",
    "gamma": "scale",
    "image_encoder.neck": "vision_encoder.neck",
    "image_encoder": "vision_encoder.backbone",
    "neck.0": "neck.conv1",
    "neck.1": "neck.layer_norm1",
    "neck.2": "neck.conv2",
    "neck.3": "neck.layer_norm2",
    "pix_feat_proj": "feature_projection",
    "patch_embed.proj": "patch_embed.projection",
    "no_mem_embed": "no_memory_embedding",
    "no_mem_pos_enc": "no_memory_positional_encoding",
    "obj_ptr": "object_pointer",
    ".norm": ".layer_norm",
    "trunk.": "",
    "out_proj": "o_proj",
    "body.": "timm_model.",
    "ff.0": "mlp.layer_norm",
    "ff.1": "mlp.up_proj",
    "ff.3": "mlp.down_proj",
}
PERCEIVER_RENAMES = {
    r"spatial_perceiver.latents": r"spatial_perceiver.latents_1d",
    r"spatial_perceiver.latents_1d_2d": r"spatial_perceiver.latents_2d",
    r"spatial_perceiver.layers.(\d+).attn.layer_norm_x": r"spatial_perceiver.layers.\1.layer_norm_input",
    r"spatial_perceiver.layers.(\d+).attn.layer_norm_latents": r"spatial_perceiver.layers.\1.layer_norm_latents",
    r"spatial_perceiver.layers.(\d+).self_attn.layer_norm": r"spatial_perceiver.layers.\1.layer_norm_self",
    r"spatial_perceiver.layers.(\d+).attn.to_q": r"spatial_perceiver.layers.\1.cross_attention.q_proj",
    r"spatial_perceiver.layers.(\d+).attn.to_kv": r"spatial_perceiver.layers.\1.cross_attention.kv_proj_combined",
    r"spatial_perceiver.layers.(\d+).attn.to_out": r"spatial_perceiver.layers.\1.cross_attention.o_proj",
    r"spatial_perceiver.layers.(\d+).self_attn.to_q": r"spatial_perceiver.layers.\1.self_attention.q_proj",
    r"spatial_perceiver.layers.(\d+).self_attn.to_kv": r"spatial_perceiver.layers.\1.self_attention.kv_proj_combined",
    r"spatial_perceiver.layers.(\d+).self_attn.to_out": r"spatial_perceiver.layers.\1.self_attention.o_proj",
    r"spatial_perceiver.layers.(\d+).attn": r"spatial_perceiver.layers.\1.cross_attention",
    r"spatial_perceiver.layers.(\d+).self_attn": r"spatial_perceiver.layers.\1.self_attention",
}
# Three-layer MLPs whose middle Linear transformers names layers.0, between proj_in and proj_out.
THREE_LAYER_MLPS = [
    r"mask_decoder.pred_obj_score_head.layers.(\d+).*",
    r".*.output_hypernetworks_mlps.(\d+).layers.(\d+).*",
    r"object_pointer_proj.layers.(\d+).*",
]
THREE_LAYER_NAMES = {0: "proj_in", 1: "layers.0", 2: "proj_out"}
# Two-layer MLPs that transformers names proj_in and proj_out.
TWO_LAYER_MLPS = {
    r"vision_encoder.backbone.blocks.(\d+).mlp.layers.(\d+).*": "layers.",
    r"mask_decoder.transformer.layers.(\d+).mlp.layers.(\d+).*": "mlp.layers.",
}
TWO_LAYER_NAMES = {0: "proj_in", 1: "proj_out"}
MASK_DOWNSAMPLER = r"memory_encoder.mask_downsampler.encoder.(\d+).*"


def validate_source(source):
    declared = PurePosixPath(str(source.get("uri", ""))).name
    if declared != CHECKPOINT:
        raise ValueError(f"Expected the checkpoint file {CHECKPOINT}, model.yaml declares {declared!r}")


def renamed_key(key):
    """The transformers name of one original checkpoint tensor."""
    for old, new in RENAMES.items():
        key = key.replace(old, new)
    for pattern, replacement in PERCEIVER_RENAMES.items():
        if re.match(pattern, key):
            key = re.sub(pattern, replacement, key)
    for pattern, prefix in TWO_LAYER_MLPS.items():
        match = re.match(pattern, key)
        if match:
            layer = int(match.group(2))
            key = key.replace(f"{prefix}{layer}", prefix.replace("layers.", "") + TWO_LAYER_NAMES[layer])
    if key.startswith("memory_attention."):
        key = key.replace("linear1", "mlp.up_proj").replace("linear2", "mlp.down_proj")
    for pattern in THREE_LAYER_MLPS:
        match = re.match(pattern, key)
        if match:
            layer = int(match.groups()[-1])
            key = key.replace(f"layers.{layer}", THREE_LAYER_NAMES[layer], 1)
    if re.match(r"vision_encoder.neck.convs.(\d+).conv", key):
        key = key.replace(".conv.", ".")
    if key.startswith("memory_encoder.o_proj."):
        key = key.replace(".o_proj.", ".projection.")
    match = re.match(MASK_DOWNSAMPLER, key)
    if match:
        layer = int(match.group(1))
        if layer == 12:
            key = key.replace(f"encoder.{layer}", "final_conv")
        elif layer % 3 == 0:
            key = key.replace(f"encoder.{layer}", f"layers.{layer // 3}.conv")
        elif layer % 3 == 1:
            key = key.replace(f"encoder.{layer}", f"layers.{layer // 3}.layer_norm")
    return key


def renamed_state_dict(original):
    """The checkpoint's tensors under transformers' EdgeTAM video model names."""
    import torch

    state = {}
    for key, value in original.items():
        key = renamed_key(key)
        if "kv_proj_combined" in key:
            # The perceiver's fused key-value projection becomes separate key and value projections.
            keys, values = torch.chunk(value, 2, dim=0)
            state[key.replace("kv_proj_combined", "k_proj")] = keys
            state[key.replace("kv_proj_combined", "v_proj")] = values
            continue
        state[key] = value
    state["shared_image_embedding.positional_embedding"] = state["prompt_encoder.shared_embedding.positional_embedding"]
    state["prompt_encoder.point_embed.weight"] = torch.cat(
        [state.pop(f"prompt_encoder.point_embed.{index}.weight") for index in range(4)]
    )
    return state


def image_encoder(state):
    """transformers' EdgeTAM vision model with the mask decoder's projections of its two high-resolution levels."""
    import torch
    from transformers import EdgeTamVisionConfig, EdgeTamVisionModel

    class ImageEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vision = EdgeTamVisionModel(EdgeTamVisionConfig(backbone_config=BACKBONE))
            self.conv_s0 = torch.nn.Conv2d(256, 32, kernel_size=1)
            self.conv_s1 = torch.nn.Conv2d(256, 64, kernel_size=1)

        def forward(self, image):
            levels = self.vision(image, return_dict=True).fpn_hidden_states
            return levels[2], self.conv_s0(levels[0]), self.conv_s1(levels[1])

    encoder = ImageEncoder().eval()
    encoder.vision.load_state_dict(
        {key[len(VISION_PREFIX) :]: value for key, value in state.items() if key.startswith(VISION_PREFIX)},
        strict=True,
    )
    for name in ("conv_s0", "conv_s1"):
        getattr(encoder, name).load_state_dict(
            {part: state[f"mask_decoder.{name}.{part}"] for part in ("weight", "bias")}, strict=True
        )
    return encoder


def seeded_image(seed):
    """A 1024x1024 RGB image of uniform pixels, normalized as the tracker normalizes frames."""
    rng = np.random.default_rng(seed)
    pixels = rng.uniform(0, 1, (1, 3, IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
    return (pixels - IMAGENET_MEAN) / IMAGENET_STD


def export_image_encoder(encoder, output):
    import onnx
    import onnxsim
    import torch

    with torch.no_grad():
        torch.onnx.export(
            encoder,
            (torch.from_numpy(seeded_image(0)),),
            output,
            input_names=["image"],
            output_names=list(OUTPUTS),
            opset_version=OPSET,
            dynamo=False,
        )
    # Fold only: onnxsim's graph optimizations turn MatMul+Add into Reshape+Gemm+Reshape, which the compiler
    # mishandles before an attention's head split.
    simplified, valid = onnxsim.simplify(onnx.load(output), perform_optimization=False)
    if not valid:
        raise RuntimeError("onnxsim could not validate the simplified image encoder")
    onnx.save(simplified, output)
    onnx.checker.check_model(output, full_check=True)


def check_image_encoder(encoder, path):
    """The ONNX's three outputs equal transformers' on seeded images."""
    import onnxruntime as ort
    import torch

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    shapes = {item.name: [int(size) for size in item.shape] for item in session.get_outputs()}
    if shapes != OUTPUTS:
        raise ValueError(f"image encoder outputs {shapes}, expected {OUTPUTS}")
    for seed in SEEDS:
        image = seeded_image(seed)
        with torch.no_grad():
            expected = [value.numpy() for value in encoder(torch.from_numpy(image))]
        actual = session.run(list(OUTPUTS), {"image": image})
        for name, reference, value in zip(OUTPUTS, expected, actual):
            error = float(np.abs(reference - value).max() / max(np.abs(reference).max(), 1e-6))
            if error > MAX_RELATIVE_ERROR:
                raise ValueError(f"seed {seed}: ONNX {name} differs from transformers' by {error:.2e} of its range")
    print(f"transformers/ONNX image encoder check passed with seeds {SEEDS.start}-{SEEDS.stop - 1}")


def convert(checkpoint, output_dir):
    import torch

    output_dir.mkdir(parents=True, exist_ok=False)
    encoder_name, weights_name = conversion_outputs(MODEL_DIR)
    state = renamed_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"])
    encoder = image_encoder(state)
    export_image_encoder(encoder, output_dir / encoder_name)
    check_image_encoder(encoder, output_dir / encoder_name)
    torch.save(state, output_dir / weights_name)
    print(f"Wrote {output_dir / encoder_name} and {output_dir / weights_name}")
    return output_dir / encoder_name


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
