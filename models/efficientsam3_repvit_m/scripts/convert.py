"""Bundle EfficientSAM3 RepViT-M's upstream weights with the upstream model's outputs on four cases.

The sam3 package builds SAM 3's text encoder, neck, encoder, decoder and heads; timm builds the RepViT-M1.1
trunk. The Stage 1 checkpoint supplies SAM 3's weights; the fine-tuned student supplies the trunk and neck.
The student image encoder around the trunk is written here after SimonZeng7108/efficientsam3 at
bd0936c788fed8d51fa799437f05abd97b401b06, which publishes no package. That student resamples the trunk's
features to a 72x72 grid of image tokens (embed_size=72 in sam3/sam3/model_builder.py), returning 288x288 masks.

No graph is changed or exported here. The bundle holds the assembled model's state dict, its tokenizer assets,
its position encoding at the token grid, the settings surgery.py needs to rebuild it, and the PyTorch model's
outputs on four synthetic cases at that upstream grid. surgery.py checks its reconstruction against these outputs
before applying the deployment grid reduction. Compiler-side grid changes do not require reconversion.
"""

import argparse
import gzip
import json
import math
import tempfile
from pathlib import Path

import numpy as np
from calibration import COCO_TOKENS, CONTEXT, END, IMAGE_SIZE, START, tokens
from model_registry.conversion import conversion_outputs
from model_registry.fetch import Fetcher
from model_registry.model import load_and_validate_model

MODEL_DIR = Path(__file__).resolve().parents[1]

CHECKPOINT = "efficient_sam3_repvit_m.pt"
# The fine-tuned student (efficientsam3_ft) recalls about three times as many COCO objects from a text prompt as the
# Stage 1 trunk and neck. It carries a MobileCLIP-S0 text encoder instead of SAM 3's, so only its trunk and neck load.
BACKBONE = "efficientsam3_repvit.pt"
BACKBONE_MODULES = ("backbone.vision_backbone.trunk.", "backbone.vision_backbone.convs.")
TRUNK_PREFIX = "backbone.vision_backbone.trunk.model.backbone.model."
SOURCE = ("file_set", CHECKPOINT, [CHECKPOINT, BACKBONE, "vocab.json", "merges.txt"])
# Preserve the upstream student's token grid in the weights and reference cases.
GRID = 72
# CLIP's Hugging Face vocabulary names the special tokens differently from sam3's tokenizer.
SPECIAL_TOKENS = {"<|startoftext|>": "<start_of_text>", "<|endoftext|>": "<end_of_text>"}
# Synthetic images with single-token, two-token and three-token prompts.
CASES = [(123, "person"), (123, "traffic light"), (124, "fire hydrant"), (124, "giraffe")]


def validate_source(source):
    declared = (source.get("type"), source.get("entrypoint"), [item.get("path") for item in source.get("files", [])])
    if declared != SOURCE:
        raise ValueError(
            f"Expected the checkpoints and CLIP tokenizer {SOURCE}, model.yaml declares {declared}"
        )


def check_tokenizer(source_dir, work_dir):
    """Build sam3's tokenizer from the pinned CLIP merges and check it against the vocabulary and COCO_TOKENS."""
    from sam3.model.tokenizer_ve import SimpleTokenizer

    bpe = work_dir / "bpe_simple_vocab_16e6.txt.gz"
    bpe.write_bytes(gzip.compress((source_dir / "merges.txt").read_bytes(), mtime=0))
    tokenizer = SimpleTokenizer(bpe_path=str(bpe))
    vocabulary = json.loads((source_dir / "vocab.json").read_text(encoding="utf-8"))
    if tokenizer.encoder != {SPECIAL_TOKENS.get(key, key): value for key, value in vocabulary.items()}:
        raise ValueError("merges.txt does not rebuild the vocab.json vocabulary")
    if [tokenizer.sot_token_id, tokenizer.eot_token_id] != [START, END]:
        raise ValueError("Unexpected start and end token IDs")
    for name in COCO_TOKENS:
        expected = tokens(name)
        if len(expected) > CONTEXT:
            raise ValueError(f"Prompt exceeds the exported context: {name}")
        if tokenizer([name], context_length=CONTEXT)[0].tolist() != expected + [0] * (CONTEXT - len(expected)):
            raise ValueError(f"COCO_TOKENS disagrees with the tokenizer for {name!r}")
    return bpe


def trunk_key(key):
    """Map an upstream RepViT ``features`` checkpoint key to timm's ``stem``/``stages`` modules."""
    rest = key.split("features.", 1)[1]
    index, rest = rest.split(".", 1)
    index = int(index)
    if index == 0:
        layer, rest = rest.split(".", 1)
        return {"0": "stem.conv1.", "2": "stem.conv2."}[layer] + rest
    targets = []
    for stage, depth in enumerate((2, 2, 12, 2)):
        if stage:
            targets += [(f"stages.{stage}.downsample.pre_block.", "block"), (f"stages.{stage}.downsample.", "down")]
        targets += [(f"stages.{stage}.blocks.{block}.", "block") for block in range(depth)]
    prefix, kind = targets[index - 1]
    names = (
        {
            "token_mixer.0.": "token_mixer.",
            "token_mixer.1.": "se.",
            "channel_mixer.m.0.": "channel_mixer.conv1.",
            "channel_mixer.m.2.": "channel_mixer.conv2.",
        }
        if kind == "block"
        else {
            "token_mixer.0.": "spatial_downsample.",
            "token_mixer.2.": "channel_downsample.",
            "channel_mixer.m.0.": "ffn.conv1.",
            "channel_mixer.m.2.": "ffn.conv2.",
        }
    )
    for old, new in names.items():
        if rest.startswith(old):
            return prefix + new + rest[len(old) :]
    raise KeyError(f"Unmapped RepViT checkpoint key: {key}")


def backbone_weights(checkpoint):
    """Return the fine-tuned student's trunk and neck under the detector's key names."""
    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
    weights = {}
    for key, value in state.items():
        if key.startswith(BACKBONE_MODULES):
            weights[TRUNK_PREFIX + trunk_key(key) if key.startswith(TRUNK_PREFIX) else key] = value
    return weights


def build_stock(checkpoint, fine_tuned, bpe):
    """Return the stock sam3 image model with the EfficientSAM3 student image encoder and its weights."""
    import timm
    import torch
    from sam3 import model_builder as builder
    from sam3.model.decoder import TransformerDecoder
    from sam3.model.model_misc import TransformerWrapper
    from torch import nn
    from torch.nn import functional as F

    class Trunk(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = nn.Module()
            self.model.stem, self.model.stages = model.stem, model.stages

        def forward(self, x):
            return self.model.stages(self.model.stem(x))

    class Student(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
            self.head = nn.Sequential(
                nn.Conv2d(512, 1024, 1, bias=False),
                nn.BatchNorm2d(1024),
                nn.GELU(),
                nn.Conv2d(1024, 1024, 3, padding=1),
            )

        def forward(self, x):
            return F.interpolate(self.head(self.backbone(x)), size=(GRID, GRID), mode="bilinear", align_corners=False)

    class OneLevel(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model, self.channel_list = model, [1024]

        def forward(self, x):
            return [self.model(x)]

    trunk = OneLevel(Student(Trunk(timm.create_model("repvit_m1_1", pretrained=False))))
    neck = builder._create_vit_neck(builder._create_position_encoding(None), trunk, enable_inst_interactivity=False)
    backbone = builder._create_vl_backbone(neck, builder._create_text_encoder(str(bpe)))
    coordinates = TransformerDecoder._get_coords
    # The stock builder precomputes decoder coordinates on CUDA; build the same cache on the CPU.
    TransformerDecoder._get_coords = staticmethod(lambda height, width, device: coordinates(height, width, "cpu"))
    try:
        decoder = builder._create_transformer_decoder()
    finally:
        TransformerDecoder._get_coords = staticmethod(coordinates)
    transformer = TransformerWrapper(encoder=builder._create_transformer_encoder(), decoder=decoder, d_model=256)
    model = builder._create_sam3_model(
        backbone,
        transformer,
        builder._create_geometry_encoder(),
        builder._create_segmentation_head(),
        builder._create_dot_product_scoring(),
        None,
        True,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if list(state) != ["model"]:
        raise ValueError(f"Expected one 'model' state dict in the checkpoint, found {list(state)}")
    state = state["model"]
    if any(not key.startswith(("detector.", "tracker.")) for key in state):
        raise ValueError("Checkpoint keys must belong to the detector or the tracker")
    detector = {}
    for key, value in state.items():
        if key.startswith("detector."):
            key = key[len("detector.") :]
            detector[TRUNK_PREFIX + trunk_key(key) if key.startswith(TRUNK_PREFIX) else key] = value
    tuned = backbone_weights(fine_tuned)
    stage1 = {key: value.shape for key, value in detector.items() if key.startswith(BACKBONE_MODULES)}
    if {key: value.shape for key, value in tuned.items()} != stage1:
        raise ValueError("The fine-tuned trunk and neck do not match the Stage 1 trunk and neck")
    detector.update(tuned)
    missing, unexpected = model.load_state_dict(detector, strict=False)
    # The checkpoint also carries the SAM 2 neck used only by the video tracker.
    stray = [key for key in unexpected if not key.startswith("backbone.vision_backbone.sam2_convs.")]
    if missing or stray:
        raise ValueError(f"Checkpoint does not match the model: missing={missing}, unexpected={stray}")
    return model.eval()


def stock_settings(stock):
    """Return the settings surgery.py rebuilds the model from, after checking the layout its graph assumes."""
    from torch import nn

    encoder, decoder, scoring = stock.transformer.encoder, stock.transformer.decoder, stock.dot_prod_scoring
    head, language = stock.segmentation_head, stock.backbone.language_backbone.encoder
    position = stock.backbone.vision_backbone.position_encoding
    if (position.num_pos_feats, position.temperature, position.normalize, position.scale) != (
        128, 10000, True, 2 * math.pi
    ):
        raise ValueError("Unexpected image position encoding")
    layers = [(layer, False) for layer in encoder.layers] + [(layer, True) for layer in stock.geometry_encoder.encode]
    for layer, geometry in layers:
        expected = (True, not geometry, geometry, False, "relu")
        actual = (
            layer.pre_norm,
            layer.pos_enc_at_attn,
            layer.pos_enc_at_cross_attn_keys,
            layer.pos_enc_at_cross_attn_queries,
            layer.activation_str,
        )
        if actual != expected:
            raise ValueError(f"Unexpected encoder layer settings: {actual}")
    if encoder.level_embed is not None or encoder.add_pooled_text_to_img_feat:
        raise ValueError("Unexpected encoder fusion")
    if (
        decoder.boxRPB != "log"
        or not decoder.box_refine
        or decoder.presence_token is None
        or not decoder.use_normed_output_consistently
    ):
        raise ValueError("Unexpected decoder configuration")
    mlps = [decoder.bbox_embed, decoder.boxRPB_embed_x, decoder.boxRPB_embed_y, decoder.ref_point_head]
    mlps += [decoder.presence_token_head, head.mask_predictor.mask_embed]
    if any(mlp.residual or not isinstance(mlp.out_norm, nn.Identity) for mlp in mlps):
        raise ValueError("Plain MLP expected")
    if not scoring.prompt_mlp.residual or scoring.prompt_mlp.num_layers != 2 or not scoring.clamp_logits:
        raise ValueError("Unexpected scoring configuration")
    if head.pixel_decoder.interpolation_mode != "nearest" or head.pixel_decoder.shared_conv or head.no_dec:
        raise ValueError("Unexpected segmentation head")
    scales = [scale for block in language.transformer.resblocks for scale in (block.ls_1, block.ls_2)]
    if any(not isinstance(scale, nn.Identity) for scale in scales):
        raise ValueError("Unexpected text layer scale")
    attentions = {name: module for name, module in stock.named_modules() if isinstance(module, nn.MultiheadAttention)}
    for name, attention in attentions.items():
        if not attention._qkv_same_embed_dim or attention.bias_k is not None or attention.add_zero_attn:
            raise ValueError(f"Unsupported attention configuration: {name}")
    text = {attention.num_heads for name, attention in attentions.items() if name.startswith("backbone.")}
    image = {attention.num_heads for name, attention in attentions.items() if not name.startswith("backbone.")}
    norms = (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)
    epsilons = {module.eps for module in stock.modules() if isinstance(module, norms)}
    groups = {norm.num_groups for norm in head.pixel_decoder.norms}
    trunk = stock.backbone.vision_backbone.modules()
    activations = {module.approximate for module in trunk if isinstance(module, nn.GELU)}
    if any(len(values) != 1 for values in (text, image, epsilons, groups)) or activations != {"none"}:
        raise ValueError(f"Mixed settings: heads {text} / {image}, eps {epsilons}, groups {groups}, GELU {activations}")
    return {
        "grid": GRID,
        "text_heads": text.pop(),
        "heads": image.pop(),
        "norm_eps": epsilons.pop(),
        "groups": groups.pop(),
        "score_scale": float(scoring.scale),
        "score_limit": float(scoring.clamp_max_val),
    }


def stock_outputs(stock, pixels, prompt):
    import torch
    from sam3.model import box_ops
    from sam3.model.data_misc import FindStage

    backbone = stock.backbone.forward_image(torch.from_numpy(pixels))
    backbone.update(stock.backbone.forward_text([prompt], device="cpu"))
    find = FindStage(
        img_ids=torch.tensor([0]),
        text_ids=torch.tensor([0]),
        input_boxes=None,
        input_boxes_mask=None,
        input_boxes_label=None,
        input_points=None,
        input_points_mask=None,
    )
    # The geometry encoder pins a host tensor for a CUDA copy; pinning has no numerical effect.
    pin_memory = torch.Tensor.pin_memory
    torch.Tensor.pin_memory = lambda tensor, *args, **kwargs: tensor
    try:
        out = stock.forward_grounding(
            backbone_out=backbone, find_input=find, find_target=None, geometric_prompt=stock._get_dummy_prompt()
        )
    finally:
        torch.Tensor.pin_memory = pin_memory
    scores = (out["pred_logits"].sigmoid() * out["presence_logit_dec"].sigmoid().unsqueeze(1))[0, :, 0]
    return {
        "text_features": backbone["language_features"][:CONTEXT, 0].numpy(),
        "scores": scores.numpy(),
        "boxes_px": (box_ops.box_cxcywh_to_xyxy(out["pred_boxes"])[0] * IMAGE_SIZE).numpy(),
        "mask_logits": out["pred_masks"][0].numpy(),
    }


def synthetic_pixels(seed):
    image = np.random.RandomState(seed).randint(0, 256, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    return np.ascontiguousarray((image.astype(np.float32) / np.float32(127.5) - 1).transpose(2, 0, 1)[None])


def convert(source, output_dir):
    import torch

    output_dir.mkdir(parents=True, exist_ok=False)
    (filename,) = conversion_outputs(MODEL_DIR)
    output = output_dir / filename
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    with tempfile.TemporaryDirectory() as work_dir:
        stock = build_stock(source, source.parent / BACKBONE, check_tokenizer(source.parent, Path(work_dir)))
        cases = []
        for seed, prompt in CASES:
            pixels = synthetic_pixels(seed)
            outputs = stock_outputs(stock, pixels, prompt)
            if not all(np.isfinite(value).all() for value in outputs.values()):
                raise ValueError(f"The upstream model returns non-finite outputs for {prompt!r}")
            cases.append(
                {
                    "prompt": prompt,
                    "pixel_values": torch.from_numpy(pixels),
                    **{name: torch.from_numpy(np.ascontiguousarray(value)) for name, value in outputs.items()},
                }
            )
        position = stock.backbone.vision_backbone.position_encoding(torch.zeros(1, 256, GRID, GRID))
        bundle = {
            "state_dict": stock.state_dict(),
            "assets": {name: (source.parent / name).read_bytes() for name in ("vocab.json", "merges.txt")},
            "position_encoding": position.clone(),
            "settings": stock_settings(stock),
            "cases": cases,
        }
        torch.save(bundle, output)
    print(f"Wrote {output} with the upstream outputs of {len(cases)} cases")
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
