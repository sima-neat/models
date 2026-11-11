"""Build EdgeTAM's four per-frame graphs for the compiler from the converted image encoder and weights, and check them.

transformers' EdgeTAM video model tracks with Python lists of earlier frames' memories. The graphs here run one frame
with fixed shapes, and every rewrite keeps the tracker's numbers:

- image_encoder adds the no-memory embedding to the vision features, so its image_embeddings feed the mask decoder
  directly on a prompted frame. memory_attention and memory_encoder subtract it again.
- memory_attention reads a fixed memory bank: MEMORY_SLOTS mask memories and OBJECT_POINTERS object pointers, each
  with a 1 or 0 in its mask input. Every key of an empty slot gets -1e4 before the softmax, which leaves the
  attention over the filled slots unchanged. Slot 0 holds the prompted frame's memory and slot k the memory of k
  frames back, since each slot's temporal encoding is a constant. Rotary encoding multiplies by constant cos and
  sin tables, with the pairwise rotation and the attention scale folded into the query and key projections; keys
  without rotary encoding get cos 1 and sin 0.
- mask_decoder embeds PROMPT_POINTS prompt slots from normalized coordinates and one-hot point types, as the
  prompt encoder does, and returns all four masks, their scores and four object pointers, so the CPU picks the
  mask. Each pointer is the no-object pointer when the object score is not positive. The prompt encoder appends one
  padding point to the user's points, and a tracked frame has two padding points. The slots hold up to two points
  and padding in the rest; identical padding tokens stay identical through the decoder, so n of them weigh as m
  when every attention that reads them as keys adds log(m / n) to their scores.
- memory_encoder encodes a mask, already upsampled to 1024x1024 and turned into probabilities, into the 512 memory
  tokens of one bank slot. Its fuser's pointwise Linears run as 1x1 convolutions.

The export check compares every graph with transformers' modules on seeded inputs. compile.py calls
export_components. To get the FP32 graphs on a host, run:

    python models/edgetam/scripts/surgery.py \
        --image-encoder image_encoder.onnx --weights weights.pth --output-dir graphs
"""

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import numpy as np
import onnx

COMPONENTS = ["image_encoder", "memory_attention", "mask_decoder", "memory_encoder"]
IMAGE_SIZE = 1024
HIDDEN = 256
EMBEDDING = [1, HIDDEN, 64, 64]
TOKENS = 64 * 64
MEMORY_DIM = 64
# A memory is 256 latents of the whole frame and 256 latents of 4x4 windows, each MEMORY_DIM wide.
MEMORY_TOKENS = 512
MEMORY_SLOTS = 7
OBJECT_POINTERS = 16
POINTER_TOKENS = HIDDEN // MEMORY_DIM
PROMPT_POINTS = 3
# The padding points transformers' prompt encoder gives a frame with user points, and a tracked frame.
PROMPTED_PADDING = 1
TRACKED_PADDING = 2
# The point types, in the order of the one-hot point_types channels.
POINT_TYPES = ["padding", "background", "foreground", "box_top_left", "box_bottom_right"]
MASKS = 4
# The decoder's object score, IoU and mask tokens, before the prompt tokens.
OUTPUT_TOKENS = 2 + MASKS
MASK_SIZE = 256
INTERFACES = {
    "image_encoder": {
        "inputs": {"image": [1, 3, IMAGE_SIZE, IMAGE_SIZE]},
        "outputs": {
            "image_embeddings": EMBEDDING,
            "high_res_features_0": [1, 32, 256, 256],
            "high_res_features_1": [1, 64, 128, 128],
        },
    },
    "memory_attention": {
        "inputs": {
            "image_embeddings": EMBEDDING,
            "memories": [1, MEMORY_SLOTS * MEMORY_TOKENS, MEMORY_DIM],
            "memory_mask": [1, MEMORY_SLOTS],
            "object_pointers": [1, OBJECT_POINTERS, HIDDEN],
            "object_pointer_mask": [1, OBJECT_POINTERS],
        },
        "outputs": {"conditioned_embeddings": EMBEDDING},
    },
    "mask_decoder": {
        "inputs": {
            "image_embeddings": EMBEDDING,
            "high_res_features_0": [1, 32, 256, 256],
            "high_res_features_1": [1, 64, 128, 128],
            "point_coords": [1, 2, 1, PROMPT_POINTS],
            "point_types": [1, len(POINT_TYPES), 1, PROMPT_POINTS],
        },
        "outputs": {
            "mask_logits": [1, MASKS, MASK_SIZE, MASK_SIZE],
            "iou_scores": [1, MASKS],
            "object_score_logits": [1, 1],
            "object_pointers": [1, MASKS, HIDDEN],
        },
    },
    "memory_encoder": {
        "inputs": {"image_embeddings": EMBEDDING, "mask": [1, 1, IMAGE_SIZE, IMAGE_SIZE]},
        "outputs": {"memory": [1, MEMORY_TOKENS, MEMORY_DIM]},
    },
}
OPSET = 17
SEEDS = range(123, 126)
# The largest difference check_graph accepts between the ONNX and transformers, relative to the output's range.
MAX_RELATIVE_ERROR = 1e-3
# The score added to every key of an empty memory slot; exp(-1e4) is 0 in float32 and BF16.
EMPTY_SLOT_SCORE = -1e4
# The no-object gate is 1 for object scores of at least 1e-4 and 0 for scores of 0 or less, as score > 0 is.
GATE_SCALE = 1e4
# EdgeTAM scales mask probabilities to logits for the memory encoder.
MASK_SCALE = 20.0
MASK_BIAS = -10.0
# The operations a W8A8 component can run in BF16, by the names the export gives them, and how many it has.
BF16_REGIONS = {
    "memory_attention": {
        # The empty-slot scores, and each attention's query-key product, softmax over the keys and weighted sum of
        # the values, with the empty-slot scores added in cross attention.
        "attention": (
            r"/(MatMul|MatMul_1|Sub_1|Mul)"
            r"|/self_attention(_1)?/(MatMul|Softmax|MatMul_1)"
            r"|/cross_attention(_1)?/(MatMul|Add_2|Softmax|MatMul_1)",
            18,
        ),
    },
    "mask_decoder": {},
    "memory_encoder": {},
    "image_encoder": {},
}
# Operations that move values without computing: a region ends at the first other operation that reads it.
LAYOUT_OPS = {"Reshape", "Transpose", "Split", "Concat", "Slice", "Squeeze", "Unsqueeze"}


def video_config():
    """transformers' EdgeTAM video configuration as edgetam.yaml declares the model.

    Its vision part is left at the default: the graphs here start from the image encoder's outputs.
    """
    from transformers import EdgeTamVideoConfig

    return EdgeTamVideoConfig(
        vision_config=None,
        enable_temporal_pos_encoding_for_object_pointers=False,
        enable_occlusion_spatial_embedding=False,
    )


def _load(module, state, prefix):
    module.load_state_dict({key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)})
    return module.eval()


def edgetam_modules(weights):
    """transformers' EdgeTAM modules after the image encoder, with the converted weights."""
    import torch
    from transformers.models.edgetam_video import modeling_edgetam_video as edgetam

    state = torch.load(weights, map_location="cpu", weights_only=True)
    config = video_config()
    config.mask_decoder_config._attn_implementation = "eager"
    modules = {
        "memory_attention": _load(edgetam.EdgeTamVideoMemoryAttention(config), state, "memory_attention."),
        "memory_encoder": _load(edgetam.EdgeTamVideoMemoryEncoder(config), state, "memory_encoder."),
        "spatial_perceiver": _load(edgetam.EdgeTamVideoPerceiverResampler(config), state, "spatial_perceiver."),
        "prompt_encoder": _load(
            edgetam.EdgeTamVideoPromptEncoder(config.prompt_encoder_config), state, "prompt_encoder."
        ),
        "mask_decoder": _load(edgetam.EdgeTamVideoMaskDecoder(config.mask_decoder_config), state, "mask_decoder."),
        "object_pointer_proj": _load(
            edgetam.EdgeTamVideoFeedForward(HIDDEN, HIDDEN, HIDDEN, 3), state, "object_pointer_proj."
        ),
    }
    tensors = {
        name: state[name]
        for name in ("no_memory_embedding", "no_object_pointer", "memory_temporal_positional_encoding")
    }
    return modules, tensors


def sine_position_encoding(channels, height, width):
    """transformers' normalized sine position encoding of a [1, channels, height, width] feature map."""
    import torch
    from transformers.models.edgetam_video.modeling_edgetam_video import EdgeTamVideoPositionEmbeddingSine

    encoding = EdgeTamVideoPositionEmbeddingSine(num_position_features=channels // 2, normalize=True)
    return encoding((1, channels, height, width), "cpu", torch.float32)


def rotation_matrix(size):
    """R with rotate_pairwise(x) = x @ R.T, taken from transformers' rotate_pairwise itself."""
    import torch
    from transformers.models.edgetam_video.modeling_edgetam_video import rotate_pairwise

    matrix = rotate_pairwise(torch.eye(size)).T
    if not torch.equal(matrix.abs().sum(0), torch.ones(size)):
        raise ValueError("transformers' rotate_pairwise is not a signed permutation of the channels")
    return matrix


def transformed_projection(linear, matrix):
    """A Linear whose output is matrix times linear's output."""
    import torch

    transformed = torch.nn.Linear(linear.in_features, linear.out_features)
    with torch.no_grad():
        transformed.weight.copy_(matrix @ linear.weight)
        transformed.bias.copy_(matrix @ linear.bias)
    return transformed


def pointwise_convolution(linear, scale=None):
    """A 1x1 convolution computing linear's output, times scale if given, at every position of a [1, C, H, W] map."""
    import torch

    if scale is None:
        scale = torch.ones(linear.out_features)
    convolution = torch.nn.Conv2d(linear.in_features, linear.out_features, kernel_size=1)
    with torch.no_grad():
        convolution.weight.copy_((scale[:, None] * linear.weight)[:, :, None, None])
        convolution.bias.copy_(scale * linear.bias)
    return convolution


def memory_position_encoding(spatial_perceiver):
    """One memory's latent position encoding, as the spatial perceiver returns it: zeros for its whole-frame latents
    and the window grid's sine encoding for its window latents."""
    import torch

    windows = int(math.sqrt(spatial_perceiver.num_latents_2d))
    grid = sine_position_encoding(MEMORY_DIM, windows, windows).flatten(2).transpose(1, 2)[0]
    return torch.cat([torch.zeros(spatial_perceiver.num_latents_1d, MEMORY_DIM), grid])


def image_position_encoding(prompt_encoder):
    """The prompt encoder's random-Fourier encoding of the 64x64 embedding grid's cell centres."""
    import torch

    rows, columns = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
    centres = torch.stack([(columns + 0.5) / 64, (rows + 0.5) / 64], dim=-1)[None]
    return prompt_encoder.shared_embedding(centres).permute(0, 3, 1, 2).clone()


def padding_scores(padding):
    """The score each decoder token adds as an attention key: log(m / n) for each of the n padding slots, which
    weigh as the m padding points transformers' prompt encoder gives the frame; 0 for every other token.

    With at most PROMPT_POINTS - 1 points, n is 1, 2 or 3: 1 and 2 for a frame with points, 3 for a tracked
    frame. (n - 1)(3 - n) and (n - 1)(n - 2) / 2 are 1 for n = 2 and n = 3 and 0 otherwise.
    """
    import torch

    slots = padding.sum(dim=1, keepdim=True)
    two, three = (slots - 1) * (3 - slots), (slots - 1) * (slots - 2) / 2
    score = two * math.log(PROMPTED_PADDING / 2) + three * math.log(TRACKED_PADDING / 3)
    prompt = padding * score
    return torch.cat([torch.zeros(1, OUTPUT_TOKENS, 1), prompt], dim=1).reshape(1, 1, 1, -1)


class EdgeTam(NamedTuple):
    """The modules surgery builds, and transformers' modules and tensors they are checked against."""

    components: dict  # component -> torch module, for every component after the image encoder
    upstream: dict  # transformers' module name -> module
    tensors: dict  # checkpoint tensor name -> tensor


def build_components(weights):
    """The memory_attention, mask_decoder and memory_encoder modules the compiler builds."""
    import torch

    modules, tensors = edgetam_modules(weights)
    no_memory = tensors["no_memory_embedding"].reshape(1, HIDDEN, 1, 1)
    rotation = rotation_matrix(HIDDEN)

    class Attention(torch.nn.Module):
        """One head's attention with rotary encoding from constant tables, and scores added before the softmax.

        The attention scale, 1/16, is folded into the query projections; a power of two, it folds exactly.
        """

        def __init__(self, attention, query_rope, key_rope):
            super().__init__()
            if math.log2(attention.scaling) != round(math.log2(attention.scaling)):
                raise ValueError(f"the attention scale {attention.scaling} is not a power of two")
            scale = attention.scaling * torch.eye(HIDDEN)
            self.q_proj = transformed_projection(attention.q_proj, scale)
            self.q_rotated = transformed_projection(attention.q_proj, scale @ rotation)
            self.k_proj, self.v_proj, self.o_proj = attention.k_proj, attention.v_proj, attention.o_proj
            self.k_rotated = transformed_projection(attention.k_proj, rotation)
            for name, table in zip(("q_cos", "q_sin", "k_cos", "k_sin"), (*query_rope, *key_rope)):
                self.register_buffer(name, table.clone())

        def forward(self, query, key, value, scores=None):
            query = self.q_proj(query) * self.q_cos + self.q_rotated(query) * self.q_sin
            key = self.k_proj(key) * self.k_cos + self.k_rotated(key) * self.k_sin
            weights = torch.matmul(query, key.transpose(1, 2))
            if scores is not None:
                weights = weights + scores
            return self.o_proj(torch.matmul(torch.softmax(weights, dim=-1), self.v_proj(value)))

    class MemoryAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            upstream = modules["memory_attention"]
            query_rope = upstream.rotary_emb()
            slot_cos, slot_sin = upstream.rotary_emb_k()
            # Each memory's whole-frame latents and the object pointers have no rotary encoding.
            plain_cos = torch.ones(MEMORY_TOKENS - slot_cos.shape[0], HIDDEN)
            plain_sin = torch.zeros(MEMORY_TOKENS - slot_sin.shape[0], HIDDEN)
            pointer_cos = torch.ones(OBJECT_POINTERS * POINTER_TOKENS, HIDDEN)
            pointer_sin = torch.zeros(OBJECT_POINTERS * POINTER_TOKENS, HIDDEN)
            key_cos = torch.cat([*[torch.cat([plain_cos, slot_cos]) for _ in range(MEMORY_SLOTS)], pointer_cos])
            key_sin = torch.cat([*[torch.cat([plain_sin, slot_sin]) for _ in range(MEMORY_SLOTS)], pointer_sin])
            self.layers = torch.nn.ModuleList()
            for layer in upstream.layers:
                block = torch.nn.Module()
                block.layer_norm1, block.layer_norm2, block.layer_norm3, block.mlp = (
                    layer.layer_norm1,
                    layer.layer_norm2,
                    layer.layer_norm3,
                    layer.mlp,
                )
                block.self_attention = Attention(layer.self_attn, query_rope, query_rope)
                block.cross_attention = Attention(layer.cross_attn_image, query_rope, (key_cos, key_sin))
                self.layers.append(block)
            self.layer_norm = upstream.layer_norm
            self.register_buffer("no_memory", no_memory.clone())
            vision_encoding = sine_position_encoding(HIDDEN, 64, 64).flatten(2).transpose(1, 2)
            self.register_buffer("query_encoding", 0.1 * vision_encoding)
            # Slot 0 is the prompted frame, whose temporal encoding is the last; slot k is k frames back.
            temporal = tensors["memory_temporal_positional_encoding"].reshape(MEMORY_SLOTS, MEMORY_DIM)
            slot_encoding = memory_position_encoding(modules["spatial_perceiver"])
            key_encoding = [slot_encoding + temporal[slot - 1] for slot in range(MEMORY_SLOTS)]
            key_encoding.append(torch.zeros(OBJECT_POINTERS * POINTER_TOKENS, MEMORY_DIM))
            self.register_buffer("key_encoding", torch.cat(key_encoding)[None])
            self.register_buffer("memory_tokens", torch.eye(MEMORY_SLOTS).repeat_interleave(MEMORY_TOKENS, dim=1))
            self.register_buffer("pointer_tokens", torch.eye(OBJECT_POINTERS).repeat_interleave(POINTER_TOKENS, dim=1))

        def forward(self, image_embeddings, memories, memory_mask, object_pointers, object_pointer_mask):
            features = (image_embeddings - self.no_memory).flatten(2).transpose(1, 2) + self.query_encoding
            # Each object pointer is POINTER_TOKENS consecutive MEMORY_DIM-wide tokens.
            keys = torch.cat([memories, object_pointers.reshape(1, -1, MEMORY_DIM)], dim=1)
            filled = torch.cat([memory_mask @ self.memory_tokens, object_pointer_mask @ self.pointer_tokens], dim=1)
            scores = ((1 - filled) * EMPTY_SLOT_SCORE)[:, None, :]
            for block in self.layers:
                query = block.layer_norm1(features)
                features = features + block.self_attention(query, query, query)
                query = block.layer_norm2(features)
                features = features + block.cross_attention(query, keys + self.key_encoding, keys, scores)
                features = features + block.mlp(block.layer_norm3(features))
            return self.layer_norm(features).transpose(1, 2).reshape(EMBEDDING)

    class DecoderAttention(torch.nn.Module):
        """transformers' mask decoder attention on [1, tokens, channels] inputs, without its point-batch axis, with
        scores added before the softmax."""

        def __init__(self, attention):
            super().__init__()
            self.q_proj, self.k_proj, self.v_proj, self.o_proj = (
                attention.q_proj,
                attention.k_proj,
                attention.v_proj,
                attention.o_proj,
            )
            self.heads, self.scaling = attention.num_attention_heads, attention.scaling

        def split_heads(self, values):
            return values.reshape(1, values.shape[1], self.heads, -1).transpose(1, 2)

        def forward(self, query, key, value, scores=None):
            query, key, value = (
                self.split_heads(self.q_proj(query)),
                self.split_heads(self.k_proj(key)),
                self.split_heads(self.v_proj(value)),
            )
            weights = torch.matmul(query, key.transpose(2, 3)) * self.scaling
            if scores is not None:
                weights = weights + scores
            attended = torch.matmul(torch.softmax(weights, dim=-1), value).transpose(1, 2)
            return self.o_proj(attended.reshape(1, attended.shape[1], -1))

    class MaskDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            prompt, decoder = modules["prompt_encoder"], modules["mask_decoder"]
            self.decoder = decoder
            self.attentions = torch.nn.ModuleList(
                [
                    torch.nn.ModuleList(
                        DecoderAttention(attention)
                        for attention in (
                            layer.self_attn,
                            layer.cross_attn_token_to_image,
                            layer.cross_attn_image_to_token,
                        )
                    )
                    for layer in decoder.transformer.layers
                ]
            )
            self.final_attention = DecoderAttention(decoder.transformer.final_attn_token_to_image)
            self.object_pointer_proj = modules["object_pointer_proj"]
            self.register_buffer("gaussian", prompt.shared_embedding.positional_embedding.clone())
            # One row per point type, in POINT_TYPES order.
            self.register_buffer("point_table", torch.cat([prompt.not_a_point_embed.weight, prompt.point_embed.weight]))
            self.register_buffer("dense", prompt.no_mask_embed.weight.reshape(1, HIDDEN, 1, 1).clone())
            self.register_buffer("image_encoding", image_position_encoding(prompt))
            tokens = torch.cat([decoder.obj_score_token.weight, decoder.iou_token.weight, decoder.mask_tokens.weight])
            self.register_buffer("output_tokens", tokens[None].clone())
            self.register_buffer("no_object_pointer", tensors["no_object_pointer"].reshape(1, 1, HIDDEN).clone())

        def forward(self, image_embeddings, high_res_features_0, high_res_features_1, point_coords, point_types):
            decoder = self.decoder
            coords = point_coords[:, :, 0, :].transpose(1, 2)
            types = point_types[:, :, 0, :].transpose(1, 2)
            projected = 2 * math.pi * (coords @ self.gaussian)
            # A padding point gets no positional encoding, only its own embedding.
            padding = types[..., :1]
            encoding = torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1) * (1 - padding)
            tokens = torch.cat([self.output_tokens, encoding + types @ self.point_table], dim=1)
            queries, keys = self.two_way_transformer(
                tokens, image_embeddings + self.dense, self.image_encoding, padding_scores(padding)
            )
            mask_tokens = queries[:, 2 : 2 + MASKS]
            image = keys.transpose(1, 2).reshape(EMBEDDING)
            upscaled = decoder.activation(
                decoder.upscale_layer_norm(decoder.upscale_conv1(image) + high_res_features_1)
            )
            upscaled = decoder.activation(decoder.upscale_conv2(upscaled) + high_res_features_0)
            hyper = torch.cat(
                [mlp(mask_tokens[:, index : index + 1]) for index, mlp in enumerate(decoder.output_hypernetworks_mlps)],
                dim=1,
            )
            masks = (hyper @ upscaled.flatten(2)).reshape(1, MASKS, MASK_SIZE, MASK_SIZE)
            iou_scores = decoder.iou_prediction_head(queries[:, 1:2]).reshape(1, MASKS)
            object_score = decoder.pred_obj_score_head(queries[:, 0:1])
            visible = torch.clamp(object_score * GATE_SCALE, 0, 1)
            pointers = self.object_pointer_proj(mask_tokens)
            pointers = self.no_object_pointer + visible * (pointers - self.no_object_pointer)
            return masks, iou_scores, object_score.reshape(1, 1), pointers

        def two_way_transformer(self, tokens, image, image_encoding, scores):
            """transformers' two-way transformer, with scores added wherever the decoder tokens are attention keys."""
            transformer = self.decoder.transformer
            image = image.flatten(2).transpose(1, 2)
            image_encoding = image_encoding.flatten(2).transpose(1, 2)
            queries, keys = tokens, image
            for layer, (self_attention, token_to_image, image_to_token) in zip(transformer.layers, self.attentions):
                if layer.skip_first_layer_pe:
                    queries = self_attention(queries, queries, queries, scores)
                else:
                    query = queries + tokens
                    queries = queries + self_attention(query, query, queries, scores)
                queries = layer.layer_norm1(queries)
                queries = layer.layer_norm2(queries + token_to_image(queries + tokens, keys + image_encoding, keys))
                queries = layer.layer_norm3(queries + layer.mlp(queries))
                keys = layer.layer_norm4(
                    keys + image_to_token(keys + image_encoding, queries + tokens, queries, scores)
                )
            attended = self.final_attention(queries + tokens, keys + image_encoding, keys)
            return transformer.layer_norm_final_attn(queries + attended), keys

    class FuserBlock(torch.nn.Module):
        """A memory fuser block with its pointwise Linears as 1x1 convolutions on the [1, C, H, W] map, and its
        layer scale folded into the second; the compiler rejects the Linears on the channels-last map."""

        def __init__(self, block):
            super().__init__()
            self.depthwise_conv, self.layer_norm, self.activation = (
                block.depthwise_conv,
                block.layer_norm,
                block.activation,
            )
            self.pointwise_conv1 = pointwise_convolution(block.pointwise_conv1)
            self.pointwise_conv2 = pointwise_convolution(block.pointwise_conv2, block.scale)

        def forward(self, features):
            update = self.layer_norm(self.depthwise_conv(features))
            return features + self.pointwise_conv2(self.activation(self.pointwise_conv1(update)))

    class MemoryEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            upstream = modules["memory_encoder"]
            self.mask_downsampler, self.feature_projection, self.projection = (
                upstream.mask_downsampler,
                upstream.feature_projection,
                upstream.projection,
            )
            self.fuser = torch.nn.Sequential(*[FuserBlock(block) for block in upstream.memory_fuser.layers])
            self.spatial_perceiver = modules["spatial_perceiver"]
            self.register_buffer("no_memory", no_memory.clone())
            self.register_buffer("encoding", sine_position_encoding(MEMORY_DIM, 64, 64))

        def forward(self, image_embeddings, mask):
            masks = self.mask_downsampler(mask * MASK_SCALE + MASK_BIAS)
            features = self.fuser(self.feature_projection(image_embeddings - self.no_memory) + masks)
            memory, _ = self.spatial_perceiver(self.projection(features), self.encoding)
            return memory

    components = {
        "memory_attention": MemoryAttention(),
        "mask_decoder": MaskDecoder(),
        "memory_encoder": MemoryEncoder(),
    }
    return EdgeTam({name: module.eval() for name, module in components.items()}, modules, tensors)


def _seeded_memory_bank(inputs, rng, seed):
    """Three filled memory slots and three object pointers."""
    inputs["memory_mask"] = (np.arange(MEMORY_SLOTS) < 3).astype(np.float32)[None]
    inputs["object_pointer_mask"] = (np.arange(OBJECT_POINTERS) < 3).astype(np.float32)[None]


def _seeded_prompt(inputs, rng, seed):
    """A tracked frame, a click and a box, in turn by seed."""
    shapes = INTERFACES["mask_decoder"]["inputs"]
    kinds = [[], ["foreground"], ["box_top_left", "box_bottom_right"]][seed % 3]
    kinds = [*kinds, *["padding"] * (PROMPT_POINTS - len(kinds))]
    inputs["point_coords"] = rng.uniform(-1, 1, shapes["point_coords"]).astype(np.float32)
    types = np.zeros(shapes["point_types"], np.float32)
    types[0, [POINT_TYPES.index(kind) for kind in kinds], 0, range(PROMPT_POINTS)] = 1
    inputs["point_types"] = types


def _seeded_mask(inputs, rng, seed):
    """Mask probabilities in [0, 1]."""
    inputs["mask"] = rng.uniform(0, 1, INTERFACES["memory_encoder"]["inputs"]["mask"]).astype(np.float32)


# Each component's inputs that are not features: masks, prompts and probabilities.
SEEDED_INPUTS = {
    "memory_attention": _seeded_memory_bank,
    "mask_decoder": _seeded_prompt,
    "memory_encoder": _seeded_mask,
}


def seeded_inputs(component, seed):
    """Seeded inputs for a component's export and check: normal features, and valid masks, prompts and
    probabilities."""
    rng = np.random.default_rng(seed)
    shapes = INTERFACES[component]["inputs"]
    inputs = {name: rng.normal(0, 1, shape).astype(np.float32) for name, shape in shapes.items()}
    SEEDED_INPUTS[component](inputs, rng, seed)
    return inputs


def _memory_attention_reference(edgetam, inputs):
    """transformers' memory attention over the filled memory slots and object pointers only."""
    import torch

    tensor = {name: torch.from_numpy(value) for name, value in inputs.items()}
    no_memory = edgetam.tensors["no_memory_embedding"].reshape(1, HIDDEN, 1, 1)
    features = (tensor["image_embeddings"] - no_memory).flatten(2).permute(2, 0, 1)
    encoding = sine_position_encoding(HIDDEN, 64, 64).flatten(2).permute(2, 0, 1)
    slots = [slot for slot in range(MEMORY_SLOTS) if inputs["memory_mask"][0, slot]]
    temporal = edgetam.tensors["memory_temporal_positional_encoding"]
    slot_encoding = memory_position_encoding(edgetam.upstream["spatial_perceiver"])
    memories = [tensor["memories"][0, slot * MEMORY_TOKENS : (slot + 1) * MEMORY_TOKENS] for slot in slots]
    encodings = [slot_encoding + temporal[slot - 1].reshape(MEMORY_DIM) for slot in slots]
    pointers = tensor["object_pointers"][0][inputs["object_pointer_mask"][0] > 0].reshape(-1, MEMORY_DIM)
    output = edgetam.upstream["memory_attention"](
        current_vision_features=features,
        memory=torch.cat([*memories, pointers])[:, None],
        current_vision_position_embeddings=encoding,
        memory_posision_embeddings=torch.cat([*encodings, torch.zeros_like(pointers)])[:, None],
        num_object_pointer_tokens=pointers.shape[0],
        num_spatial_memory_tokens=len(slots),
    )
    # As transformers' tracker reshapes the conditioned features for the mask decoder.
    return [output.squeeze(1).transpose(1, 2).reshape(EMBEDDING)]


def _mask_decoder_reference(edgetam, inputs):
    """transformers' prompt encoder and mask decoder on the real prompt points only, with both mask outputs."""
    import torch

    tensor = {name: torch.from_numpy(value) for name, value in inputs.items()}
    prompt, decoder = edgetam.upstream["prompt_encoder"], edgetam.upstream["mask_decoder"]
    # Only the user's points, and a tracked frame's one padding point; the prompt encoder appends another.
    types = inputs["point_types"][0, :, 0].argmax(0)
    points = [slot for slot, kind in enumerate(types) if kind != POINT_TYPES.index("padding")]
    # transformers' point labels count from background as 0, one below POINT_TYPES; -1 is padding.
    labels = torch.tensor([[[int(types[slot]) - 1 for slot in points] or [-1]]])
    # The prompt encoder shifts pixel coordinates to the pixel centre and normalizes them by the image size.
    pixels = (tensor["point_coords"][0, :, 0].T + 1) / 2 * IMAGE_SIZE - 0.5
    pixels = pixels[points] if points else torch.zeros(1, 2)
    sparse = prompt._embed_points(pixels[None, None], labels, pad=True)
    dense = prompt.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(1, -1, 64, 64)
    outputs = {}
    for multimask in (False, True):
        decoder.dynamic_multimask_via_stability = False
        outputs[multimask] = decoder(
            image_embeddings=tensor["image_embeddings"],
            image_positional_embeddings=image_position_encoding(prompt),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=multimask,
            high_resolution_features=[tensor["high_res_features_0"], tensor["high_res_features_1"]],
        )
    masks = torch.cat([outputs[False][0], outputs[True][0]], dim=2)[:, 0]
    scores = torch.cat([outputs[False][1], outputs[True][1]], dim=2)[:, 0]
    tokens = torch.cat([outputs[False][2], outputs[True][2]], dim=2)[:, 0]
    object_score = outputs[False][3][:, 0]
    pointers = edgetam.upstream["object_pointer_proj"](tokens)
    visible = (object_score > 0).float()[:, None]
    pointers = visible * pointers + (1 - visible) * edgetam.tensors["no_object_pointer"]
    return [masks, scores, object_score, pointers]


def _memory_encoder_reference(edgetam, inputs):
    """transformers' memory encoder and spatial perceiver."""
    import torch

    tensor = {name: torch.from_numpy(value) for name, value in inputs.items()}
    no_memory = edgetam.tensors["no_memory_embedding"].reshape(1, HIDDEN, 1, 1)
    features, encoding = edgetam.upstream["memory_encoder"](
        tensor["image_embeddings"] - no_memory, tensor["mask"] * MASK_SCALE + MASK_BIAS
    )
    memory, _ = edgetam.upstream["spatial_perceiver"](features, encoding)
    return [memory]


# transformers' own computation of what each component returns.
REFERENCES = {
    "memory_attention": _memory_attention_reference,
    "mask_decoder": _mask_decoder_reference,
    "memory_encoder": _memory_encoder_reference,
}


def export_graph(module, component, output):
    import onnxsim
    import torch

    interface = INTERFACES[component]
    inputs = tuple(torch.from_numpy(value) for value in seeded_inputs(component, 0).values())
    with torch.no_grad():
        torch.onnx.export(
            module,
            inputs,
            output,
            input_names=list(interface["inputs"]),
            output_names=list(interface["outputs"]),
            opset_version=OPSET,
            dynamo=False,
        )
    # Fold only: onnxsim's graph optimizations turn MatMul+Add into Reshape+Gemm+Reshape, which the compiler
    # mishandles before an attention's head split.
    simplified, valid = onnxsim.simplify(onnx.load(output), perform_optimization=False)
    if not valid:
        raise RuntimeError(f"onnxsim could not validate the simplified {component}")
    onnx.save(simplified, output)


def add_no_memory_embedding(source, output, no_memory):
    """The converted image encoder with the no-memory embedding added to its vision features."""
    model = onnx.load(source)
    outputs = {value.name: value for value in model.graph.output}
    # convert.py's OUTPUTS; surgery.py cannot import convert.py, which is not a compile input.
    if list(outputs) != ["vision_features", "high_res_features_0", "high_res_features_1"]:
        raise ValueError(f"unexpected image encoder outputs {list(outputs)}")
    embedding = onnx.numpy_helper.from_array(
        no_memory.reshape(1, HIDDEN, 1, 1).astype(np.float32), "no_memory_embedding"
    )
    model.graph.initializer.append(embedding)
    model.graph.node.append(
        onnx.helper.make_node(
            "Add", ["vision_features", "no_memory_embedding"], ["image_embeddings"], name="/no_memory/Add"
        )
    )
    outputs["vision_features"].name = "image_embeddings"
    onnx.save(model, output)


def validate_export(path, component):
    model = onnx.load(path)
    onnx.checker.check_model(model, full_check=True)
    interface = INTERFACES[component]
    for values, expected in ((model.graph.input, interface["inputs"]), (model.graph.output, interface["outputs"])):
        shapes = {value.name: [dim.dim_value for dim in value.type.tensor_type.shape.dim] for value in values}
        if shapes != expected:
            raise ValueError(f"{component} has interface {shapes}, expected {expected}")


def check_graph(path, component, edgetam):
    """Fail unless the ONNX's outputs equal transformers' modules' on seeded inputs.

    The measured differences are about 1e-6 of each output's range; MAX_RELATIVE_ERROR only allows for float32
    accumulation order."""
    import onnxruntime as ort
    import torch

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    names = list(INTERFACES[component]["outputs"])
    for seed in SEEDS:
        inputs = seeded_inputs(component, seed)
        with torch.no_grad():
            expected = [value.numpy() for value in REFERENCES[component](edgetam, inputs)]
        actual = session.run(names, inputs)
        for name, reference, value in zip(names, expected, actual):
            error = float(np.abs(reference - value).max() / max(np.abs(reference).max(), 1e-6))
            if error > MAX_RELATIVE_ERROR:
                raise ValueError(f"seed {seed}: ONNX {component} {name} differs from transformers' by {error:.2e}")
    print(f"transformers/ONNX {component} check passed with seeds {SEEDS.start}-{SEEDS.stop - 1}", flush=True)


def export_components(image_encoder, weights, output_dir):
    """Write each component's FP32 graph, checked against transformers, as <output_dir>/<component>.onnx."""
    output_dir.mkdir(parents=True, exist_ok=True)
    edgetam = build_components(weights)
    paths = {component: output_dir / f"{component}.onnx" for component in COMPONENTS}
    add_no_memory_embedding(image_encoder, paths["image_encoder"], edgetam.tensors["no_memory_embedding"].numpy())
    validate_export(paths["image_encoder"], "image_encoder")
    for component, module in edgetam.components.items():
        export_graph(module, component, paths[component])
        validate_export(paths[component], component)
        check_graph(paths[component], component, edgetam)
    return paths


def _annotated_operation(node, readers):
    """The operation a precision annotation follows: the Relu the compiler fuses into a convolution, or the node."""
    after = readers[node.output[0]]
    return after[0] if node.op_type == "Conv" and [reader.op_type for reader in after] == ["Relu"] else node


def _first_operations(readers, value):
    """The operations that read value, looking through layout operations."""
    operations = []
    for reader in readers[value]:
        if reader.op_type in LAYOUT_OPS:
            for output in reader.output:
                operations.extend(_first_operations(readers, output))
        else:
            operations.append(reader)
    return operations


def precision_marks(model, component, regions):
    """Return the precision to annotate after each operation, by name: bfloat16 for the regions' operations and int8
    for the operations that first read their results.

    The compiler gives an annotation's precision to the operation producing its input, and each later operation takes
    the precision its inputs share, so an int8 mark on each first reader ends a region.
    """
    readers = defaultdict(list)
    for node in model.graph.node:
        for value in node.input:
            readers[value].append(node)
    promoted = {}
    for region in regions:
        pattern, count = BF16_REGIONS[component][region]
        operations = [node for node in model.graph.node if re.fullmatch(pattern, node.name)]
        if len(operations) != count:
            raise ValueError(f"expected {count} {component} {region} operations, found {len(operations)}")
        promoted.update({node.name: node for node in operations})
    marks = {_annotated_operation(node, readers).name: "bfloat16" for node in promoted.values()}
    for node in promoted.values():
        for output in node.output:
            for reader in _first_operations(readers, output):
                if reader.name not in promoted:
                    marks.setdefault(_annotated_operation(reader, readers).name, "int8")
    return marks


def annotate_bf16_regions(source, output, component, regions):
    """Write the graph with the regions' operations marked BF16 for the compiler; return the marks per precision.

    ONNX Runtime cannot run the annotated graph, so checks against FP32 keep using the source.
    """
    model = onnx.load(source)
    marks = precision_marks(model, component, regions)
    nodes = []
    for node in model.graph.node:
        nodes.append(node)
        if node.name in marks:
            value = node.output[0]
            node.output[0] = f"{value}/unannotated"
            nodes.append(
                onnx.helper.make_node(
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
    model.opset_import.append(onnx.helper.make_opsetid("ai.sima", 1))
    onnx.checker.check_model(model)
    onnx.save(model, output)
    return {precision: list(marks.values()).count(precision) for precision in ("bfloat16", "int8")}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image-encoder", type=Path, required=True, help="the converted image_encoder.onnx")
    parser.add_argument("--weights", type=Path, required=True, help="the converted weights.pth")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in export_components(args.image_encoder, args.weights, args.output_dir).values():
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
