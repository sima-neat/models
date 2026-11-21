"""Fixed-capacity SAM decoding with exact active-token semantics and the upstream checkpoint modules.

Inactive prompt slots must be excluded wherever sparse tokens are attention keys. Zero embeddings alone do not
remove a token from softmax. Dense image-key attentions are unchanged. Arithmetic masking avoids an infinite
sentinel or boolean-select operator, and inactive logits cannot dominate the softmax normalization.
"""

import math

import torch


def masked_softmax(logits, valid_keys):
    """Softmax over active keys only; every row must contain at least one active key.

    Replace inactive logits by the active mean before softmax, then remove and renormalize their probability.
    The replacement cannot exceed the active maximum, so even arbitrarily large inactive logits cannot cause
    the active probabilities to underflow. Only ordinary arithmetic/reductions are required by the export.
    """
    mask = valid_keys[:, None, None, :]
    active = logits * mask
    reference = (active / mask.sum(dim=-1, keepdim=True)).sum(dim=-1, keepdim=True)
    weights = torch.softmax(active + reference * (1 - mask), dim=-1) * mask
    return weights / weights.sum(dim=-1, keepdim=True)


def masked_attention(attention, query, key, value, valid_keys):
    """Apply an existing SAM Attention module while excluding inactive keys."""
    query = attention._separate_heads(attention.q_proj(query), attention.num_heads)
    key = attention._separate_heads(attention.k_proj(key), attention.num_heads)
    value = attention._separate_heads(attention.v_proj(value), attention.num_heads)
    logits = (query @ key.transpose(-2, -1)) / math.sqrt(attention.internal_dim // attention.num_heads)
    attended = masked_softmax(logits, valid_keys) @ value
    return attention.out_proj(attention._recombine_heads(attended))


def masked_transformer(transformer, image, image_pe, tokens, valid_keys):
    """The upstream two-way transformer; mask both self-attention and image-to-token attention keys."""
    keys = image.flatten(2).transpose(1, 2)
    key_pe = image_pe.flatten(2).transpose(1, 2)
    queries = tokens
    for layer in transformer.layers:
        if layer.skip_first_layer_pe:
            queries = masked_attention(layer.self_attn, queries, queries, queries, valid_keys)
        else:
            query = queries + tokens
            queries = queries + masked_attention(layer.self_attn, query, query, queries, valid_keys)
        queries = layer.norm1(queries)
        queries = layer.norm2(queries + layer.cross_attn_token_to_image(q=queries + tokens, k=keys + key_pe, v=keys))
        queries = layer.norm3(queries + layer.mlp(queries))
        keys = layer.norm4(
            keys
            + masked_attention(layer.cross_attn_image_to_token, keys + key_pe, queries + tokens, queries, valid_keys)
        )
    queries = transformer.norm_final_attn(
        queries + transformer.final_attn_token_to_image(q=queries + tokens, k=keys + key_pe, v=keys)
    )
    return queries, keys


def predict_masks(decoder, image, image_pe, sparse, dense, valid_prompts):
    """Reuse SAM's learned output tokens, transformer layers, mask heads and IoU head unchanged."""
    output_tokens = torch.cat([decoder.iou_token.weight, decoder.mask_tokens.weight], dim=0)[None]
    tokens = torch.cat([output_tokens, sparse], dim=1)
    valid_keys = torch.cat([torch.ones_like(output_tokens[:, :, 0]), valid_prompts], dim=1)
    tokens, encoded = masked_transformer(decoder.transformer, image + dense, image_pe, tokens, valid_keys)
    encoded = encoded.transpose(1, 2).reshape_as(image)
    upscaled = decoder.output_upscaling(encoded)
    hyper = torch.stack(
        [head(tokens[:, index + 1]) for index, head in enumerate(decoder.output_hypernetworks_mlps)], dim=1
    )
    masks = (hyper @ upscaled.flatten(2)).reshape(image.shape[0], decoder.num_mask_tokens, *upscaled.shape[2:])
    return masks, decoder.iou_prediction_head(tokens[:, 0])
