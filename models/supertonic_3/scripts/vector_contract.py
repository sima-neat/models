"""Fixed source and compiler contracts for the Supertonic 3 vector field."""

from __future__ import annotations

from collections import OrderedDict

SOURCE_SHA256 = "883ac868ea0275ef0e991524dc64f16b3c0376efd7c320af6b53f5b780d7c61c"

SOURCE_INPUT_SHAPES = OrderedDict(
    [
        ("noisy_latent", (1, 144, 192)),
        ("text_emb", (1, 256, 192)),
        ("style_ttl", (1, 50, 256)),
        ("latent_mask", (1, 1, 192)),
        ("text_mask", (1, 1, 192)),
        ("current_step", (1,)),
        ("total_step", (1,)),
    ]
)
SOURCE_OUTPUT_SHAPES = OrderedDict([("denoised_latent", (1, 144, 192))])

PRE_EXTERNAL_INPUT_SHAPES = OrderedDict(
    [
        ("noisy_latent", (1, 144, 1, 192)),
        ("text_emb", (1, 256, 1, 192)),
        ("style_ttl", (1, 256, 1, 50)),
        ("style_key", (1, 256, 1, 50)),
        ("latent_mask", (1, 1, 1, 192)),
        # LLiMa-compatible attention score layout: [B, key_length, 1, 1].
        ("text_mask", (1, 192, 1, 1)),
        ("current_step", (1, 1, 1, 1)),
        ("total_step", (1, 1, 1, 1)),
    ]
)

COMPILED_INPUT_SHAPES = OrderedDict(
    [
        ("noisy_latent", (1, 144, 1, 192)),
        ("text_emb", (1, 256, 1, 192)),
        ("style_ttl", (1, 256, 1, 50)),
        ("style_key", (1, 256, 1, 50)),
        ("latent_mask", (1, 1, 1, 192)),
        # LLiMa-compatible attention score layout: [B, key_length, 1, 1].
        ("text_mask", (1, 192, 1, 1)),
        ("time_sinusoidal", (1, 64, 1, 1)),
        ("rope_tables", (1, 128, 1, 192)),
    ]
)
COMPILED_OUTPUT_SHAPES = OrderedDict([("velocity", (1, 144, 1, 192))])

SYMBOLIC_DIMENSIONS = {
    "batch_size": 1,
    "text_length": 192,
    "latent_length": 192,
}
