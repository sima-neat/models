"""Mask2Former's interface, shared by conversion, compile and the application: component shapes, the routing of
tensors between components, the image preprocessing and the instance decoding. It needs only NumPy and Pillow, and
no model weights.

The model runs as sixteen components. The backbone (Swin-T) turns a 384x384 image into four feature maps, at 1/4
to 1/32 of the input, as two components: backbone_0 (stages 1 and 2) and backbone_1 (stages 3 and 4). The pixel
decoder turns them into per-pixel mask features at 1/4 and three multi-scale feature maps at 1/32, 1/16 and 1/8, as
thirteen components. Each of its six deformable attention layers runs as two: pixel_decoder_{2i} runs heads 0-3 of
layer i and returns their share of the attention output projection (partial{i}), and pixel_decoder_{2i+1} runs heads
4-7, adds that share and completes the layer (tokens{i}). pixel_decoder_0 also projects the feature maps to tokens
(embedded), and pixel_decoder_12 builds the 1/4 level and the mask features. The transformer decoder takes the last
layer's tokens as the multi-scale features. They are split so because compiled whole, the backbone needs more
memory to compile than a CI runner has, and so does a single deformable layer once the compiler's check simulation
is counted. The transformer decoder predicts, for each of 100 queries, class logits over the 80 COCO classes plus
"no object", and mask logits at 1/4 of the input.

Each component input takes the earlier component output of the same name (route).
"""

from __future__ import annotations

import numpy as np

PIXEL_LAYERS = 6
PIXEL_COMPONENTS = 2 * PIXEL_LAYERS + 1
COMPONENTS = (
    "backbone_0",
    "backbone_1",
    *(f"pixel_decoder_{index}" for index in range(PIXEL_COMPONENTS)),
    "transformer_decoder",
)
IMAGE_SIZE = 384
QUERIES = 100
CLASSES = 80
MASK_SIZE = IMAGE_SIZE // 4
SCALE_SIZES = (IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8)
TOKENS = sum(size * size for size in SCALE_SIZES)
TOKEN_SHAPE = (1, 256, 1, TOKENS)
MASK_FEATURES = (1, 256, MASK_SIZE, MASK_SIZE)
BACKBONE_SHAPES = tuple((1, 96 * 2**i, IMAGE_SIZE // 4 // 2**i, IMAGE_SIZE // 4 // 2**i) for i in range(4))
FEATURES = {f"features{i}": shape for i, shape in enumerate(BACKBONE_SHAPES)}


def _layer_input(layer: int) -> str:
    """The tokens deformable layer `layer` reads: the projected feature maps, then each layer's output."""
    return "embedded" if layer == 0 else f"tokens{layer - 1}"


INPUT_SHAPES = {
    "backbone_0": {"image": (1, 3, IMAGE_SIZE, IMAGE_SIZE)},
    # Stage 3's input: stage 2's output after patch merging.
    "backbone_1": {"stage3_input": BACKBONE_SHAPES[2]},
    "pixel_decoder_0": {name: FEATURES[name] for name in ("features1", "features2", "features3")},
    **{f"pixel_decoder_{2 * layer}": {f"tokens{layer - 1}": TOKEN_SHAPE} for layer in range(1, PIXEL_LAYERS)},
    **{
        f"pixel_decoder_{2 * layer + 1}": {_layer_input(layer): TOKEN_SHAPE, f"partial{layer}": TOKEN_SHAPE}
        for layer in range(PIXEL_LAYERS)
    },
    f"pixel_decoder_{PIXEL_COMPONENTS - 1}": {
        f"tokens{PIXEL_LAYERS - 1}": TOKEN_SHAPE,
        "features0": FEATURES["features0"],
    },
    # The mask features, and the multi-scale features as the last deformable layer's tokens, 1/32 to 1/8.
    "transformer_decoder": {"mask_features": MASK_FEATURES, f"tokens{PIXEL_LAYERS - 1}": TOKEN_SHAPE},
}
INPUT_SHAPES = {component: INPUT_SHAPES[component] for component in COMPONENTS}
OUTPUT_SHAPES = {
    "backbone_0": {
        "features0": FEATURES["features0"],
        "features1": FEATURES["features1"],
        "stage3_input": BACKBONE_SHAPES[2],
    },
    "backbone_1": {"features2": FEATURES["features2"], "features3": FEATURES["features3"]},
    "pixel_decoder_0": {"embedded": TOKEN_SHAPE, "partial0": TOKEN_SHAPE},
    **{f"pixel_decoder_{2 * layer}": {f"partial{layer}": TOKEN_SHAPE} for layer in range(1, PIXEL_LAYERS)},
    **{f"pixel_decoder_{2 * layer + 1}": {f"tokens{layer}": TOKEN_SHAPE} for layer in range(PIXEL_LAYERS)},
    # Only the mask features: the compiler gives each independent output branch its own MLA segment.
    f"pixel_decoder_{PIXEL_COMPONENTS - 1}": {"mask_features": MASK_FEATURES},
    "transformer_decoder": {
        # Class logits per query, classes along the channels: [1, classes + 1, 1, queries].
        "class_logits": (1, CLASSES + 1, 1, QUERIES),
        "mask_logits": (1, QUERIES, MASK_SIZE, MASK_SIZE),
    },
}
OUTPUT_SHAPES = {component: OUTPUT_SHAPES[component] for component in COMPONENTS}
# The checkpoint's ImageNet normalization, on 0-255 RGB.
PIXEL_MEAN = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
PIXEL_STD = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)


def route(component: str, available: dict[str, object]) -> list[object]:
    """A component's inputs in INPUT_SHAPES order, from the outputs so far (and the image) by name."""
    return [available[name] for name in INPUT_SHAPES[component]]


def tensor_name(component: str, name: str) -> str:
    """A component's tensor in the side-by-side ONNX that conversion writes."""
    return f"{component}_{name}"


def preprocess(image: np.ndarray) -> np.ndarray:
    """An RGB uint8 HWC image as the backbone's float32 HWC input: stretched bilinearly to IMAGE_SIZE x IMAGE_SIZE
    and normalized, as transformers' Mask2Former image processor does. At this input size, stretching scored
    higher COCO mask AP than resizing the longer side and padding (float32, 200 seeded val2017 images: 0.397 vs
    0.374)."""
    from PIL import Image

    resized = Image.fromarray(image).resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    return (np.asarray(resized, np.float32) - PIXEL_MEAN) / PIXEL_STD


def _resize(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Bilinear resize of a float32 map to (height, width)."""
    from PIL import Image

    height, width = size
    return np.asarray(
        Image.fromarray(mask.astype(np.float32), mode="F").resize((width, height), Image.Resampling.BILINEAR)
    )


def decode(
    class_logits: np.ndarray, mask_logits: np.ndarray, original_size: tuple[int, int], top_k: int = QUERIES
) -> list[tuple[int, float, np.ndarray]]:
    """The model's outputs as instances (class index, score, binary mask at the original size), as Mask2Former's
    instance inference does: the top_k (query, class) pairs by class probability, "no object" excluded; each mask's
    logits upsampled to the input, resized to the image and thresholded at 0; each score the
    class probability times the mean mask probability inside the mask."""
    logits = np.asarray(class_logits, np.float32).reshape(CLASSES + 1, QUERIES).T
    logits = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities = (probabilities / probabilities.sum(axis=1, keepdims=True))[:, :CLASSES]
    flat = probabilities.ravel()
    order = np.argsort(-flat, kind="stable")[:top_k]
    masks = np.asarray(mask_logits, np.float32).reshape(QUERIES, MASK_SIZE, MASK_SIZE)
    instances = []
    for index in order:
        query, label = divmod(int(index), CLASSES)
        logit = _resize(_resize(masks[query], (IMAGE_SIZE, IMAGE_SIZE)), original_size)
        binary = logit > 0
        mask_probability = 1 / (1 + np.exp(-logit))
        mask_score = float((mask_probability * binary).sum() / (binary.sum() + 1e-6))
        instances.append((label, float(flat[index]) * mask_score, binary))
    return instances
