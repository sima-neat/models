# YOLO-Worldv2-L

![YOLO-Worldv2-L model hero](hero.png)

## Overview

YOLO-Worldv2-L finds the objects you name in text and returns a box, a class,
and a score for each. Use it to detect objects outside any fixed class list,
without training. Encode each of up to 80 names once and reuse its vector for
every image. Changing the list can change the other names' detections.

## Inputs

| Input | Size | Holds |
|---|---|---|
| `token_embeddings` | 1×1×77×512 | one name's token vectors |
| `eos_selector` | 1×1×96×1 | zeros, with a one at the end-of-text position |
| `images` | 640×640×3 (height, width, channel) | one RGB image with values from 0 to 1 |

Text preprocessing, once per name, with the package's `prompt_assets` files:

1. Tokenize with `tokenizer.json` and right-pad to 77 tokens with end-of-text ID 49407.
2. Look each token up in `token_embedding_table.npy` to fill `token_embeddings`.
3. Set `eos_selector` to one at the first end-of-text position. Only its first 77 entries are used.

Image preprocessing:

1. Resize with bilinear resampling so the longer side is 640.
2. Center it on a 640×640 gray (114, 114, 114) frame.
3. Divide by 255.

## Between the components

1. Run the text encoder on each name, divide its 512-value `text_embeddings` by its length, and cache it.
2. Fill the detector's `text_embeddings`, 1×80×512, with the cached vectors, one per slot.
3. Set `prompt_mask`, 1×80, to one for used slots, else zero.
4. With no names, skip the detector.

## Outputs

| Output | Grid | Per cell |
|---|---|---|
| `bbox_dfl_s8`, `bbox_dfl_s16`, `bbox_dfl_s32` | 80×80, 40×40, 20×20 (stride 8, 16, 32) | 16 bin logits per box edge: left, top, right, bottom |
| `class_prob_s8`, `class_prob_s16`, `class_prob_s32` | same | 80 class probabilities |

Postprocessing:

1. Softmax each edge's 16 bins and sum bin index (0 to 15) times probability. This is the edge's distance in cells.
2. The corners are the cell center (c + 0.5, r + 0.5) minus (left, top) and plus (right, bottom), times the stride.
3. Keep the cell-class pairs that score above your threshold.
4. Within each class, remove boxes that overlap a higher-scoring box by more than 0.7.
5. Keep at most 300 boxes.
6. Remove the padding, divide by the resize scale, and clip to the image.

Class index i is the name in slot i.

## Building applications

Run the package with the [Neat Model API](https://developer.sima.ai/software/develop-apps/development-workflow/model). For complete applications, start from the [Neat example apps](https://developer.sima.ai/examples).
