# EfficientSAM ViT-T

![EfficientSAM ViT-T model hero](hero.png)

## Overview

EfficientSAM ViT-T cuts out the object a user marks in an image. Mark the object
with a box or a few clicks, and it returns the object's mask, with no fixed list
of classes. Use it for photo editing, interactive labeling, or measuring a
selected item. Your application runs two components: an image encoder once per
image, and a prompt decoder once per prompt, which returns three candidate masks
with a quality score for each.

## Inputs

One 1024×1024×3 (height, width, channel) RGB image with values from 0 to 1.

Preprocessing:

1. Stretch to 1024×1024 with bilinear resampling, without keeping the aspect ratio.
2. Divide by 255.

## Between the components

1. Run `image_encoder` on the image for `image_embeddings`, 64×64×256 (height, width, channel).
2. For each prompt, run `prompt_decoder` on `image_embeddings` and the prompt's `points`.

**Prompt:** up to 6 points. A box is its top-left and bottom-right corners, and a click is a positive or a negative point. `points` holds one row of 8 values per point: for a point (x, y) in a width×height image, (x × 1024 / width + 0.5) / 1024 and (y × 1024 / height + 0.5) / 1024, then the point's type as one-hot values in the order padding, negative point, positive point, box top-left, box bottom-right, then 0. Fill the unused rows with −0.5 / 1024 for both coordinates and the padding type.

## Outputs

| Output | Size | Holds |
|---|---|---|
| `low_res_masks` | 256×256×3 | three candidate masks as logits, for the stretched 1024×1024 image |
| `iou_predictions` | 3 | the predicted quality of each mask |

Postprocessing:

1. Take the mask with the highest score in `iou_predictions`.
2. Resize its logits to the image's size with bilinear interpolation.
3. Keep the pixels whose logit is above 0.

## Building applications

Run the package with the [Neat Model API](https://developer.sima.ai/software/develop-apps/development-workflow/model). For complete applications, start from the [Neat example apps](https://developer.sima.ai/examples).
