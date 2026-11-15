# EdgeTAM

![EdgeTAM model hero](hero.png)

## Overview

EdgeTAM follows one object through a video. Mark the object on the first frame
with a click or a box, and it returns the object's mask on every frame. Use it
to cut a moving object out of footage or to track a user-selected item. Your
application runs four components on each frame: an image encoder, a memory
attention that recalls earlier frames, a mask decoder that returns four
candidate masks, and a memory encoder that stores the chosen one.

## Inputs

One 1024×1024×3 (height, width, channel) RGB frame with ImageNet normalization.

Preprocessing:

1. Stretch to 1024×1024 with bicubic resampling.
2. Divide by 255, subtract (0.485, 0.456, 0.406), and divide by (0.229, 0.224, 0.225).

## Between the components

On each frame:

1. Run `image_encoder` for `image_embeddings`, `high_res_features_0`, and `high_res_features_1`.
2. After the prompted frame, run `memory_attention` on `image_embeddings` and the memory bank. Its `conditioned_embeddings` replace the embeddings in step 3.
3. Run `mask_decoder` on the embeddings, both `high_res_features`, and the prompt.
4. Choose from `mask_logits` (1×256×256×4). With one point or none, take the best of masks 1 to 3 by `iou_scores` and its `object_pointers` row. With two points, take pointer 0 and mask 0, or the best of masks 1 to 3 if under 98% of mask 0's pixels above −0.05 are above 0.05.
5. If `object_score_logits` is 0 or below, set the mask to −1024.
6. Resize the mask bilinearly to 1024×1024. Threshold it at 0 on the prompted frame, else apply a sigmoid.
7. Run `memory_encoder` on it and step 1's `image_embeddings`.

**Prompt:** on the first frame, a click, a click and a background click, or a box's two corners. `point_coords` holds (x + 0.5)/512 − 1 and (y + 0.5)/512 − 1, for x and y in 1024×1024 pixels. `point_types` is one-hot over padding, background, foreground, box top-left, and box bottom-right. Unused slots, and all slots on later frames, are padding.

**Memory bank:** `memories` holds seven `memory_encoder` `memory` outputs: the prompted frame's, then the latest six, newest first. `object_pointers` holds sixteen chosen pointers in that order. `memory_mask` and `object_pointer_mask` are 1 for a filled slot, else 0.

## Outputs

The chosen mask after step 5, 256×256 logits.

Postprocessing:

1. Resize it bilinearly to the frame's size.
2. Pixels above 0 belong to the object.

## Building applications

Run the package with the [Neat Model API](https://developer.sima.ai/software/develop-apps/development-workflow/model). For complete applications, start from the [Neat example apps](https://developer.sima.ai/examples).
