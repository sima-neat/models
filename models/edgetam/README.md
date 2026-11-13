# EdgeTAM

![EdgeTAM model hero](hero.png)

## Overview

EdgeTAM is Meta's on-device video object segmentation model. Given a click, two
clicks or a box on one object in a video's first frame, it segments that object
in every later frame. It runs as four components per frame: an image encoder, a
memory attention that recalls earlier frames, a mask decoder that returns four
candidate masks with quality scores, and a memory encoder that remembers the
chosen mask.

## Inputs and outputs

**Frames.** Stretch each RGB frame to 1024×1024 (bicubic), scale to 0–1,
subtract (0.485, 0.456, 0.406) and divide by (0.229, 0.224, 0.225).

**Per frame:**

1. Run `image_encoder` for `image_embeddings`, `high_res_features_0` and `high_res_features_1`.
2. On tracked frames, run `memory_attention` on `image_embeddings` and the memory bank; its `conditioned_embeddings` replace the embeddings in step 3.
3. Run `mask_decoder` on the embeddings, both `high_res_features` and the prompt. Prompt the first frame with a click, a click and a background click, or a box's two corners. `point_coords` holds (x + 0.5) / 512 − 1 and (y + 0.5) / 512 − 1 in 1024×1024 pixels; `point_types` is one-hot over padding, background, foreground, box top-left and box bottom-right. Pad unused slots; tracked frames are all padding.
4. From `mask_logits` (1×256×256×4, channels-last), with one point or none take the best of masks 1 to 3 by `iou_scores` and its `object_pointers` row. With two points take pointer 0 and mask 0, unless under 98% of its pixels above −0.05 are above 0.05; then the best of masks 1 to 3.
5. If `object_score_logits` is 0 or below, set the mask to −1024.
6. Resize the mask bilinearly to 1024×1024 for `memory_encoder`, with the frame's `image_embeddings`: thresholded at 0 on the prompted frame, through a sigmoid otherwise.

**Memory bank.** `memories` holds seven slots of `memory_encoder`'s `memory`:
the prompted frame, then the latest six, newest first. `object_pointers` holds
sixteen chosen pointers in that order. Each mask is 1 for a filled slot, else
0. The object is where the chosen mask is above 0.

## Building applications

Run the package with the [Neat Model API](https://developer.sima.ai/software/develop-apps/development-workflow/model). For complete applications, start from the [Neat example apps](https://developer.sima.ai/examples).
