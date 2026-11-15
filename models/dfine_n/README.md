# D-FINE Nano

![D-FINE Nano model hero](hero.png)

## Overview

D-FINE Nano finds everyday objects in an image, such as people, vehicles, and
animals, and returns a box, a class, and a score for each one. Use it to count,
track, or monitor objects in camera streams. It knows the 80 COCO classes. The
model runs in two parts, and your application picks the 300 best candidates on
the CPU between them. Each object is predicted once, so there are no overlapping
boxes to remove.

## Inputs

One 640×640×3 (height, width, channel) RGB image with values from 0 to 1.

Preprocessing:

1. Stretch to 640×640 with bilinear resampling, without padding or cropping.
2. Divide by 255.

## Between the components

| Backbone output | Size |
|---|---|
| `attention_memory` | 2,000 rows of 128 values |
| `query_content` | 2,000 rows of 128 values |
| `proposal_scores` | 2,000 values |
| `reference_logits` | 2,000 rows of 4 values |

1. Take the 300 highest `proposal_scores`. Break ties by the lower index.
2. Gather those rows of `query_content` and `reference_logits` in that order, without changing their values.
3. Run the decoder on both and the whole `attention_memory`.

## Outputs

| Output | Size | Holds |
|---|---|---|
| `pred_boxes` | 300 rows of 4 values | boxes as (cx, cy, w, h), fractions of the image |
| `pred_logits` | 300 rows of 80 values | class logits per box |

Postprocessing:

1. Apply a sigmoid to `pred_logits` to get class scores.
2. Rank all 300×80 box-class pairs by score and keep the 300 best. One box can appear with several classes.
3. The box corners are (cx - w/2, cy - h/2) and (cx + w/2, cy + h/2).
4. Multiply x by the image width and y by the image height, clip to the image, and drop boxes left empty.

Class index i is the i-th COCO category ID in ascending order.

## Building applications

Run the package with the [Neat Model
API](https://developer.sima.ai/software/develop-apps/development-workflow/model).
For complete applications, start from the [Neat example
apps](https://developer.sima.ai/examples).
