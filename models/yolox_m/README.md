# YOLOX-m

![YOLOX-m model hero](hero.png)

## Overview

YOLOX-m finds everyday objects in an image, such as people, vehicles, and
animals, and returns a box, a class, and a score for each one. Use it to count,
track, or monitor objects in live camera streams. It is trained on the 80 COCO
classes. It expects blue, green, red color order and unscaled pixel values.

## Inputs

One 640×640×3 (height, width, channel) BGR image with values from 0 to 255.

Preprocessing:

1. Resize with bilinear resampling so the longer side is 640.
2. Center it on a 640×640 gray (114, 114, 114) frame.
3. Keep the blue, green, red channel order and the 0 to 255 values.

## Outputs

| Output | Grid | Per cell |
|---|---|---|
| `bbox_0` to `bbox_2` | 80×80, 40×40, 20×20 (stride 8, 16, 32) | dx, dy, log width, and log height |
| `objectness_logit_0` to `objectness_logit_2` | same | one objectness logit |
| `class_logit_0` to `class_logit_2` | same | 80 class logits |

Postprocessing:

[BoxDecode](https://developer.sima.ai/software/reference/boxdecode_decode_types) with decode type `YoloX` and option `Split3Interleaved` runs these steps for you:

1. For column c, row r, and stride s, the box center is ((dx + c) × s, (dy + r) × s).
2. The width and height are exp(log width) × s and exp(log height) × s.
3. The box corners are the center minus and plus half the width and height.
4. Apply a sigmoid to all logits. The score is the objectness times the highest class probability.
5. Keep the cells whose score passes your threshold.
6. Within each class, remove each box that overlaps a higher-scoring box by more than 0.65.
7. Remove the padding, divide by the resize scale, and clip to the image.

Class index i is the i-th COCO category ID in ascending order.

## Building applications

Run the package with the [Neat Model API](https://developer.sima.ai/software/develop-apps/development-workflow/model). For complete applications, start from the [Neat example apps](https://developer.sima.ai/examples).
