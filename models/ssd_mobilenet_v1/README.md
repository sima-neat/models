# SSD-MobileNet V1

![SSD-MobileNet V1 model hero](hero.png)

## Overview

SSD-MobileNet V1 is a single-shot object detector (SSD) with a MobileNet V1
backbone, trained on COCO. It takes a 300×300 RGB image and returns raw box and
class predictions for 1,917 anchors over six grid levels. An anchor is a fixed
reference box that each prediction adjusts. Neat's SSD box decoder turns these
predictions into detections.

## Inputs and outputs

**Input.** Resize the image to 300×300 with bilinear resampling, without keeping
the aspect ratio. Scale the RGB values to -1 to 1: divide by 255, subtract 0.5,
and divide by 0.5.

**Outputs.** Twelve tensors over six levels, with grids of 19, 10, 5, 3, 2, and
1 cells and three anchors per cell at level 0 and six at the other levels:

- `ssd_loc_0` to `ssd_loc_5`: four box offsets for each anchor.
- `ssd_confidence_0` to `ssd_confidence_5`: 91 class logits for each anchor.

Neat's `Ssd` box decoder generates the anchors, decodes the boxes, scores the
classes, and removes overlapping boxes. It reads the box outputs first, then the
class outputs, each in level order.

Class index i is COCO category ID i: index 0 is the background, and 10 indexes
are unused.

## Building applications

Run the package with the [Neat Model
API](https://developer.sima.ai/software/develop-apps/development-workflow/model).
For complete applications, start from the [Neat example
apps](https://developer.sima.ai/examples).
