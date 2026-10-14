# SSD-MobileNet V3 Large

![SSD-MobileNet V3 Large model hero](hero.png)

## Overview

SSD-MobileNet V3 Large is SSDlite, a lighter single-shot detector (SSD), with a
MobileNet V3 Large backbone. TorchVision trained it on COCO. It takes a 320×320
RGB image and returns raw box and class predictions for 3,234 anchors over six
grid levels. An anchor is a fixed reference box that each prediction adjusts.
Neat's SSD box decoder turns these predictions into detections.

## Inputs and outputs

**Input.** Resize the image to 320×320 with bilinear resampling, without keeping
the aspect ratio. Scale the RGB values to 0–1, subtract the ImageNet mean
(0.485, 0.456, 0.406), and divide by the standard deviation (0.229, 0.224,
0.225).

**Outputs.** Twelve tensors over six levels, with grids of 20, 10, 5, 3, 2, and
1 cells and six anchors per cell:

- `ssd_loc_0` to `ssd_loc_5`: four box offsets for each anchor.
- `ssd_confidence_0` to `ssd_confidence_5`: 91 class logits for each anchor. Index 0 is the background.

Neat's `Ssd` box decoder generates the anchors, decodes the boxes, scores the
classes, and removes overlapping boxes. It reads the box outputs first, then the
class outputs, each in level order.

## Building applications

Run the package with the [Neat Model
API](https://developer.sima.ai/software/develop-apps/development-workflow/model).
For complete applications, start from the [Neat example
apps](https://developer.sima.ai/examples).
