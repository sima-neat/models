# YOLOv8x-obb

![YOLOv8x-obb model hero](hero.png)

## Overview

YOLOv8x-obb finds objects in aerial images, such as planes and ships, and
returns a rotated box, a class, and a score for each one. Use it for aerial
surveys or port monitoring. It knows only the 15 DOTA aerial classes. Your
application cuts large images into overlapping windows first. It is the largest
of the five YOLOv8 oriented-box sizes from Ultralytics; the larger sizes are
more accurate and slower.

## Inputs

One 1024×1024×3 (height, width, channel) RGB window with values from 0 to 1.

Preprocessing:

1. Cut 1024×1024 windows every 824 pixels along each side, so neighbors overlap by 200.
2. Move a side's last window back to end at the image edge.
3. A side up to 1024 gets one window.
4. Resize each window with bilinear resampling so the longer side is 1024.
5. Center it on a 1024×1024 gray (114, 114, 114) frame.
6. Divide by 255.

## Outputs

| Output | Grid | Per cell |
|---|---|---|
| `bbox_0` to `bbox_2` | 128×128, 64×64, 32×32 (stride 8, 16, 32) | 16 logits per edge of the unrotated box: left, top, right, bottom |
| `class_logit_0` to `class_logit_2` | same | 15 class logits |
| `angle_0` to `angle_2` | same | one angle logit |

Postprocessing:

1. Each edge's distance in cells is the softmax of its 16 logits, weighted by bin index 0 to 15 and summed.
2. The angle is (sigmoid(logit) − 0.25) × π radians.
3. The cell in column c and row r has its center at (c + 0.5, r + 0.5).
4. The box is left + right wide and top + bottom high.
5. Its center is the cell center plus half of (right − left, bottom − top), rotated by the angle.
6. Multiply center, width, and height by the stride.
7. Apply a sigmoid to the class logits to get class scores.
8. Keep cell-class pairs scoring above your threshold.
9. Per class, drop every box whose probabilistic IoU with a higher-scoring box is at least 0.7.
10. Keep the 300 best.
11. Remove the padding, divide by the resize scale, and add the window's offset.
12. Per class, drop every box whose probabilistic IoU with a higher-scoring box from any window is at least 0.3.

Classes 0 to 14: plane, ship, storage-tank, baseball-diamond, tennis-court, basketball-court, ground-track-field, harbor, bridge, large-vehicle, small-vehicle, helicopter, roundabout, soccer-ball-field, swimming-pool.

## Building applications

Run the package with the [Neat Model API](https://developer.sima.ai/software/develop-apps/development-workflow/model).
For complete applications, start from the [Neat example apps](https://developer.sima.ai/examples).
