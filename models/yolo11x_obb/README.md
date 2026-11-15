# YOLO11x-obb

![YOLO11x-obb model hero](hero.png)

## Overview

YOLO11x-obb finds objects in aerial images, such as planes and ships, and
returns a rotated box, a class, and a score for each one. Use it for aerial
surveys or port monitoring. It knows only the 15 DOTA aerial classes. Your
application cuts large images into overlapping windows first. It is the largest
of the five YOLO11 oriented-box sizes from Ultralytics; the larger sizes are
more accurate and slower.

## Inputs

1024×1024 RGB windows of the image that overlap by 200 pixels, with values from 0 to 1.

- With Neat: pass each window as an image, with letterbox resizing and normalization on.
- Yourself: letterbox each window to 1024×1024 with bilinear resampling and centered gray (114, 114, 114) padding, divide by 255, and pass a 1024×1024×3 (height, width, channel) float32 tensor with Neat's preprocessing off.

## Outputs

Detections, each a rotated box, a class index, and a score.

- With Neat: the `YoloV8Obb` box decoder runs steps 1 to 5. Set `top_k`, `score_threshold`, and `nms_iou_threshold`.
- Yourself: decode these height, width, channel tensors.

| Tensor | Grid | Per cell |
|---|---|---|
| `bbox_0` to `bbox_2` | 128×128, 64×64, 32×32 (stride 8, 16, 32) | 64 logits: 16 bins per side (left, top, right, bottom) |
| `class_logit_0` to `class_logit_2` | same | 15 class logits |
| `angle_0` to `angle_2` | same | one angle logit |

1. The cell in column c and row r has its center at (c + 0.5, r + 0.5).
2. Each side's distance in cells is the mean bin index, weighted by a softmax over its 16 bins.
3. Apply a sigmoid to the class logits. The angle in radians is (sigmoid(angle logit) − 0.25) × π.
4. The box is left plus right wide and top plus bottom high. Its center is the cell center plus half of (right − left, bottom − top), rotated by the angle. Multiply by the stride.
5. Keep the scores above your threshold. Per class, drop every box whose probabilistic IoU with a higher-scoring box is 0.7 or more.
6. Remove the letterbox padding and scale, and add the window's offset.
7. Per class, drop every box whose probabilistic IoU with a higher-scoring box from any window is 0.3 or more.

Classes 0 to 14: plane, ship, storage-tank, baseball-diamond, tennis-court, basketball-court, ground-track-field, harbor, bridge, large-vehicle, small-vehicle, helicopter, roundabout, soccer-ball-field, swimming-pool.

## Building applications

Run the package with the [Neat Model API](https://developer.sima.ai/software/develop-apps/development-workflow/model).
For complete applications, start from the [Neat example apps](https://developer.sima.ai/examples).
