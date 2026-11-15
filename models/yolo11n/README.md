# YOLO11n-det

![YOLO11n-det model hero](hero.png)

## Overview

YOLO11n-det finds everyday objects in an image, such as people, vehicles, and
animals, and returns a box, a class, and a score for each one. Use it to count,
track, or monitor objects in live camera streams. It knows only the 80 object
classes of the COCO dataset. It is the smallest of the five YOLO11 detector
sizes from Ultralytics; the larger sizes are more accurate and slower. It
predicts several overlapping boxes for one object, and decoding removes the
duplicates.

## Inputs

One 640×640 RGB image with values from 0 to 1.

- With Neat: pass your frame as an image, with letterbox resizing and normalization on.
- Yourself: letterbox the image, that is, resize it with bilinear resampling so its longer side is 640 and center it on a 640×640 gray (114, 114, 114) frame. Divide by 255 and pass a 640×640×3 (height, width, channel) float32 tensor with Neat's preprocessing off.

## Outputs

Detections, each a box, a class index, and a score.

- With Neat: the `YoloV8` box decoder returns the detections. Set `top_k`, `score_threshold` (such as 0.25), and `nms_iou_threshold` (such as 0.7).
- Yourself: decode these height, width, channel tensors.

| Tensor | Grid | Per cell |
|---|---|---|
| `bbox_0` to `bbox_2` | 80×80, 40×40, 20×20 (stride 8, 16, 32) | 64 logits: 16 bins for each of the left, top, right, and bottom edge distances |
| `class_logit_0` to `class_logit_2` | same | 80 class logits |

1. The cell in column c and row r has its center at (c + 0.5, r + 0.5).
2. Apply a softmax over each edge's 16 bins. The edge distance in cells is the probability-weighted mean bin index, from 0 to 15.
3. The box corners are the center minus the left and top distances, and the center plus the right and bottom distances, times the stride.
4. Apply a sigmoid to the class logits to get class scores, and keep the cell-class pairs above your score threshold.
5. Among boxes of the same class, drop every box that overlaps a higher-scoring kept box by more than 0.7 (non-maximum suppression).
6. Remove the letterbox padding, divide by the letterbox scale, and clip to the image.

Class index i is the i-th COCO category ID in ascending order.

## Building applications

Run the package with the [Neat Model API](https://developer.sima.ai/software/develop-apps/development-workflow/model).
For complete applications, start from the [Neat example apps](https://developer.sima.ai/examples).
