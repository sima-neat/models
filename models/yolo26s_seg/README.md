# YOLO26s-seg

![YOLO26s-seg model hero](hero.png)

## Overview

YOLO26s-seg finds everyday objects in an image and returns a box, a class, a
score, and a pixel mask that traces the outline of each one. Use it to cut
objects out, measure their area, or blur them. It knows only the 80 object
classes of the COCO dataset. It is the second smallest of the five YOLO26
segmentation sizes from Ultralytics; the larger sizes are more accurate and
slower. Each object is predicted once, so your application does not need to
remove overlapping boxes.

## Inputs and outputs

### Inputs

One 640×640 RGB image with values from 0 to 1.

- With Neat: pass your frame as an image, with letterbox resizing and normalization on.
- Yourself: letterbox the image, that is, resize it with bilinear resampling so its longer side is 640 and center it on a 640×640 gray (114, 114, 114) frame. Divide by 255 and pass a 640×640×3 (height, width, channel) float32 tensor with Neat's preprocessing off.

### Outputs

Detections, each a box, a class index, a score, and a mask.

- With Neat: the `YoloV26Seg` box decoder decodes these outputs. Set `top_k` and `score_threshold` (such as 0.25).
- Yourself: decode these height, width, channel tensors.

| Tensor | Grid | Per cell |
|---|---|---|
| `bbox_0` to `bbox_2` | 80×80, 40×40, 20×20 (stride 8, 16, 32) | distances from the cell center to the left, top, right, and bottom box edges, in cells |
| `class_logit_0` to `class_logit_2` | same | 80 class logits |
| `mask_coeff_0` to `mask_coeff_2` | same | 32 mask coefficients |
| `mask_proto` | 160×160 (stride 4) | 32 mask patterns (prototypes) |

1. The cell in column c and row r has its center at (c + 0.5, r + 0.5).
2. The box corners are the center minus the left and top distances, and the center plus the right and bottom distances, times the stride.
3. Apply a sigmoid to the class logits to get class scores.
4. Keep the 300 cells with the highest class score, then the 300 best cell-class pairs among them.
5. Multiply `mask_proto` by each detection's coefficients and resize bilinearly to 640×640. Positive pixels inside the box belong to the object.
6. Remove the letterbox padding, divide by the letterbox scale, and clip to the image. Map the masks the same way.

Class index i is the i-th COCO category ID in ascending order.

## Building applications

Run the package with the [Neat Model API](https://developer.sima.ai/software/develop-apps/development-workflow/model).
For complete applications, start from the [Neat example apps](https://developer.sima.ai/examples).
