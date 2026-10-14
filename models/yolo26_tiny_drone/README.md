# YOLO26n-P2 Tiny Drone

![YOLO26n-P2 Tiny Drone model hero](hero.png)

## Overview

YOLO26n-P2 Tiny Drone is a one-class detector for very small drones in
high-resolution frames. It is based on the Nano size of Ultralytics YOLO26, with
an extra output level at stride 4 (P2) for small objects. It takes a 1600×1600
RGB image and returns raw box and score outputs at four scales. Your application
decodes them, keeps the confident boxes, removes overlapping ones, and can pass
the result to a tracker.

## Inputs and outputs

**Input.** Letterbox the image to 1600×1600: resize it with bilinear resampling
so its longer side is 1600, center it, and fill the rest with gray (114, 114,
114). Scale the RGB values to 0–1.

**Outputs.** Eight tensors at strides 4, 8, 16, and 32 (grids of 400, 200, 100,
and 50 cells):

- `bbox_0` to `bbox_3`: the distances from the cell center to the box's left, top, right, and bottom edges, in cells.
- `class_logit_0` to `class_logit_3`: one drone logit for each cell.

1. Apply a sigmoid to the logits and keep the cells that score above 0.05.
2. The cell in column c and row r has its center at (c + 0.5, r + 0.5).
3. The box corners are the center minus the left and top distances, and the center plus the right and bottom distances, times the stride.
4. Remove overlapping boxes (non-maximum suppression) whose intersection over union is above 0.6.
5. Remove the letterbox padding, divide by the letterbox scale, and clip to the image.

Neat's `YoloV26` box decoder can also decode these outputs.

## Building applications

Run the package with the [Neat Model
API](https://developer.sima.ai/software/develop-apps/development-workflow/model).
For complete applications, start from the [Neat example
apps](https://developer.sima.ai/examples).
