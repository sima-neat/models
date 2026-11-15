# YOLO26n-pose

![YOLO26n-pose model hero](hero.png)

## Overview

YOLO26n-pose finds each person in an image and adds 17 body keypoints to the
box, such as the eyes, shoulders, and knees. Use it for activity analysis,
fitness coaching, or fall detection. It returns a box, a score, and the
keypoints for each person, each keypoint with a position and a visibility score.
It detects people only. Each person is predicted once, so your application does
not need to remove overlapping boxes.

## Inputs

One 640×640×3 (height, width, channel) RGB image with values from 0 to 1.

Preprocessing:

1. Resize with bilinear resampling so the longer side is 640.
2. Center it on a 640×640 gray (114, 114, 114) frame.
3. Divide by 255.

## Outputs

| Output | Grid | Per cell |
|---|---|---|
| `bbox_0` to `bbox_2` | 80×80, 40×40, 20×20 (stride 8, 16, 32) | distances from the cell center to the left, top, right, and bottom box edges, in cells |
| `class_logit_0` to `class_logit_2` | same | one person logit |
| `keypoint_0` to `keypoint_2` | same | 17 keypoints in COCO order, each an x offset, a y offset, and a visibility logit |

Postprocessing:

1. The cell in column c and row r has its center at (c + 0.5, r + 0.5).
2. The box corners are the center minus the left and top distances, and the center plus the right and bottom distances, times the stride.
3. Apply a sigmoid to the person logits to get person scores.
4. Keep the 300 cells with the highest person score.
5. Each keypoint is at the cell center plus its x and y offsets, times the stride.
6. Apply a sigmoid to each visibility logit to get the visibility score.
7. Remove the padding, divide by the resize scale, and clip to the image. Map the keypoints the same way.

## Building applications

Run the package with the [Neat Model API](https://developer.sima.ai/software/develop-apps/development-workflow/model).
For complete applications, start from the [Neat example apps](https://developer.sima.ai/examples).
