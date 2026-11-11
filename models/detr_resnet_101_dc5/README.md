# DETR ResNet-101 DC5

![DETR ResNet-101 DC5 model hero](hero.png)

## Overview

DETR ResNet-101 DC5 is Facebook AI Research's original DETR transformer object detector
with a ResNet-101 backbone whose last stage is dilated for finer features, trained on COCO. It takes one RGB image and returns 100
detections, each a box with scores for 91 COCO category slots and a "no
object" class. It runs as one component that holds the backbone, the
transformer encoder and decoder, and both prediction heads. DETR predicts each
object once, so the application does not need to remove overlapping boxes.

## Inputs and outputs

**Input.** Resize the image with bilinear resampling to exactly 1280×720
(width × height), without padding, so a 720p camera frame passes unchanged.
Padding lowers the model's accuracy, because the model reads it as part of the
scene. Scale the RGB values to 0–1, subtract the ImageNet mean (0.485, 0.456,
0.406), and divide by the standard deviation (0.229, 0.224, 0.225). Pass this
image as a 720×1280×3 (height, width, channel) float tensor, or let Neat
stretch and normalize the frame with its ImageNet preset.

**Outputs.** The package returns `logits`, 92 class logits for each of 100
queries, and `pred_boxes`, 100 boxes as normalized (cx, cy, w, h).

1. Apply a softmax over the 92 logits of each query.
2. Drop the last class, which means no object, and take the best remaining class and its probability as the score.
3. Keep the queries whose score passes your threshold.
4. Convert each box to corners: (cx - w/2, cy - h/2) and (cx + w/2, cy + h/2).
5. Multiply x by the image width and y by the image height to reach image pixels.

Class index i is COCO category ID i. Eleven of the first 91 indexes are unused.
No Neat box decoder reads these outputs, so decode them on the CPU.

## Building applications

Run the package with the [Neat Model
API](https://developer.sima.ai/software/develop-apps/development-workflow/model).
For complete applications, start from the [Neat example
apps](https://developer.sima.ai/examples).
