# DETR ResNet-50

![DETR ResNet-50 model hero](hero.png)

## Overview

DETR ResNet-50 finds everyday objects in an image and returns a box, a class,
and a score for each. Use it to count, track, or monitor objects in camera
streams. It is trained on the COCO classes and returns at most 100 detections
per image. It is the smaller of the two original DETR detectors, with a
ResNet-50 backbone. Each object is predicted once, so there are no overlapping
boxes to remove.

## Inputs

One 640×640×3 (height, width, channel) RGB image, normalized with the ImageNet mean and standard deviation.

Preprocessing:

1. Stretch to 640×640 with bilinear resampling, without padding or cropping.
2. Divide by 255.
3. Subtract (0.485, 0.456, 0.406) and divide by (0.229, 0.224, 0.225), per channel.

## Outputs

| Output | Size | Holds |
|---|---|---|
| `logits` | 100 rows of 92 values | class logits per query |
| `box_logits` | 100 rows of 4 values | box logits per query |

Postprocessing:

[BoxDecode](https://developer.sima.ai/software/reference/boxdecode_decode_types) with decode type `Detr` runs these steps for you:

1. Apply a softmax to each query's 92 logits.
2. Drop the last class, which means no object. The best remaining class and its probability are the class and the score.
3. Keep the queries whose score passes your threshold.
4. Apply a sigmoid to the 4 box logits to get the box as (cx, cy, w, h), fractions of the image.
5. The box corners are (cx - w/2, cy - h/2) and (cx + w/2, cy + h/2).
6. Multiply x by the image width and y by the image height.

Class index i is COCO category ID i. Eleven of the first 91 indexes are unused.

## Building applications

Run the package with the [Neat Model
API](https://developer.sima.ai/software/develop-apps/development-workflow/model).
For complete applications, start from the [Neat example
apps](https://developer.sima.ai/examples).
