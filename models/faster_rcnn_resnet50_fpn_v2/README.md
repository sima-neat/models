# Faster R-CNN ResNet-50 FPN v2

![Faster R-CNN ResNet-50 FPN v2 model hero](hero.png)

## Overview

Faster R-CNN ResNet-50 FPN v2 finds everyday objects in an image and returns a
box, a class, and a score for each, up to 100 per image. Use it to count,
locate, or inventory objects in photos and camera frames, where small and
crowded objects matter more than speed. It is trained on the COCO classes. The
model runs in two parts, and your application proposes and pools 1,000
candidate regions on the CPU between them.

## Inputs

One 1344×1344×3 (height, width, channel) RGB image, normalized with the ImageNet mean and standard deviation, with the resized picture in its top-left corner.

Preprocessing:

1. Divide by 255.
2. Subtract (0.485, 0.456, 0.406) and divide by (0.229, 0.224, 0.225), per channel.
3. Resize bilinearly, keeping the aspect ratio, to a short side of 800 with the long side at most 1333.
4. Place it top-left in a 1344×1344×3 tensor of zeros.

## Between the components

The backbone returns `backbone`: one row per cell of maps P2 (336×336) to P6 (21×21), in order, holding 256 map values, three objectness scores (multiply by 16), and four box offsets per anchor (multiply by 2).

1. Per level, move the 1,000 best anchors by their offsets, clip to the resized image, and drop boxes overlapping a better one above 0.7 intersection over union (IoU).
2. Keep the 1,000 best boxes across levels.
3. Pool each box into 7×7 cells from the map its size selects.
4. Place region i at row i ÷ 40, column i mod 40 of a 25×40 grid of 8×8 cells, zero elsewhere: the 200×320×256 input `rois`.

## Outputs

| Output | Size | Holds |
|---|---|---|
| `class_logits` | 25×40 grid of 91 values | class logits per region |
| `box_regression` | 25×40 grid of 364 values | four box offsets per class and region |

Postprocessing:

1. Apply a softmax to each region's logits and keep region-class pairs above 0.05.
2. Divide the offsets by 10, 10, 5 and 5, move the region's center by the first two, and scale its size by the exponent of the last two.
3. Per class, drop overlaps above 0.5 IoU; keep the 100 best overall.
4. Divide by the resize scale for original-image coordinates.

Class index i is COCO category ID i: index 0 is the background, and 10 indexes are unused.

## Building applications

Run the package with the [Neat Model
API](https://developer.sima.ai/software/develop-apps/development-workflow/model).
For complete applications, start from the [Neat example
apps](https://developer.sima.ai/examples).
