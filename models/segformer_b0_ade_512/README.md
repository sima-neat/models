# SegFormer-B0

![SegFormer-B0 model hero](hero.png)

## Overview

SegFormer-B0 labels every pixel of a scene with one of 150 classes, such as
wall, sky, floor, tree, person, and car. Use it to find free floor for a robot,
to separate road, sky, and buildings in outdoor scenes, or to replace a
background. It returns one class per pixel and does not tell objects of the
same class apart. It is the smallest size of NVIDIA's SegFormer, fine-tuned on
ADE20K. It predicts at a quarter of the input resolution, and the result
is resized to the image.

## Inputs

One 512×512 RGB image, normalized with the ImageNet mean and standard deviation.

- With Neat: pass your frame as an image, with stretch resizing and the ImageNet normalization preset.
- Yourself: stretch to 512×512 with bilinear resampling, with no crop or padding. Scale to 0–1, subtract (0.485, 0.456, 0.406), and divide by (0.229, 0.224, 0.225). Pass a 512×512×3 (height, width, channel) float tensor.

## Outputs

One class index for each pixel.

- With Neat: the semantic-segmentation decode returns one class index per pixel. Set `semantic_segmentation.enable` and the output width and height in the model options.
- Yourself: decode `logits`, which holds 150 class scores for each cell of a 128×128 grid.

1. Resize `logits` bilinearly to the image's height and width.
2. At each pixel, take the class with the highest score.

Class index i is class i of the standard 150-class ADE20K scene-parsing order,
from 0 wall, 1 building, 2 sky, and 3 floor to 149 flag. This is the `id2label`
order in the original model's `config.json`.

## Building applications

Run the package with the [Neat Model
API](https://developer.sima.ai/software/develop-apps/development-workflow/model).
For complete applications, start from the [Neat example
apps](https://developer.sima.ai/examples).
