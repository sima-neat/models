# SegFormer-B0 ADE20K

![SegFormer-B0 ADE20K model hero](hero.png)

## Overview

SegFormer-B0 is the smallest size of NVIDIA's SegFormer semantic segmentation
transformer, fine-tuned on ADE20K. It labels every pixel of a scene with one of
150 classes, such as wall, sky, floor, tree, person and car. It takes a 640×480
RGB image and returns class scores on a 160×120 grid, a quarter of the input
resolution. A hierarchical transformer encoder reads the image at four scales,
and a light all-MLP head fuses them. The model runs as one component.

## Inputs and outputs

**Input.** Resize the image bilinearly to fit within 640×480 while preserving
its aspect ratio. Round the resized dimensions up and center the image with
RGB (114, 114, 114) padding. All sizes here are width × height. Scale the RGB
values to 0–1, subtract the ImageNet mean (0.485, 0.456, 0.406), and divide by
the standard deviation (0.229, 0.224, 0.225). Neat
can do this with its letterbox resize and the ImageNet normalization preset.

**Outputs.** `logits` holds 150 class scores for each cell of a 160×120 grid.

1. Map the original image's pixel centers through the resized image and padding into the quarter-resolution logits, sampling bilinearly with `align_corners=False` to exclude padding.
2. At each pixel, take the class with the highest score.

Class index i is class i of the standard 150-class ADE20K scene-parsing order,
from 0 wall, 1 building, 2 sky and 3 floor to 149 flag, which is the `id2label`
order of the upstream `config.json`. No Neat decoder handles this output; run
the steps on the CPU.

## Building applications

Run the package with the [Neat Model
API](https://developer.sima.ai/software/develop-apps/development-workflow/model).
For complete applications, start from the [Neat example
apps](https://developer.sima.ai/examples).
