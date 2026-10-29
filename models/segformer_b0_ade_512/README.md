# SegFormer-B0 ADE20K

![SegFormer-B0 ADE20K model hero](hero.png)

## Overview

SegFormer-B0 is the smallest SegFormer, NVIDIA's semantic segmentation
transformer, fine-tuned on ADE20K at 512×512. It labels every pixel of a scene
with one of 150 classes, such as wall, sky, floor, tree, person and car. A
hierarchical transformer encoder reads the image at four scales, and a light
all-MLP head fuses them. The model runs as one component, in a BF16 (`w16a16`)
or an INT8 (`w8a8`) build.

No registry dataset holds ADE20K labels, so the compile stage compares each
build's logits with the ONNX source on COCO 2017 validation images instead of
scoring mIoU. The NVIDIA Source Code License allows research or evaluation use
only, not commercial use.

## Inputs and outputs

**Input.** Stretch the image to 512×512 with bilinear resampling, with no crop
or padding. Scale the RGB values to 0–1, subtract the ImageNet mean (0.485,
0.456, 0.406), and divide by the standard deviation (0.229, 0.224, 0.225). Neat
can do this with its stretch resize and the ImageNet normalization preset.

**Outputs.** `logits` holds 150 class scores on a 128×128 grid, a quarter of the
input resolution. Resize the logits bilinearly to the image size and take the
highest-scoring class at each pixel. Class index i is entry i of `id2label` in
the upstream `config.json`, the ADE20K order without its "other" class.

## Building applications

Run the package with the [Neat Model
API](https://developer.sima.ai/software/develop-apps/development-workflow/model).
For complete applications, start from the [Neat example
apps](https://developer.sima.ai/examples).
