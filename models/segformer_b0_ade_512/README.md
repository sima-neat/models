# SegFormer-B0 ADE20K

![SegFormer-B0 ADE20K model hero](hero.png)

## Overview

SegFormer-B0 is the smallest SegFormer, NVIDIA's semantic segmentation
transformer, fine-tuned on ADE20K at 512×512. It labels every pixel of a scene
with one of 150 classes, such as wall, sky, floor, tree, person and car. A
hierarchical transformer encoder reads the image at four scales, and a light
all-MLP head fuses them.

## Inputs and outputs

**Input.** `pixel_values`, a 512×512 RGB image: stretch the image to 512×512
with bilinear resampling (no crop or padding), scale it to 0–1, and normalize it
with the ImageNet mean (0.485, 0.456, 0.406) and standard deviation (0.229,
0.224, 0.225).

**Output.** `logits`, 150 class scores on a 128×128 grid, a quarter of the
input resolution. To get a label map, resize the logits bilinearly to the image
size and take the highest-scoring class at each pixel. The class names are
`id2label` in the upstream `config.json`.

## Builds

| Build | Weights and activations |
| --- | --- |
| `w16a16` | BF16 |
| `w8a8` | INT8, calibrated on 50 COCO 2017 images |

## Accuracy

No registry dataset holds ADE20K labels, so the compile stage reports numerical
fidelity instead: it compares each build's logits with the ONNX source on 20
COCO 2017 validation images.

## Source and license

Converted from
[nvidia/segformer-b0-finetuned-ade-512-512](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512),
pinned by revision and SHA-256. The ONNX export is checked against PyTorch.

The [NVIDIA Source Code License](https://github.com/NVlabs/SegFormer/blob/master/LICENSE)
allows use only for research or evaluation, not commercially, so the build
cannot be redistributed.
