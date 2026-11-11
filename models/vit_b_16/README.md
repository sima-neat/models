# ViT-B/16

![ViT-B/16 model hero](hero.png)

## Overview

ViT-B/16 is Google's Vision Transformer for image classification. It cuts a
224×224 RGB image into 16×16 patches and reads them as a sequence with 12
transformer layers. Google pretrained it on ImageNet-21k and fine-tuned it on
the 1,000 classes of ImageNet-1k. It returns one score for each class. Your
application turns the scores into probabilities with a softmax and reports the
most likely classes. The model runs as one component.

## Inputs and outputs

**Input.** Resize the image straight to 224×224 with bilinear resampling,
without keeping its aspect ratio or cropping. Scale the RGB values to 0–1,
subtract 0.5, and divide by 0.5, so each value lies between −1 and 1. The accuracy script passes
this image as a 224×224×3 (height, width, channel) float tensor, with Neat's
resize, color conversion, and normalization turned off.

**Outputs.** `logits` holds 1,000 class scores.

1. Apply a softmax to `logits` to get class probabilities.
2. Take the five highest probabilities as the top-5 classes.

Class index i is the i-th ImageNet-1k WordNet ID in sorted order, which is the
standard ImageNet-1k class order.

## Building applications

Run the package with the [Neat Model
API](https://developer.sima.ai/software/develop-apps/development-workflow/model).
For complete applications, start from the [Neat example
apps](https://developer.sima.ai/examples).
