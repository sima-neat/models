# Inception v3

![Inception v3 model hero](hero.png)

## Overview

Inception v3 is a convolutional network for image classification whose Inception modules run 1×1, 3×3 and factorized larger convolutions side by side, trained on the 1,000 classes of ImageNet-1k. The package uses
torchvision's ImageNet weights (IMAGENET1K_V1). It takes a 299×299 RGB image and
returns one score for each class. Your application turns the scores into
probabilities with a softmax and reports the most likely classes. The model runs
as one component.

## Inputs and outputs

**Input.** Resize the image with bicubic resampling so its shorter side is 341,
then crop the center 299×299. Scale the RGB values to 0–1, subtract the mean
(0.5, 0.5, 0.5), and divide by the standard deviation (0.5, 0.5, 0.5). The
accuracy script passes this image as a 299×299×3 (height, width,
channel) float tensor, with Neat's resize, color conversion, and normalization
turned off.

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
