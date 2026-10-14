# ResNet-50

![ResNet-50 model hero](hero.png)

## Overview

ResNet-50 is Microsoft's 50-layer residual network for image classification,
trained on the 1,000 classes of ImageNet-1k. It takes a 224×224 RGB image and
returns one score for each class. Your application turns the scores into
probabilities with a softmax and reports the most likely classes. The model runs
as one component.

## Inputs and outputs

**Input.** Resize the image with bicubic resampling so its shorter side is 256,
then crop the center 224×224. Scale the RGB values to 0–1, subtract the ImageNet
mean (0.485, 0.456, 0.406), and divide by the standard deviation (0.229, 0.224,
0.225). The reference scripts pass this image as a 224×224×3 (height, width,
channel) float tensor, with Neat's resize, color conversion, and normalization
turned off.

**Outputs.** `logits` holds 1,000 class scores. Apply a softmax and take the
five highest. Class index i is the i-th ImageNet-1k WordNet ID in sorted order,
which is the standard ImageNet-1k class order.

## Building applications

Run the package with the [Neat Model
API](https://developer.sima.ai/software/develop-apps/development-workflow/model).
For complete applications, start from the [Neat example
apps](https://developer.sima.ai/examples).
