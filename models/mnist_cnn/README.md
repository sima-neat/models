# MNIST CNN

![MNIST CNN model hero](hero.png)

## Overview

MNIST CNN recognizes a handwritten digit. It is the ONNX Model Zoo's small
convolutional network, trained in CNTK on the MNIST dataset of handwritten
digits. It takes one 28×28 grayscale image of a single digit, white on a black
background, and returns one score for each of the ten digits, 0 to 9. Your
application takes the highest score as the digit, or applies a softmax for
probabilities. The model runs as one component.

## Inputs

One input, `Input3`: a 28×28×1 (height, width, channel) grayscale digit image with values from 0 to 1.

Preprocessing:

1. Crop the digit.
2. Convert it to grayscale, with the digit white on a black background as in MNIST.
3. Resize it to 28×28.
4. Divide the pixel values by 255; the model applies no other normalization.

## Outputs

One output, `Plus214_Output_0`, with 10 digit scores before a softmax.

Postprocessing:

1. Take the index of the highest score as the digit.
2. Apply a softmax to the scores when the application needs a confidence.

Score index i is the digit i, from 0 to 9.

## Building applications

Run the package with the [Neat Model
API](https://developer.sima.ai/software/develop-apps/development-workflow/model).
For complete applications, start from the [Neat example
apps](https://developer.sima.ai/examples).
