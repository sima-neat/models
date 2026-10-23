# NetVLAD

![NetVLAD model hero](hero.png)

## Overview

NetVLAD turns a whole image into one compact place descriptor for loop
closure and visual place recognition. These weights come from Patch-NetVLAD: a
VGG16 backbone, NetVLAD pooling over 64 learned clusters, and a whitening
projection, trained on Pittsburgh street-view images. It takes a 480×640 RGB
image and returns one 4,096-value descriptor of unit length. Your application
compares the descriptor of each new frame with those of stored keyframes; the
closest ones are candidate revisits, which it confirms with geometric
verification before updating its pose graph. The model runs as one component.

## Inputs and outputs

**Input.** Resize the image to 640 wide by 480 high with bilinear resampling,
without cropping or padding. Scale the RGB values to 0–1, subtract the ImageNet
mean (0.485, 0.456, 0.406), and divide by the standard deviation (0.229, 0.224,
0.225). Pass the result as a 480×640×3 (height, width, channel) float tensor.

**Outputs.** `descriptor` holds 4,096 float values with a length (L2 norm) of 1.
The whole computation, including the cluster soft-assignment, residual
aggregation, normalizations and projection, runs inside the model.

1. Store the descriptor of each keyframe in a nearest-neighbor index.
2. For a new frame, take the dot product with each stored descriptor; for unit vectors it equals cosine similarity, and a larger value means a more similar place.
3. Skip the most recent keyframes, which always look alike, and keep the best matches above a similarity threshold tuned on your own route.
4. Confirm each candidate with local feature matching and geometric verification before adding a loop-closure constraint.

The output has no classes, and Neat has no decoder for it; the application
compares descriptors itself.

## Building applications

Run the package with the [Neat Model API](https://developer.sima.ai/software/develop-apps/development-workflow/model).
For complete applications, start from the [Neat example apps](https://developer.sima.ai/examples).
