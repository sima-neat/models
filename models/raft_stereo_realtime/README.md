# RAFT-Stereo (realtime)

![RAFT-Stereo (realtime) model hero](hero.png)

## Overview

RAFT-Stereo is a stereo-matching model from Princeton University's Vision &
Learning Lab, and this is its realtime configuration, the fastest one the
authors release. It takes one frame holding the left and right images of a
rectified stereo camera pair side by side and returns a disparity map for the
left image: for each pixel, how many pixels to the left the same point appears
in the right image; nearer surfaces have larger disparity. It correlates the
two images' features and refines the disparity over 7 iterations, all in one
component.

## Inputs and outputs

**Input.** One RGB frame, `stereo`, with a rectified pair side by side: the left
image in the left half, the right image in the right half, and a scene point on
the same row in both. The model reads it as 540×1920, two 540×960 images.
Prepare it either way:

1. With Neat: pass the frame as an RGB image with stretch resizing and the ImageNet normalization preset.
2. Yourself: scale each image to 540×960, join the two, scale RGB to 0–1, subtract the ImageNet mean (0.485, 0.456, 0.406), divide by the standard deviation (0.229, 0.224, 0.225), and pass a 540×1920×3 float tensor with Neat's preprocessing off.

**Outputs.** `disparity` holds one value per pixel of the 540×960 left image,
in pixels.

1. If you scaled your images, resize the map to your left image and multiply each value by its width divided by 960.
2. A pixel at column x with disparity d matches column x − d of the same row in the right image.
3. For d above 0, depth equals the focal length in pixels times the baseline, divided by d.
4. Treat disparities near 0 as too far away to measure.

No Neat decoder handles this output.

## Building applications

Run the package with the [Neat Model API](https://developer.sima.ai/software/develop-apps/development-workflow/model).
For complete applications, start from the [Neat example apps](https://developer.sima.ai/examples).
