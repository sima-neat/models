# LightStereo-S

![LightStereo-S model hero](hero.png)

## Overview

LightStereo-S is the smallest LightStereo stereo-matching model from OpenStereo,
trained on the synthetic Scene Flow dataset. It takes one frame holding the left
and right images of a rectified stereo camera pair side by side and returns a
disparity map for the left image: for each pixel, how many pixels to the left
the same point appears in the right image; nearer surfaces have larger
disparity. The model runs as one component and covers disparities up to 188
pixels.

## Inputs and outputs

**Input.** One RGB frame, `stereo`, with a rectified pair side by side: the left
image in the left half, the right image in the right half, and a scene point on
the same row in both. The model reads it as 544×1920. Prepare it either way:

1. With Neat: pass the frame as an RGB image with stretch resizing and the ImageNet normalization preset.
2. Yourself, as the original evaluation does: scale each image to at most 544×960, pad it to 544×960 by repeating its top row above and its rightmost column to the right, join the two, scale RGB to 0–1, subtract the ImageNet mean (0.485, 0.456, 0.406), divide by the standard deviation (0.229, 0.224, 0.225), and pass a 544×1920×3 float tensor with Neat's preprocessing off.

**Outputs.** `disparity` holds one value per pixel of the 544×960 left image, in
pixels, from 0 to 188.

1. After Neat's preparation, resize the map to your left image and multiply each value by its width divided by 960.
2. After your own, remove the padded rows and columns and divide the values by any factor you scaled by.
3. A pixel at column x with disparity d matches column x − d of the same row in the right image.
4. For d above 0, depth equals the focal length in pixels times the baseline, divided by d.
5. Treat disparities near 0 as too far away to measure.

No Neat decoder handles this output.

## Building applications

Run the package with the [Neat Model API](https://developer.sima.ai/software/develop-apps/development-workflow/model).
For complete applications, start from the [Neat example apps](https://developer.sima.ai/examples).
