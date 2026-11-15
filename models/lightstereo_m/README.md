# LightStereo-M

![LightStereo-M model hero](hero.png)

## Overview

LightStereo-M uses the two images of a stereo camera to estimate how far away
each pixel is. Use it for obstacle avoidance, 3D measurement, and navigation. It
returns a disparity map for the left image: for each pixel, how many pixels to
the left the same point appears in the right image. Nearer surfaces have larger
disparity, and your application turns disparity into distance with the camera's
focal length and baseline. The pair must be rectified and passed side by side in
one frame. It covers disparities up to 188 pixels and was trained on the
synthetic Scene Flow dataset.

## Inputs

One 540×1920×3 (height, width, channel) RGB frame, `stereo`, normalized with the ImageNet mean and standard deviation. It holds a rectified pair side by side, with each scene point on the same row in both images.

Preprocessing:

1. Scale each image to 960×540.
2. Join the two side by side: the left image in the left half, the right image in the right half.
3. Divide by 255.
4. Subtract (0.485, 0.456, 0.406) and divide by (0.229, 0.224, 0.225).

## Outputs

`disparity` holds one value per pixel of the 960×540 left image, in pixels, from 0 to 188.

Postprocessing:

1. If you scaled your images, resize the map to your left image and multiply each value by its width divided by 960.
2. A pixel at column x with disparity d matches column x − d of the same row in the right image.
3. For d above 0, depth = focal length in pixels × baseline / d.
4. Treat disparities near 0 as too far away to measure.

## Building applications

Run the package with the [Neat Model API](https://developer.sima.ai/software/develop-apps/development-workflow/model).
For complete applications, start from the [Neat example apps](https://developer.sima.ai/examples).
