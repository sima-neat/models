# Depth Anything 3 Small 504×280

![Depth Anything 3 Small 504×280 model hero](hero.png)

## Overview

Depth Anything 3 Small estimates relative depth from one RGB image. It returns a dense depth map and a matching confidence map, allowing applications to distinguish nearby surfaces from distant ones and identify uncertain regions. This single-view entry processes one image at a time at 280×504 pixels, suitable for a 16:9 camera view. It uses one compiled model and needs no camera pose, focal length, second view, or text prompt. Depth values do not represent meters.

## Inputs and outputs

Prepare `image` as RGB using these steps:

1. Resize while preserving aspect ratio until the image covers 280×504 pixels; use area interpolation when shrinking and bicubic interpolation when enlarging.
2. Center-crop to 280×504 pixels.
3. Divide pixel values by 255, subtract RGB means `[0.485, 0.456, 0.406]`, and divide by `[0.229, 0.224, 0.225]`.
4. Supply the preprocessed float32 tensor with explicit HWC layout using Neat's tensor interface.

`depth` and `depth_conf` each contain one value per image pixel. Larger `depth` means farther away, with an unknown scale and shift in inverse depth. `depth_conf` is a confidence score; larger values mean greater confidence. Read the two outputs in that order and reshape each to 280×504 for display. No detection decoder is needed. These entries omit the upstream camera-pose and auxiliary ray outputs.

## Building applications

Run the package with the [Neat Model API](https://developer.sima.ai/software/develop-apps/development-workflow/model). For complete applications, start from the [Neat example apps](https://developer.sima.ai/examples).
