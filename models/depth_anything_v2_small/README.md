# Depth Anything V2 Small

![Depth Anything V2 Small model hero](hero.png)

## Overview

Depth Anything V2 Small estimates how far away every pixel of an image is, from
a single camera and without a depth sensor. Use it for obstacle avoidance, for
separating foreground from background, or for 3D effects. It returns a depth
map where larger values are closer. The depth is relative: the values have an
unknown scale and offset, so your application aligns them to real distances
when it needs them. It is the Small size of the Depth Anything V2 family.

## Inputs and outputs

**Input.** One 518×686 (height × width) RGB image, normalized with the ImageNet mean and standard deviation.

- With Neat: pass your frame as an image, with stretch resizing and the ImageNet normalization preset.
- Yourself: scale RGB to 0–1, resize to 518×686 with bicubic resampling, subtract (0.485, 0.456, 0.406), divide by (0.229, 0.224, 0.225), and pass a 518×686×3 (height, width, channel) float tensor with Neat's preprocessing off.

The input size fits a 640×480 frame. A frame of another shape is stretched.

**Outputs.** One map of relative inverse depth, where larger is closer. Neat has no decoder for it. `predicted_depth` holds 518×686 values.

1. Resize `predicted_depth` to your image's resolution with bilinear interpolation, keeping the corner pixels aligned.
2. For a relative depth view, use the values directly.
3. For metric depth, fit a scale s and an offset t so that s × value + t matches 1 / distance at points of known distance, such as from a depth sensor.
4. Take each pixel's distance as 1 / (s × value + t), capped at the farthest distance your scene can contain.

## Building applications

Run the package with the [Neat Model API](https://developer.sima.ai/software/develop-apps/development-workflow/model). For complete applications, start from the [Neat example apps](https://developer.sima.ai/examples).
