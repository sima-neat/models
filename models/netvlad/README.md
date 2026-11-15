# NetVLAD

![NetVLAD model hero](hero.png)

## Overview

NetVLAD turns an image of a place into one descriptor, a list of 4,096 numbers,
so that two views of the same place get similar descriptors. Use it for visual
place recognition and loop closure, where a robot or vehicle notices that it has
returned to a place it mapped before. Your application compares the descriptor
of each new frame with those of stored keyframes. The closest ones are candidate
revisits. It confirms each one by matching local features between the two images
before it corrects its map. The weights come from Patch-NetVLAD and were trained
on Pittsburgh street-view images.

## Inputs

One 480×640×3 (height, width, channel) RGB image, normalized with the ImageNet mean and standard deviation.

Preprocessing:

1. Resize to 640 wide by 480 high with an antialiased bilinear filter, without cropping or padding.
2. Divide by 255.
3. Subtract the mean (0.485, 0.456, 0.406).
4. Divide by the standard deviation (0.229, 0.224, 0.225).

## Outputs

| Output | Size | Holds |
|---|---|---|
| `descriptor` | 4,096 values | the place descriptor, with a length (L2 norm) of 1 |

Postprocessing:

1. To compare two images, take the dot product of their descriptors, which equals their cosine similarity. A higher value means a more similar place.

## Building applications

Run the package with the [Neat Model
API](https://developer.sima.ai/software/develop-apps/development-workflow/model).
For complete applications, start from the [Neat example
apps](https://developer.sima.ai/examples).
