# SuperPoint

![SuperPoint model hero](hero.png)

## Overview

SuperPoint finds distinctive points in an image, such as corners and texture
details, and describes each with a 256-number descriptor. Matching descriptors
between two images, such as consecutive drone frames, links the same scene
points for visual odometry, mapping, and image alignment. SuperGlue can match
its keypoints. It takes one grayscale image and returns a keypoint score map and
a descriptor map, which your application or Neat turns into keypoints.

## Inputs and outputs

**Input.** `image`: one 640 × 480 grayscale image scaled to [0, 1]. Resize the
image, then mix red, green, and blue as 0.2989, 0.5870, and 0.1140. Neat's
RGB-to-grayscale preprocessing keeps only the red channel instead, so for the
trained colour mix, convert to grayscale before passing the image to Neat.

**Outputs.** Both are on an 80 × 60 grid, one cell per 8 × 8 pixels:

- `logits`: 65 values per cell, one for each pixel of the cell plus a "no keypoint" value.
- `descriptors`: 256 values per cell, not yet scaled to unit length.

To get keypoints:

1. Apply softmax over the 65 values of each cell, drop the last, and place the other 64 on the cell's pixels, row by row.
2. Keep a pixel only if it has the highest score within 4 pixels after suppression, as the transformers SuperPoint does.
3. Keep pixels scoring above 0.005, at least 4 pixels from every edge, and the best ones if you need fewer.
4. Scale each cell's descriptor to unit length, interpolate it at each keypoint, and scale the result to unit length.

Neat's SuperPoint decoder runs these steps with the `LightGlueV1` profile and a
score threshold of 0.005. Its `MagicLeapDemoV1` profile runs Magic Leap's demo
decode instead, which the model page's results use: it keeps pixels scoring at
least 0.015 before suppression and returns the best 600.

## Building applications

Run the package with the [Neat Model API](https://developer.sima.ai/software/develop-apps/development-workflow/model). For complete applications, start from the [Neat example apps](https://developer.sima.ai/examples).
