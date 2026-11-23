# SuperPoint

![SuperPoint model hero](hero.png)

## Overview

SuperPoint finds distinctive points in an image, such as corners and texture
details, and returns each as a keypoint with a position, a score, and a
256-number descriptor. Match the descriptors of two images, such as consecutive
drone frames, to link the same scene points for visual odometry, mapping, and
image alignment. SuperPoint does not match them itself: compare the descriptors
in your application, or use a matcher such as SuperGlue or LightGlue.

## Inputs

One 480×640×1 (height, width, channel) grayscale image, `image`, with values from 0 to 1.

Preprocessing:

1. Resize the RGB image to 640×480 (width × height) with area interpolation, without keeping the aspect ratio.
2. Divide by 255.
3. Mix the channels as 0.2989 × red + 0.5870 × green + 0.1140 × blue.

## Outputs

Both outputs are on an 80×60 grid, one cell per 8×8 pixels.

| Output | Per cell |
|---|---|
| `logits` | 65 values: one per pixel of the cell plus a "no keypoint" value |
| `descriptors` | 256 values, not yet scaled to unit length |

Postprocessing:

[BoxDecode](https://developer.sima.ai/software/reference/boxdecode_decode_types) with decode type `SuperPoint` and profile `MagicLeapDemoV1` runs these steps for you:

1. Apply softmax over each cell's 65 values, drop the last, and place the other 64 on the cell's pixels, row by row. This gives a 640×480 score map.
2. Keep the pixels scoring at least 0.015.
3. Go through the kept pixels from the highest score down. Keep each one as a keypoint and remove every other pixel within 4 pixels of it.
4. Drop keypoints within 4 pixels of an image edge.
5. Keep the 1,000 highest-scoring keypoints.
6. Scale each cell's descriptor to unit length, interpolate it bilinearly at each keypoint, and scale the result to unit length.

## Building applications

Run the package with the [Neat Model API](https://developer.sima.ai/software/develop-apps/development-workflow/model). For complete applications, start from the [Neat example apps](https://developer.sima.ai/examples).
