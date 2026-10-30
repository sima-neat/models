# LightGlue (SuperPoint)

![LightGlue (SuperPoint) model hero](hero.png)

## Overview

LightGlue is a feature matcher from ETH Zurich. Given the SuperPoint keypoints
and descriptors of two images, it finds which keypoints show the same point of
the scene, for uses such as relative pose estimation, image stitching and 3D
reconstruction. It runs as one component on 512 keypoints per image and
returns a soft assignment between the two images' keypoints, from which the
CPU extracts the matches. The package holds the matcher only: run SuperPoint,
which this package does not include, to find the keypoints first.

## Inputs and outputs

**Input.** Run SuperPoint on each image stretched to 640×480 and keep its best
512 keypoints. Pass `encoding0` and `encoding1`, each 1×512×128: each keypoint's
position encoding, computed on the CPU in double precision. Normalize the
keypoint's pixel coordinates as (x − 320) / 320 and (y − 240) / 320, multiply by
LightGlue's 32×2 keypoint projection (listed in the package's
`scripts/calibration.py`) to get 32 phases, repeat each phase twice, and pass the
64 cosines followed by the 64 sines. Pass `descriptors0` and `descriptors1`, each
1×512×256: SuperPoint's unit-length descriptors, in the same order. For an image
with fewer than 512 keypoints, fill the remaining rows with the encoding of
(0, 0) and zero descriptors, and ignore matches to those rows.

**Outputs.** `assignment` is 1×512×512: the probability that keypoint i of the
first image and keypoint j of the second show the same point.

1. For each keypoint i, take the keypoint j with the highest assignment in row i.
2. Keep the pair when i is also the highest in column j, and its assignment is above 0.1.

## Building applications

Run the package with the [Neat Model API](https://developer.sima.ai/software/develop-apps/development-workflow/model). For complete applications, start from the [Neat example apps](https://developer.sima.ai/examples).
