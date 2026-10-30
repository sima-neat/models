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
512 keypoints. Pass `keypoints0` and `keypoints1`, each 1×512×2: the keypoints'
pixel coordinates (x, y), minus (320, 240), divided by 320. Pass `descriptors0`
and `descriptors1`, each 1×512×256: SuperPoint's unit-length descriptors, in the
same order. For an image with fewer than 512 keypoints, fill the remaining rows
with zero coordinates and zero descriptors, and ignore matches to those rows.

**Outputs.** `assignment` is 1×512×512: the probability that keypoint i of the
first image and keypoint j of the second show the same point.

1. For each keypoint i, take the keypoint j with the highest assignment in row i.
2. Keep the pair when i is also the highest in column j, and its assignment is above 0.1.

## Building applications

Run the package with the [Neat Model API](https://developer.sima.ai/software/develop-apps/development-workflow/model). For complete applications, start from the [Neat example apps](https://developer.sima.ai/examples).
