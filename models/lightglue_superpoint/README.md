# LightGlue

![LightGlue model hero](hero.png)

## Overview

LightGlue finds which keypoints of two images show the same point of the scene.
Use it to estimate how a camera moved between two images, to stitch images, and
for 3D reconstruction. It returns, for every keypoint pair, the probability that
the two match, and your application picks the matches on the CPU. It does not
detect keypoints: run SuperPoint on each image first. It takes exactly 512
keypoints per image. It is a feature matcher from ETH Zurich.

## Inputs

Four float tensors: a position encoding and the descriptors of 512 SuperPoint keypoints for each of two images (0 and 1).

| Input | Size | Holds |
|---|---|---|
| `encoding0`, `encoding1` | 1×1×512×128 | each keypoint's position encoding |
| `descriptors0`, `descriptors1` | 1×1×512×256 | unit-length SuperPoint descriptors, in the same keypoint order |

Preprocessing:

1. Resize each image to 640×480 without keeping the aspect ratio, and run SuperPoint on it.
2. Decode its keypoints with suppression within 4 pixels, a score threshold of 0.0005, and no keypoints within 4 pixels of an edge.
3. Keep the 512 highest-scoring keypoints, best first. Skip a pair when an image yields fewer than 512.
4. Normalize each position as ((x − 320) / 320, (y − 240) / 320).
5. Multiply it by LightGlue's 32×2 keypoint projection, the checkpoint's `positional_encoder.projector.weight`, to get 32 phases.
6. Build the encoding from the 32 cosines followed by the 32 sines, each value repeated twice.
7. Pass the tensors in the order `encoding0`, `descriptors0`, `encoding1`, `descriptors1`.

## Outputs

`assignment` is 1×1×512×512. As a 512×512 matrix, entry (i, j) is the probability that keypoint i of image 0 matches keypoint j of image 1.

Postprocessing:

1. For each keypoint i, take the keypoint j with the highest assignment in row i.
2. Keep the pair when i is also the highest in column j, and its assignment is above 0.1.

## Building applications

Run the package with the [Neat Model API](https://developer.sima.ai/software/develop-apps/development-workflow/model). For complete applications, start from the [Neat example apps](https://developer.sima.ai/examples).
