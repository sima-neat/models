# SuperGlue Outdoor

![SuperGlue Outdoor model hero](hero.png)

## Overview

SuperGlue Outdoor matches keypoints between two images of the same outdoor
scene, such as consecutive drone frames. Use it for visual odometry, mapping,
and image alignment. It returns a score for every keypoint pair, plus an
unmatched option for each keypoint, and your application turns the scores into
matches on the CPU. It does not detect keypoints: run SuperPoint on each image
first and pass its keypoints, scores, and descriptors. It takes exactly 512
keypoints per image, so skip a pair when an image yields fewer.

## Inputs

Six float tensors: the positions, scores, and descriptors of 512 SuperPoint keypoints for each of two images (0 and 1).

| Input | Size | Holds |
|---|---|---|
| `keypoints0`, `keypoints1` | 1×1×512×2 | normalized (x, y) keypoint positions |
| `scores0`, `scores1` | 1×1×512×1 | keypoint scores |
| `descriptors0`, `descriptors1` | 1×1×512×256 | unit-length SuperPoint descriptors |

Preprocessing:

1. Resize each image to 640×480 without keeping the aspect ratio, and run SuperPoint on it.
2. Decode its keypoints with suppression within 4 pixels, a score threshold of 0.0005, and no keypoints within 4 pixels of an edge.
3. Keep the 512 highest-scoring keypoints, best first. Skip a pair when an image yields fewer than 512.
4. Normalize each position as ((x − 320) / 448, (y − 240) / 448).

## Outputs

`couplings` is 1×1×513×513. As a 513×513 matrix, entry (i, j) scores image-0 keypoint i against image-1 keypoint j. Index 512 is each image's unmatched option.

Postprocessing:

1. Subtract the matrix maximum and take the exponent of every entry.
2. Run 20 Sinkhorn iterations: alternately rescale rows and columns so rows 0–511 each sum to 1/1024 and row 512 to 512/1024, and the same for columns.
3. Multiply the result by 1024.
4. For each image-0 keypoint, find its best image-1 keypoint, ignoring index 512.
5. Keep the pair only if that image-1 keypoint's best is the same image-0 keypoint.
6. The pair's value is its match score. Keep matches scoring above 0.2.

## Building applications

Run the package with the [Neat Model API](https://developer.sima.ai/software/develop-apps/development-workflow/model). For complete applications, start from the [Neat example apps](https://developer.sima.ai/examples).
