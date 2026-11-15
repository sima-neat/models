# LightGlue

![LightGlue model hero](hero.png)

## Overview

LightGlue finds which keypoints of two images show the same point of the
scene. Use it to estimate how a camera moved between two images, to stitch
images, and for 3D reconstruction. It returns, for every keypoint pair, the
probability that the two match, and your application picks the matches on the
CPU. It does not detect keypoints: run SuperPoint on each image first. It takes
exactly 512 keypoints per image. It is a feature matcher from ETH Zurich.

## Inputs and outputs

**Input.** Four tensors: a position encoding and the descriptors of 512 SuperPoint keypoints for each of two images.

1. Run SuperPoint on each image stretched to 640×480, with a score threshold of 0.0005 (not SuperPoint's usual 0.005), suppression within 4 pixels, and no keypoints within 4 pixels of an edge. The `LightGlueV1` profile of Neat's SuperPoint decoder uses these settings.
2. Keep the 512 highest-scoring keypoints. With that decoder, set `top_k` to 512: the profile keeps 600.
3. Skip an image with fewer than 512 keypoints: without a padding mask, placeholder rows would change the real matches.
4. Pass the tensors in the table's order, with Neat's preprocessing off.

| Tensor | Size | Holds |
|---|---|---|
| `encoding0` | 1×1×512×128 | each keypoint's position encoding, computed below |
| `descriptors0` | 1×1×512×256 | SuperPoint's unit-length descriptors, in the same order |
| `encoding1`, `descriptors1` | same | the same for the second image |

For the encoding, normalize the pixel coordinates as (x − 320) / 320 and (y − 240) / 320. Multiply by LightGlue's 32×2 keypoint projection, the checkpoint's `positional_encoder.projector.weight`, to get 32 phases. Repeat each phase twice, and pass the 64 cosines followed by the 64 sines.

**Outputs.** Matches between the two images' keypoints. Neat has no decoder for them.

`assignment` is 1×1×512×512. As a 512×512 matrix, entry (i, j) is the probability that keypoint i of the first image matches keypoint j of the second.

1. For each keypoint i, take the keypoint j with the highest assignment in row i.
2. Keep the pair when i is also the highest in column j, and its assignment is above 0.1.

## Building applications

Run the package with the [Neat Model API](https://developer.sima.ai/software/develop-apps/development-workflow/model). For complete applications, start from the [Neat example apps](https://developer.sima.ai/examples).
