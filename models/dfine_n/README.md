# D-FINE Nano

![D-FINE Nano model hero](hero.png)

## Overview

D-FINE Nano detects 80 COCO object classes using a backbone and decoder.
The backbone extracts features and proposes candidate locations; the decoder
refines the best 300 into boxes and class scores. Select candidates on the
CPU, retaining the original feature tensor. Native preprocessing accepts
original-size RGB images; the decoder's boxes and scores are decoded on the
CPU into image coordinates.

## Inputs and outputs

**Input.** Stretch RGB images to 640×640 with bilinear resampling. Scale pixels
to 0–1 without ImageNet normalization; retain original dimensions for decoding.

**Between components.** The backbone returns `attention_memory` and
`query_content`, each shaped (1, 2000, 128), `proposal_scores` (1, 2000), and
`reference_logits` (1, 2000, 4). Select the 300 highest proposal scores, breaking
ties by lower proposal index. Gather the corresponding content and raw reference
rows. Pass these and the original attention tensor to the decoder; retain
their storage until execution completes.

**Outputs.** The decoder returns `pred_boxes` (1, 300, 4), normalized
(cx, cy, width, height), and `pred_logits` (1, 300, 80).

Decode them on the CPU, as `scripts/accuracy.py` does:

1. Apply sigmoid; rank all 300×80 query/class pairs.
2. Keep 300 pairs, including different classes per box; do not suppress overlaps.
3. Convert to corners, scale to original dimensions, clip to the image, and drop boxes left empty.
4. Map class indices through sorted COCO category IDs, including their gaps.

Public tensor boundaries remain FP32. Choose a package that balances accuracy
and throughput for your application.

## Building applications

Run the package with the [Neat Model
API](https://developer.sima.ai/software/develop-apps/development-workflow/model).
For complete applications, start from the [Neat example
apps](https://developer.sima.ai/examples).
