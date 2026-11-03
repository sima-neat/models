# YOLOv8l-obb

![YOLOv8l-obb model hero](hero.png)

## Overview

YOLOv8l-obb is the Large size of the Ultralytics YOLOv8 oriented-box model,
trained on the 15 DOTA aerial object classes, such as planes and ships. It
finds each object in a 1024×1024 RGB tile of an aerial image and returns a
rotated box around it. It returns raw box, class, and angle outputs at three
scales, which your application decodes with the steps below.

## Inputs and outputs

**Input.** Cut larger images into 1024×1024 windows that overlap by 200 pixels.
Letterbox each window to 1024×1024 with gray (114, 114, 114) padding, scaling it
up with bilinear resampling only when it is smaller. Scale the RGB values to
0–1. Pass the tile as a 1024×1024×3 (height, width, channel) float tensor, with
Neat's resize, color conversion, and normalization turned off.

**Outputs.** Each tensor is in height, width, channel order, at stride 8, 16, or
32 (grids of 128, 64, and 32 cells):

- `bbox_0` to `bbox_2`: 64 box logits, 16 for each of the left, top, right, and bottom sides, in that order.
- `class_logit_0` to `class_logit_2`: 15 class logits.
- `angle_0` to `angle_2`: one angle logit.

1. The cell in column c and row r has its center at (c + 0.5, r + 0.5).
2. For each side, apply a softmax to its 16 logits. The side's distance from the cell center, in cells and in the box's rotated frame, is the sum of each bin index (0 to 15) times its weight.
3. Apply a sigmoid to the class logits. The angle in radians is (sigmoid(angle logit) − 0.25) × π.
4. The box is left plus right wide and top plus bottom high. Its center is the cell center plus half of (right − left, bottom − top), rotated by the angle. Multiply by the stride.
5. Keep the cell-class pairs that score above your threshold. Per class, drop boxes that overlap a higher-scoring box by your NMS threshold or more, measured as probabilistic IoU of rotated boxes.
6. Remove the letterbox padding and scale, and add the window's offset.
7. Per class, drop boxes that overlap a higher-scoring box from another window by 0.3 or more.

The accuracy measurement keeps scores above 0.001, uses 0.7 for step 5, and
keeps at most 300 boxes per tile.

Class index i is, in order: plane, ship, storage-tank, baseball-diamond,
tennis-court, basketball-court, ground-track-field, harbor, bridge,
large-vehicle, small-vehicle, helicopter, roundabout, soccer-ball-field,
swimming-pool. Neat's `YoloV8Obb` box decoder can also run steps 1 to 5.

## Building applications

Run the package with the [Neat Model
API](https://developer.sima.ai/software/develop-apps/development-workflow/model).
For complete applications, start from the [Neat example
apps](https://developer.sima.ai/examples).
