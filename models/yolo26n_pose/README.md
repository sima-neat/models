# YOLO26n-pose

## Overview

YOLO26n-pose is the Nano edition of Ultralytics' YOLO26 pose estimator. It
finds people in a letterboxed 640×640 RGB image and returns raw box distances,
person logits, and 17 COCO keypoints on three feature grids. The application
decodes them into scored people with boxes and keypoints; YOLO26 needs no
non-maximum suppression. Pose accuracy is measured as COCO keypoint mAP on a
Modalix DevKit over the 2,346 COCO 2017 validation images that contain a person
with labelled keypoints. YOLO26 is licensed under AGPL-3.0, which restricts
redistribution of the model and of applications built on it.

## Usage

Download a published build from the `develop` catalog:

```bash
sima-cli models download \
  --id yolo26n_pose \
  --variant modalix_int8 \
  --branch develop
```

Use `--variant modalix_bf16` for the BF16 build.

The MPK returns raw head tensors. Decode them in the application as described
below; `scripts/accuracy.py` contains a reference decoder.

## Model interface

Each image is letterboxed as in Ultralytics inference: scaled with bilinear
sampling so its longer side is 640 pixels, centered, and padded to 640×640 with
gray value 114. The RGB pixels are then scaled to `[0, 1]`. The ONNX input
`images` is that image as an NCHW FP32 tensor with shape `1×3×640×640`. With
Neat, pass the same image as an HWC FP32 tensor with shape `640×640×3` and turn
off Neat's resize, color conversion, and normalization.

The model emits nine tensors: a box, a person, and a keypoint tensor for each of
strides 8, 16, and 32. The MPK returns them in NHWC order:

| Output | MPK shape | Contents |
| --- | --- | --- |
| `bbox_0`, `bbox_1`, `bbox_2` | `1×80×80×4`, `1×40×40×4`, `1×20×20×4` | left, top, right, and bottom distances from the cell center, in grid cells |
| `class_logit_0`, `class_logit_1`, `class_logit_2` | `1×80×80×1`, `1×40×40×1`, `1×20×20×1` | person logit |
| `keypoint_0`, `keypoint_1`, `keypoint_2` | `1×80×80×51`, `1×40×40×51`, `1×20×20×51` | for each of the 17 COCO keypoints in order: x and y offsets from the cell center in grid cells, then a visibility logit |

To decode, give the cell in row `r` and column `c` the center `(c + 0.5, r + 0.5)`.
The person score is the sigmoid of the logit. Ultralytics' end-to-end selection
takes the 300 cells with the highest score. A box's corners are the center minus
the left and top distances and the center plus the right and bottom distances,
multiplied by the stride. A keypoint is the center plus its x and y offsets,
multiplied by the stride; its visibility is the sigmoid of its logit. Finally,
subtract the letterbox padding and divide by the letterbox scale, and clip boxes
and keypoints to the original image.

## Available builds

The registry declares two build profiles with one `model` component each. Both
compile the whole network into one MLA segment; the compile recipe rejects a
build that places any layer on the A65 or APU.

- `modalix_bf16` runs weights and activations in BF16 and needs no calibration
  data.
- `modalix_int8` quantizes activations per tensor and weights per channel to
  INT8. It calibrates with histogram MSE over 2,048 bins on 50 COCO images and
  applies regular bias correction.

## Quality and performance

On a Modalix DevKit, `scripts/accuracy.py` measures the 10 COCO keypoint AP and
AR metrics and the person-box mAP and AP50 on the 2,346 val2017 images that
contain a person with labelled keypoints, the images Ultralytics validates pose
models on. The `coco_keypoint_accuracy` result is published as an accuracy run
for each build, and the model browser shows it beside Ultralytics'
[published score](https://github.com/ultralytics/ultralytics/blob/f2d3aed634a5b0e4828024718d4a61ab2f83fb19/docs/en/models/yolo26.md):
keypoint mAP 50–95 0.569 and mAP 50 0.830.

The script letterboxes each image on the CPU, runs the MPK, and decodes it with
Ultralytics' end-to-end selection, keeping up to 300 people with a score above
0.001. Keypoint scoring keeps the 20 highest-scoring people per image, as COCO
does. Ultralytics' published score batches validation images as padded
rectangles; this package always uses a 640×640 square.

The benchmark script runs Neat Core's default synthetic benchmark on the MPK,
and CI publishes one result for each build.

## Source and license

The source is the `yolo26n-pose.pt` checkpoint from Ultralytics' YOLO26
repository on Hugging Face, exported to ONNX with Ultralytics 8.4.0. Normal
compilation consumes the pinned ONNX and does not install the exporter
dependencies.

- [YOLO26 model card](https://huggingface.co/Ultralytics/YOLO26)
- [Pinned checkpoint revision](https://huggingface.co/Ultralytics/YOLO26/tree/070ac3c51435984d992ba92eb3f234daf4e4500d)
- [Ultralytics 8.4.0 source](https://github.com/ultralytics/ultralytics/tree/f2d3aed634a5b0e4828024718d4a61ab2f83fb19)

YOLO26 is distributed under AGPL-3.0. Its redistribution terms apply to the
compiled MPKs and to applications that include them.

## Reproducing the artifact

### Conversion

The `conversion.default` entry in `model.yaml` pins the checkpoint, and
`scripts/requirements-convert.txt` pins the export dependencies. They install
Ultralytics as `ultralytics-opencv-headless`, which ships the same code as
`ultralytics` with headless OpenCV, so conversion needs no system graphics
libraries. `convert.py` exports with Ultralytics at opset 12 and folds constants
with onnxsim, so surgery starts from a static graph. On Linux ARM64, the shared
runner script converts in a fresh Python 3.12 environment built from those
requirements:

```bash
scripts/run_model_conversion.sh models/yolo26n_pose default /tmp/yolo26n_pose-export
```

To publish a conversion, manually run **Vulcan Model Convert** with model
`yolo26n_pose`. The workflow validates the ONNX, compares the nine raw head
tensors with PyTorch on ten synthetic inputs, and uploads `model.onnx`. Copy the
printed source pin into `model.yaml` only after surgery and compiler validation.
The smoke comparison does not measure model accuracy.

### Calibration and compilation

`surgery.py` cuts the graph at the last convolution of each one-to-one Pose
branch and names the nine outputs. Removing the in-graph box and keypoint decode
and top-300 selection lets the whole network run on the MLA. The cut changes no
values: the head tensors match the full graph exactly.

The INT8 build calibrates on 50 images drawn with a fixed seed from the
`cal2017` split of the COCO 2017 mini dataset, so no scored val2017 image
calibrates the model. In the pinned Model Compiler environment, compile one
build with the shared runner script. It resolves the source and the calibration
archive, runs `scripts/compile.py`, and validates the MPK:

```bash
scripts/run_model_compilation.sh models/yolo26n_pose modalix_int8 model /tmp/yolo26n_pose
```

The MPK lands in `/tmp/yolo26n_pose/yolo26n_pose/modalix_int8/model/`. Use
`modalix_bf16` for the BF16 build, which needs no calibration archive. Cut
graphs and generated models stay in a temporary directory outside Git.

### DevKit accuracy

The `coco_keypoint_accuracy` accuracy entry declares the COCO 2017 `keypoints`
dataset archive, which holds the original val2017 images and the official
person keypoint annotations, the published metrics, and Ultralytics' reference
score. On a Modalix DevKit, run the accuracy stage on the output of
`scripts/run_model_compilation.sh`, setting `REGISTRY_PYTHON` to a Python that
has the full Model Registry tooling and Dataset Registry:

```bash
scripts/run_model_accuracy.sh models/yolo26n_pose modalix_int8 \
  /tmp/yolo26n_pose /media/nvme/neat-tests/yolo26n_pose-accuracy
```

The script installs `scripts/requirements-accuracy.txt` beside PyNeat's
environment, extracts the archive under `/media/nvme/neat-tests/`, and removes
both when it exits. See
[Accuracy on the DevKit](../../docs/local-accuracy-development.md#accuracy-on-the-devkit)
for the result format and publication.
