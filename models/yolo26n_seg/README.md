# YOLO26n-seg

![YOLO26n-seg model hero](hero.png)

## Overview

YOLO26n-seg is the Nano edition of Ultralytics' YOLO26 instance segmentation
model for the 80 COCO classes. It accepts a letterboxed 640×640 RGB image and
returns raw box distances, class logits and mask coefficients on three feature
grids, plus 32 mask prototypes. The application decodes them into scored boxes,
each with a pixel mask; YOLO26 needs no non-maximum suppression. Mask accuracy
is measured as COCO mask mAP on a Modalix DevKit. YOLO26 is licensed under
AGPL-3.0, which restricts redistribution of the model and of applications built
on it.

## Usage

Download a published build from the `develop` catalog:

```bash
sima-cli models download \
  --id yolo26n_seg \
  --variant modalix_int8 \
  --branch develop
```

Use `--variant modalix_bf16` for the BF16 build.

The MPK returns raw head tensors and prototypes. Decode them in the application
as described below; `scripts/accuracy.py` contains a reference decoder.

## Model interface

Each image is letterboxed as in Ultralytics inference: scaled with bilinear
sampling so its longer side is 640 pixels, centered, and padded to 640×640 with
gray value 114. The RGB pixels are then scaled to `[0, 1]`. The ONNX input
`images` is that image as an NCHW FP32 tensor with shape `1×3×640×640`. With
Neat, pass the same image as an HWC FP32 tensor with shape `640×640×3` and turn
off Neat's resize, color conversion, and normalization.

The model emits ten tensors: a box, class and mask-coefficient tensor for each
of strides 8, 16, and 32, then the prototypes. The MPK returns them in NHWC
order:

| Output | MPK shape | Contents |
| --- | --- | --- |
| `bbox_0`, `bbox_1`, `bbox_2` | `1×80×80×4`, `1×40×40×4`, `1×20×20×4` | left, top, right, and bottom distances from the cell center, in grid cells |
| `class_logit_0`, `class_logit_1`, `class_logit_2` | `1×80×80×80`, `1×40×40×80`, `1×20×20×80` | logits for the 80 COCO classes |
| `mask_coeff_0`, `mask_coeff_1`, `mask_coeff_2` | `1×80×80×32`, `1×40×40×32`, `1×20×20×32` | coefficients over the 32 mask prototypes |
| `mask_proto` | `1×160×160×32` | mask prototypes at stride 4 |

To decode boxes, give the cell in row `r` and column `c` the center
`(c + 0.5, r + 0.5)`. Its box corners are the center minus the left and top
distances and the center plus the right and bottom distances, multiplied by the
stride. Class scores are the sigmoid of the logits. Ultralytics' end-to-end
selection takes the 300 cells with the highest class score, then the 300
highest-scoring cell and class pairs among them. Class index `i` is the `i`-th
COCO category ID in ascending order.

A selected detection's mask logits are the prototypes multiplied by that cell's
32 coefficients. Upsample them bilinearly to 640×640, keep only the pixels
inside the detection's box, and set a pixel where its logit is positive. To
return boxes and masks to the original image, remove the letterbox padding,
then scale by the inverse letterbox scale and clip boxes to the image.

## Available builds

The registry declares two build profiles with one `model` component each. Both
compile the whole network into one MLA segment; the compile recipe rejects a
build that places any layer on the A65 or APU.

- `modalix_bf16` runs weights and activations in BF16 and needs no calibration
  data.
- `modalix_int8` quantizes activations per tensor and weights per channel to
  INT8. It calibrates with histogram MSE over 1,024 bins on 50 COCO images and
  applies regular bias correction.

## Quality and performance

On a Modalix DevKit, `scripts/accuracy.py` measures the 12 COCO mask AP and AR
metrics and the box mAP 50–95 and AP50 on all 5,000 val2017 images. The
`coco_accuracy` result is published as an accuracy run for each build, and the
model browser shows it beside Ultralytics'
[published scores](https://github.com/ultralytics/ultralytics/blob/f2d3aed634a5b0e4828024718d4a61ab2f83fb19/docs/en/models/yolo26.md)
for the end-to-end head: mask mAP 50–95 0.339 and box mAP 50–95 0.396.

The script letterboxes each image on the CPU, runs the MPK, and decodes it with
Ultralytics' end-to-end selection, keeping up to 300 detections with a score
above 0.001. Given the letterboxed image, it computes the masks of Ultralytics'
CPU validation pixel for pixel, and it scores masks and boxes from the same
detections. Two differences from the published protocol remain:

- Ultralytics batches validation images as padded rectangles; this package
  always uses a 640×640 square.
- Ultralytics' validation loader rounds the resized image size up, while this
  package rounds it to the nearest pixel, as Ultralytics' prediction letterbox
  does. The size differs by one pixel for 210 of the 5,000 images.

The benchmark script runs Neat Core's default synthetic benchmark on the MPK
with Neat's YOLO26 segmentation decoder, and CI publishes one result for each
build. The decoder's default score threshold is 0, so it keeps up to 100
detections of each synthetic frame and computes their 160×160 masks on the
A65; the FPS includes that mask work.

## Source and license

The source is the `yolo26n-seg.pt` checkpoint from Ultralytics' YOLO26
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
scripts/run_model_conversion.sh models/yolo26n_seg default /tmp/yolo26n_seg-export
```

To publish a conversion, manually run **Vulcan Model Convert** with model
`yolo26n_seg`. The workflow validates the ONNX, compares the nine raw head
tensors and the prototypes with PyTorch on ten synthetic inputs, and uploads
`model.onnx`. Copy the printed source pin into `model.yaml` only after surgery
and compiler validation. The smoke comparison does not measure model accuracy.

### Calibration and compilation

`surgery.py` cuts the graph at the last convolution of each one-to-one Segment
branch and at the prototype output, and names the ten outputs. Removing the
in-graph box decode, top-300 selection and coefficient gather lets the whole
network run on the MLA. The cut changes no values: the head tensors match the
full graph exactly.

The INT8 build calibrates on 50 images drawn with a fixed seed from the
`cal2017` train2017 images of the COCO 2017 `cal` dataset, so no scored
val2017 image calibrates the model. In the pinned Model Compiler environment, compile one
build with the shared runner script. It resolves the source and the calibration
archive, runs `scripts/compile.py`, and validates the MPK:

```bash
scripts/run_model_compilation.sh models/yolo26n_seg modalix_int8 model /tmp/yolo26n_seg
```

The MPK lands in `/tmp/yolo26n_seg/yolo26n_seg/modalix_int8/model/`. Use
`modalix_bf16` for the BF16 build, which needs no calibration archive. Cut
graphs and generated models stay in a temporary directory outside Git.

### DevKit accuracy

The `coco_accuracy` accuracy entry declares the COCO 2017 `val` dataset
archive, the published metrics, and Ultralytics' reference scores. On a Modalix
DevKit, run the accuracy stage on the output of
`scripts/run_model_compilation.sh`, setting `REGISTRY_PYTHON` to a Python that
has the full Model Registry tooling and Dataset Registry:

```bash
scripts/run_model_accuracy.sh models/yolo26n_seg modalix_int8 \
  /tmp/yolo26n_seg /media/nvme/neat-tests/yolo26n_seg-accuracy
```

The script installs `scripts/requirements-accuracy.txt` beside PyNeat's
environment, extracts the archive under `/media/nvme/neat-tests/`, and removes
both when it exits. See
[Accuracy on the DevKit](../../docs/local-accuracy-development.md#accuracy-on-the-devkit)
for the result format and publication.
