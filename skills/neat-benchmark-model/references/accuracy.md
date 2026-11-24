# Accuracy

Accuracy asks how well the compiled model does its task, such as top-1 accuracy
for a classifier or COCO mAP for a detector. It needs the model's own decoding
and scoring, so a model's `scripts/accuracy.py` runs every compiled MPK of a
build on the DevKit with PyNeat. Two other measurements are not accuracy:

- Numerical fidelity compares the compiled model's numbers with the source
  model's during compilation. It is a numerical diagnostic.
- Calibration data feeds quantization and proves nothing about quality.

[Accuracy and fidelity](https://github.com/sima-neat/models/blob/main/docs/local-accuracy-development.md)
holds the full rules.

## Get the dataset

A dataset goes by its source ID, `<id>-<variant>`, such as `coco-2017-val`. The
model's `model.yaml` pins each one under `accuracy.datasets` with its `id`,
`variant`, `sha256`, and `size_bytes`. Extract it on the DevKit under
`/media/nvme/datasets/<source id>`.

- **From the original dataset.** This route needs no SiMa access. Download the
  original dataset and arrange it the way the model's scripts read it.
  YOLO26n's scripts read `val2017/` and `annotations/instances_val2017.json` for
  COCO 2017, as
  [Getting started](https://github.com/sima-neat/models/blob/main/docs/getting-started.md#3-get-the-datasets)
  shows. Results match the published ones only when the scripts read the same
  samples.
- **From the Dataset Registry, on SiMa's network.** The Dataset Registry is the
  SiMa.ai store of the exact dataset archives CI uses. It needs a connection to
  the SiMa private network or VPN and read access to SiMa's Artifactory, as
  [Datasets](https://github.com/sima-neat/models/blob/main/docs/datasets.md#set-up-artifactory-access-first)
  describes. `dataset-registry resolve` downloads and verifies the archive, then
  prints JSON whose `path` names it. Stream it onto the DevKit without storing
  the archive there, so a dropped connection leaves no half-copied file:

  ```bash
  archive="$(dataset-registry resolve <id> --variant <variant> | jq -r .path)"
  ssh <user>@<devkit-address> 'mkdir -p /media/nvme/datasets/<source id> && tar -xzf - -C /media/nvme/datasets/<source id>' < "$archive"
  ```

Check the result with `ls`. If it prints one folder instead of the split
folders, that folder is the dataset directory to pass.

## The accuracy script

`scripts/accuracy.py` receives:

| Argument | Value |
| --- | --- |
| `--mpk <component>=<path>` | once for each component, `model=<path>` for a single-component model |
| `--dataset <source id>=<directory>` | once for each dataset, the extracted archive |
| `--resources <directory>` | when the compile declared supporting `runtime_files`, the profile's output folder with the same relative paths as the package |
| `--output <directory>` | where the report goes |

Existing scripts also accept `--limit <images>` to evaluate a subset for a
quick check. CI passes it only when `model.yaml` sets `accuracy.samples`, so
compare a run with the same limit, or none when the field is unset. Keep the
limit at or below the dataset's sample count, because most scripts draw that
many samples with a fixed seed and stop with an error when the dataset holds
fewer. Optional `scripts/requirements-accuracy.txt` pins extra wheels, which
install into a separate folder beside PyNeat's environment and never into it.
A script that imports another file under `scripts/`, such as ResNet-50's
`calibration.py`, needs that file beside it. `calibration.py` imports
`model_inputs`, so such a run also needs `tools/model_inputs` on `PYTHONPATH`.

The script scores the whole model on each dataset and hands the scores to
`model_report.from_accuracy()`, keyed by the dataset source IDs:

```python
dataset_id, dataset_dir = args.dataset.split("=", 1)  # imagenet-2012-val, <directory>
...
report = model_report.from_accuracy({
    dataset_id: {
        "metrics": {"top1_accuracy": top1 / len(names), "top5_accuracy": top5 / len(names)},
        "split": VALIDATION_SPLIT,
        "evaluated_samples": len(names),
        "available_samples": len(available),
        "details": {"inference_seconds": round(time.monotonic() - started, 1)},
    },
})
report.save(args.output)
```

How the script evaluates each dataset is up to the model. ResNet-50's
[`accuracy.py`](https://github.com/sima-neat/models/blob/main/models/resnet_50/scripts/accuracy.py)
builds a runner from the first input tensor and runs every validation image
through it:

```python
model = pyneat.Model(str(parse_mpks(args.mpk)))
runner = None
for image in images:
    tensor = pyneat.Tensor.from_numpy(image, copy=False)
    if runner is None:
        runner = model.build([tensor])
    logits = runner.run([tensor], timeout_ms=120000)[0].to_numpy(copy=False).reshape(-1)
```

`from_accuracy()` checks each dataset's result:

- It needs `metrics`, `split`, `evaluated_samples`, and `available_samples`, and
  allows only `details` besides them.
- Every metric must be an accuracy metric of `metrics.json`, inside its bounds,
  as [metrics](metrics.md#accuracy-metrics) describes.
- `evaluated_samples` must be positive and at most `available_samples`.

`report.save(output)` writes `model-report.json` with the stage, the PyNeat
version and build, and `datasets`, which gives for each source ID its split,
sample counts, one number per metric, and `details`. The model, build profile,
MPK digests, and board are not report fields.

## Required metrics

Each dataset that models evaluate has a file `datasets/<id>.yaml` in the Models
repository's [`datasets`](https://github.com/sima-neat/models/tree/main/datasets)
folder. It lists, per model task, the metrics every accuracy report on that
dataset must hold. A result must hold the metrics listed for any of the model's
tasks, and may hold other known accuracy metrics. The files list these today:

| Dataset | Task | Required metrics |
| --- | --- | --- |
| ImageNet 2012 | image classification, image-text embedding | `top1_accuracy` |
| ImageNet 2012 | image embedding | `knn_top1_accuracy` |
| COCO 2017 | object detection | `coco_map` |
| COCO 2017 | instance segmentation | `coco_mask_map` |
| COCO 2017 | pose estimation | `coco_keypoint_map` |
| ADE20K | semantic segmentation | `miou` |
| DAVIS 2017 | video object segmentation | `jf_mean`, `j_mean`, `f_mean` |
| DOTA v1.0 | oriented object detection | `dota_obb_map50` |
| HPatches | feature point detection | `keypoint_repeatability`, `homography_accuracy` |
| NYU Depth V2 | depth estimation | `abs_rel`, `delta1` |
| Pittsburgh 30k | loop closure | `recall_at_1`, `recall_at_5`, `recall_at_10` |
| Scene Flow | depth estimation | `epe` |
| YFCC100M | feature matching | `pose_auc_5`, `pose_auc_10`, `pose_auc_20` |

## FP32 baseline

When no published score matches the package's setup, a model can report an FP32
baseline: what the unquantized source model scores on the same inputs and
protocol. A standalone `scripts/fp32_baseline_accuracy.py` runs an FP32 ONNX
with ONNX Runtime on a host and scores it with the model's own `accuracy.py`, so
only the model execution differs from the DevKit measurement. Its docstring
names the ONNX it reads.
