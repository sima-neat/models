# Getting started

This is the starting guide for this repository. It says what the repository is for,
shows the path every model takes, and then walks through that path once with real
commands.

## What this repository is for

This repository hosts the models you find in the
[Model Browser](https://developer.sima.ai/models/). Each model has a page there with
its measured performance and accuracy on a Modalix DevKit.

You can reproduce every step behind such a page on your own machine. Each model has a
folder, `models/<model id>/`, with the scripts that convert, compile, and measure it.
The GitHub link in the header of a model page opens that folder.

Some model pages offer no download, because the model's license forbids redistribution.
You can still rebuild every model from its scripts.

## The path every model takes

```text
┌──────────────────┐  convert.py   ┌─────────────┐  compile.py   ┌───────────────┐
│ Official model   │──────────────►│ ONNX model  │──────────────►│ MPK           │
│ pinned in        │               │             │               │ runs on the   │
│ model.yaml       │               │             │               │ DevKit        │
└──────────────────┘               └─────────────┘               └───────────────┘
                                                                         │
                                         performance.py and accuracy.py  │
                                                                         ▼
┌──────────────────┐         compare with         ┌─────────────────────────────┐
│ Model page in    │◄─────────────────────────────│ Your results                │
│ the Model Browser│                              │ model-report.json           │
└──────────────────┘                              └─────────────────────────────┘
```

1. `model.yaml` names the official model and pins its files, so everyone starts from the
   same bytes.
2. `convert.py` downloads the official model and exports it to ONNX.
3. `compile.py` prepares the ONNX for the chip and compiles it into an MPK, the archive
   a DevKit runs. An 8-bit variant also needs calibration images.
4. `performance.py` measures speed on the DevKit, and `accuracy.py` scores the model on
   its dataset.
5. You compare your results with the model page.

CI runs the same scripts to produce the numbers on the model page. That is why your
results can match them.

## Reproduce a model: the example

The page uses one model from start to finish, so you can copy every command as written.
The example is YOLO26n, the smallest YOLO26 object detector. Its folder is
`models/yolo26n`.

A model can be compiled in several variants, called build profiles. The commands build
`w8a8`, the 8-bit variant of YOLO26n. To reproduce another model or variant, replace
`yolo26n` and `w8a8` in the commands. Step 2 shows which variants a model supports.
[Concepts](concepts.md) defines the terms this page uses.

## Before you start

| You need | Check it |
| --- | --- |
| A clone of this repository. Run each command from its root. | `ls models/yolo26n/model.yaml` |
| Python 3.12 and internet access, for the conversion. | `python3.12 --version` |
| The Model Compiler, at the version [`deps/manifest.json`](../deps/manifest.json) pins under `model-compiler`. Activate its environment in the shell that runs step 5. In the Neat SDK, run `activate-model-compiler`. | `python -c 'import afe; print("Model Compiler Python APIs available")'` prints that sentence. |
| A Modalix DevKit that your machine reaches over SSH, with PyNeat, the Python package that runs a compiled model. Step 8 also needs internet access on the DevKit. [DevKit performance](benchmarking.md) describes the setup. | `~/pyneat/bin/python -c 'import pyneat; print(pyneat.__version__)'` prints the PyNeat version. |

Steps 4 to 6 run in one clone, because each step reads what the step before wrote to
`build/`.

## Steps

| Step | Runs on | Script in `models/yolo26n/scripts/` | Result |
| --- | --- | --- | --- |
| Convert | a machine with Python 3.12 | `convert.py` | `model.onnx` |
| Compile | the Model Compiler | `compile.py` | one MPK, the archive a DevKit runs |
| Measure performance | the DevKit | `performance.py` | `model-report.json` |
| Measure accuracy | the DevKit | `accuracy.py` | `model-report.json` |

A model without one of these scripts has no such step. A model with `compile.sh` in place
of `compile.py` is a [language model](#language-models).

### 1. Read `model.yaml`

Open [`models/yolo26n/model.yaml`](../models/yolo26n/model.yaml). It declares everything
the scripts use:

| Field | What it tells you | In `yolo26n` |
| --- | --- | --- |
| `upstream` | the official model | the Hugging Face repository `Ultralytics/YOLO26`, at a pinned revision |
| `conversion.source` | the checkpoint `convert.py` downloads | `yolo26n.pt`, with its SHA-256 and size |
| `compilation.source` | where the ONNX to compile comes from | `conversion`, so step 4 produces it |
| `build_profiles` | the variants you can build | `w16a16`, `w8a8`, and `w8a16` |
| `calibrations` | the dataset that calibrates a build profile | `coco-2017-cal`, used by `w8a8` |
| `accuracy.datasets` | the datasets accuracy is measured on | `coco-2017-val` |
| `reference_metrics` | the scores the official model publishes | `coco_map: 0.401` |

A dataset goes by its source ID, `<id>-<variant>`. The pin with `id: coco-2017` and
`variant: val` is `coco-2017-val`.

### 2. Choose a build profile

The entries under `build_profiles` in `model.yaml` are the variants you can build. Each
name gives the bit widths of the weights and of the activations. YOLO26n declares
`w16a16`, `w8a8`, and `w8a16`.

Check three things before you choose one:

- `compile.py` builds only a declared build profile. For any other name it stops with an error that names the build profile.
- A build profile that is commented out in `model.yaml` is not supported. Leave it commented out.
- A build profile with a `calibration_dataset` line needs that dataset in step 5. In YOLO26n only `w8a8` has one.

The published results differ between build profiles.
`models/yolo26n/tests/tests.yaml` lists the expected values of each one.

### 3. Get the datasets

The `w8a8` build needs two datasets. `coco-2017-cal` calibrates the compile, which means
the compiler runs sample images to choose the INT8 value ranges. `coco-2017-val` supplies
the images for the performance and accuracy measurements.

The Dataset Registry is the SiMa.ai store of the exact dataset archives that CI uses.
Take the first route when you have access to it, and the second when you do not.

**With access to the Dataset Registry.** Set up access as [Datasets](datasets.md)
describes, and install `model-registry` as [Qualify locally](local-development.md)
describes. Then resolve both archives in the Model Compiler environment. The commands
need `jq`:

```bash
calibration="$(model-registry fetch-calibration models/yolo26n \
  --build-profile w8a8 --component model --json | jq -r .path)"
validation="$(dataset-registry resolve coco-2017 --variant val | jq -r .path)"
```

Each command downloads one archive, verifies it, and stores its path. `--component model`
names the model's one component, whose ID is `model`.

**Without access.** Download the original dataset and arrange it the way the scripts read
it:

| Script and argument | What it reads |
| --- | --- |
| `compile.py --dataset` | A directory or a `.tar.gz` archive. It takes the `.jpg`, `.jpeg`, and `.webp` images in a folder named `cal2017` and picks 20 of them with a fixed seed. |
| `performance.py --dataset coco-2017-val=<directory>` | The `.jpg` images in `<directory>/val2017/`. It picks one with a fixed seed. |
| `accuracy.py --dataset coco-2017-val=<directory>` | `<directory>/val2017/` and `<directory>/annotations/instances_val2017.json`. It scores every image the annotation file lists. |

The COCO 2017 validation download uses the `val2017` and `annotations` names. `cal2017`
is not part of the COCO download. Create it and copy at least 20 COCO images into it,
taken from outside `val2017`, so that calibration and evaluation use different images.
Point `calibration` at the directory that holds `cal2017`:

```bash
calibration=/data/coco-2017-cal
```

Your results match the published ones only when the scripts read the same samples. CI
calibrates `w8a8` from the registered `coco-2017-cal` archive, so other images in
`cal2017` give a different INT8 model. The `w16a16` and `w8a16` build profiles use no
calibration dataset.

### 4. Convert the checkpoint to ONNX

Run this on the machine with your clone:

```bash
scripts/run_model_conversion.sh models/yolo26n default build/yolo26n/onnx
```

The runner creates a fresh Python environment in `build/yolo26n/onnx-venv`. It installs
this repository and `models/yolo26n/scripts/requirements-convert.txt` there, then runs
`convert.py`. `convert.py` downloads `yolo26n.pt`, checks it against the SHA-256 in
`model.yaml`, and exports it to ONNX. `default` names the model's one conversion.

The last line reads `Synthetic-input PyTorch/ONNX head check passed with seeds 123-132`,
followed by the full path of `build/yolo26n/onnx/model.onnx`. `build/` is in
`.gitignore`, so the outputs stay out of Git.

Other models differ here:

- A model that lists `conversion.outputs` in `model.yaml` writes those files instead of `model.onnx`.
- A model that pins an ONNX file under `compilation.source`, as [`models/resnet_50`](../models/resnet_50/model.yaml) does, has no `convert.py`. Skip this step. `model-registry fetch models/resnet_50` downloads the file, verifies it, and prints its path.

### 5. Compile the model

Run this in the same clone, with the Model Compiler environment active:

```bash
PYTHONPATH=tools python models/yolo26n/scripts/compile.py \
  --model build/yolo26n/onnx/model.onnx \
  --output-dir build/yolo26n/w8a8 \
  --build-profile w8a8 \
  --compile-invocation w8a8 \
  --dataset "$calibration"
```

`compile.py` works in three parts:

1. It rewrites the graph with `surgery.py`, which cuts YOLO26n at its raw detection heads, before the box decode. A model without `surgery.py` compiles the ONNX as it is.
2. It quantizes the graph. The `w8a8` build profile uses 20 calibration images to choose its INT8 ranges.
3. It writes one MPK per model component. An MPK is the archive a DevKit runs.

`PYTHONPATH=tools` lets `compile.py` import two packages from your clone: `model_registry`,
which reads `model.yaml`, and `model_inputs`, which lists the calibration images.
`--compile-invocation` repeats the build profile name, because a
build profile compiles in one invocation named after it.

The compiler ends with a summary whose last line starts with `Compiled model:`. `compile.py`
then copies the archive to `build/yolo26n/w8a8/model/yolo26n_w8a8_mpk.tar.gz`. Check it:

```bash
PYTHONPATH=tools python -m model_registry verify-mpk build/yolo26n/w8a8/model
```

The command prints `Valid artifact: yolo26n_w8a8_mpk.tar.gz`.

Other models differ here:

- The `w16a16` and `w8a16` build profiles take no `--dataset`.
- A model that declares `numerical_fidelity` in `model.yaml`, such as `resnet_50`, also takes `--fidelity-dataset`. Pass the archive or directory of the dataset that `numerical_fidelity.dataset` names. Without it, `compile.py` stops before it compiles.
- A model with several components, such as `rf_detr_nano`, writes one folder and one MPK per component.
- `scripts/run_model_compilation.sh` is the runner CI uses. For a model whose `compilation.source` is `conversion`, it fetches the ONNX that CI published, which needs CI's registry credentials. Call `compile.py` with your own ONNX instead.

### 6. Copy the build and the dataset to the DevKit

Set `devkit` to your DevKit's user and address, then copy the model folder, the two
shared modules its scripts import, and the MPK:

```bash
devkit=sima@192.168.1.20
tar -czf - models/yolo26n tools/model_report tools/model_inputs build/yolo26n/w8a8/model \
  | ssh "$devkit" 'mkdir -p /media/nvme/neat-tests/yolo26n && tar -xzf - -C /media/nvme/neat-tests/yolo26n'
```

Copy the whole model folder, because `accuracy.py` imports `calibration.py` from the same
`scripts/` folder. Use `/media/nvme`, because the NVMe drive keeps large files off the
root filesystem.

The validation dataset belongs in `/media/nvme/datasets/coco-2017-val` on the DevKit.
Copy your own directory there, or extract the Dataset Registry archive from step 3
straight onto the DevKit:

```bash
ssh "$devkit" 'mkdir -p /media/nvme/datasets/coco-2017-val && tar -xzf - -C /media/nvme/datasets/coco-2017-val' < "$validation"
```

The command stores no archive on the DevKit, so a connection that drops leaves no
half-copied file. Run it again in that case. Then check the result:

```bash
ssh "$devkit" 'ls /media/nvme/datasets/coco-2017-val'
```

`ls` prints `annotations` and `val2017`. If it prints one folder instead, that folder is
the dataset directory to pass in steps 7 and 8.

### 7. Measure performance

On the DevKit, read the two clocks that throughput depends on:

```bash
cat /sys/class/devfreq/5000000.mla/cur_freq
cat /sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq
```

The commands print `1000000000` and `1400000`. The first is the clock of the MLA, the
DevKit's machine learning accelerator, at 1 GHz. The second is the CPU clock at 1.4 GHz.
CI measures at these clocks, so a lower clock gives a lower throughput than the model
page shows. [If something fails](#if-something-fails) shows how to raise a lower clock.

Then run the script:

```bash
cd /media/nvme/neat-tests/yolo26n
PYTHONPATH=tools ~/pyneat/bin/python models/yolo26n/scripts/performance.py \
  --mpk build/yolo26n/w8a8/model/yolo26n_w8a8_mpk.tar.gz \
  --dataset coco-2017-val=/media/nvme/datasets/coco-2017-val \
  --output results/performance
```

`PYTHONPATH=tools` lets the script import `model_report`, which writes the report, and
`model_inputs`, which builds the benchmark frame. The
script places one `val2017` image in a 1280x720 frame. It measures latency over 1000
inferences and throughput over at least 60 seconds, then writes
`results/performance/model-report.json`.

Other models differ here:

- A model with several components takes one `--mpk <component>=<archive>` per component.
- A model without an accuracy dataset takes no `--dataset`.
- A model with `scripts/requirements-performance.txt` needs those packages first. Install them as step 8 installs the accuracy packages.
- [DevKit performance](benchmarking.md) covers these cases and the `--resources` argument.

### 8. Measure accuracy

Run this on the DevKit:

```bash
cd /media/nvme/neat-tests/yolo26n
~/pyneat/bin/python -m pip install --no-deps --only-binary=:all: \
  --target site-packages -r models/yolo26n/scripts/requirements-accuracy.txt
PYTHONPATH=tools:site-packages ~/pyneat/bin/python models/yolo26n/scripts/accuracy.py \
  --mpk model=build/yolo26n/w8a8/model/yolo26n_w8a8_mpk.tar.gz \
  --dataset coco-2017-val=/media/nvme/datasets/coco-2017-val \
  --output results/accuracy
```

The `pip` command needs internet access on the DevKit. It installs the packages
`accuracy.py` imports into the separate folder `site-packages`, so that PyNeat's own
packages stay unchanged. In `--mpk model=...`, `model` is the ID of the model's one
component.

`accuracy.py` runs every listed image through the MPK, decodes the boxes, and scores them
against the annotations. It prints its progress every 250 images. It ends with one JSON
line that holds each score, and writes `results/accuracy/model-report.json`.

Add `--limit 100` for a first trial on 100 images. A trial does not compare with the
model page.

Other models differ here:

- A model without `scripts/accuracy.py` has no accuracy stage. Skip this step.
- A model without `scripts/requirements-accuracy.txt` needs no `pip` command.
- A model with several components takes one `--mpk <component>=<archive>` per component.
- When a model's `model.yaml` sets `accuracy.samples`, CI scores at most that many samples of each dataset. Pass the same number as `--limit` to compare with the model page.
- [Accuracy and fidelity](local-accuracy-development.md) describes the arguments of `accuracy.py`.

### 9. Compare with the model page

Open the model's page in the Model Browser and select the **W8A8** tab. Compare these
fields of the two reports with the page:

| Report | Field in `model-report.json` | Label on the model page |
| --- | --- | --- |
| `results/performance` | `metrics.throughput_fps` | Throughput |
| `results/performance` | `metrics.latency_avg_ms` | Average latency |
| `results/accuracy` | `datasets["coco-2017-val"].metrics.coco_map` | mAP 50–95, shown as a percentage |

The page's SoC power tile adds up separate power readings and matches no single field
of the report, so compare throughput, latency, and accuracy.

The clone also holds the published values. `models/yolo26n/tests/tests.yaml` lists the
expected values of a build profile once CI has published its report. For `w8a8` they are
`throughput_fps: 964.5` and `coco_map: 0.3808`. CI fails a model's accuracy test when a
score falls more than 5% below its expected value, and its performance test when
throughput falls more than 10%.

## Language models

A model whose `model.yaml` sets `compilation.format: huggingface` is a language,
vision-language, or speech model. It has no `convert.py` and no `accuracy.py`. Its `scripts/compile.sh`
compiles the pinned Hugging Face checkpoint, and `scripts/run_llima_compilation.sh` runs
that script:

```bash
PYTHONPATH=tools scripts/run_llima_compilation.sh models/gemma_4_e2b_it w4a16 w4a16 build
```

The arguments are the model folder, the build profile, the compile invocation, which
repeats the build profile, and the output folder. Run it in a Model Compiler environment
that has `llima-compile` and `llima-deploy`. `PYTHONPATH=tools` lets the runner call
`model_registry` from your clone. The runner downloads the checkpoint, compiles it, and
writes a `deployment/` folder under `build/gemma_4_e2b_it/w4a16/model/`. [LLiMa language and
vision-language models](adding-a-model.md#llima-language-and-vision-language-models) has
the details.

The result is a folder, not an MPK, so the DevKit commands of steps 7 and 8 do not apply.
A language model's `performance.py` takes the folder with `--deployment` and writes its report
to `--output`. The speech models have no `performance.py`.
[LLiMa text and image performance](benchmarking.md#llima-text-and-image-performance)
describes that run.

## If something fails

| What you see | Cause | Fix |
| --- | --- | --- |
| `yolo26n: resolve conversion with plan-stages before compilation` | `model-registry fetch` or `scripts/run_model_compilation.sh` looked for the ONNX that CI published. | Pass your own ONNX to `compile.py` with `--model`, as in step 5. |
| `python3.12: command not found` | `scripts/run_model_conversion.sh` starts the conversion with `python3.12`. | Install Python 3.12, or set `CONVERSION_PYTHON` to its path. |
| `FileExistsError` from `convert.py` or `compile.py` | The output directory exists. Both scripts create it themselves. | Delete the directory, or name a new one. |
| `No module named 'afe'` | The command does not run in the Model Compiler environment. | Activate the Model Compiler environment, then run the command again. In the Neat SDK, run `activate-model-compiler`. |
| `unknown build profile: <name>` | `--build-profile` names a variant that `model.yaml` does not declare. | Pass a name from `build_profiles`, as step 2 describes. |
| `model.yaml declares numerical_fidelity; pass --fidelity-dataset` | The model measures numerical fidelity during the compile. | Pass `--fidelity-dataset` with the dataset that `numerical_fidelity.dataset` names. |
| `No module named 'model_registry'` | `PYTHONPATH` lacks `tools`, or the command runs outside the repository root. | Run the command from the repository root with `PYTHONPATH=tools`. |
| `the INT8 build requires --dataset from dataset-registry` | `compile.py` ran for `w8a8` without `--dataset`. | Pass the calibration archive or directory from step 3. |
| `cannot select 20 calibration images from 0 cal2017 images` | The calibration input has no folder named `cal2017`, or the folder holds fewer than 20 images. | Put at least 20 images in a folder named `cal2017`. |
| `--model must be the verified ONNX source, not <path>` | The `--model` path does not end in `.onnx`. | Pass `build/yolo26n/onnx/model.onnx`. |
| `has no images under val2017/` | The dataset directory does not hold `val2017/` directly. | Pass the folder that contains `val2017/` and `annotations/`. |
| `No module named 'model_report'` | `PYTHONPATH` lacks `tools`, or the command runs outside the folder that holds `tools/`. | Run from `/media/nvme/neat-tests/yolo26n` with `PYTHONPATH=tools`. |
| `No module named 'model_inputs'` | The DevKit folder lacks `tools/model_inputs`, which the model's scripts import. | Copy it as [step 6](#6-copy-the-build-and-the-dataset-to-the-devkit) does. |
| `No module named 'faster_coco_eval'` | The accuracy packages are missing, or `PYTHONPATH` lacks `site-packages`. | Run both commands of step 8 from the same folder. |
| A clock in step 7 prints a lower value | The DevKit runs the MLA or the CPU below the clock CI measures at. | Set both clocks with the commands below, then read them again. |
| The MLA clock stays below `1000000000` after you set it | The DevKit started with a lower limit for the MLA clock. `cat /sys/class/devfreq/5000000.mla/max_freq` prints that limit. | Restart the DevKit with `sudo reboot` and read the clock again. Repeat until it prints `1000000000`. |
| The performance report has no `avg_power_watts` | Sentinel, the DevKit's telemetry service, gave no power readings. | Set up the DevKit as [DevKit performance](benchmarking.md) describes. |
| `dataset-registry` cannot download an archive | The machine has no access to the dataset storage. | Follow the access steps in [Datasets](datasets.md). |

Set the MLA clock to 1 GHz and the CPU clock to 1.4 GHz:

```bash
echo 1000000000 | sudo tee /sys/class/devfreq/5000000.mla/userspace/set_freq
echo 1400000 | sudo tee /sys/devices/system/cpu/cpufreq/policy0/scaling_setspeed
```

The first command prints `1000000000` even when the DevKit keeps the lower clock, so read
both clocks again afterwards.

## Next

To add a model or change one, continue with [Define the model
contract](adding-a-model.md).
