---
name: neat-benchmark-model
description: Measure a compiled model's performance and accuracy on a SiMa Modalix DevKit the way the Models repository does. Use to benchmark an MPK's throughput, latency, power, or energy with PyNeat, write a standalone performance script, read a model-report.json, score accuracy on a dataset, or compare a result with a published model page or an earlier run.
---

# Neat Benchmark Model

The measurement runs on the DevKit in PyNeat, the Python package that runs a
compiled model, at `~/pyneat/bin/python`. A model's performance script calls
Neat Core's benchmark and hands the result to `model_report`, which writes
`model-report.json`. `model_report` is the Models repository's
[`tools/model_report`](https://github.com/sima-neat/models/tree/main/tools/model_report).
It needs only the Python standard library, so it runs in PyNeat's environment.

This skill measures MPKs. Compiling a model belongs to the `neat-compile-model`
skill. Language and vision-language models run MoLE workloads instead, which
[DevKit performance](https://github.com/sima-neat/models/blob/main/docs/benchmarking.md#llima-text-and-image-performance)
describes.

## Steps

1. **Prepare the DevKit.**
   - To compare with published numbers, run the platform version and the Core
     package that the Models repository's
     [`deps/manifest.json`](https://github.com/sima-neat/models/blob/main/deps/manifest.json)
     pins as `platform-version` and `neat-core.install_target`. CI installs Core
     with `sima-cli neat install --env production --type minimal core@<branch>:<tag>`.
     Core's board installer also installs Sentinel, the telemetry daemon that
     supplies power, temperature, CPU, and memory readings.
   - Read the two clocks that throughput depends on. Published results use the
     MLA at 1 GHz and the CPU at 1.4 GHz:

     ```bash
     cat /sys/class/devfreq/5000000.mla/cur_freq                  # expect 1000000000
     cat /sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq  # expect 1400000
     ```

     When a clock reads lower, set it. Setting the CPU clock requires the
     `userspace` governor. The MLA command prints `1000000000` even when the
     DevKit keeps the lower clock, so read both clocks again afterwards:

     ```bash
     echo 1000000000 | sudo tee /sys/class/devfreq/5000000.mla/userspace/set_freq
     echo 1400000 | sudo tee /sys/devices/system/cpu/cpufreq/policy0/scaling_setspeed
     ```

     If the MLA clock stays low, `cat /sys/class/devfreq/5000000.mla/max_freq`
     shows the limit the DevKit started with. Restart it with `sudo reboot` and
     read the clock again, until it prints `1000000000`.
   - From the machine that holds the build, copy the model's scripts,
     `tools/model_report`, and the MPKs to the DevKit's NVMe drive, which keeps
     large files off the root filesystem. Copy the whole model folder, because
     `accuracy.py` can import another file from its `scripts/` folder:

     ```bash
     tar -czf - models/<model-id> tools/model_report <build-folder> \
       | ssh <user>@<devkit-address> 'mkdir -p /media/nvme/neat-tests/<run> && tar -xzf - -C /media/nvme/neat-tests/<run>'
     ```

   - When the model has an accuracy dataset, extract it under
     `/media/nvme/datasets/<source id>` as
     [accuracy](references/accuracy.md#get-the-dataset) describes.

   Done when `~/pyneat/bin/python -c 'import pyneat; print(pyneat.__version__)'`
   prints a version, both clocks read their target, the run folder holds the
   MPKs, the scripts, and `tools/model_report`, and the dataset folder holds the
   split the scripts read.

2. **Write or reuse a performance script.** A published model's own
   `scripts/performance.py` is the script CI runs, so reuse it as it is. For
   your own model, start from [the performance script](references/performance-script.md),
   which gives the single-component and multi-component shapes and the
   benchmark input rules. Done when the script takes `--mpk`, `--output`,
   `--dataset <source id>=<directory>` when the model has an accuracy dataset,
   and `--resources` when it declares supporting files, measures one fixed
   input, and hands Core's result to `model_report.from_performance()` with a
   `workload` line.

3. **Run it.** From the run folder:

   ```bash
   set -o pipefail
   PYTHONPATH=tools ~/pyneat/bin/python models/<model-id>/scripts/performance.py \
     --mpk <path to the MPK> \
     --dataset <source id>=/media/nvme/datasets/<source id> \
     --output results/performance 2>&1 | tee performance.log
   ```

   A multi-component model takes one `--mpk <component>=<archive>` per
   component, in `model.yaml` order. A model with
   `scripts/requirements-performance.txt` needs those packages in a separate
   folder first, installed as step 5 installs the accuracy packages, with that
   folder added to `PYTHONPATH`. Done when the command exits with 0 and
   `results/performance/` holds `model-report.json` and Core's
   `benchmark-report.json`. Keep the log. A missing report is not a successful
   benchmark.

4. **Read the report.** [Metrics](references/metrics.md) lists every metric
   with its unit, better direction, and source. Read `metrics`, `workload`,
   `runtime.pyneat`, and each component's `mla_layer_stats` and
   `mla_stage_stats` status. Done when you have `throughput_fps`,
   `latency_avg_ms`, `avg_power_watts`, and `energy_per_inference_joules` with
   their units, the reason for any that is absent, the PyNeat version, and the
   workload line.

5. **Measure accuracy.** Run this step when the model has a
   `scripts/accuracy.py`. [Accuracy](references/accuracy.md) gives the script's
   arguments, the report it writes, and how to write one for your own model:

   ```bash
   ~/pyneat/bin/python -m pip install --no-deps --only-binary=:all: \
     --target site-packages -r models/<model-id>/scripts/requirements-accuracy.txt
   PYTHONPATH=tools:site-packages ~/pyneat/bin/python models/<model-id>/scripts/accuracy.py \
     --mpk model=<path to the MPK> \
     --dataset <source id>=/media/nvme/datasets/<source id> \
     --output results/accuracy
   ```

   The `pip` command needs internet access on the DevKit, and a model without
   `requirements-accuracy.txt` needs no `pip` command. `--limit <images>` gives a
   quick trial on a subset. Done when `results/accuracy/model-report.json` holds
   a result for each dataset, with every metric that dataset requires, from a
   run without `--limit`, or with `--limit` set to `accuracy.samples` when the
   model's `model.yaml` sets it, because CI passes that number.

6. **Compare with a baseline.** The baseline is the published model page, the
   expected values in the model's `tests/tests.yaml`, or an earlier run of your
   own. [Comparable measurements](references/comparability.md) lists what must
   match and the fields to compare. Done when each compared pair matches on MPK,
   runtime, clocks, workload, and dataset samples, or the mismatch is named
   beside the numbers, and each result states what it measured, in the form
   `<model-id> <profile>: throughput_fps <value>, MLA 1 GHz, CPU 1.4 GHz, PyNeat <version>`.
