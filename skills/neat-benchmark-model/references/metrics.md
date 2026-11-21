# Metrics

[`metrics.json`](https://github.com/sima-neat/models/blob/main/tools/model_report/metrics.json)
in `tools/model_report` is the one metric vocabulary. Each row gives a metric's
label, unit, better direction, and the Core report field or telemetry statistic
it comes from. A report's `metrics` hold one number per measured metric, under
these names.

## Performance metrics

Every Core performance report must hold the four required metrics. A missing
required metric fails the stage in CI.

| Metric | Label | Unit | Better | Comes from |
| --- | --- | --- | --- | --- |
| `latency_avg_ms` (required) | Average latency | ms | lower | Core `latency_ms` |
| `latency_p50_ms` | Latency p50 | ms | lower | Core `latency_p50_ms` |
| `latency_p95_ms` | Latency p95 | ms | lower | Core `latency_p95_ms` |
| `latency_p99_ms` | Latency p99 | ms | lower | Core `latency_p99_ms` |
| `throughput_fps` (required) | Throughput | inferences/s | higher | Core `fps` |
| `avg_power_watts` (required) | SoM power | W | lower | mean of telemetry `power_som_watts` |
| `energy_per_inference_joules` (required) | Energy / inference | J/inference | lower | `avg_power_watts / throughput_fps` |
| `peak_power_watts` | Peak SoM power | W | lower | maximum of `power_som_watts` |
| `avg_mla_power_watts`, `peak_mla_power_watts` | MLA power | W | lower | mean and maximum of `power_mla_watts` |
| `avg_die_temperature_c`, `peak_die_temperature_c` | SoC temperature | °C | lower | the hottest on-die sensor at each sample |
| `avg_board_temperature_c`, `peak_board_temperature_c` | SoM temperature | °C | lower | the hotter of the top-side and bottom-side board sensors |
| `avg_cpu_usage_pct`, `peak_cpu_usage_pct` | CPU usage, all cores | % | lower | CPU usage averaged over all cores |
| `avg_cpu_core_usage_pct`, `peak_cpu_core_usage_pct` | CPU usage, busiest core | % | lower | the highest per-core average and the highest per-core peak, which can be different cores |
| `mla_memory_avg_mib`, `mla_memory_peak_mib` | MLA memory | MiB | lower | buffers from the MLA memory pool, plus shared buffers only the MLA uses |
| `ev74_memory_avg_mib`, `ev74_memory_peak_mib` | EV74 memory | MiB | lower | buffers EV74 holds, such as the frames it preprocesses for the MLA |
| `avg_linux_memory_mib`, `peak_linux_memory_mib` | Board Linux memory used | MiB | neutral | Linux `MemTotal - MemAvailable` for the whole board |

How the numbers are filled:

- Core reports mean latency and its p50, p95, and p99, throughput in logical
  inferences per second, and average SoM power.
- Throughput counts every inference in a batch, so `model_report` derives
  energy per inference by dividing average power by throughput alone. The model
  page shows it in mJ/inference.
- The telemetry metrics come from Sentinel, each over the throughput pass. A
  telemetry field with one entry per core or sensor reports its busiest entry.
- Core writes 0 for a headline value it did not measure, and Sentinel gives 0
  samples for a reading it did not take. Those metrics are absent from the
  report. A report without `avg_power_watts` means Sentinel gave no power
  readings.
- MLA and EV74 memory add up to the model's accelerator memory, because
  Sentinel gives each buffer one owner. Linux memory covers every process on the
  board, so it is board context, not the model's memory.
- EV74 memory needs a Sentinel that reports `ev74_mem_allocated_mb`. With an
  older daemon the report has no EV74 memory.
- Individual temperature sensors and CPU cores are not metrics. They stay in
  the report's `telemetry`.

## Accuracy metrics

Accuracy scores use the metric's unit. A ratio runs from 0 to 1. `abs_rel` is a
nonnegative ratio without an upper bound, `d1_all` is a percentage from 0 to
100, and `epe` is in pixels. `model_report.from_accuracy()` rejects an unknown
accuracy metric and a value outside its bounds.

| Metrics | Reported by | Unit | Better |
| --- | --- | --- | --- |
| `top1_accuracy`, `top5_accuracy` | image classification and image-text embedding on ImageNet 2012 | ratio | higher |
| `knn_top1_accuracy` | image embedding on ImageNet 2012 | ratio | higher |
| `coco_map`, `coco_ap50`, `coco_ap75`, `coco_ap_small`, `coco_ap_medium`, `coco_ap_large`, `coco_ar1`, `coco_ar10`, `coco_ar100`, `coco_ar_small`, `coco_ar_medium`, `coco_ar_large` | object detection on COCO 2017 | ratio | higher |
| `coco_mask_map`, `coco_mask_ap50`, `coco_mask_ap75`, `coco_mask_ap_small`, `coco_mask_ap_medium`, `coco_mask_ap_large`, `coco_mask_ar1`, `coco_mask_ar10`, `coco_mask_ar100`, `coco_mask_ar_small`, `coco_mask_ar_medium`, `coco_mask_ar_large` | instance segmentation on COCO 2017 | ratio | higher |
| `coco_box_map`, `coco_box_ap50` | the boxes of instance segmentation models | ratio | higher |
| `coco_keypoint_map`, `coco_keypoint_ap50`, `coco_keypoint_ap75`, `coco_keypoint_ap_medium`, `coco_keypoint_ap_large`, `coco_keypoint_ar`, `coco_keypoint_ar50`, `coco_keypoint_ar75`, `coco_keypoint_ar_medium`, `coco_keypoint_ar_large` | pose estimation on COCO 2017 | ratio | higher |
| `coco_person_box_map`, `coco_person_box_ap50` | the person boxes of pose models | ratio | higher |
| `dota_obb_map50`, `dota_obb_map` | oriented object detection on DOTA v1.0 | ratio | higher |
| `keypoint_repeatability`, `homography_accuracy` | feature point detection on HPatches | ratio | higher |
| `pose_auc_5`, `pose_auc_10`, `pose_auc_20`, `match_precision`, `matching_score` | feature matching on YFCC100M | ratio | higher |
| `jf_mean`, `j_mean`, `f_mean` | video object segmentation on DAVIS 2017 | ratio | higher |
| `recall_at_1`, `recall_at_5`, `recall_at_10` | loop closure on Pittsburgh 30k | ratio | higher |
| `miou` | semantic segmentation on ADE20K | ratio | higher |
| `delta1` | depth estimation on NYU Depth V2 | ratio | higher |
| `abs_rel` | depth estimation on NYU Depth V2 | ratio, no upper bound | lower |
| `epe`, `d1_all` | stereo depth on Scene Flow | px, % | lower |

## Language model metrics

Language and vision-language models report these per workload instead:
`ttft_ms` (time to first token, ms, lower), `prefill_tps` (tokens/s, higher),
`decode_tps` (generation speed, tokens/s, higher), `draft_contribution_pct` (%,
higher), and `decode_speedup` (×, higher).
