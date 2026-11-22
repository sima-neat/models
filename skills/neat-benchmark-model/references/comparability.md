# Comparable measurements

Two measurements compare only when everything that decides the result is the
same. The Models repository's published results fix each of these:

| Must match | Why | Where to check |
| --- | --- | --- |
| The MPK | Published results differ between build profiles. An INT8 build calibrated on other images is a different model. | the archive name, and its build profile |
| The runtime | The DevKit runs the platform version and Core package that `deps/manifest.json` pins. Component reports must share one runtime. | `runtime.pyneat` and `runtime.build` in the report |
| The MLA and CPU clocks | Throughput and power depend on both clocks, because the CPU feeds the MLA and runs postprocessing. Model outputs do not depend on them. | the two clock reads before the run, MLA 1 GHz and CPU 1.4 GHz |
| The workload | Preprocessing, the frame, and decoder settings are part of the measurement. | the report's `workload` line |
| The measurement settings | They size the latency pass and the throughput window. | `num_samples` 1000 and `min_duration_ms` 60000 in the script |
| The scope | A whole-model report and a per-component report measure different things. | a per-component report sets `scope: components` and holds `metrics` under each component. A whole-model report has no `scope` field and holds `metrics` at its top level. |
| The telemetry source | An older Sentinel reports no EV74 memory. | the presence of `ev74_memory_avg_mib` |
| For accuracy, the dataset and its samples | A `--limit` run scores a subset. A score compares with a reference only for the same dataset `id`. | each dataset's `split`, `evaluated_samples`, and `available_samples` |

Results measured before the scripts took a dataset image used Core's synthetic
inputs without image preprocessing, and YOLO26 then decoded up to 100
detections at any score. Do not compare those results with current ones. Older
reports also counted MLA memory differently, including buffers EV74 hands to
the MLA.

## Compare with a published model page

Open the model's page in the [Model Browser](https://developer.sima.ai/models/)
and select the build profile's tab. Compare these fields:

| Report | Field in `model-report.json` | Label on the model page |
| --- | --- | --- |
| performance | `metrics.throughput_fps` | Throughput |
| performance | `metrics.latency_avg_ms` | Average latency |
| accuracy | `datasets["<source id>"].metrics.<metric>` | the metric's label, such as mAP 50–95 for `coco_map`, shown as a percentage |

The page's SoC power tile adds up separate power readings and matches no single
field of the report, so compare throughput, latency, and accuracy.

## Compare with the expected values

Each published model's `tests/tests.yaml` in the Models repository holds the
expected values of each build profile, taken from the latest published results
and rounded to 4 decimal places for accuracy and 1 for throughput. YOLO26n
expects `coco_map: 0.3808` for `w8a8` on `coco-2017-val`.

CI tests a stage's values only when its entry in `tests.yaml` leaves
`run_tests` at `true`. An entry with `run_tests: false` gives a `skip_reason`
instead. The Models tests fail an accuracy score more than 5% below its expected
value, and a throughput more than 10% below. They also fail an accuracy score
more than 8% below the model's reference score for that dataset: its FP32
baseline, else the score its authors published for the same split. A metric
whose better direction is lower may rise by that much instead. Every model
shares these limits, so a result inside them passes the model's test when that
test runs.
