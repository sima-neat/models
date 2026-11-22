# Download a model with sima-cli

At the end of this page you have a model package on your machine, and you have read
its measured results in the terminal. Use it when you want a model from the
[Model Browser](https://developer.sima.ai/models/) without opening the browser.

The examples use RF-DETR Nano (`rf_detr_nano`) and its `w8a16_w16a16` variant. A
variant is one build profile of a model.

## Before you start

- `sima-cli` is installed and has the `models` commands of this page. The help of
  `models list` names `--select-branch`:

  ```bash
  sima-cli models list --help
  ```

- You need no account and no credentials. The commands read the public Model Registry.

## Steps

1. Find the model. The command reads the Main catalog and opens a list of the models
   that match your words.

   ```bash
   sima-cli models list -q rf_detr_nano
   ```

   You see RF-DETR Nano (`rf_detr_nano`) with the number of its variants. Press Enter
   to select it, then select the variant.

2. Read the model card. It shows the model's details, the files of its package with
   their sizes and SHA-256 values, and its validation, accuracy, and benchmark results.
   For RF-DETR Nano, **Accuracy** lists the COCO scores and **Benchmark** lists
   throughput, latency, power, and memory on a Modalix DevKit. For an LLM or VLM,
   **Benchmark** lists time to first token, prefill speed, and decode speed for each
   prompt length.

3. Answer `y` at `Download this model artifact?` to download the package. Answer `n`
   to leave.

4. To download without the list, name the model and the variant. Write the variant ID
   in lowercase.

   ```bash
   sima-cli models download --id rf_detr_nano --variant w8a16_w16a16
   ```

   You see the path of the package:

   ```text
   Downloaded to models/rf_detr_nano/w8a16_w16a16/rf_detr_nano_w8a16_w16a16_default.zip
   ```

   The ZIP holds one MPK per model component and the model's license file. An LLM, a
   VLM, or a speech model downloads as a folder of files instead of a ZIP.

5. To save the package somewhere else, add `--output`. The command creates
   `<model id>/<variant>/` under that folder.

   ```bash
   sima-cli models download --id rf_detr_nano --variant w8a16_w16a16 --output ~/neat-models
   ```

## Use another branch

Both commands read the Main catalog. To read another branch, such as Develop, name it
with `--branch`:

```bash
sima-cli models list --branch develop
```

To choose the branch from a list, use `--select-branch`:

```bash
sima-cli models list --select-branch
```

To print the branches that have models:

```bash
sima-cli models branches
```

## If something fails

| Message | Cause | Fix |
|---|---|---|
| `Model 'lightstereo_s' is not downloadable: The LicenseRef-OpenStereo-Academic license does not permit redistribution.` | The model's license forbids redistribution. | Rebuild the model from its scripts. See [Getting started](getting-started.md). |
| `No completed model build found for model 'rf_detr_nano', variant 'W8A16_W16A16'.` | The model ID or the variant ID does not match a build on that branch. | Copy both IDs from `sima-cli models list`. Write the variant ID in lowercase. |
| `No published model catalog for branch 'my-branch'` | The branch has no catalog. | Run `sima-cli models branches` and choose a listed branch. |
| `Destination already exists: ... Use --force to replace it.` | You downloaded this package before. | Add `--force` to replace it, or use another `--output` folder. |
| `was blocked by the registry CDN (HTTP 403)` | The registry may be limiting your requests. | Wait a few minutes, then run the command again. |
| `Model selection requires an interactive terminal. Use --json to list models.` | `models list` ran without a terminal, for example in a script. | Add `--json` to print the list, then use `models download`. |

## Next

To rebuild a model from its scripts and measure it yourself, read
[Getting started](getting-started.md).
