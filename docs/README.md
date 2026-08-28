# Model Registry developer documentation

Use this documentation when adding a model or changing the model lifecycle. The
shortest path for a new model is:

1. Read [Adding a model](adding-a-model.md) and confirm that the source,
   license, and intended Modalix variants are suitable.
2. Use the [`model.yaml` reference](metadata-schema.md) to describe the model
   without relying on mutable upstream documentation.
3. Follow [Local development and testing](local-development.md) until the model
   and complete catalog validate locally.
4. Complete the checklist in [Contributing with Git and GitHub](contributing.md)
   before opening a pull request.

## Guides

| Guide | Use it for |
| --- | --- |
| [Adding a model](adding-a-model.md) | The end-to-end model contribution path |
| [`model.yaml` reference](metadata-schema.md) | Field intent, constraints, and examples |
| [Local development and testing](local-development.md) | Environment setup, validation, fetches, tests, and optional compilation |
| [Contributing with Git and GitHub](contributing.md) | Commit contents, pull requests, CI, and review evidence |
| [Model taxonomy](model-taxonomy.md) | Sanctioned category IDs and taxonomy changes |
| [Synthetic model benchmarking](benchmarking.md) | Benchmark manifests, DevKit execution, and results |
| [Architecture](architecture.md) | Repository boundaries, trust model, and publication lifecycle |

The JSON Schemas under [`schemas/`](https://github.com/sima-neat/models/tree/develop/schemas)
and the taxonomy files under
[`taxonomies/`](https://github.com/sima-neat/models/tree/develop/taxonomies) are authoritative. The guides explain how to
work with those contracts; they do not replace them.

## Where validation happens

| Stage | What it proves | Typical environment |
| --- | --- | --- |
| Local validation | YAML, schema, taxonomy, paths, source policy, checksums, matrix selection, and host-side tests | Developer workstation |
| Compilation | The pinned recipe produces exactly one valid MPK with the repository compiler version | Trusted Vulcan Model Compiler runner |
| Benchmarking | The compiled route executes on the pinned Neat and Modalix platform and produces performance evidence | Modalix DevKit runner |
| Promotion | Required evidence, provenance, licensing, and maturity are acceptable for release | Maintainer review and protected workflows |

A locally valid model is ready for review, not automatically qualified for
release. Pull requests never execute contributor-controlled model recipes on a
self-hosted runner.
