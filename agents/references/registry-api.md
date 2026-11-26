# Registry API

How this repository talks to the Vulcan artifact registry: its two endpoints, how a request is signed and sent,
how a run is published, and every endpoint each command or page calls. Read it before you add a registry call,
change a request body, or move code that sends one. What each run records is in [publication](publication.md),
and which job calls each command is in [CI flow](ci-flow.md).

## The registry

The registry is a JSON API in REST style under `/v1`. Paths name resources, `GET` reads, `POST` creates or runs an
action such as `finalize`, and `PUT` sets one part of a run. Files live in the registry's versioned S3 bucket. The
API stores each file's key, exact version, SHA-256, and size, and returns a presigned `download_url` when it is
read.

Each Vulcan environment (`staging`, `production`) serves two endpoints:

| Endpoint | Serves | Where its URL comes from |
| --- | --- | --- |
| public | unsigned reads | `public_url` in `config/registry.json`, and the GitHub environment variable `VULCAN_ARTIFACT_REGISTRY_PUBLIC_READ_URL` |
| IAM API | every write, and the reads that need credentials | the GitHub environment variable `VULCAN_ARTIFACT_REGISTRY_API_URL` |

`config/registry.json` holds only what may be published. The IAM API, bucket, KMS key, and roles stay in each
GitHub environment's variables, and the workflows pass them to each command. When a public endpoint moves,
change `config/registry.json` and the matching GitHub variable together.

## Code

Every registry request from Python is sent by `tools/model_registry/registry_api/`:

| Module | Owns |
| --- | --- |
| `config.py` | `public_url(root, environment)`, read from `config/registry.json`. |
| `transport.py` | `SigV4RegistryClient`, which signs with `curl --aws-sigv4` and passes the credentials on curl's stdin, never as arguments. `read_registry_json`, which signs a read only under `MODEL_REGISTRY_IAM_READ_API_URL`. `canonical_sha256`, `RegistryRequestError`, `is_catalog_conflict`, and the retried statuses. |
| `storage.py` | `S3ArtifactStore.ensure_native_sha256`, which uploads a file once with a native SHA-256 and returns its exact version. |
| `runs.py` | `REPOSITORY_PATH`, `create_run`, and `OpenRun`, which sends each request of a run with the key `<stable_key>-<step>`. |

A signed request makes up to five attempts on a failed connection and on HTTP 429, 502, 503, and 504, with exponential
backoff. An unsigned read retries the same statuses with jitter, because every build's benchmark reads at once.

Two callers build their requests outside `runs.py`:

- `report_publication.py` sends performance and accuracy runs through `SigV4RegistryClient` without `OpenRun`. It is a benchmark
  fingerprint input, so moving it onto `OpenRun` reruns every benchmark. Move it when a benchmark rerun is already
  due.
- `model-branch-lifecycle.yml` purges or restores a deleted or recreated branch with curl, on a runner that has no
  checkout of this repository.

## Publishing a run

A publisher derives a `stable_key` from what it publishes and the workflow run ID, then:

1. `create_run(client, body, stable_key)` creates the run, or returns the run an earlier attempt created under the
   same key.
2. `run.put("inputs/<name>" | "parameters" | "test-results/<suite>/<name>", body, step)` sets each part.
3. `store.ensure_native_sha256(...)` uploads a file, and `run.put_artifact(name, bucket, stored, record, step)`
   registers that exact S3 version.
4. `run.add_measurements(measurements, step)` and then `run.finalize()`.

A deployment file's key is `file-` and the `canonical_sha256` of the run's key and the file's artifact name, because a
run holds many files. Keep each `step` name and key format when you move a request. The registry matches a retried request by its idempotency key, so a
renamed step creates a second record instead of resuming the first.

## Endpoints

Repository paths start with `REPOSITORY_PATH`, `/v1/repositories/sima-neat/models`.

| Endpoint | Signed | Called by |
| --- | --- | --- |
| `POST /v1/runs`, then `PUT /v1/runs/{id}/parameters`, `/inputs/{name}`, `/artifacts/{name}`, `/test-results/{suite}/{name}`, `POST /measurements/batch`, `POST /finalize` | yes | `publish`, `publish-package`, `publish-catalog` (documentation), `publish-conversion`, `plan-stages` (compile plan), `publish-selection` (qualification), `publish-report` |
| `GET /v1/runs/{id}/provenance` | no | `plan-stages` (compile plan), the model browser |
| `GET /v1/runs/{id}`, `/test-results`, `/artifacts/{name}/content` | no | the model browser |
| `GET /v1/runs/{id}/artifacts/{name}` | no, or yes under `MODEL_REGISTRY_IAM_READ_API_URL` | `fetch` (conversions), `plan-stages`, `stage-registry-build`, `download-build`, `publish-catalog`, the model browser |
| `GET {repository}/catalogs/{revision}` | no | the model browser |
| `GET {repository}/catalogs/current?git_ref=` | no | `plan-stages`, `publish-catalog`, `publish-branch-deployment`, `scripts/model_promotion.py`, `check-main-merge`, `plan-promoted-benchmarks`, the model browser |
| `POST {repository}/catalogs/{revision}/snapshots` | yes | `publish-catalog`, `scripts/model_promotion.py` (benchmark refresh) |
| `GET {repository}/resolve?run_type=&fingerprint=` | no | `plan-stages`, the model browser |
| `GET {repository}/runs`, `/branches` | no | the model browser |
| `GET {repository}/models/filters`, `POST {repository}/catalog-selections`, `/catalog-selections/default` | read no, write yes | `publish-selection` |
| `POST {repository}/deployments`, `GET` and `PUT /runs/{id}` and `POST /finalize` under `{repository}/deployments/{id}`, `POST {repository}/branch-deployments` | yes | `publish-candidate`, `publish-branch-deployment` |
| `POST {repository}/promotions`, `GET {repository}/promotions/{id}`, `POST .../{approve,reject,cancel,withdraw}` | yes | `scripts/model_promotion.py` |
| `POST {repository}/branches/{purge,restore}`, `GET /v1/purge-jobs/{id}` | yes | `model-branch-lifecycle.yml` |

The model browser reads with unsigned `fetch` from `artifact-browser/api.js`. `website/scripts/portal-config.mjs`
takes both public URLs from `config/registry.json` at build time.
