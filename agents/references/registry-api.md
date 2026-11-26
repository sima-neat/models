# Registry API

How this repository talks to the Vulcan artifact registry: its two endpoints, where each setting comes from, how a
request is signed and sent, and every endpoint each command or page calls. Read it before you add a registry call,
change a request body, or move code that sends one. What each run records is in [publication](publication.md),
and which job calls each command is in [CI flow](ci-flow.md).

## The registry

The registry is a JSON API in REST style under `/v1`. Paths name resources, `GET` reads, `POST` creates or runs an
action such as `finalize`, and `PUT` sets one part of a run. Files live in the registry's versioned S3 bucket. The
API stores each file's key, exact version, SHA-256, and size, and returns a presigned `download_url` when it is
read.

Each Vulcan environment (`staging`, `production`) serves two endpoints: a public endpoint for unsigned reads, and an
IAM API for every write and for the reads that need credentials.

## Settings

| Setting | Comes from |
| --- | --- |
| AWS region, buckets, KMS keys | the `resolve-vulcan-config` jobs, which call `sima-neat/.github/.github/workflows/vulcan-resolve-config.yml` with the environment and pass its outputs, such as `aws_region`, to each job |
| IAM API URL, publisher and promotion roles | the GitHub environment variables `VULCAN_ARTIFACT_REGISTRY_API_URL`, `VULCAN_ARTIFACT_REGISTRY_PUBLISHER_ROLE_ARN`, and `VULCAN_ARTIFACT_REGISTRY_PROMOTION_ROLE_ARN` |
| public endpoint in CI | the GitHub environment variable `VULCAN_ARTIFACT_REGISTRY_PUBLIC_READ_URL` |
| public endpoint in the model browser and `check-main-merge` | `website/scripts/portal-config.mjs`, `artifact-browser/config.js`, `artifact-browser/environment.js`, and `scripts/check_main_merge.sh` |

Read the region from the resolved config. A workflow that hard-codes it signs for the wrong region when Vulcan moves
an environment.

## Sending a request

`SigV4RegistryClient` in `registry_http.py` sends every signed request from Python. It signs with
`curl --aws-sigv4` and gives curl the credentials through `--config -` on stdin, because any process on the runner
can read curl's arguments. A body goes through a temporary file. Each request carries an `idempotency-key`, and a
run's requests use `<stable_key>-<step>`, so a retried job resumes the run it started. Keep each step name and key
format when you move a request, because a renamed step creates a second record. A signed request makes up to five
attempts on a failed connection and on HTTP 429, 502, 503, and 504, with exponential backoff.

Reads go through `read_registry_json_authenticated` in `registry_http.py`. It signs a read only under
`MODEL_REGISTRY_IAM_READ_API_URL`. Every other read goes to `read_registry_json` in `fetch.py` unsigned and retries
the same statuses with jitter, because every build's benchmark reads at once.

`model-branch-lifecycle.yml` sends its requests with curl in the workflow, because its job has no checkout of this
repository. It passes the credentials on stdin the same way.

## Code

| Module | Sends |
| --- | --- |
| `registry.py` | compile, package, documentation, fidelity, MLA stage simulation, and deployment runs, catalog snapshots, and the reads that stage a build. `S3ArtifactStore` uploads a file once with a native SHA-256 and returns its exact version |
| `report_publication.py` | performance and accuracy runs |
| `conversion.py` | conversion plan and publication runs |
| `stage_plan.py` | the reads that plan compiles and benchmarks |
| `catalog_selection.py` | the qualification run and catalog selections |
| `candidates.py` | on-demand candidate and branch deployments |
| `promotions.py`, `scripts/model_promotion.py` | promotions and the benchmark refresh |
| `fetch.py` | build file downloads on a DevKit |

## Endpoints

`{repository}` is `/v1/repositories/sima-neat/models`.

| Endpoint | Signed | Called by |
| --- | --- | --- |
| `POST /v1/runs`, then `PUT /v1/runs/{id}/parameters`, `/inputs/{name}`, `/artifacts/{name}`, `/test-results/{suite}/{name}`, `POST /measurements/batch`, `POST /finalize` | yes | `publish`, `publish-package`, `publish-catalog` (documentation), `publish-conversion`, `plan-stages` (compile plan), `publish-selection` (qualification), `publish-report` |
| `GET /v1/runs/{id}/provenance` | no | `plan-stages` (compile plan), the model browser |
| `GET /v1/runs/{id}`, `/test-results`, `/artifacts/{name}/content` | no | the model browser |
| `GET /v1/runs/{id}/artifacts/{name}` | no, or yes under `MODEL_REGISTRY_IAM_READ_API_URL` | conversion fetches, `plan-stages`, `stage-registry-build`, `download-build`, `publish-catalog`, the model browser |
| `GET {repository}/catalogs/{revision}` | no | the model browser |
| `GET {repository}/catalogs/current?git_ref=` | no | `plan-stages`, `publish-catalog`, `publish-branch-deployment`, `scripts/model_promotion.py`, `check-main-merge`, `plan-promoted-benchmarks`, the model browser |
| `POST {repository}/catalogs/{revision}/snapshots` | yes | `publish-catalog`, `scripts/model_promotion.py` (benchmark refresh) |
| `GET {repository}/resolve?run_type=&fingerprint=` | no | `plan-stages`, the model browser |
| `GET {repository}/runs`, `/branches` | no | the model browser |
| `GET {repository}/models/filters`, `POST {repository}/catalog-selections`, `/catalog-selections/default` | read no, write yes | `publish-selection` |
| `POST {repository}/deployments`, `GET` and `PUT /runs/{id}` and `POST /finalize` under `{repository}/deployments/{id}`, `POST {repository}/branch-deployments` | yes | `publish-candidate`, `publish-branch-deployment` |
| `POST {repository}/promotions`, `GET {repository}/promotions/{id}`, `POST .../{approve,reject,cancel,withdraw}` | yes | `scripts/model_promotion.py` |
| `POST {repository}/branches/{purge,restore}`, `GET /v1/purge-jobs/{id}` | yes | `model-branch-lifecycle.yml` |

The model browser reads with unsigned `fetch` from `artifact-browser/api.js`.
