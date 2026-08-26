# Model Artifact Browser

A client-only web interface for the public SiMa.ai Model Registry read APIs.
The browser calls the registry directly; there is no application backend,
credential handling, or proxy middleware. Downloads use the temporary URL
returned by the registry directly.

Run from the Models repository:

```bash
npx --yes serve artifact-browser --listen 4173
```

Then open <http://127.0.0.1:4173>. Any static file server works. To use another
registry environment, change `apiBase`, `environment`, and `repository` in
`config.js` before serving the directory:

```js
window.MODEL_REGISTRY_CONFIG = Object.freeze({
  apiBase: "https://example.test",
  environment: "Production",
  repository: Object.freeze({ organization: "sima-neat", name: "models" }),
});
```

To preview the complete ResNet-50 schema-v2 benchmark UI without a registry or
DevKit connection, run:

```bash
python3 scripts/preview_artifact_browser.py
```

Open the URL printed by the script. The local server provides a realistic model
compile, provenance, validation, and three-trial benchmark result. The fixture is
stored in `artifact-browser/fixtures/performance-result-v2.json` and can be edited
to exercise other metric values or availability states.

## Staging deployment

Changes to `artifact-browser/**` on `main` are published by
`vulcan-model-browser.yml` to the staging docs bucket under `models/`. The
Developer Center CloudFront distribution routes that prefix to the docs bucket,
making the browser available at <https://developer-stg.sima.ai/models>.

The workflow can also be started manually. It resolves the bucket, publisher
role, KMS key, and CloudFront distribution from the shared staging Vulcan
configuration; those deployment values are not hard-coded in this repository.

The default is the WAF-protected public staging registry at
`https://models-registry.stg.neat.sima.ai`. The API must allow cross-origin
`GET` requests from the static app's origin. The separate
`artifacts-registry.stg.neat.sima.ai` endpoint is reserved for IAM-signed
publisher traffic and rejects direct public reads.

The initial view uses one paginated runs request and saves the latest catalog in
browser storage so repeat visits can render immediately while refreshing. Full
provenance, validation, and metric views load only after a build is selected.
Public discovery requests completed builds only. The primary search matches
model and variant names together with category names; secondary branch,
compiler, type, task, architecture, paradigm, and input facets are collapsed by
default and display an active-filter count when used.
The branch filter defaults to `main`. A branch selected by the user is saved per
registry and repository in browser storage; if that branch disappears, the
portal falls back to `main` (or all branches when `main` is unavailable).
Pagination is branch-aware, so **Load earlier builds** is shown only when the
registry reports another page for the currently selected branch.
