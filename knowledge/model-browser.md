# Model browser code

How the model browser's code is organized: which module draws what, and how one build
serves the customer view and the developer view. Read it before you add a section to the
model page, move code between modules under `artifact-browser/`, or hide something from
customers. The rules, checks, and deployment steps are in
[the browser README](../artifact-browser/README.md). What a visitor sees is in
[the model browser page](../docs/model-browser.md).

## Module layout

Each module is named after what it draws. `app.js` routes and handles events,
`model-view.js` assembles the model page, and each section of that page has its own
module.

```text
state.js ── shows = audienceFacts(uiMode) ◄── audience.js

app.js  (routing, events)
  ├─► download-dialog.js
  └─► model-view.js  (assembles the model page)
        ├─► performance-section.js ─┐
        ├─► execution-section.js   ─┼─► page-parts.js
        └─► accuracy-section.js    ─┘
```

| Module | Holds | Main exports |
| --- | --- | --- |
| `audience.js` | the list of what only one view shows | `audienceFacts` |
| `state.js` | shared page state, configuration, and the audience facts of this build | `shows`, `state`, `elements` |
| `model-view.js` | the model page: header, variants, overview, component rows, Details, license, the loading skeleton, and the error panel | `renderDetail`, `detailSkeleton`, `errorPanel` |
| `performance-section.js` | Performance: the tiles, `DETAIL_CARDS`, power rails, the CSV, and the generative benchmark | `performanceSummary`, `generativeBenchmark` |
| `execution-section.js` | Execution Graph with its tooltip, Runtime Layer Analysis, and MLA Utilization | `executionGraphSection`, `layerAnalysisSection`, `renderMlaUtilization` |
| `accuracy-section.js` | Accuracy: measured scores, coverage, published and FP32 baseline scores | `accuracyPanel`, and `accuracyHeadline`, which Compare and the catalog also use |
| `page-parts.js` | markup and helpers that sections share | `tile`, `keyValue`, `infoMark`, `componentSwitch`, `measurementSource`, `trustedHttpsUrl` |
| `download-dialog.js` | the download dialog and its state | `downloadVariant`, `downloadPackage`, `downloadArtifact` |

A section module never imports `model-view.js` or another section, so the model page has
no import cycle. Markup that two sections need goes in `page-parts.js`. The README's
[module table](../artifact-browser/README.md#how-the-page-is-built) lists every module of
the browser, including the catalog, Compare, and report readers.

When you add a module, add it to `fingerprintedAssets` in
`website/scripts/assemble-portal.mjs`, to the README's module table, and to the table
above when it belongs to the model page.

## Customer view and developer view

The build selects one audience through `uiMode`: `customer` on `main`, `developer`
elsewhere (`website/scripts/portal-config.mjs`). The customer view is the developer view
without some parts, and with the version filters. `audienceFacts` in `audience.js` names
each of those parts once:

```js
export function audienceFacts(uiMode) {
  const developer = uiMode !== "customer";
  return { rawResult: developer, buildDetails: developer, /* ... */ };
}
```

`developer` is a true or false value, not a view. It is true in the developer build and
false in the customer build, so the same function serves both. `state.js` calls it once
and exports the result as `shows`. Page code asks for one fact by name where it builds
the markup:

```js
const raw = shows.rawResult ? `<button ...>Raw result</button>` : "";
```

| Build | `developer` | `shows.rawResult` | Result |
| --- | --- | --- | --- |
| developer | `true` | `true` | the Raw result button is drawn |
| customer | `false` | `false` | the button is left out |

Every fact but one equals `developer`. `versionFilters` is `!developer`, because only
customers see the version filters.

| Fact | What only the developer view shows |
| --- | --- |
| `buildHistory` | build history, and a link to one registry run |
| `compilerFilter` | the compiler filter that lists the compilers of the branch's builds |
| `buildDetails` | the Details section of a model page: components, build provenance, package and source inputs |
| `measurementSource` | the commit and workflow that measured a result |
| `rawResult` | the model report as the registry stores it |
| `boardMemory` | memory every process on the board uses. `api.js` removes that reading from a performance report when the fact is false |
| `mlaUtilization` | MLA Utilization under Performance |
| `evaluationDetails` | how an accuracy run was evaluated |
| `referenceTables` | the tables of published and FP32 baseline scores |
| `lowAccuracyVariants` | variants whose accuracy falls more than 8% below the source model's reference score. `api.js` removes them from the catalog when the fact is false |
| `commandBranch` | the `--branch` option in the copied CLI command. The customer command names no branch and downloads from `main` |

| Fact | What only the customer view shows |
| --- | --- |
| `versionFilters` | the Platform, Neat Core and Compiler filters of the catalog |

Three customer-view rules live outside this list, and the header of `audience.js` names
them. `styles.css` hides `.developer-only` markup, and the branch and compiler tags.
`environment.js` keeps customers on Main. `styles.css` also shows the Compiler filter to
customers when the version filters are shown.

### Catalog versions

`catalog-versions.js` owns the published versions, the selected one, and the URL
parameters `platform_version`, `core_version` and `compiler`. Other modules ask it.

- `app.js` loads the registry's list once, when `shows.versionFilters` is true. The
  list comes from `GET .../models/filters`. When that request fails, the page has no
  version filters and reads the registry as before.
- A version is one `selections` entry of that list. The Platform and Neat Core
  filters are built from those entries, never from the top-level version arrays.
- With a version selected, `selectionId()` returns its ID. The catalog read and each
  benchmark `/resolve` read then send `selection_id`. The catalog read sends it in
  place of `git_ref`.
- Every cache that holds catalog or benchmark data has the selection ID in its key:
  the saved catalog, the benchmark results, and the variants in `state.variantCache`.
- A catalog document names no compiler for its builds. When a customer chooses a
  compiler, `catalogWithCompiler` in `api.js` reads
  `GET .../models?toolchain_version=<compiler>&limit=200` with `selection_id`, or with
  `git_ref` of Main for Latest, and follows `next_cursor`. It copies each listed build's
  `toolchain_version` onto the catalog's builds, so the existing compiler filter keeps
  the matching models and variants. The answer is kept per source and compiler. A
  failed request shows a toast and clears the compiler choice.
- The customer view never loads the branch's run history for this. That stays behind
  `shows.catalogRunMetadata`.

A new registry read that depends on the catalog must send `selectionId()` when it is
set, and a new cache must put it in its key.

### Hide something new from customers

1. Add one named fact to `audienceFacts` in `audience.js`, with a comment that says what
   it covers.
2. Test `shows.<fact>` where the markup is built. Page code never reads `uiMode` or tests
   the audience itself.
3. Add the fact to the table above, and describe the visible difference in
   [the model browser page](../docs/model-browser.md).
4. Run the checks in the README for both audiences: `MODEL_BROWSER_UI_MODE=customer` and
   the default `developer`.
