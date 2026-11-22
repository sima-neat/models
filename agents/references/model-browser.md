# Model browser code

How the model browser's code is organized: which module draws what, and how one build
serves the customer view and the developer view. Read it before you add a section to the
model page, move code between modules under `artifact-browser/`, or hide something from
customers. The rules, checks, and deployment steps are in
[the browser README](../../artifact-browser/README.md). What a visitor sees is in
[the model browser page](../../docs/model-browser.md).

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
| `audience.js` | the list of what only the developer view shows | `audienceFacts` |
| `state.js` | shared page state, configuration, and the audience facts of this build | `shows`, `state`, `elements` |
| `model-view.js` | the model page: header, variants, overview, component rows, Details, license, the loading skeleton, and the error panel | `renderDetail`, `detailSkeleton`, `errorPanel` |
| `performance-section.js` | Performance: the tiles, `DETAIL_CARDS`, power rails, the CSV, and the generative benchmark | `performanceSummary`, `generativeBenchmark` |
| `execution-section.js` | Execution Graph with its tooltip, Runtime Layer Analysis, and MLA Utilization | `executionGraphSection`, `layerAnalysisSection`, `renderMlaUtilization` |
| `accuracy-section.js` | Accuracy: measured scores, coverage, published and FP32 baseline scores | `accuracyPanel`, and `accuracyHeadline`, which Compare and the catalog also use |
| `page-parts.js` | markup and helpers that sections share | `tile`, `keyValue`, `infoMark`, `componentSwitch`, `measurementSource`, `trustedHttpsUrl` |
| `download-dialog.js` | the download dialog and its state | `downloadVariant`, `downloadPackage`, `downloadArtifact` |

A section module never imports `model-view.js` or another section, so the model page has
no import cycle. Markup that two sections need goes in `page-parts.js`. The README's
[module table](../../artifact-browser/README.md#how-the-page-is-built) lists every module of
the browser, including the catalog, Compare, and report readers.

When you add a module, add it to `fingerprintedAssets` in
`website/scripts/assemble-portal.mjs`, to the README's module table, and to the table
above when it belongs to the model page.

## Customer view and developer view

The build selects one audience through `uiMode`: `developer` by default, and `customer`
only when `MODEL_BROWSER_UI_MODE=customer` is set (`portalConfig` in
`website/scripts/portal-config.mjs`). No branch selects it. The customer view is the developer view
without some parts. `audienceFacts` in `audience.js` names each of those parts once:

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

Every fact equals `developer` because nothing is customer-only. A part that only
customers see would be a line such as `customerNotice: !developer`.

| Fact | What only the developer view shows |
| --- | --- |
| `buildHistory` | build history, and a link to one registry run |
| `compilerFilter` | the compiler filter of the catalog |
| `buildDetails` | the Details section of a model page: components, build provenance, package and source inputs |
| `measurementSource` | the commit and workflow that measured a result |
| `rawResult` | the model report as the registry stores it |
| `boardMemory` | memory every process on the board uses. `api.js` removes that reading from a performance report when the fact is false |
| `mlaUtilization` | MLA Utilization under Performance |
| `evaluationDetails` | how an accuracy run was evaluated |
| `referenceTables` | the tables of published and FP32 baseline scores |
| `commandBranch` | the `--branch` option in the copied CLI command. The customer command names no branch and downloads from `main` |

Three customer-view rules live outside this list, and the header of `audience.js` names
them. `styles.css` hides `.developer-only` markup, and the branch and compiler tags.
`environment.js` keeps customers on Main.

### Hide something new from customers

1. Add one named fact to `audienceFacts` in `audience.js`, with a comment that says what
   it covers.
2. Test `shows.<fact>` where the markup is built. Page code never reads `uiMode` or tests
   the audience itself.
3. Add the fact to the table above, and describe the visible difference in
   [the model browser page](../../docs/model-browser.md).
4. Run the checks in the README for both audiences: `MODEL_BROWSER_UI_MODE=customer` and
   the default `developer`.
