// Accuracy section of the model page: measured scores, coverage, and published and FP32 baseline scores.
import {
  REFERENCE_ORIGINS, accuracyDisplay, accuracyMetricLabel, datasetName, formatAccuracy, orderedMetricNames,
  referenceNotes, referenceOrigin, referenceScores, registryArchive,
} from "./accuracy-metrics.js";
import { formatDate } from "./metrics.js";
import { readAccuracy } from "./model-report.js";
import { shows } from "./state.js";
import { icons, escapeHtml, disclosure } from "./ui.js";
import {
  humanize, infoMark, keyValue, measurementSource, targetName, trustedHttpsUrl, variantSectionTitle,
} from "./page-parts.js";

// The first metrics in display order are shown as tiles; any further metrics are listed in a table.
const ACCURACY_SUMMARY_TILES = 4;

// The registry does not preserve a result's metric order, so known metrics follow the label order and
// any others follow them in the order they arrive.
function orderedAccuracyMetrics(metrics) {
  return orderedMetricNames(Object.keys(metrics)).map((name) => [name, metrics[name]]);
}

// The first accuracy metric of a build's first dataset and the source model's reference score, for the Compare
// page and the catalog.
export function accuracyHeadline(variant) {
  const result = readAccuracy(variant.accuracy);
  if (result.kind !== "supported" || !result.value.datasets.length) return null;
  const { metrics, dataset } = result.value.datasets[0];
  const [name, value] = orderedAccuracyMetrics(metrics)[0] || [];
  if (!name) return null;
  const display = accuracyDisplay(name, value);
  return {
    label: accuracyMetricLabel(name), value: formatAccuracy(name, value),
    reference: formatAccuracy(name, referenceScores(variant.reference, dataset.id)[name], { blank: true }),
    numericValue: display.number,
    unit: display.unit,
    direction: display.direction,
    dataset,
    comparisonKey: dataset.id ? JSON.stringify([dataset.id, dataset.variant, dataset.split, dataset.sha256, name]) : null,
  };
}

// The build profile's accuracy run, with the published and FP32 baseline scores on the same dataset.
export function accuracyPanel(variant, modelId) {
  const reference = variant.reference || [];
  const referenceMarkup = shows.referenceTables ? referenceDisclosures(reference) : "";
  const result = readAccuracy(variant.accuracy);
  if (result.kind === "missing") return referenceAccuracySection(reference, variant);
  if (result.kind === "unsupported") {
    return `<section id="accuracy-section" class="accuracy-section" tabindex="-1">
      ${variantSectionTitle("Accuracy", variant)}
      <p class="muted-message">This result format is not supported by this browser.</p>
      ${referenceMarkup}
    </section>`;
  }
  const { datasets, tool, timestamp } = result.value;
  const measured = `${timestamp ? `Measured ${escapeHtml(formatDate(timestamp, false))}` : ""}${measurementSource(variant.accuracy.run, modelId)}`;
  const target = targetName(variant.accuracy.run);
  const results = datasets.map((entry) => accuracyResult(
    entry, tool, target, datasetReferences(reference, entry.dataset.id), measured, datasets.length > 1,
  )).join("");
  // One dataset is described beside the section title; several each describe their own result.
  return `<section id="accuracy-section" class="accuracy-section" tabindex="-1">
    ${variantSectionTitle("Accuracy", variant, datasets.length === 1 ? evaluatedOn(datasets[0].dataset, target) : "")}
    ${results}
    ${referenceMarkup}
  </section>`;
}

// Each origin's reference scores on a dataset, keyed by origin.
function scoresByOrigin(reference, datasetId) {
  return Object.fromEntries(Object.keys(REFERENCE_ORIGINS).map((origin) => [origin, referenceScores(reference, datasetId, origin)]));
}

// Where each origin's reference scores on a dataset come from, for the origins that name an https source.
function referenceSources(reference, datasetId) {
  return Object.fromEntries(Object.keys(REFERENCE_ORIGINS).map((origin) => {
    const entry = reference.find((item) => item.dataset?.id === datasetId && referenceOrigin(item) === origin);
    return [origin, trustedHttpsUrl(entry?.source)?.href];
  }));
}

// What a dataset's result is compared with: each origin's reference scores and their sources.
function datasetReferences(reference, datasetId) {
  return { scores: scoresByOrigin(reference, datasetId), sources: referenceSources(reference, datasetId) };
}

// One origin's note under an accuracy tile, such as "Published 39.50%", linked to its source when it has one.
function referenceNote(name, origin, scores, sources) {
  const [note] = referenceNotes(name, { [origin]: scores[origin] });
  if (!note) return "";
  const text = escapeHtml(note);
  return `<small>${sources[origin]
    ? `<a class="measurement-link" href="${escapeHtml(sources[origin])}" target="_blank" rel="noreferrer">${text}${icons.external}</a>`
    : text}</small>`;
}

function accuracyMetricTiles(metrics, scores, sources) {
  return orderedAccuracyMetrics(metrics).slice(0, ACCURACY_SUMMARY_TILES).map(([name, value], index) => {
    const notes = Object.keys(REFERENCE_ORIGINS).map((origin) => referenceNote(name, origin, scores, sources)).join("");
    return `<div class="accuracy-metric${index === 0 ? " primary" : ""}"><span>${escapeHtml(accuracyMetricLabel(name))}</span><strong>${escapeHtml(formatAccuracy(name, value))}</strong>${notes || "<small></small>"}</div>`;
  }).join("");
}

// The Published column always shows; the FP32 baseline column only when the model has a baseline here.
function accuracyMetricTable(metrics, scores) {
  const origins = Object.keys(REFERENCE_ORIGINS).filter((origin) => origin === "published" || Object.keys(scores[origin]).length);
  const rows = orderedAccuracyMetrics(metrics).slice(ACCURACY_SUMMARY_TILES).map(([name, value]) => (
    `<tr><th scope="row">${escapeHtml(accuracyMetricLabel(name))}</th><td>${escapeHtml(formatAccuracy(name, value))}</td>${origins.map((origin) => `<td>${escapeHtml(formatAccuracy(name, scores[origin][name]))}</td>`).join("")}</tr>`
  )).join("");
  if (!rows) return "";
  const headers = origins.map((origin) => `<th scope="col">${escapeHtml(REFERENCE_ORIGINS[origin].label)}</th>`).join("");
  return `<div class="metric-table"><h4>Additional metrics</h4><div class="metric-table-wrap"><table><thead><tr><th scope="col">Metric</th><th scope="col">Value</th>${headers}</tr></thead><tbody>${rows}</tbody></table></div></div>`;
}

// Conditions are written as a phrase, such as "640x640, FP32"; notes may already end a sentence.
function sentence(text = "") {
  return !text || /[.!?]$/.test(text) ? text : `${text}.`;
}

// One row per origin the model has, listing that origin's entries.
function referenceDisclosures(reference) {
  return Object.entries(REFERENCE_ORIGINS).map(([origin, { title, id }]) => {
    const entries = reference.filter((entry) => referenceOrigin(entry) === origin);
    if (!entries.length) return "";
    return disclosure(id, title, referenceTables(entries), `${entries.length} ${entries.length === 1 ? "source" : "sources"}`);
  }).join("");
}

// Each reference entry with the dataset and conditions it was measured on, and where it came from.
function referenceTables(reference) {
  return reference.map((entry) => {
    const { label } = REFERENCE_ORIGINS[referenceOrigin(entry)] || REFERENCE_ORIGINS.published;
    const metrics = entry.metrics || {};
    const rows = orderedMetricNames(Object.keys(metrics)).map((name) => (
      `<tr><th scope="row">${escapeHtml(accuracyMetricLabel(name))}</th><td>${escapeHtml(formatAccuracy(name, metrics[name]))}</td></tr>`
    )).join("");
    const source = trustedHttpsUrl(entry.source);
    const link = source
      ? ` <a class="measurement-link" href="${escapeHtml(source.href)}" target="_blank" rel="noreferrer">Source${icons.external}</a>`
      : entry.script ? ` Measured by ${escapeHtml(entry.script)}.` : "";
    return `<div class="metric-table published-scores"><h4>${escapeHtml(`${entry.dataset?.id || ""} ${entry.dataset?.split || ""}`.trim())}</h4>
      <p class="disclosure-text">${escapeHtml(sentence(entry.conditions))}${entry.note ? ` ${escapeHtml(sentence(entry.note))}` : ""}${link}</p>
      <div class="metric-table-wrap"><table><thead><tr><th scope="col">Metric</th><th scope="col">${escapeHtml(label)}</th></tr></thead><tbody>${rows}</tbody></table></div></div>`;
  }).join("");
}

// A build without an accuracy result still shows its reference scores, each origin under its own explanation.
const REFERENCE_SCOPES = {
  published: "The model's authors published these scores for the source model.",
  fp32_baseline: "The FP32 source model scores these on the package's own inputs.",
};

function referenceAccuracySection(reference, variant) {
  if (!reference.length) return "";
  const groups = Object.keys(REFERENCE_ORIGINS).map((origin) => {
    const entries = reference.filter((entry) => referenceOrigin(entry) === origin);
    return entries.length ? `<p class="accuracy-scope">${escapeHtml(REFERENCE_SCOPES[origin])}</p>${referenceTables(entries)}` : "";
  }).join("");
  return `<section id="accuracy-section" class="accuracy-section" tabindex="-1">
    ${variantSectionTitle("Accuracy", variant)}
    <p class="accuracy-scope">No accuracy result for this build yet.</p>
    ${groups}
  </section>`;
}

// A run that evaluated every sample of its split has full coverage; any other run evaluated a sampled subset.
function fullCoverage(dataset) {
  return Number.isInteger(dataset.evaluated_samples) && dataset.evaluated_samples === dataset.available_samples;
}

// How much of the dataset the run evaluated, such as "using 5,000 samples" or "using 500 of its 5,000 samples".
function sampleCount(dataset) {
  if (!Number.isInteger(dataset.evaluated_samples) || !Number.isInteger(dataset.available_samples)) return "";
  const evaluated = dataset.evaluated_samples.toLocaleString();
  return fullCoverage(dataset)
    ? `using ${evaluated} samples`
    : `using ${evaluated} of its ${dataset.available_samples.toLocaleString()} samples`;
}

// How an accuracy result was measured, such as "The compiled model ran on the Modalix DevKit and was evaluated
// on COCO 2017 using 5,000 samples."
function evaluatedOn(dataset, target) {
  const samples = sampleCount(dataset);
  return `The compiled model ran on the ${target} DevKit and was evaluated on ${datasetName(dataset) || "an unnamed dataset"}${samples ? ` ${samples}` : ""}.`;
}

// One dataset's result. With several, each is titled by its model.yaml name.
function accuracyResult({ name, metrics, dataset, details }, tool, target, { scores, sources }, measured, titled) {
  const full = fullCoverage(dataset);
  const coverageLabel = full ? "Full coverage" : "Sampled coverage";
  const coverageExplanation = full
    ? "Every sample in the dataset split was evaluated."
    : "A deterministic subset of the dataset split was evaluated.";
  const context = `${keyValue("Dataset archive", registryArchive(dataset))}${keyValue("Measured on", "DevKit")}${keyValue("Runtime", tool)}${Object.entries(details)
    .map(([key, value]) => keyValue(humanize(key), typeof value === "object" ? JSON.stringify(value) : String(value))).join("")}`;
  const evaluation = `<p class="disclosure-text"><strong>${coverageLabel}.</strong> ${escapeHtml(coverageExplanation)}</p>
      <dl class="key-grid">${context}</dl>
      ${accuracyMetricTable(metrics, scores)}`;
  return `<div class="accuracy-dataset" data-accuracy-dataset="${escapeHtml(name)}">
    ${titled ? `<div class="accuracy-dataset-title"><h4>${escapeHtml(name)}${infoMark(evaluatedOn(dataset, target))}</h4></div>` : ""}
    <p class="accuracy-scope">${measured}</p>
    <div class="accuracy-summary">${accuracyMetricTiles(metrics, scores, sources)}</div>
    ${shows.evaluationDetails ? disclosure(`accuracy:${name}`, "Evaluation details", evaluation, coverageLabel) : ""}
  </div>`;
}
