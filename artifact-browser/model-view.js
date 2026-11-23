// Model page: performance, accuracy, model card, build details, and license.
import { mlaStageSimulation } from "./mla-utilization.js";
import { readPerformance } from "./model-report.js";
import { renderMarkdown, resolveDocumentationLink } from "./markdown.js";
import { shows, state, elements, catalogMode, currentVariant, showView, taxonomyLabel } from "./state.js";
import {
  icons, escapeHtml, branchName, shortSha, profileLabel, variantLabel, precisionLabel, showToast, disclosure,
} from "./ui.js";
import { formatDate, formatDuration, formatBytes } from "./metrics.js";
import {
  apiText, profileGroups, variantPackage, modelPackArtifacts, catalogModelForRun, compilerFilteredBuilds,
  documentationHeroUrl, documentationHeroFallbackUrl,
} from "./api.js";
import {
  packageFormat, artifactDelivery, deploymentSize, downloadOffered, huggingFaceLocation,
} from "./downloads.js";
import { accuracyPanel } from "./accuracy-section.js";
import {
  executionGraphSection, hideExecutionTooltip, layerAnalysisSection, renderMlaUtilization,
} from "./execution-section.js";
import {
  componentPanels, githubModelUrl, humanize, keyValue, targetName, trustedHttpsUrl, variantSectionTitle,
} from "./page-parts.js";
import { generativeBenchmark, performanceSummary } from "./performance-section.js";

function artifactValidationMark(tests = []) {
  const passed = tests.some((test) => ["mpk-validation", "deployment-validation"].includes(test.name) && test.status === "passed");
  if (!passed) return "";
  const label = tests.some((test) => test.name === "deployment-validation") ? "Model deployment" : "MPK package";
  const message = `${label} integrity verified: file structure, checksum, and build provenance match.`;
  return `<span class="mpk-validation-mark" tabindex="0" role="img" aria-label="${message}" data-tooltip="${message}">✓</span>`;
}

function huggingFaceRepositoryUrl(sourceReference = "") {
  const match = /^hf:\/\/(models|datasets|spaces)\/([^/@?#]+)\/([^/@?#]+)@[a-f0-9]{40}\/.+$/i.exec(String(sourceReference));
  if (!match) return null;
  const [, repositoryType, owner, repository] = match;
  const typePrefix = repositoryType === "models" ? "" : `${repositoryType}/`;
  return new URL(`https://huggingface.co/${typePrefix}${encodeURIComponent(owner)}/${encodeURIComponent(repository)}`);
}

// The original model is the declared `upstream.model_card`, whatever ONNX was compiled; a pinned
// Hugging Face source without a card falls back to its repository.
function originalModelLink(input = {}) {
  const url = trustedHttpsUrl(input.metadata?.model_card) || huggingFaceRepositoryUrl(input.source_reference);
  if (!url) return null;
  const huggingFace = url.hostname === "huggingface.co";
  return {
    href: url.href,
    huggingFace,
    label: `Open the original model on ${huggingFace ? "Hugging Face" : url.hostname}`,
  };
}

// The example application is the Apps portal page `example_app` names in model.yaml.
function exampleAppLink(exampleApp) {
  const url = trustedHttpsUrl(exampleApp);
  if (!url) return "";
  const label = "Open an example application that runs this model";
  return `<a class="example-app-link" href="${escapeHtml(url.href)}" target="_blank" rel="noreferrer" aria-label="${label}" data-tooltip="${label}">${icons.apps}</a>`;
}

function sourceInputPanel(inputs) {
  if (!inputs.length) return "";
  const meta = inputs.map((input) => formatBytes(input.size_bytes)).join(" · ");
  return disclosure("source", inputs.length > 1 ? "Source models" : "Source model", inputs.map((input) => `<div class="source-block">
    <strong>${escapeHtml(input.name)}</strong>
    <p>${escapeHtml(input.source_reference || input.input_type)}</p>
    <div><span>${escapeHtml(formatBytes(input.size_bytes))}</span><code>${escapeHtml(input.sha256)}</code><button type="button" class="copy-button" data-copy="${escapeHtml(input.sha256)}" aria-label="Copy source checksum">${icons.copy}</button></div>
  </div>`).join(""), meta);
}

export function detailSkeleton() {
  hideExecutionTooltip();
  showView("model");
  elements["detail-pane"].innerHTML = `<div class="detail-loading"><span></span><span></span><span></span><span></span></div>`;
}

export function showEmptyDetail() {
  hideExecutionTooltip();
  elements["detail-pane"].innerHTML = "";
  showView("catalog");
}

function detailTag(icon, value, kind) {
  return `<span class="detail-tag detail-tag-${kind}">${icon}<span>${escapeHtml(value)}</span></span>`;
}

// The component ids in the order the pipeline runs them, when the pipeline names exactly the
// variant's components; a pipeline built from another branch's model.yaml may not.
function pipelineOrder(variant) {
  const order = (variant.pipeline || []).filter((stage) => "component" in stage).map((stage) => stage.component);
  const ids = variant.details.map((detail) => detail.summary.run.metadata.component_id);
  return order.length === ids.length && order.every((id) => ids.includes(id)) ? order : null;
}

// One line from input to output: numbered component chips, with any CPU step between them as a dashed chip.
function pipelineStrip(pipeline) {
  let number = 0;
  const stages = pipeline.map((stage) => ("component" in stage
    ? `<li><span class="pipeline-chip"><span class="component-number">${++number}</span>${escapeHtml(humanize(stage.component))}</span></li>`
    : `<li><span class="pipeline-chip cpu"><span class="pipeline-cpu">CPU</span>${escapeHtml(stage.cpu)}</span></li>`));
  return `<ol class="component-pipeline" aria-label="Order the components run in"><li class="pipeline-end">Input</li>${stages.join("")}<li class="pipeline-end">Output</li></ol>`;
}

// The build profile's numbers for the whole model, with each component's layer analysis, MLA utilization and
// execution graph.
// How the variant's performance was measured, for the info mark beside the section title.
function performanceMethod(variant) {
  const result = readPerformance(variant.performance);
  if (result.kind !== "supported") return "";
  // A generation report comes from an LLM or VLM, which LLiMa runs.
  const runtime = result.value.generation ? "LLiMa" : "the Neat Model API";
  return `Measured on the ${targetName(variant.performance.run)} DevKit, running the compiled model with ${runtime}.`;
}

function modelPerformance(variant, modelId) {
  const benchmarkRun = variant.performance;
  const order = pipelineOrder(variant) || variant.details.map((detail) => detail.summary.run.metadata.component_id);
  const result = readPerformance(benchmarkRun, order);
  if (result.kind === "missing") return `<p class="muted-message">Not measured at this revision.</p>`;
  if (result.kind === "unsupported") return `<p class="muted-message">This result format is not supported by this browser.</p>`;
  const { run } = benchmarkRun;
  if (result.value.generation) {
    const compilation = variant.details[0]?.provenance.parameters?.parameters || {};
    const method = compilation.compiled_parameters?.speculative_method || compilation.compile_parameters?.speculative_method;
    return generativeBenchmark(result.value, run, method);
  }
  const components = result.value.components.map((component) => ({ ...component, label: humanize(component.id) }));
  const summary = result.value.scope === "components"
    ? componentPanels(components.map(({ label }) => label), components.map((component) =>
      performanceSummary({ ...result.value, ...component, componentId: component.id, components: [component] },
        run, modelId, component.id)))
    : performanceSummary(result.value, run, modelId);
  return `<div class="component-performance">
    ${summary}
    ${layerAnalysisSection(components.map(({ label, layerAnalysis }) => ({ label, analysis: layerAnalysis })), run.id)}
    ${renderMlaUtilization(components, mlaStageSimulation(variant.packageTests?.test_results), componentPrecisions(variant),
      result.value.measurements.throughput_fps?.value, run.id)}
    ${executionGraphSection(components.map(({ id, label, executionGraph }) => ({ id: `${run.id}:${id}`, label, graph: executionGraph })), run.id)}
  </div>`;
}

// Each component's compile precision, such as w8a8.
function componentPrecisions(variant) {
  return Object.fromEntries(variant.details.map((detail) => [detail.summary.run.metadata.component_id,
    detail.provenance.parameters?.parameters?.compile_parameters?.precision]));
}

function compileDuration(run) {
  return formatDuration(run.metadata.compile_timing.started_at, run.metadata.compile_timing.finished_at);
}

function componentRow(detail, single) {
  const run = detail.summary.run;
  const component = run.metadata.component_id;
  const artifacts = modelPackArtifacts(detail.summary.artifacts);
  const parameters = detail.provenance.parameters?.parameters?.compile_parameters || {};
  const mpkBytes = artifacts.reduce((total, artifact) => total + (Number.isFinite(deploymentSize(artifact)) ? deploymentSize(artifact) : 0), 0);
  const artifactRows = artifacts.length
    ? artifacts.map((artifact) => keyValue(artifactDelivery(artifact) === "archive" ? "SHA-256" : "Manifest SHA-256", artifact.sha256, { copy: true, mono: true })).join("")
    : keyValue("SHA-256", "No artifact metadata was registered for this run.");
  const meta = [mpkBytes ? `${run.metadata?.artifact_kind === "llima" ? "Model" : "MPK"} ${formatBytes(mpkBytes)}` : "", parameters.precision && precisionLabel(parameters.precision)]
    .filter(Boolean).join(" · ");
  const body = `<dl class="key-grid">
      ${artifactRows}
      ${keyValue("Run ID", run.id, { copy: true, mono: true })}
      ${keyValue("Fingerprint", run.run_fingerprint, { copy: true, mono: true })}
      ${keyValue("Compile recipe duration", compileDuration(run))}
      ${keyValue("Registry publication duration", formatDuration(run.started_at, run.finished_at))}
    </dl>
    ${Object.keys(parameters).length ? disclosure(`parameters:${run.id}`, "Compile parameters", `<pre>${escapeHtml(JSON.stringify(parameters, null, 2))}</pre>`, "JSON") : ""}`;
  return disclosure(`component:${run.id}`, single ? "Files and checksums" : component, body, meta);
}

// Numerical fidelity compares the whole model's outputs with the source model's, on the package run.
function fidelityPanel(tests, runId) {
  const result = tests.find((item) =>
    item.suite === "correctness"
    && item.name === "numerical-fidelity"
    && item.summary?.schema_version === 1
  );
  const report = result?.summary;
  if (!report) return "";
  const metrics = (report.metrics || []).map((metric) => `
    <div class="fidelity-metric"><span>${escapeHtml(metric.name)} · Cosine similarity</span><strong>${Number(metric.cosine_similarity).toFixed(4)}</strong></div>
    <div class="fidelity-metric"><span>${escapeHtml(metric.name)} · Normalized RMSE</span><strong>${Number(metric.normalized_rmse).toFixed(4)}</strong></div>
  `).join("");
  return `<div class="fidelity-panel">
    <div class="group-title"><h4>Numerical fidelity</h4><button class="action-button" type="button" data-download="numerical-fidelity" data-run-id="${escapeHtml(runId)}">${icons.download}<span>Download JSON</span></button></div>
    <p class="fidelity-scope">${escapeHtml(report.status)} · ${Number(report.samples).toLocaleString()} samples · ${escapeHtml(report.dataset?.id || "Dataset unavailable")} ${escapeHtml(report.dataset?.variant || "")}</p>
    <div class="fidelity-metrics">${metrics}</div>
    <dl class="key-grid">${keyValue("Reference", report.reference)}${keyValue("Candidate", report.candidate)}</dl>
    <p class="muted-message">Output similarity, not task accuracy. Output reordering can affect these metrics.</p>
  </div>`;
}

// `listed` is the license's entry in licenses.json: its name, and what it asks of someone who redistributes
// the model. A license the list lacks shows its identifier and what model.yaml records about redistribution.
function licensePanel(license, listed) {
  if (!license?.spdx) return "";
  const terms = listed?.summary || (license.redistribution ? `Redistribution ${license.redistribution}.` : "");
  return `<section id="license-section" class="license-section" tabindex="-1"><div class="detail-section-title"><h3>License</h3></div>
    <p class="license-line"><strong>${escapeHtml(listed?.name || license.spdx)}</strong></p>
    ${terms ? `<p class="license-summary">${escapeHtml(terms)}</p>` : ""}
    ${license.attribution ? `<p class="license-attribution">${escapeHtml(license.attribution)}</p>` : ""}
  </section>`;
}

function documentationKey(documentation) {
  return `${documentation.run_id}:${documentation.artifact_sha256}`;
}

// The README's title, hero, and Overview are already on the page, so the full card starts at the section
// after Overview. The registry rejects a README without that opening.
function modelCardRemainder(markdown) {
  const lines = markdown.replace(/\r\n?/g, "\n").split("\n");
  const overview = lines.findIndex((line) => /^##\s+Overview\s*$/.test(line));
  const next = overview < 0 ? -1 : lines.findIndex((line, index) => index > overview && /^##\s/.test(line));
  return next < 0 ? null : lines.slice(next).join("\n");
}

function modelCardBody(documentation) {
  const markdown = state.documentationCache.get(documentationKey(documentation));
  if (markdown === undefined) return `<p class="muted-message" data-model-card-pending>Loading usage instructions…</p>`;
  const remainder = modelCardRemainder(markdown);
  const heroUrl = documentationHeroUrl(documentation);
  return `<div class="model-documentation-body">${renderMarkdown(remainder ?? markdown, {
    stripTitle: remainder === null,
    resolveImage: (source) => (source === documentation.hero?.markdown_path ? heroUrl : null),
    resolveLink: (source) => resolveDocumentationLink(documentation, source),
  })}</div>`;
}

function modelHero(model) {
  const documentation = model?.documentation;
  if (!documentation) return "";
  const heroUrl = documentationHeroUrl(documentation);
  return heroUrl
    ? `<figure class="model-card-hero"><img src="${escapeHtml(heroUrl)}" data-hero-fallback="${escapeHtml(documentationHeroFallbackUrl(documentation))}" alt="${escapeHtml(documentation.hero.alt_text || `${model.display_name || model.model_id} model hero`)}" decoding="async" /></figure>`
    : "";
}

// The README's Overview text opens the section without a title of its own, followed by the remaining sections.
function overviewSection(model) {
  const documentation = model?.documentation;
  if (!documentation?.overview_markdown) return "";
  const instructions = documentation.run_id && documentation.artifact_sha256
    ? `<div data-model-card-body>${modelCardBody(documentation)}</div>`
    : "";
  return `<section id="overview-section" class="overview-section" tabindex="-1" aria-label="Overview">
    <div class="model-documentation-body">${renderMarkdown(documentation.overview_markdown, { resolveLink: (source) => resolveDocumentationLink(documentation, source) })}</div>
    ${instructions}
  </section>`;
}

// Loads the remaining README sections inline after checking the catalog checksum.
async function loadModelCard() {
  const variant = currentVariant();
  const documentation = variant ? catalogModelForRun(variant.anchor)?.documentation : null;
  const body = () => elements["detail-pane"].querySelector("[data-model-card-body]");
  if (!documentation || !body()?.querySelector("[data-model-card-pending]")) return;
  const requestId = ++state.documentationRequestId;
  try {
    const markdown = await apiText(`/v1/runs/${encodeURIComponent(documentation.run_id)}/artifacts/${encodeURIComponent(documentation.artifact_name || "readme")}/content`);
    const digest = [...new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(markdown)))]
      .map((byte) => byte.toString(16).padStart(2, "0")).join("");
    if (digest !== documentation.artifact_sha256) throw new Error("The model card checksum did not match the catalog.");
    state.documentationCache.set(documentationKey(documentation), markdown);
  } catch (error) {
    if (requestId !== state.documentationRequestId || state.selectedVariantKey !== variant.key) return;
    const target = body();
    if (target) target.innerHTML = `<p class="muted-message" data-model-card-pending>The usage instructions could not be loaded.</p><button class="text-button" type="button" data-action="retry">Retry</button>`;
    showToast(error.message, true);
    return;
  }
  if (requestId !== state.documentationRequestId || state.selectedVariantKey !== variant.key) return;
  const target = body();
  if (target) target.innerHTML = modelCardBody(documentation);
}

// Variant tabs switch between the latest builds in the current catalog, so a build reached from
// history shows none.
// The catalog model whose current build is this variant, or null for an older or history-only build.
function catalogModelOfVariant(variant) {
  const model = catalogMode() ? state.models.find((item) => item.model_id === variant.modelId) : null;
  return model?.builds.some((build) => variant.runs.some((run) => run.id === build.run_id)) ? model : null;
}

function detailBuildSwitcher(variant) {
  const model = catalogModelOfVariant(variant);
  const inCatalog = Boolean(model);
  const profiles = inCatalog ? profileGroups({ ...model, builds: compilerFilteredBuilds(model) }) : [];
  const tabs = profiles.map(([profile]) => {
    const selected = profile === variant.profile;
    return `<button class="detail-variant-tab" type="button" role="tab" data-detail-model-id="${escapeHtml(model.model_id)}" data-detail-profile-id="${escapeHtml(profile)}" aria-selected="${selected}" tabindex="${selected ? "0" : "-1"}">${escapeHtml(variantLabel(profile))}</button>`;
  }).join("");
  return `<section class="detail-build-switcher" aria-label="Variant selection">
    <strong>${profiles.length > 1 ? "Variants" : "Variant"}</strong>
    ${profiles.length ? `<div class="detail-variant-tabs" role="tablist" aria-label="${escapeHtml(model.display_name || model.model_id)} variants">${tabs}</div>` : `<span class="detail-variant-single">${escapeHtml(profileLabel(variant.anchor))}</span>`}
    ${profiles.length > 1 ? `<p class="detail-variant-help">Select a variant. Performance and accuracy are shown for the selected variant.</p>` : ""}
  </section>`;
}

function uniqueBy(items, key) {
  return [...new Map(items.map((item) => [key(item), item])).values()];
}

function sectionNavigation(sections) {
  return `<nav class="subnav" aria-label="Model page sections">${sections.map(([id, label]) => (
    `<button type="button" data-scroll-to="${id}">${label}</button>`
  )).join("")}</nav>`;
}

// The package is the glance view of Details: its size and checksum, and the download.
function packageCard(fullPackage, packageArtifact) {
  if (!fullPackage) return "";
  const folder = packageArtifact && artifactDelivery(packageArtifact) !== "archive";
  const location = packageArtifact && huggingFaceLocation(packageArtifact);
  return `<div class="package-card">
    <div class="package-card-copy"><strong>Full package</strong><span>${folder ? "Download the model folder with sima-cli using its verified manifest." : packageFormat(packageArtifact) === "MPK" ? "The compiled model in one verified MPK." : "All components and declared runtime resources in one verified ZIP."}</span>${location ? `<a href="${escapeHtml(location.url)}" target="_blank" rel="noreferrer">Deployment on Hugging Face ↗</a>` : ""}</div>
    ${packageArtifact ? `<dl class="package-facts"><div><dt>Size</dt><dd>${escapeHtml(formatBytes(deploymentSize(packageArtifact)))}</dd></div><div><dt>${folder ? "Manifest SHA-256" : "SHA-256"}</dt><dd class="artifact-hash"><code>${escapeHtml(packageArtifact.sha256)}</code><button type="button" class="copy-button" data-copy="${escapeHtml(packageArtifact.sha256)}" aria-label="Copy package checksum">${icons.copy}</button></dd></div></dl>` : ""}
    <div class="package-actions"><button class="action-button" type="button" data-download="package" data-run-id="${escapeHtml(fullPackage.run_id)}">${icons.download}<span>Download package</span></button></div>
  </div>`;
}

export function renderDetail(variant) {
  hideExecutionTooltip();
  showView("model");
  const run = variant.anchor;
  const details = variant.details;
  const allTests = details.flatMap((detail) => detail.tests.test_results || []);
  const inputs = uniqueBy(details.flatMap((detail) => detail.provenance.inputs || []), (input) => input.sha256 || input.name);
  const model = variant.modelId || run.run_type;
  const catalogModel = catalogModelForRun(run);
  const displayName = catalogModel?.display_name || model;
  // A reused build keeps the categories it was compiled with; the catalog has the current ones.
  const categories = catalogModel?.categories || run.model_categories || {};
  const compilation = details[0]?.provenance.parameters?.parameters || {};
  const parameters = { ...compilation.compile_parameters, ...compilation.compiled_parameters };
  const hasImageResolution = [parameters.input_height, parameters.input_width].every((value) => Number.isInteger(value) && value > 0);
  const compiledLimits = ["llm", "vlm"].includes(categories.model_type) ? [
    Number.isInteger(parameters.max_num_tokens) && parameters.max_num_tokens > 0
      ? keyValue("Maximum context", `${parameters.max_num_tokens.toLocaleString("en-US")} tokens (input + output)`) : "",
    hasImageResolution ? keyValue("Image input resolution", `${parameters.input_height} × ${parameters.input_width} pixels`) : "",
  ].join("") : "";
  // The catalog carries the license model.yaml declares now. A build's own metadata keeps the license it
  // was compiled with.
  const license = catalogModel?.license || run.metadata?.license;
  const offered = downloadOffered(license?.spdx, variant.licenses);
  // CI reuses a build until its compile inputs change, so the build's commit can predate the scripts
  // that measured it. The source link opens the commit the model card comes from; Build provenance
  // keeps the build's own commit.
  const sourceCommit = catalogModel?.documentation?.source_commit || run.commit_sha;
  const sourceUrl = githubModelUrl(run, model, sourceCommit);
  const sourceLabel = `View ${model} source on GitHub at commit ${shortSha(sourceCommit)}`;
  const accuracyMarkup = accuracyPanel(variant, model);
  const overviewMarkup = overviewSection(catalogModel);
  const hasArtifacts = details.some((detail) => modelPackArtifacts(detail.summary.artifacts).length);
  // Every component's MPK must pass validation for the build to carry the mark.
  const validated = details.every((detail) => artifactValidationMark(detail.tests.test_results || []));
  const fullPackage = variantPackage(variant.runs);
  const packageArtifact = (variant.packageSummary?.artifacts || []).find((item) => item.artifact_type === "model-package");
  const downloadArtifact = packageArtifact || details.flatMap((detail) => modelPackArtifacts(detail.summary.artifacts))[0];
  const downloadLabel = downloadArtifact && artifactDelivery(downloadArtifact) !== "archive" ? "Download folder" : "Download";
  const originalModel = inputs.map(originalModelLink).find(Boolean);
  const builtAt = Math.max(...variant.runs.map((item) => Date.parse(item.finished_at || item.created_at || "") || 0));
  const singleComponent = details.length === 1;
  const order = pipelineOrder(variant);
  const backLabel = elements["result-title"].textContent || "Models";
  const downloadAction = !offered ? `<button class="download-unavailable" type="button" data-scroll-to="license-section">${icons.license}Download restricted by license</button>`
    : fullPackage
    ? `<button class="download-action" type="button" data-download="package" data-run-id="${escapeHtml(fullPackage.run_id)}">${icons.download}<span>${downloadLabel}</span>${packageArtifact ? `<small>${escapeHtml(formatBytes(deploymentSize(packageArtifact)))}</small>` : ""}</button>`
    : hasArtifacts ? `<button class="download-action" type="button" data-download-variant="${escapeHtml(run.id)}" data-run-id="${escapeHtml(run.id)}">${icons.download}<span>${downloadLabel}</span></button>` : "";
  const sections = [
    overviewMarkup && ["overview-section", "Overview"],
    ["components-section", "Performance"],
    accuracyMarkup && ["accuracy-section", "Accuracy"],
    shows.buildDetails && ["details-section", "Details"],
    license?.spdx && ["license-section", "License"],
  ].filter(Boolean);

  elements["detail-pane"].innerHTML = `<div class="detail-content">
    <nav class="crumbs" aria-label="Breadcrumb"><button class="close-detail" type="button">${escapeHtml(backLabel)}</button><span aria-hidden="true">/</span><span aria-current="page">${escapeHtml(displayName)}</span></nav>
    <header class="detail-title">
      <div class="detail-title-copy">
        <h2>${escapeHtml(displayName)}</h2>
        <div class="detail-tags">${(categories.tasks || []).map((task) => detailTag(icons.task, taxonomyLabel("tasks", task), "task")).join("")}${categories.architecture?.id ? detailTag(icons.architecture, taxonomyLabel("architectures", categories.architecture.id), "architecture") : ""}${categories.model_type ? detailTag(icons.modelType, taxonomyLabel("model_types", categories.model_type), "model-type") : ""}${detailTag(icons.compiler, `${run.toolchain_name || "compiler"} ${run.toolchain_version || ""}`.trim(), "compiler")}${detailTag(icons.branch, branchName(run.git_ref), "branch")}${license?.spdx ? detailTag(icons.license, license.spdx, "license") : ""}</div>
        ${catalogModel?.description ? `<p class="lede">${escapeHtml(catalogModel.description)}</p>` : ""}
      </div>
      ${modelHero(catalogModel)}
    </header>
    ${detailBuildSwitcher(variant)}
    <div class="detail-title-actions">
          ${downloadAction}
          <div class="detail-links">${validated ? artifactValidationMark(allTests) : ""}${originalModel ? `<a class="original-model-link${originalModel.huggingFace ? "" : " external-model-link"}" href="${escapeHtml(originalModel.href)}" target="_blank" rel="noreferrer" aria-label="${escapeHtml(originalModel.label)}" data-tooltip="${escapeHtml(originalModel.label)}">${originalModel.huggingFace ? icons.huggingFace : icons.external}</a>` : ""}${sourceUrl ? `<a class="model-source-link" href="${escapeHtml(sourceUrl)}" target="_blank" rel="noreferrer" aria-label="${escapeHtml(sourceLabel)}" data-tooltip="${escapeHtml(sourceLabel)}">${icons.github}</a>` : ""}${exampleAppLink(variant.exampleApp)}</div>
    </div>
    ${compiledLimits ? `<dl class="key-grid">${compiledLimits}</dl>` : ""}
    ${sectionNavigation(sections)}

    ${overviewMarkup}

    <section id="components-section" class="component-section" tabindex="-1">${variantSectionTitle(`Performance on ${targetName(run)}`, variant, performanceMethod(variant))}
      ${singleComponent ? "" : `<p class="section-lede">${variant.performance?.report?.scope === "components"
        ? "Each component is benchmarked independently. Select a component to view its measurements."
        : `This model runs as ${details.length} components, measured running together.`}</p>`}
      ${order ? pipelineStrip(variant.pipeline) : ""}
      ${modelPerformance(variant, model)}
    </section>

    ${accuracyMarkup}

    ${!shows.buildDetails ? "" : `<section id="details-section" class="details-section" tabindex="-1"><div class="detail-section-title"><h3>Details</h3></div>
      ${offered ? packageCard(fullPackage, packageArtifact) : ""}
      ${fullPackage ? fidelityPanel(variant.packageTests?.test_results || [], fullPackage.run_id) : ""}
      <div class="disclosure-list">
        ${disclosure("provenance", "Build provenance", `<dl class="key-grid">
          ${keyValue("Commit", run.commit_sha, { copy: true, mono: true })}
          ${keyValue("Branch", branchName(run.git_ref))}
          ${keyValue("Built", formatDate(builtAt || run.created_at))}
          ${keyValue("Target", run.target_platform)}
          ${keyValue("Compiler", run.toolchain_version)}
        </dl>${run.ci_run_url ? `<a class="external-link" href="${escapeHtml(run.ci_run_url)}" target="_blank" rel="noreferrer">Open GitHub workflow ${icons.external}</a>` : ""}`, `${shortSha(run.commit_sha).slice(0, 7)} · ${branchName(run.git_ref)}`)}
        ${sourceInputPanel(inputs)}
        ${details.map((detail) => componentRow(detail, singleComponent)).join("")}
      </div>
    </section>`}

    ${licensePanel(license, variant.licenses?.[license?.spdx])}
  </div>`;
  loadModelCard();
}

export function errorPanel(message, title = "Registry unavailable", { closable = false } = {}) {
  return `<div class="error-state">${closable ? `<nav class="crumbs" aria-label="Breadcrumb"><button class="close-detail" type="button">Back to models</button></nav>` : ""}<span>!</span><p class="kicker">Connection error</p><h2>${escapeHtml(title)}</h2><p>${escapeHtml(message)}</p><button type="button" data-action="retry">Try again</button></div>`;
}
