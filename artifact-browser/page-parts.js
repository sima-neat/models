// Markup and helpers that several sections of the model page share.
import { shows, state } from "./state.js";
import { icons, escapeHtml, shortSha, variantLabel } from "./ui.js";

export function githubModelUrl(run, modelId, sourceCommit = run.commit_sha) {
  const organization = String(state.repository.organization || "");
  const repository = String(state.repository.name || "");
  const configuredRepository = `${organization}/${repository}`.toLowerCase();
  const runRepository = String(run.repository || "").toLowerCase();
  const commit = String(sourceCommit || "");
  if (!/^[A-Za-z0-9_.-]+$/.test(organization)
    || !/^[A-Za-z0-9_.-]+$/.test(repository)
    || runRepository !== configuredRepository
    || !/^[a-f0-9]{40}$/i.test(commit)
    || !/^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$/.test(modelId)) return null;
  return `https://github.com/${encodeURIComponent(organization)}/${encodeURIComponent(repository)}/tree/${commit}/models/${encodeURIComponent(modelId)}`;
}

// A reused build is measured by later runs, so the developer view names the commit and workflow that
// measured each result, appended to the result's context line.
export function measurementSource(run, modelId) {
  const commit = String(run?.commit_sha || "");
  if (!shows.measurementSource || !/^[a-f0-9]{40}$/i.test(commit)) return "";
  const sourceUrl = githubModelUrl(run, modelId);
  const commitMarkup = sourceUrl
    ? `<a class="measurement-link" href="${escapeHtml(sourceUrl)}" target="_blank" rel="noreferrer" aria-label="View the ${escapeHtml(modelId)} source that measured this result, at commit ${escapeHtml(shortSha(commit))}">${escapeHtml(commit.slice(0, 7))}${icons.external}</a>`
    : escapeHtml(commit.slice(0, 7));
  const workflow = run.ci_run_url
    ? ` · <a class="measurement-link" href="${escapeHtml(run.ci_run_url)}" target="_blank" rel="noreferrer">GitHub workflow${icons.external}</a>`
    : "";
  return ` · Commit ${commitMarkup}${workflow}`;
}

export function trustedHttpsUrl(value = "") {
  try {
    const url = new URL(String(value));
    return url.protocol === "https:" && !url.username && !url.password ? url : null;
  } catch {
    return null;
  }
}

export function keyValue(label, value, { copy = false, mono = false } = {}) {
  const safeValue = escapeHtml(value ?? "—");
  return `<div class="key-value"><dt>${escapeHtml(label)}</dt><dd class="${mono ? "mono" : ""}"><span title="${safeValue}">${safeValue}</span>${copy && value ? `<button type="button" class="copy-button" data-copy="${safeValue}" aria-label="Copy ${escapeHtml(label)}">${icons.copy}</button>` : ""}</dd></div>`;
}

export function tile(label, text, unit, qualifier = "") {
  return `<div class="kpi"><span>${escapeHtml(label)}${qualifier ? `<span class="qualifier">(${escapeHtml(qualifier)})</span>` : ""}</span><strong>${escapeHtml(text)}<small>${escapeHtml(unit)}</small></strong></div>`;
}

// The board a run targets, as the page names it, such as "Modalix".
export function targetName(run) {
  const target = run?.target_platform || "";
  return target ? target.charAt(0).toUpperCase() + target.slice(1) : "the target";
}

export function infoMark(text) {
  return text ? `<span class="info-mark" tabindex="0" role="img" aria-label="${escapeHtml(text)}" data-tooltip="${escapeHtml(text)}">i</span>` : "";
}

// Buttons that pick one of a pipeline's components; `attribute` names the control for its handler.
export function componentSwitch(labels, selected, attribute) {
  return `<div class="layer-order-toggle component-switch" role="group" aria-label="Component">${labels.map((label, index) => `<button type="button" ${attribute} value="${index}" aria-pressed="${index === selected}">${escapeHtml(label)}</button>`).join("")}</div>`;
}

// Every component's panel, with a switch that shows one at a time when there are several.
export function componentPanels(labels, panels) {
  const switcher = labels.length > 1 ? componentSwitch(labels, 0, "data-component-switch") : "";
  return `<div data-component-panels>${switcher}${panels.map((markup, index) =>
    `<div data-component-panel="${index}"${index ? " hidden" : ""}>${markup}</div>`).join("")}</div>`;
}

export function selectComponentPanel(event) {
  const button = event.target.closest("[data-component-switch]");
  if (!button) return;
  const panel = button.closest("[data-component-panels]");
  panel.querySelectorAll("[data-component-switch]").forEach((item) => item.setAttribute("aria-pressed", String(item === button)));
  panel.querySelectorAll("[data-component-panel]").forEach((item) => { item.hidden = item.dataset.componentPanel !== button.value; });
}

// Component and role ids read as words: "feature_extractor" becomes "Feature extractor".
export function humanize(id) {
  const words = String(id).replaceAll("_", " ");
  return words.charAt(0).toUpperCase() + words.slice(1);
}

// Performance and Accuracy describe the selected variant, so each title names it. `info` says how the
// section's results were measured, behind an info mark.
export function variantSectionTitle(title, variant, info = "") {
  return `<div class="detail-section-title"><h3>${escapeHtml(title)} <span class="section-variant">${escapeHtml(variantLabel(variant.profile))}</span>${infoMark(info)}</h3></div>`;
}
