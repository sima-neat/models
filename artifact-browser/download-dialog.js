// The download dialog: a package's file name and license, its CLI command, and the browser downloads it starts.
import {
  variantDownloads, prepareDownloads, startDownloads, artifactDelivery, deploymentSize,
  huggingFaceLocation, modelDownloadCommand,
} from "./downloads.js";
import { runProfile } from "./branch-builds.js";
import { config, shows, state, elements } from "./state.js";
import { icons, escapeHtml, branchName, showToast, profileLabel, variantLabel } from "./ui.js";
import { formatBytes } from "./metrics.js";
import { api } from "./api.js";
import { selectedVersion } from "./catalog-versions.js";

let pendingDownloads = null;
let downloadRequest = 0;
let pendingCommand = null;

function resetDownloadDialog({ title, description, confirmLabel }) {
  pendingDownloads = null;
  pendingCommand = null;
  elements["download-copy-cli"].hidden = true;
  elements["download-title"].textContent = title;
  elements["download-description"].textContent = description;
  elements["download-list"].innerHTML = "";
  elements["download-list"].hidden = false;
  elements["download-package-info"].hidden = true;
  elements["download-package-info"].classList.remove("download-package-summary");
  elements["download-package-info"].innerHTML = "";
  elements["download-dialog"].querySelector(".download-note").hidden = true;
  elements["download-status"].textContent = "";
  elements["download-confirm"].disabled = true;
  elements["download-confirm"].hidden = false;
  elements["download-confirm"].textContent = confirmLabel;
}

function catalogDownloadCommand(run) {
  const current = state.models.some((model) => model.model_id === run.metadata?.model_id
    && (model.builds?.some((build) => build.run_id === run.id)
      || model.packages?.some((item) => item.run_id === run.id)));
  return current ? modelDownloadCommand(run.metadata.model_id, runProfile(run),
    shows.commandBranch ? elements["branch-filter"].value || branchName(run.git_ref) : null, config.environment?.toLowerCase() === "staging",
    selectedVersion()) : null;
}

function showFolderDownload(run, downloads) {
  const dialog = elements["download-dialog"];
  pendingCommand = catalogDownloadCommand(run);
  elements["download-title"].textContent = "Download model folder";
  elements["download-description"].textContent = "This model is provided as a folder of individual files, similar to a Hugging Face repository, rather than a ZIP archive. Use sima-cli 2.1.19 or later to download the files in parallel and assemble the model folder.";
  elements["download-list"].hidden = true;
  elements["download-package-info"].hidden = false;
  elements["download-package-info"].innerHTML = downloads.map((item) => {
    const location = item.delivery === "huggingface" ? huggingFaceLocation(item.artifact) : null;
    return `<dt>Deployment</dt><dd>${escapeHtml(item.component)}</dd>
      <dt>Files</dt><dd>${escapeHtml(item.artifact.metadata?.file_count ?? "Not recorded")}</dd>
      <dt>Download size</dt><dd>${escapeHtml(formatBytes(item.size))}</dd>
      <dt>Storage</dt><dd>${item.delivery === "huggingface" ? "Hugging Face" : item.delivery === "archive" ? "Archive" : "Registry folder"}</dd>
      ${location ? `<dt>Repository</dt><dd><a href="${escapeHtml(location.url)}" target="_blank" rel="noreferrer">${escapeHtml(location.repo_id)} ↗</a></dd><dt>Revision</dt><dd><code>${escapeHtml(location.revision)}</code></dd>${location.subdirectory ? `<dt>Folder</dt><dd>${escapeHtml(location.subdirectory)}</dd>` : ""}` : item.delivery === "huggingface" ? `<dt>Repository</dt><dd>Pinned repository location unavailable. Refresh the catalog to retry.</dd>` : ""}`;
  }).join("") + (pendingCommand ? `<dt>Command</dt><dd><code>${escapeHtml(pendingCommand)}</code></dd>`
    : `<dt>Command</dt><dd>Open the current catalog variant for its verified CLI download command.</dd>`);
  const note = dialog.querySelector(".download-note");
  note.hidden = false;
  note.textContent = downloads.some((item) => item.delivery === "huggingface")
    ? "The repository link opens the compiled deployment at its pinned revision. Use that revision and folder when downloading from Hugging Face."
    : "The CLI uses the manifest to preserve the folder structure and verify each file’s checksum.";
  elements["download-confirm"].innerHTML = `${icons.copy}<span>Copy CLI command</span>`;
  elements["download-confirm"].hidden = !pendingCommand;
  elements["download-confirm"].disabled = false;
}

export async function downloadVariant(runId) {
  const request = ++downloadRequest;
  const catalogModels = elements["history-filter"].checked ? [] : state.models;
  const dialog = elements["download-dialog"];
  resetDownloadDialog({
    title: "Download components",
    description: "Checking the selected variant’s components…",
    confirmLabel: "Download all files",
  });
  dialog.showModal();
  try {
    const summary = await api(`/v1/runs/${encodeURIComponent(runId)}`);
    const model = catalogModels.find((model) => model.model_id === summary.run.metadata?.model_id);
    const downloads = await prepareDownloads(await variantDownloads(summary.run, api, model), api);
    if (request !== downloadRequest || !dialog.open) return;
    if (downloads.some((item) => item.delivery !== "archive" && item.delivery)) {
      showFolderDownload(summary.run, downloads);
      return;
    }
    pendingDownloads = downloads;
    pendingCommand = catalogDownloadCommand(summary.run);
    elements["download-copy-cli"].hidden = !pendingCommand;
    const profile = profileLabel(summary.run);
    const fileLabel = downloads.length === 1 ? "file" : "files";
    elements["download-description"].textContent = `${profile} contains ${downloads.length} model ${fileLabel}. ${downloads.length === 1 ? "Confirm to download this variant." : "Download all components to use this variant."}`;
    const note = elements["download-dialog"].querySelector(".download-note");
    note.hidden = downloads.length === 1;
    note.textContent = "Files download separately and in parallel. Your browser may ask you to allow multiple downloads.";
    elements["download-list"].innerHTML = downloads.map((item) => `<li><div><strong>${escapeHtml(item.component)}</strong><span>${escapeHtml(item.filename)}</span></div><span>${formatBytes(item.size)}</span></li>`).join("");
    elements["download-confirm"].textContent = `Download ${downloads.length} ${fileLabel}`;
    elements["download-confirm"].disabled = false;
  } catch (error) {
    if (request === downloadRequest && dialog.open) elements["download-status"].textContent = error.message;
  }
}

export async function downloadPackage(runId) {
  const request = ++downloadRequest;
  const dialog = elements["download-dialog"];
  resetDownloadDialog({
    title: "Download model package",
    description: "Checking the package manifest and download metadata…",
    confirmLabel: "Download",
  });
  dialog.showModal();
  try {
    const summary = await api(`/v1/runs/${encodeURIComponent(runId)}`);
    const run = summary.run;
    const artifact = (summary.artifacts || []).find((item) => item.artifact_type === "model-package");
    if (!run || !artifact) throw new Error("No complete package is available for this build.");
    const metadata = run.metadata || {};
    const profile = variantLabel(metadata.build_profile_id);
    // The registry serves the package under its published filename: a ZIP, or the MPK of a build published before ZIPs.
    const downloads = await prepareDownloads([{
      runId: run.id,
      component: "Full package",
      name: artifact.name,
      size: deploymentSize(artifact), artifact, delivery: artifactDelivery(artifact),
    }], api);
    const filename = downloads[0].filename;
    if (request !== downloadRequest || !dialog.open) return;
    if (downloads.some((item) => item.delivery !== "archive" && item.delivery)) {
      showFolderDownload(run, downloads);
      return;
    }
    pendingDownloads = downloads;
    pendingCommand = catalogDownloadCommand(run);
    elements["download-copy-cli"].hidden = !pendingCommand;
    const license = metadata.license || {};
    const redistribution = license.redistribution
      ? `${license.spdx || "License"} · redistribution ${license.redistribution}`
      : license.spdx || "Not recorded";
    const modelName = state.models.find((model) => model.model_id === metadata.model_id)?.display_name
      || metadata.model_id || "model";
    elements["download-title"].textContent = `Download ${modelName}`;
    elements["download-description"].textContent = `${profile} · ${formatBytes(deploymentSize(artifact))}`;
    elements["download-package-info"].hidden = false;
    elements["download-package-info"].classList.add("download-package-summary");
    elements["download-package-info"].innerHTML = `
      <div class="download-filename"><dt>File name</dt><dd><code>${escapeHtml(filename)}</code></dd></div>
      <div class="download-license"><dt>License</dt><dd>${escapeHtml(redistribution)}</dd></div>`;
    const note = elements["download-dialog"].querySelector(".download-note");
    note.hidden = true;
    note.textContent = "";
    elements["download-list"].hidden = true;
    elements["download-confirm"].textContent = "Download";
    elements["download-confirm"].disabled = false;
  } catch (error) {
    if (request === downloadRequest && dialog.open) elements["download-status"].textContent = error.message;
  }
}

elements["download-copy-cli"].innerHTML = `${icons.copy}<span>Copy CLI command</span>`;
elements["download-cancel"].addEventListener("click", () => elements["download-dialog"].close());
elements["download-copy-cli"].addEventListener("click", () => {
  if (pendingCommand) copyText(pendingCommand);
});
elements["download-dialog"].addEventListener("close", () => {
  downloadRequest += 1;
  pendingDownloads = null;
  pendingCommand = null;
});
elements["download-confirm"].addEventListener("click", async () => {
  if (!pendingDownloads && pendingCommand) { await copyText(pendingCommand); return; }
  if (!pendingDownloads) return;
  const request = downloadRequest;
  elements["download-confirm"].disabled = true;
  elements["download-status"].textContent = "Preparing all downloads…";
  try {
    const ready = await prepareDownloads(pendingDownloads, api);
    if (request !== downloadRequest || !elements["download-dialog"].open) return;
    startDownloads(ready, document);
    elements["download-dialog"].close();
    showToast(ready.length === 1 ? "Download requested." : `${ready.length} downloads requested. If prompted, allow multiple downloads in your browser.`);
  } catch (error) {
    if (request === downloadRequest && elements["download-dialog"].open) {
      elements["download-status"].textContent = error.message;
      elements["download-confirm"].disabled = false;
    }
  }
});

export async function downloadArtifact(runId, name, button) {
  const prior = button.innerHTML;
  const priorLabel = button.getAttribute("aria-label");
  button.disabled = true;
  button.setAttribute("aria-busy", "true");
  button.setAttribute("aria-label", "Preparing model artifact download");
  button.innerHTML = `<span class="spinner"></span>`;
  try {
    const payload = await api(`/v1/runs/${runId}/artifacts/${encodeURIComponent(name)}`);
    if (!payload.artifact?.download_url) throw new Error("The registry did not return a download URL.");
    showToast(`Download ready for ${name}. The link is valid for ${Math.round((payload.artifact.download_url_expires_in || 300) / 60)} minutes.`);
    location.href = payload.artifact.download_url;
  } catch (error) {
    showToast(error.message, true);
  } finally {
    button.disabled = false;
    button.removeAttribute("aria-busy");
    if (priorLabel) button.setAttribute("aria-label", priorLabel);
    button.innerHTML = prior;
  }
}

export async function copyText(value) {
  try {
    await navigator.clipboard.writeText(value);
    showToast("Copied to clipboard.");
  } catch {
    showToast("Clipboard access is unavailable.", true);
  }
}
