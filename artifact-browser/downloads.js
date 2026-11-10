import { deploymentCatalog, variantBuildRuns } from "./branch-builds.js";

// A model's compiled artifacts are offered only under a license that licenses.json lists as downloadable.
// An unlisted license, or a list that did not load, offers none.
export function downloadOffered(licenseId, licenses) {
  return licenses?.[licenseId]?.downloadable === true;
}

// Delivery comes from the resolved artifact, never from the source checkpoint URL.
export function artifactDelivery(artifact = {}) {
  if (artifact.metadata?.storage_backend === "huggingface") return "huggingface";
  if (artifact.metadata?.layout === "directory") return "directory";
  return "archive";
}

export function deploymentSize(artifact = {}) {
  return artifactDelivery(artifact) === "archive" ? artifact.size_bytes : artifact.metadata?.deployment_size_bytes;
}

export function huggingFaceLocation(artifact = {}) {
  const location = artifact.metadata?.huggingface;
  if (!location || !/^[a-zA-Z0-9_.-]+\/[a-zA-Z0-9_.-]+$/.test(location.repo_id)
    || !/^[a-f0-9]{40}$/i.test(location.revision)) return null;
  const parts = location.subdirectory ? location.subdirectory.split("/") : [];
  if (parts.some((part) => !part || part === "." || part === ".." || part.includes("\\"))) return null;
  return { ...location, url: `https://huggingface.co/${location.repo_id}/tree/${location.revision}${parts.length ? `/${parts.map(encodeURIComponent).join("/")}` : ""}` };
}

// Keep component discovery independent of the visible search and compiler filters.
export async function variantDownloads(run, api, catalogModel = null) {
  if (catalogModel) catalogModel = deploymentCatalog(catalogModel);
  const modelId = run.metadata?.model_id;
  const profile = run.metadata?.build_profile_id;
  const packs = (summary) => (summary.artifacts || []).filter((item) => item.artifact_type === "model-pack");
  const roots = (catalogModel?.branch_deployments?.[profile] || catalogModel?.approved_deployments?.[profile])
    ? catalogModel.builds.filter((build) => build.build_profile_id === profile) : [];
  const selected = roots.some((build) => build.run_id === run.id) ? roots : [];
  const runs = selected.length ? await Promise.all(selected.map(async (build) =>
    (await api(`/v1/runs/${encodeURIComponent(build.run_id)}`)).run))
    : modelId && profile ? await variantBuildRuns(run, api) : [run];
  const downloads = await Promise.all(runs.map(async (item) => {
    const summary = await api(`/v1/runs/${encodeURIComponent(item.id)}`);
    const root = selected.find((build) => build.run_id === item.id);
    const artifacts = packs(summary).filter((artifact) => !root || artifact.name === root.artifact_name);
    const component = item.metadata.component_id;
    if (!artifacts.length) throw new Error(`No model file is available for ${component}. No downloads were started.`);
    return artifacts.map((artifact) => ({
      runId: item.id, component, name: artifact.name, size: deploymentSize(artifact),
      delivery: artifactDelivery(artifact), artifact,
    }));
  }));
  return downloads.flat().sort((a, b) => a.component.localeCompare(b.component));
}

// Directory manifests describe folders; archive packages contain an MPK or ZIP.
export function packageFormat(artifact, filename = "") {
  if (artifact && artifactDelivery(artifact) !== "archive") return "Folder";
  return artifact?.media_type === "application/gzip" || filename.endsWith("_mpk.tar.gz") ? "MPK" : "ZIP";
}

export async function prepareDownloads(downloads, api) {
  // Resolve every link before starting any transfer, so preparation can be retried safely.
  return Promise.all(downloads.map(async (item) => {
    const payload = await api(`/v1/runs/${encodeURIComponent(item.runId)}/artifacts/${encodeURIComponent(item.name)}`);
    if (!payload.artifact) throw new Error(`No download URL for ${item.component}. No downloads were started.`);
    const artifact = { ...item.artifact, ...payload.artifact, metadata: { ...item.artifact?.metadata, ...payload.artifact.metadata } };
    const delivery = artifactDelivery(artifact);
    if (delivery !== "archive") return { ...item, delivery, artifact, size: deploymentSize(artifact), url: null };
    if (!artifact.download_url) throw new Error(`No download URL for ${item.component}. No downloads were started.`);
    // The logical registry name (e.g. "mpk") is not the archive filename.
    const filename = artifact.s3_key
      ? artifact.s3_key.split("/").pop()
      : decodeURIComponent(new URL(artifact.download_url, "https://registry.invalid").pathname.split("/").pop());
    return { ...item, filename: filename || "Filename unavailable", url: artifact.download_url };
  }));
}

export function startDownloads(downloads, document) {
  if (downloads.some((item) => item.delivery && item.delivery !== "archive")) throw new Error("Folder deployments require the download instructions.");
  // Separate browsing contexts prevent one file navigation from cancelling another.
  // Stream directly to the browser download manager; never buffer model packs in memory.
  downloads.forEach((item) => {
    const frame = document.createElement("iframe");
    frame.hidden = true;
    frame.title = `Download ${item.component}`;
    frame.src = item.url;
    document.body.append(frame);
  });
}

// Quote catalog values as shell arguments; branch names can contain shell metacharacters.
export function modelDownloadCommand(modelId, profile, branch, staging = false) {
  const quote = value => /^[a-zA-Z0-9_./-]+$/.test(String(value)) ? String(value)
    : "'" + String(value).replaceAll("'", "'\"'\"'") + "'";
  return `sima-cli models download --id ${quote(modelId)} --variant ${quote(profile)} --branch ${quote(branch)}${staging ? " --stg" : ""}`;
}
