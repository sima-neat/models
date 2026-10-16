import { variantBuildRuns } from "./branch-builds.js";

// Keep component discovery independent of the visible search and compiler filters.
export async function variantDownloads(run, api) {
  const modelId = run.metadata?.model_id;
  const profile = run.metadata?.build_profile_id || run.metadata?.variant_id;
  const packs = (summary) => (summary.artifacts || []).filter((item) => item.artifact_type === "model-pack");
  const runs = modelId && profile ? await variantBuildRuns(run, api) : [run];
  const downloads = await Promise.all(runs.map(async (item) => {
    const summary = await api(`/v1/runs/${encodeURIComponent(item.id)}`);
    const artifacts = packs(summary);
    const component = item.metadata?.component_id || "model";
    if (!artifacts.length) throw new Error(`No model file is available for ${component}. No downloads were started.`);
    return artifacts.map((artifact) => ({ runId: item.id, component, name: artifact.name, size: artifact.size_bytes }));
  }));
  return downloads.flat().sort((a, b) => a.component.localeCompare(b.component));
}

// A single-component package is published as its MPK; any other package is a ZIP.
export function packageFormat(artifact, filename = "") {
  return artifact?.media_type === "application/gzip" || filename.endsWith("_mpk.tar.gz") ? "MPK" : "ZIP";
}

export async function prepareDownloads(downloads, api) {
  // Resolve every link before starting any transfer, so preparation can be retried safely.
  return Promise.all(downloads.map(async (item) => {
    const payload = await api(`/v1/runs/${encodeURIComponent(item.runId)}/artifacts/${encodeURIComponent(item.name)}`);
    if (!payload.artifact?.download_url) throw new Error(`No download URL for ${item.component}. No downloads were started.`);
    const artifact = payload.artifact;
    // The logical registry name (e.g. "mpk") is not the archive filename.
    const filename = artifact.s3_key
      ? artifact.s3_key.split("/").pop()
      : decodeURIComponent(new URL(artifact.download_url, "https://registry.invalid").pathname.split("/").pop());
    return { ...item, filename: filename || "Filename unavailable", url: artifact.download_url };
  }));
}

export function startDownloads(downloads, document) {
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
