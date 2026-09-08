// Keep component discovery independent of the visible search and compiler filters.
export async function variantDownloads(run, api, catalogModel = null) {
  const modelId = run.metadata?.model_id;
  const profile = run.metadata?.build_profile_id || run.metadata?.variant_id;
  const packs = (summary) => (summary.artifacts || []).filter((item) => item.artifact_type === "model-pack");
  let runs = [run];
  if (modelId && profile) {
    const [organization, repository] = run.repository.split("/");
    const query = new URLSearchParams({
      git_ref: run.git_ref, status: "completed", run_type: "model_compile", q: modelId, limit: "100",
    });
    const candidates = [];
    do {
      const payload = await api(`/v1/repositories/${encodeURIComponent(organization)}/${encodeURIComponent(repository)}/runs?${query}`);
      candidates.push(...(payload.runs || []));
      if (!payload.has_more || !payload.next_cursor) break;
      query.set("cursor", payload.next_cursor);
    } while (true);
    const sameBuild = candidates.filter((item) => item.metadata?.model_id === modelId
      && (item.metadata.build_profile_id || item.metadata.variant_id) === profile
      && item.git_ref === run.git_ref && item.commit_sha === run.commit_sha
      && item.toolchain_name === run.toolchain_name && item.toolchain_version === run.toolchain_version
      && item.target_platform === run.target_platform
      && (run.metadata?.github_run_id
        ? item.metadata?.github_run_id === run.metadata.github_run_id
        : run.ci_run_url ? item.ci_run_url === run.ci_run_url : item.id === run.id));
    const components = new Map();
    sameBuild.sort((a, b) => String(a.created_at).localeCompare(String(b.created_at)))
      .forEach((item) => components.set(item.metadata?.component_id || "model", item));
    components.set(run.metadata?.component_id || "model", run);
    runs = [...components.values()];
  }
  const downloads = await Promise.all(runs.map(async (item) => {
    const summary = await api(`/v1/runs/${encodeURIComponent(item.id)}`);
    const artifacts = packs(summary);
    const component = item.metadata?.component_id || "model";
    if (!artifacts.length) throw new Error(`No model file is available for ${component}. No downloads were started.`);
    const build = catalogModel?.model_id === modelId ? catalogModel.builds.find((build) =>
      build.run_id === item.id && build.component_id === component && build.build_profile_id === profile) : null;
    const downloadFilename = build?.artifact_suffix ? `${modelId}_${build.artifact_suffix}_mpk.tar.gz` : null;
    return artifacts.map((artifact) => ({
      runId: item.id, component, name: artifact.name, size: artifact.size_bytes, downloadFilename,
    }));
  }));
  return downloads.flat().sort((a, b) => a.component.localeCompare(b.component));
}

export async function prepareDownloads(downloads, api) {
  // Resolve every link before starting any transfer, so preparation can be retried safely.
  return Promise.all(downloads.map(async (item) => {
    const query = item.downloadFilename ? `?${new URLSearchParams({download_filename: item.downloadFilename})}` : "";
    const payload = await api(`/v1/runs/${encodeURIComponent(item.runId)}/artifacts/${encodeURIComponent(item.name)}${query}`);
    if (!payload.artifact?.download_url) throw new Error(`No download URL for ${item.component}. No downloads were started.`);
    const artifact = payload.artifact;
    // The logical registry name (e.g. "mpk") is not the archive filename.
    const filename = artifact.s3_key
      ? artifact.s3_key.split("/").pop()
      : decodeURIComponent(new URL(artifact.download_url, "https://registry.invalid").pathname.split("/").pop());
    return { ...item, filename: item.downloadFilename || filename || "Filename unavailable", url: artifact.download_url };
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
