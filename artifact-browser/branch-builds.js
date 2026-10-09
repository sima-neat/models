// The model catalog describes models; only completed compile runs prove that a
// branch produced a build. Keep components from one workflow together.
export async function allPages(fetchPage, basePath, query, collection) {
  const items = [];
  const seen = new Set();
  let cursor = null;
  do {
    if (cursor) query.set("cursor", cursor);
    const page = await fetchPage(`${basePath}?${query}`);
    if (!Array.isArray(page[collection])) throw new Error(`The registry did not return ${collection}.`);
    items.push(...page[collection]);
    const next = page.next_cursor ?? page.cursor ?? null;
    if (page.has_more && (!next || seen.has(next))) throw new Error(`The registry returned an incomplete ${collection} page.`);
    cursor = page.has_more ? next : null;
    if (cursor) seen.add(cursor);
  } while (cursor);
  return items;
}

export function branchModels(compileRuns, packageRuns, catalogModels, compiler = "") {
  const definitions = new Map(catalogModels.map((model) => [model.model_id, model]));
  const groups = new Map();
  for (const run of compileRuns) {
    const meta = run.metadata || {};
    if (run.status !== "completed" || run.run_type !== "model_compile" || !meta.model_id) continue;
    if (compiler && run.toolchain_version !== compiler) continue;
    const profile = meta.build_profile_id || meta.variant_id || "default";
    const workflow = meta.github_run_id || run.ci_run_url || meta.compile_invocation_id || run.id;
    const key = JSON.stringify([meta.model_id, profile, run.commit_sha, run.toolchain_name, run.toolchain_version, run.target_platform, workflow]);
    if (!groups.has(key)) groups.set(key, { modelId: meta.model_id, profile, runs: [], time: 0 });
    const group = groups.get(key);
    group.runs.push(run);
    group.time = Math.max(group.time, Date.parse(run.finished_at || run.created_at || "") || 0);
  }

  const latest = new Map();
  for (const group of groups.values()) {
    const key = JSON.stringify([group.modelId, group.profile]);
    const previous = latest.get(key);
    if (!previous || group.time > previous.time || (group.time === previous.time && group.runs[0].id > previous.runs[0].id)) {
      latest.set(key, group);
    }
  }

  const models = new Map();
  for (const group of latest.values()) {
    const definition = definitions.get(group.modelId) || { model_id: group.modelId, display_name: group.modelId };
    if (!models.has(group.modelId)) models.set(group.modelId, { ...definition, builds: [], packages: [] });
    const model = models.get(group.modelId);
    const components = new Map();
    for (const run of group.runs) {
      const component = run.metadata?.component_id || "model";
      const previous = components.get(component);
      if (!previous || (Date.parse(run.finished_at || run.created_at || "") || 0)
        > (Date.parse(previous.finished_at || previous.created_at || "") || 0)) components.set(component, run);
    }
    for (const run of components.values()) {
      const component = run.metadata?.component_id || "model";
      const exactBuild = definition.builds?.find((build) => build.run_id === run.id);
      const catalogBuild = exactBuild
        || definition.builds?.find((build) => (build.build_profile_id || build.variant_id) === group.profile
          && (build.component_id || "model") === component);
      model.builds.push({
        run_id: run.id,
        run,
        build_profile_id: group.profile,
        component_id: component,
        artifact_suffix: catalogBuild?.artifact_suffix,
        // The accuracy key names the MPKs it measured, so only the catalog build of this run carries it.
        validation_fingerprint: exactBuild?.validation_fingerprint,
        toolchain_name: run.toolchain_name,
        toolchain_version: run.toolchain_version,
        target_platform: run.target_platform,
        created_at: run.created_at,
        finished_at: run.finished_at,
      });
    }
  }

  for (const model of models.values()) {
    for (const run of packageRuns) {
      const meta = run.metadata || {};
      const components = meta.component_run_ids || {};
      const componentIds = Object.values(components);
      const builds = model.builds.filter((build) => build.build_profile_id === meta.build_profile_id);
      if (run.status !== "completed" || run.run_type !== "model_package" || meta.model_id !== model.model_id
        || !builds.length || componentIds.length !== builds.length
        || !builds.every((build) => components[build.component_id] === build.run_id
          && run.git_ref === build.run.git_ref && run.commit_sha === build.run.commit_sha)) continue;
      model.packages.push({ run_id: run.id, build_profile_id: meta.build_profile_id, component_run_ids: components });
    }
    model.packages.sort((a, b) => {
      const left = packageRuns.find((run) => run.id === a.run_id);
      const right = packageRuns.find((run) => run.id === b.run_id);
      return (Date.parse(right?.finished_at || right?.created_at || "") || 0)
        - (Date.parse(left?.finished_at || left?.created_at || "") || 0);
    });
    model.builds.sort((a, b) => a.component_id.localeCompare(b.component_id));
  }
  return [...models.values()].sort((a, b) => a.model_id.localeCompare(b.model_id));
}
