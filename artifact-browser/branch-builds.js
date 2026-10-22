// A branch shows what it delivers: its catalog's builds, each with the compile run that produced it.
// Keep components from one workflow together.
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

// One workflow compiles every component of a build; a run without CI provenance stands alone.
export function workflowIdentity(run) {
  return run.metadata.github_run_id || run.id;
}

export function runProfile(run) {
  return run.metadata.build_profile_id;
}

function runTime(run) {
  return Date.parse(run.finished_at || run.created_at || "") || 0;
}

// A variant build is the component runs of one model and build profile from one workflow.
export function variantBuildKey(run) {
  return JSON.stringify([
    run.metadata?.model_id, runProfile(run), run.git_ref, run.commit_sha,
    run.toolchain_name, run.toolchain_version, run.target_platform, workflowIdentity(run),
  ]);
}

// Groups runs into variant builds in the order the builds first appear. A rerun of a component
// replaces the earlier run of that component; the "model" component, else the first, anchors the build.
export function variantBuilds(runs) {
  const groups = new Map();
  for (const run of runs) {
    const key = variantBuildKey(run);
    if (!groups.has(key)) {
      groups.set(key, { key, modelId: run.metadata?.model_id, profile: runProfile(run), components: new Map(), time: 0 });
    }
    const group = groups.get(key);
    const component = run.metadata.component_id;
    const previous = group.components.get(component);
    if (!previous || runTime(run) > runTime(previous)) group.components.set(component, run);
    group.time = Math.max(group.time, runTime(run));
  }
  return [...groups.values()].map(({ components, ...group }) => {
    const sorted = [...components.entries()].sort(([left], [right]) => left.localeCompare(right)).map(([, run]) => run);
    return { ...group, runs: sorted, anchor: components.get("model") || sorted[0] };
  });
}

// Finds every component run of the build that produced `run`, whatever the visible filters are.
export async function variantBuildRuns(run, fetchPage) {
  const modelId = run.metadata?.model_id;
  if (!modelId) return [run];
  const [organization, repository] = run.repository.split("/");
  const query = new URLSearchParams({
    git_ref: run.git_ref, status: "completed", run_type: "model_compile", q: modelId, limit: "100",
  });
  const basePath = `/v1/repositories/${encodeURIComponent(organization)}/${encodeURIComponent(repository)}/runs`;
  const key = variantBuildKey(run);
  const siblings = (await allPages(fetchPage, basePath, query, "runs"))
    .filter((item) => item.id !== run.id && variantBuildKey(item) === key);
  const component = run.metadata.component_id;
  // The requested run always stands for its own component.
  const others = siblings.filter((item) => item.metadata.component_id !== component);
  return variantBuilds([run, ...others])[0].runs;
}

// A profile's deployment selection replaces its catalog builds and packages.
export function deploymentCatalog(model) {
  const selections = { ...model.approved_deployments, ...model.branch_deployments };
  const profiles = new Set(Object.keys(selections));
  const builds = (model.builds || []).filter(build => !profiles.has(build.build_profile_id));
  const packages = (model.packages || []).filter(pack => !profiles.has(pack.build_profile_id));
  for (const [profile, selection] of Object.entries(selections)) {
    const roots = selection.artifacts || [];
    const components = roots.filter(root => root.component_id && root.run_id);
    for (const root of components) {
      // The catalog build of this exact component run, not merely of its profile.
      const original = (model.builds || []).find(build => build.run_id === root.run_id
        && build.build_profile_id === profile && build.component_id === root.component_id);
      builds.push({ ...original, ...root, build_profile_id: profile });
    }
    for (const root of roots.filter(root => root.package_id && root.run_id)) {
      const ids = model.build_requirements?.[profile]?.packages
        ?.find(item => item.id === root.package_id)?.component_ids || [];
      packages.push({ ...root, build_profile_id: profile,
        component_run_ids: Object.fromEntries(components.filter(item => ids.includes(item.component_id))
          .map(item => [item.component_id, item.run_id])) });
    }
  }
  return { ...model, builds, packages };
}

// Resolves the compile runs behind catalog builds that `knownRuns` lacks, such as builds a new
// branch inherits. The catalog names each run and its commit; the run carries its own branch.
export async function catalogBuildRuns(catalogModels, knownRuns, fetchPage, repositoryPath) {
  catalogModels = catalogModels.map(deploymentCatalog);
  const known = new Set(knownRuns.map((run) => run.id));
  const missing = new Map(catalogModels.flatMap((model) => model.builds || [])
    .filter((build) => build.run_id && !known.has(build.run_id))
    .map((build) => [build.run_id, build]));
  const commits = [...new Set([...missing.values()].map((build) => build.commit_sha).filter(Boolean))];
  const byCommit = await Promise.all(commits.map((commit) => allPages(fetchPage, `${repositoryPath}/runs`,
    new URLSearchParams({ commit_sha: commit, status: "completed", run_type: "model_compile", limit: "100" }), "runs")));
  const runs = byCommit.flat().filter((run) => missing.has(run.id));
  // The run list omits superseded runs; a catalog may still name one.
  const found = new Set(runs.map((run) => run.id));
  const rest = await Promise.all([...missing.keys()].filter((id) => !found.has(id))
    .map(async (id) => (await fetchPage(`/v1/runs/${encodeURIComponent(id)}`)).run));
  return [...runs, ...rest];
}

// Shows the catalog's build of each model and build profile, unless `compileRuns` holds a newer
// completed build of it, which the catalog may not record yet.
export function branchModels(compileRuns, packageRuns, catalogModels, compiler = "") {
  catalogModels = catalogModels.map(deploymentCatalog);
  const definitions = new Map(catalogModels.map((model) => [model.model_id, model]));
  const runs = compileRuns.filter((run) => {
    const meta = run.metadata || {};
    return run.status === "completed" && run.run_type === "model_compile" && meta.model_id
      && (!compiler || run.toolchain_version === compiler);
  });
  const variantKey = (modelId, profile) => JSON.stringify([modelId, profile]);

  const latest = new Map();
  const runsById = new Map(runs.map((run) => [run.id, run]));
  for (const model of catalogModels) {
    for (const build of model.builds || []) {
      const run = runsById.get(build.run_id);
      if (!run || build.build_profile_id !== runProfile(run)
        || build.component_id !== run.metadata.component_id) continue;
      const key = variantKey(model.model_id, runProfile(run));
      if (!latest.has(key)) latest.set(key, { modelId: model.model_id, profile: runProfile(run), runs: [], time: 0, catalog: true });
      const group = latest.get(key);
      group.runs.push(run);
      group.time = Math.max(group.time, runTime(run));
    }
  }
  // A renamed or removed profile keeps its completed runs; show only the profiles the catalog declares.
  const declaredProfiles = new Map([...definitions].map(([modelId, model]) => [modelId,
    new Set([...(model.builds || []), ...(model.pending_builds || [])].map((build) => build.build_profile_id))]));
  for (const group of variantBuilds(runs)) {
    // A model removed from the catalog keeps its completed runs too; a branch without a catalog shows every run.
    if (definitions.size && !definitions.has(group.modelId)) continue;
    if (declaredProfiles.get(group.modelId)?.has(group.profile) === false) continue;
    const definition = definitions.get(group.modelId);
    if (definition?.build_on_demand || definition?.branch_deployments?.[group.profile]
      || definition?.approved_deployments?.[group.profile]) continue;
    const key = variantKey(group.modelId, group.profile);
    const previous = latest.get(key);
    // A build holding every run the catalog names is the catalog's build, with all its components.
    const isCatalogBuild = previous?.catalog && previous.runs.every((run) => group.runs.includes(run));
    const newer = !previous || group.time > previous.time
      || (!previous.catalog && group.time === previous.time && group.runs[0].id > previous.runs[0].id);
    if (isCatalogBuild || newer) latest.set(key, group);
  }

  const models = new Map();
  for (const group of latest.values()) {
    const definition = definitions.get(group.modelId) || { model_id: group.modelId, display_name: group.modelId };
    if (!models.has(group.modelId)) models.set(group.modelId, { ...definition, benchmark_fingerprints: {}, builds: [], packages: [] });
    const model = models.get(group.modelId);
    const exactBuilds = group.runs.map((run) => definition.builds?.find((build) => build.run_id === run.id
      && build.build_profile_id === group.profile && build.component_id === run.metadata.component_id));
    // A benchmark fingerprint names the MPKs it measured, so only a profile showing the catalog's own build keeps it.
    if (exactBuilds.every(Boolean) && definition.benchmark_fingerprints?.[group.profile]) {
      model.benchmark_fingerprints[group.profile] = definition.benchmark_fingerprints[group.profile];
    }
    group.runs.forEach((run, index) => {
      model.builds.push({
        ...exactBuilds[index],
        run_id: run.id,
        run,
        build_profile_id: group.profile,
        component_id: run.metadata.component_id,
        toolchain_name: run.toolchain_name,
        toolchain_version: run.toolchain_version,
        target_platform: run.target_platform,
        created_at: run.created_at,
        finished_at: run.finished_at,
      });
    });
  }

  for (const model of models.values()) {
    // A package belongs to the shown build when it bundles exactly that build's component runs.
    const candidates = [
      ...packageRuns.filter((run) => !model.build_on_demand
        && run.status === "completed" && run.run_type === "model_package"
        && run.metadata?.model_id === model.model_id
        && !model.branch_deployments?.[run.metadata.build_profile_id]
        && !model.approved_deployments?.[run.metadata.build_profile_id])
        .map((run) => ({ run_id: run.id, build_profile_id: run.metadata.build_profile_id, component_run_ids: run.metadata.component_run_ids })),
      ...(definitions.get(model.model_id)?.packages || []),
    ];
    for (const item of candidates) {
      const components = item.component_run_ids || {};
      const builds = model.builds.filter((build) => build.build_profile_id === item.build_profile_id);
      if (!builds.length || Object.keys(components).length !== builds.length
        || !builds.every((build) => components[build.component_id] === build.run_id)
        || model.packages.some((known) => known.run_id === item.run_id)) continue;
      model.packages.push({ ...item, run_id: item.run_id, build_profile_id: item.build_profile_id, component_run_ids: components });
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
