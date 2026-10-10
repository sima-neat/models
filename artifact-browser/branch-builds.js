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

// One workflow compiles every component of a build; a run without CI provenance stands alone.
export function workflowIdentity(run) {
  const meta = run.metadata || {};
  return meta.github_run_id || run.ci_run_url || meta.compile_invocation_id || run.id;
}

export function runProfile(run) {
  return run.metadata?.build_profile_id || run.metadata?.variant_id || "default";
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
    const component = run.metadata?.component_id || "model";
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
  const component = run.metadata?.component_id || "model";
  // The requested run always stands for its own component.
  const others = siblings.filter((item) => (item.metadata?.component_id || "model") !== component);
  return variantBuilds([run, ...others])[0].runs;
}
