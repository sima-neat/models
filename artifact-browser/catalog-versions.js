// The published catalog versions a customer can browse: the registry's list of them, the selected one, and
// the page URL parameters that carry the choice. A version is one Platform with one Neat Core. With none
// selected the page shows Latest, the live catalog of the default branch.

const PLATFORM_PARAM = "platform_version";
const CORE_PARAM = "core_version";
const COMPILER_PARAM = "compiler";

// The registry's answer, or null when it lists no versions: the route failed or is not deployed.
let versions = null;
let discovery = null;
let selected = null;

async function fetchCatalogVersions(fetchPage, repositoryPath, gitRef) {
  const path = `${repositoryPath}/models/filters`;
  const [published, branch] = await Promise.all([
    fetchPage(path),
    fetchPage(`${path}?${new URLSearchParams({ git_ref: gitRef })}`).catch(() => null),
  ]);
  if (!Array.isArray(published.selections)) throw new Error("The registry did not return its published versions.");
  return {
    selections: published.selections.filter((selection) => !selection.retired),
    latestCompilers: branch?.toolchain_versions || [],
  };
}

// Asks the registry once per page load. A failed request leaves the page without version filters.
export function loadCatalogVersions(fetchPage, repositoryPath, gitRef) {
  discovery ||= fetchCatalogVersions(fetchPage, repositoryPath, gitRef).catch(() => null).then((answer) => {
    versions = answer;
  });
  return discovery;
}

export function versionFiltersAvailable() {
  return Boolean(versions);
}

export function publishedSelections() {
  return versions?.selections || [];
}

export function platformVersions(selections) {
  return [...new Set(selections.map((selection) => selection.platform_version))].sort();
}

function platformSelections(selections, platform) {
  return selections.filter((selection) => selection.platform_version === platform)
    .sort((left, right) => (left.neat_core_version < right.neat_core_version ? -1 : left.neat_core_version > right.neat_core_version ? 1 : 0));
}

export function coreVersions(selections, platform) {
  return platformSelections(selections, platform).map((selection) => selection.neat_core_version);
}

// The Neat Core a platform opens with: the one the registry marks as its default, else the last by name.
export function defaultSelection(selections, platform) {
  const published = platformSelections(selections, platform);
  return published.find((selection) => selection.is_default) || published.at(-1) || null;
}

// Selects the published version of `platform` with `core`, or with the platform's default Neat Core when
// `core` is empty. Latest is selected when `platform` is empty or the version is not published.
export function selectVersion(platform, core = "") {
  const published = publishedSelections();
  selected = (core
    ? published.find((selection) => selection.platform_version === platform && selection.neat_core_version === core)
    : defaultSelection(published, platform)) || null;
  return selected;
}

export function selectedVersion() {
  return selected;
}

// What a pinned registry read sends as `selection_id`, or null for Latest.
export function selectionId() {
  return selected?.selection_id || null;
}

// The compilers that built the shown catalog: the selected version's, or for Latest the default branch's.
export function compilerVersions() {
  return selected ? selected.toolchain_versions || [] : versions?.latestCompilers || [];
}

// Selects the version a link names. Returns false when the link names one that is not published.
export function selectLinkedVersion(params) {
  const platform = params.get(PLATFORM_PARAM);
  if (!versions || !platform) {
    selected = null;
    return true;
  }
  return Boolean(selectVersion(platform, params.get(CORE_PARAM) || ""));
}

export function linkedCompiler(params) {
  return params.get(COMPILER_PARAM) || "";
}

// Latest and any compiler leave their parameters out.
export function writeVersionParams(url, compiler) {
  const values = [
    [PLATFORM_PARAM, selected?.platform_version],
    [CORE_PARAM, selected?.neat_core_version],
    [COMPILER_PARAM, compiler],
  ];
  values.forEach(([name, value]) => {
    if (value) url.searchParams.set(name, value);
    else url.searchParams.delete(name);
  });
}
