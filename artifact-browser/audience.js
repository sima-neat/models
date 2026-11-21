// What each audience of the Model Browser sees. The build selects the audience; the customer view is the
// developer view without the parts below, and with the version filters, which only it has. Page code asks
// for one of these facts by name, so this list is the whole difference between the two views, together
// with three rules outside it: styles.css hides `.developer-only` markup and the branch and compiler tags,
// and environment.js keeps customers on Main. styles.css also shows the Compiler filter to customers
// when the version filters are shown.
export function audienceFacts(uiMode) {
  const developer = uiMode !== "customer";
  return {
    // Build history, and a link to one registry run.
    buildHistory: developer,
    // The compiler filter that lists the compilers of the branch's builds.
    compilerFilter: developer,
    // Customer only: the Platform and Neat Core filters, and a Compiler filter that lists the compilers of the
    // chosen version, when the registry lists its published versions.
    versionFilters: !developer,
    // Branch runs can augment the published catalog for developers.
    catalogRunMetadata: developer,
    // The Details section of a model page: components, build provenance, package and source inputs.
    buildDetails: developer,
    // The commit and workflow that measured a result.
    measurementSource: developer,
    // The model report as the registry stores it.
    rawResult: developer,
    // Memory every process on the board uses, which describes the board rather than the model.
    boardMemory: developer,
    // MLA Utilization under Performance.
    mlaUtilization: developer,
    // How an accuracy run was evaluated, and the tables of published and FP32 baseline scores.
    evaluationDetails: developer,
    referenceTables: developer,
    // Variants whose accuracy falls more than 8% below the source model's reference score.
    lowAccuracyVariants: developer,
    // The branch in the copied CLI command. Customers download from main, the sima-cli default.
    commandBranch: developer,
  };
}
