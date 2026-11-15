// What each audience of the Model Browser sees. The build selects the audience; the customer view is the
// developer view without the parts below. Page code asks for one of these facts by name, so this list is
// the whole difference between the two views, together with three rules outside it: styles.css hides
// `.developer-only` markup and the branch and compiler tags, and environment.js keeps customers on Main.
export function audienceFacts(uiMode) {
  const developer = uiMode !== "customer";
  return {
    // Build history, and a link to one registry run.
    buildHistory: developer,
    // The compiler filter of the catalog.
    compilerFilter: developer,
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
  };
}
