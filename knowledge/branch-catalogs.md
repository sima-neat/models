# Branch catalogs

How a branch's catalog is built from its own and its base branches' catalogs, what a push
compiles and benchmarks as a result, what a merge does, and how concurrent pushes resolve.
Read it before you change `tools/model_registry/stage_plan.py` or `publish_catalog`, or
when a push compiled more or less than you expected. The catalog document itself is in
[publication](publication.md#catalog-snapshots).

## The model

Each environment has one registry. In it, every branch has its own catalog, keyed by Git
ref, and every run is shared by all branches. A catalog lists the builds a branch delivers
at one revision. It may list builds that another branch compiled.

`is_long_lived` in `stage_plan.py` splits branches in two kinds:

| Kind | Branches | Catalog sources |
| --- | --- | --- |
| long-lived | `develop`, `main`, `release-*`, `integration/*` | its own catalog only. On its first push, the closest base branch that has a catalog. |
| feature | every other branch | its own catalog, then each base branch's catalog, closest first |

## Base branches

Git does not record which branch another was cut from. The `select` job therefore runs
`resolve-base` (`resolve_base_branches`), which works as follows:

- A `base_branch` dispatch input wins. The list is that one branch, which must exist on
  `origin`.
- Otherwise the candidates are `develop` and every `release-*` and `integration/*` branch
  on `origin` that shares history with the head. `main` keeps its own catalog but is never
  a resolved base.
- Candidates are ordered by distance from the merge base: the head's commits plus the
  candidate's commits, then fewer candidate commits, then `develop` first.
- A feature branch is never a candidate. A sibling that shares a commit would otherwise
  look closer than the branch both were cut from.

The list can be empty, for example on a push to `develop` with no release branch.

## The prior catalog

`read_catalog_sources` returns the source catalogs in priority order.
`merge_catalog_sources` combines them into one prior catalog. Each model build profile
takes its builds, packages, approved deployments, and benchmark fingerprints from one
source:

1. the first source whose builds carry every compile fingerprint the head declares for
   that profile
2. else the first source that lists the profile as pending at exactly those fingerprints
3. else the first source that has the profile at all

For a profile with an approved deployment, rule 1 compares the fingerprints in that
deployment's requirements, not the builds (`_carries_fingerprints`).

Rule 1 lets a feature branch that merged `develop` reuse both its own builds and the builds
`develop` recompiled. Rule 2 gives the branch the base branch's entry while the base branch
compiles, so the branch shows the base branch's previous build until its next push. Rule 3
keeps a profile's previous build and its benchmarks visible while the replacement compiles.

Two callers use the prior catalog. `plan_stages` decides against it, and `publish_catalog`
seeds a new catalog revision from it. Base branches seed a new revision only. Once the
branch's catalog is at the pushed revision, `publish_catalog` reads the branch's own
catalog alone.

## What a push compiles and benchmarks

`plan_stages` walks every compile invocation the checkout declares. It skips an on-demand
model unless the dispatch selected it ([on-demand builds](on-demand-builds.md)).

1. On a feature branch, it leaves a build to the closest base branch that has a catalog
   when that catalog lists the profile as pending at the head fingerprints. The base
   branch compiles and benchmarks it.
2. It compiles a build when the prior catalog has no build of some component with the
   head compile fingerprint, or when the dispatch sets `force_rebuild_all`.
3. Otherwise the build is reused. For each stage the profile declares, the plan computes
   the benchmark fingerprint from the reused builds and queues the stage unless a result
   exists.

| Decision | Where it looks |
| --- | --- |
| compile | the builds in the prior catalog |
| benchmark | a completed run with the benchmark fingerprint, from any branch (`stage_run_exists`). A feature branch also accepts a fingerprint that the closest base branch's catalog records. |

A benchmark fingerprint names every component's compile run. It therefore matches only a
benchmark of the same build, and a lookup across branches is safe.

A forced rebuild ignores step 1 and the base branch's recorded fingerprints. Without a
registry reader (`plan-stages --offline`), every selected automatic build with a direct
source compiles. A selected model with a conversion source fails the plan, because
`plan_conversions` needs a reader and a registry client.

## Merge

A merge is a push to `develop`, which compares with its own catalog. Builds whose compile
fingerprint the merge changed compile again on `develop`, and their benchmarks run again.
The builds the feature branch compiled are not reused.

A feature branch that later merges `develop` does not repeat that work. It leaves to
`develop` each profile `develop` is compiling and each benchmark whose fingerprint
`develop`'s catalog records.

## Concurrent pushes

- A newer push to the same ref cancels the running orchestrator with its compiles and
  benchmarks. Builds and benchmark runs that already published stay. The next run plans
  the builds left pending and the benchmarks without a completed run.
- A manual dispatch has its own concurrency group. It runs to completion, also beside
  a push run of the same ref.
- The builds of one run publish in parallel, each on its own. Every catalog write is a
  compare-and-swap that re-reads the catalog on a conflict
  ([catalog snapshots](publication.md#catalog-snapshots)).
- A base branch that changes during a run does not alter the branch's revision. After
  initialization the branch reads only its own catalog, and its pending list can only
  shrink (`_remaining_pending_builds`).

## What the model browser shows

The customer browser reads the published Main snapshot once. On production it reads
the snapshot its deploy pinned, so a later change to Main reaches it at the next deploy
([Pin the production catalog](../artifact-browser/README.md#pin-the-production-catalog)). The developer
browser renders the selected branch's snapshot while the branch's compile and
package lists load. Each browser saves the definitions and taxonomy locally for
repeat visits.

`branchModels` in `artifact-browser/branch-builds.js` assembles the view:

- It lists published builds, including inherited ones. A deployment selection
  replaces its profile's builds and packages
  (`deploymentCatalog`, [on-demand builds](on-demand-builds.md)).
- The developer view can overlay a newer completed build compiled on the
  selected branch when both builds' ages are known. It also shows completed
  builds for declared pending profiles. The customer view shows what the
  snapshot publishes.
- The developer view looks up missing exact run IDs in batches of up to 40.
  The run identifies its build's source branch, commit, and compiler. Without
  that data, the catalog can still show the published build, and a compiler
  filter only matches builds with a known compiler.
- A profile keeps its benchmark fingerprints only while the view shows the
  catalog's own build. A package must match every shown component's run ID.

## Design decisions

| Decision | Why |
| --- | --- |
| Catalogs are per branch and runs are shared. | A branch can deliver a build it did not compile, and a new branch needs no copy of anything. |
| The compile decision reads the catalog, and the benchmark decision reads runs. | The catalog pins which compile runs a branch delivers. A benchmark fingerprint already names those runs, so any matching run is a result for the same build. |
| A long-lived branch reads only its own catalog. | A merge recompiles what it changed, so every automatic build `develop` delivers comes from a `develop` run. |
| A feature branch leaves pending profiles and recorded benchmarks to its closest base branch. | The work a merge started on `develop` is not repeated on each branch that merges `develop`. |
| Feature branches are never base branches. | Graph distance cannot tell a sibling from the branch both were cut from. |
| Base branches seed a new revision only. | During a revision, a base branch could otherwise restore a package the revision withheld, or replace a published one when the base branch finishes rebuilding first. |
| A pending profile keeps its prior build. | The model stays visible, with its benchmarks, while the replacement compiles. |
| Packages are immutable within a revision. | A forced rebuild of a published package needs a new commit ([forced rebuilds](publication.md#forced-rebuilds)). |
| Catalog writes are compare-and-swap with a re-read. | Each build publishes as soon as it is done. A publisher that loses adds its build to the winner's catalog. |
