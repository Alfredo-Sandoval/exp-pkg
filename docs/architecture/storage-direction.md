# Storage Direction

<div class="page-intro">
<p>
If direct <code>.xpkg</code> archive handling feels slightly off in the current
xpkg story, that reaction is reasonable. The public product contract is now
workspace folder + private <code>.xpkg/</code> state + portable
<code>.expkg</code> export, while the implementation still relies on
archive-shaped compatibility internals for some save, migration, and durable
commit flows.
</p>
</div>

!!! info
    Status: current implementation notes. Today the committed source of truth is
    the durable store head under <code>.xpkg/</code>, while
    <code>.xpkg/state/current.json</code> is a rebuildable cache.

## Current Truth

Today xpkg has four storage ideas in play, but the live workspace path has
already moved forward:

- workspace root as the editable project boundary
- `.xpkg/` as the private mutable store boundary
- `.expkg` as the portable packed artifact
- `.xpkg` as the low-level direct archive format

The normal workspace save/load/import/migrate flow now treats the durable store
head as committed truth and uses `.xpkg/state/current.json` as a rebuildable
local cache. Direct archive reads still remain in the codebase for older
workflows, migration, fixtures, and explicit archive-facing tools.

That split explains the current tension. The public contract is workspace-first,
but parts of the implementation still stage, validate, and commit archive files
to get complete round-trip behavior.

## Why The Archive Engine Is Still Here

### 1. It is still the complete storage engine

The current round-trip serializer still lives in the archive layer:

- `xpkg.io.archive_format.write_archive`
- `xpkg.io.archive_format.update_labels_archive`
- `xpkg.io.archive_format.read_archive`

Those functions already know how to carry labels, predictions, segmentation,
metrics, metadata, and manifest information together. The workspace layer does
not yet have an independent backend with the same coverage.

### 2. Workspace saves still stage archives

The workspace code is already public-facing, but parts of its save path still
run through staged archive files.

Archive dependency is now concentrated in compatibility and migration seams:

- `migrate_legacy_archive(...)`
- `import_dlc_csv_workspace(...)`
- `import_dlc_h5_workspace(...)`
- `import_sleap_package_workspace(...)`
- archive fallback when opening an older workspace that does not yet have a
  native snapshot

### 3. The durable store still commits immutable archive objects

The new private store is real. It has recovery semantics, journaled commit
boundaries, and immutable objects under `.xpkg/`.

But the object it currently commits is still an archive file:

- `ArchiveStore.create_from_archive(...)`
- `ArchiveStore.current_archive_path()`
- `ArchiveStore.commit_new_archive(...)`

So the store is newer than the archive format, but it still wraps the archive
format instead of replacing it.

### 4. Migration and fixtures still matter

Existing adapters, tests, fixtures, and migration flows still move through the
archive compatibility engine. Keeping that engine available has practical value
while the workspace-first surface hardens.

## Why It Feels Wrong

The discomfort is structural, not cosmetic.

- The public story says workspace and experiment packaging, but part of the
  save engine still speaks archive.
- The workspace API is thin and clean, but its implementation still depends on
  a lower-level compatibility format for the committed payload.
- The public contract has moved faster than the underlying storage cutover.

So when someone asks why direct archive IO still exists, the honest answer is:
because it is still the only fully implemented round-trip storage backend, even
though it is no longer the product contract we want to lead with.

## Recommended Position

xpkg should treat the direct archive layer as a transition mechanism, not the
product identity.

That means:

- keep `.xpkg` archive handling available for migration, fixtures, import, and
  low-level compatibility work
- stop expanding the public product story around direct archive workflows
- keep archive terminology out of the primary artifact contract
- be explicit that the workspace/store layer is the future-facing boundary

In other words: the current archive engine can stay for a while, but it should
feel narrow and transitional rather than native and aspirational.

## Cutover Paths

### Option A: Keep archive payloads as private commit units for now

This is the least disruptive path.

- keep the current save path working
- reduce user-facing emphasis on direct archive handling
- make the workspace/store layer the primary public contract
- avoid designing new features directly against archive mutation

This is the best short-term option if stability matters more than immediate
storage surgery.

### Option B: Replace archive-backed commits with workspace-native state

This is the real cutover.

- workspace save stops calling `write_archive(...)` directly
- the store commits normalized workspace state instead of a single archive file
- labels, segmentation, predictions, and metadata get a backend-neutral storage
  contract
- the archive layer becomes export/import compatibility instead of the
  canonical save engine

This is the most aligned end-state, but it is a real refactor, not a wording
change.

## What Needs To Happen Before Archive Handling Can Shrink

The current code suggests a practical sequence.

1. Split `xpkg.io.project_workspace` into smaller layers.
   Right now descriptor logic, media management, imports, migration, save
   staging, packing, and validation all live in one module.
2. Separate store protocol from archive semantics.
   `ArchiveStore` currently has real durability logic, but its roots still point
   at immutable archive files.
3. Extract a backend-neutral save model.
   The workspace layer needs a storage contract for labels, segmentation,
   predictions, and metadata that does not require writing an archive first.
4. Broaden the workspace service beyond labels-only save/load semantics.
   The current service still exposes `load_labels()` and `save_labels(...)` as
   its main mutable operations.

Until those changes happen, direct archive handling will keep showing up
because the runtime still depends on it for the actual durable payload.

## Bottom Line

Direct `.xpkg` archive handling is still here because it remains the only
complete round-trip storage engine in the implementation.

That is useful in the short term, but it also explains why the code still feels
partly anchored to an older storage model. The right posture is to keep archive
handling available as a compatibility substrate while moving product language,
public contracts, and future storage work toward the workspace/store boundary.
