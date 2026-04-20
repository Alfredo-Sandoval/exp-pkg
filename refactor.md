# Refactor Plan

This file tracks the storage cutover so the repo does not lose the thread
mid-refactor.

## Current Decisions

- Public editable project = workspace folder
- Private durable/runtime state = `.xpkg/`
- Rebuildable local cache = `.xpkg/state/current.json`
- Durable committed source of truth = store head under `.xpkg/`
- Normal workspace head payload = workspace-native snapshot root
- Portable packed artifact = `.expkg`
- Legacy `.xpkg` archive handling = explicit migration seam plus internal compatibility tooling

## Current Reality

The storage model is now materially cut over:

- workspace save/import/migrate commit a snapshot root into the durable store
- `.xpkg/state/current.json` is written from committed workspace-native state
- workspace load prefers the snapshot cache only when its embedded
  `xpkg_commit_id` matches the durable head
- stale or missing snapshot caches rebuild from the committed durable head
- pre-cutover workspaces with only `.xpkg/state/current.xpkg` now require
  explicit `migrate_legacy_archive(...)` cutover before workspace-first
  load/save helpers are used
- public archive-first facades (`xpkg.adapters`, `xpkg convert`,
  `current_project_archive_path(...)`, `import_legacy_archive(...)`) have been removed

Archive support still exists, but it is no longer the normal committed
workspace write path or a first-class downstream API.

## Ordered Phases

### Phase 1: Rename the packed artifact

Goal:

- replace `.poseproj` with `.expkg`

Status:

- completed

### Phase 2: Split workspace responsibilities

Goal:

- break `src/xpkg/io/project_workspace.py` into smaller boundaries

Status:

- completed enough for the storage cutover

Progress:

- extracted descriptor/layout helpers into `src/xpkg/io/project_layout.py`
- extracted pack/unpack/validation helpers into `src/xpkg/io/project_artifact.py`
- reduced `src/xpkg/io/project_workspace.py` to workspace/store/import/media work
- added `src/xpkg/io/workspace_snapshot_backend.py` for the native snapshot path
- made `WorkspaceService.describe()` report current workspace state rather than
  archive-only state

### Phase 3: Move the durable head from archive commits to snapshot commits

Goal:

- make workspace-native state the committed durable contract

Status:

- completed for the normal workspace path

Progress:

- the durable store now supports generic named roots instead of archive-only
  runtime behavior
- normal workspace save/import/migrate commit `roots["snapshot"]`
- `.xpkg/state/current.json` is rebuilt from committed workspace-native state
- stale snapshot protection still keys off the durable commit id
- archive-backed durable heads remain readable where they still exist, but
  pre-cutover workspaces no longer auto-adopt legacy archive state

### Phase 4: Demote archive handling to explicit compatibility

Goal:

- keep direct `.xpkg` archive handling available only where it is still needed

Progress:

- durable-store commit roots now hydrate through typed `RootEntry` values
  instead of raw root dictionaries
- the public API/docs/CLI now route project-facing work through
  `WorkspaceService`, `workspace.imports.*`, and `migrate_legacy_archive(...)`
- direct archive conversion/export convenience surfaces were removed from the
  main public facades

Current allowed uses:

- explicit legacy archive migration
- internal archive materialization where a compatibility workflow truly needs it
- compatibility fixtures and tests

Current disallowed direction:

- no new normal workspace save/load flows should depend on archive-first
  commits
- no new public examples should market direct `.xpkg` conversion as the native
  workflow

## Remaining Seams

- keep the retained legacy migration seam narrow and explicit
- keep shrinking archive-first assumptions in internal tests/docs when safe

## Known Hotspots

- `src/xpkg/io/project_workspace.py`
- `src/xpkg/io/workspace_snapshot_backend.py`
- `src/xpkg/io/archive_store/store.py`
- `src/xpkg/io/labels/serialization.py`
- `tests/test_workspace_store_cache_sync.py`
- `tests/test_project_workspace.py`

## Verification Expectations

For each phase:

- update docs in the same change as code
- keep targeted pytest coverage passing for touched surfaces
- prefer cutover over compatibility aliases unless there is a concrete migration
  reason

## Notes

- `docs/architecture/storage-direction.md` holds the storage rationale
- this file is the execution tracker for the remaining storage seams
