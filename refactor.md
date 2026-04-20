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
- Direct `.xpkg` archive handling = compatibility/import/export surface

## Current Reality

The storage model is now meaningfully cut over:

- workspace save/import/migrate commit a snapshot root into the durable store
- `.xpkg/state/current.json` is written from committed workspace-native state
- workspace load prefers the snapshot cache only when its embedded
  `xpkg_commit_id` matches the durable head
- stale or missing snapshot caches rebuild from the committed durable head
- legacy archive-backed workspaces still open through explicit archive fallback

Archive support still exists, but it is no longer the normal committed
workspace write path.

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
- archive-backed heads remain readable for older workspaces

### Phase 4: Demote archive handling to explicit compatibility

Goal:

- keep direct `.xpkg` archive handling available only where it is still needed

Current allowed uses:

- explicit archive import/export workflows
- migration from older archive-backed state
- compatibility fixtures and tests
- lazy archive materialization for archive-facing helpers

Current disallowed direction:

- no new normal workspace save/load flows should depend on archive-first
  commits

## Remaining Seams

- decide how long the workspace loader should keep automatic archive fallback
  for older workspaces
- consider adding typed root-entry helpers in the durable-store schema instead
  of raw root dictionaries
- consider whether explicit archive materialization should stay a helper-level
  behavior or move behind a more explicit export API
- keep shrinking archive-first assumptions in tests and docs

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
