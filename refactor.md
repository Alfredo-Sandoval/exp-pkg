# Refactor Plan

This file tracks the storage cutover so the repo does not lose the thread
mid-refactor.

## Current Decisions

- Public editable project = workspace folder
- Private mutable/runtime state = `.posetta/`
- Portable packed artifact = `.expkg`
- Canonical native bundle = `.sta`
- `.siesta` is not a future-facing native format
- `.siesta` should end as migration/import-only and then be removable

## Current Reality

The public contract and the live workspace path are now closer.

Today the workspace path uses a native JSON snapshot as its source of truth:

- workspace save/load/import/migrate write and read
  `.posetta/state/current.json`
- workspace-native predictions are preserved in the snapshot payload as a
  detached `predictions` section
- archive reads remain as a fallback for older workspaces and for explicit
  bundle-facing workflows

The remaining archive dependency is now below the normal workspace hot path:

- the durable store still commits immutable archive objects in
  `src/posetta/io/siesta_store/store.py`
- explicit `.sta` / legacy `.siesta` workflows still use
  `src/posetta/io/siesta_format/`

## Ordered Phases

### Phase 1: Rename the packed artifact

Goal:

- replace `.poseproj` with `.expkg`

Scope:

- constants and helper names in `src/posetta/io/project_workspace.py`
- public exports in `src/posetta/formats/project.py` and `src/posetta/formats/__init__.py`
- CLI text and validation messages in `src/posetta/cli.py`
- load guards in `src/posetta/io/labels/serialization.py`
- tests and docs

Status:

- completed

### Phase 2: Split workspace responsibilities

Goal:

- break `src/posetta/io/project_workspace.py` into smaller boundaries

Target split:

- descriptor/layout helpers
- workspace save/commit path
- migration/import path
- pack/unpack path
- validation path

Reason:

- the file currently mixes public artifact semantics, media management, import
  orchestration, archive staging, and validation in one place

Progress:

- extracted descriptor/layout helpers into `src/posetta/io/project_layout.py`
- extracted pack/unpack/validation helpers into `src/posetta/io/project_artifact.py`
- reduced `src/posetta/io/project_workspace.py` to store/save/import/media work
- switched the canonical native bundle suffix from `.siesta` to `.sta`
- changed conversion results and CLI plumbing from `siesta_path` to
  `bundle_path`
- made workspace state default to `current.sta` while still accepting
  `current.siesta` during transition
- extracted staged bundle rewrite/update/migration mechanics into
  `src/posetta/io/workspace_bundle_backend.py`
- removed direct HDF5 bundle surgery from `src/posetta/io/project_workspace.py`
- switched workspace load/save/import/migrate to the native snapshot path in
  `src/posetta/io/workspace_snapshot_backend.py`
- made `WorkspaceService.describe()` report current workspace state rather than
  archive-only state

Remaining:

- move durable store commits from immutable archive blobs to workspace-native
  snapshots
- decide whether archive fallback should stay in the workspace loader or move to
  explicit migration only

### Phase 3: Stop using `.siesta` as the workspace save engine

Goal:

- make workspace-native state the source of truth

Required outcome:

- workspace save no longer writes a staged `.sta` bundle first
- `.posetta/` stores canonical state directly
- save/update/load behavior prefers the snapshot path and only falls back to
  archives for older workspaces

Status:

- completed for the normal workspace path

Progress:

- added a workspace-native snapshot backend in
  `src/posetta/io/workspace_snapshot_backend.py`
- workspace saves now materialize `.posetta/state/current.json` from the
  committed workspace state
- workspace loads now prefer `.posetta/state/current.json` over the current
  bundle path
- labels JSON now roundtrips named tracks and frame-level segmentation
- workspace snapshots now preserve detached predictions alongside labels

Remaining:

- stop rebuilding the snapshot from a committed `.sta` bundle and write it
  directly from workspace-native state
- move the durable store from archive commits to snapshot/state commits
- remove the remaining archive-first assumptions from pack/validate/store APIs

### Phase 4: Demote `.siesta` to one-way compatibility

Goal:

- keep `.siesta` only for import/migration during the transition

Allowed uses:

- explicit legacy import
- fixtures needed during cutover
- migration tooling
- archive fallback for older workspaces while the durable store still points at
  archive blobs

Disallowed direction:

- no new native save/load flows should target `.siesta`

### Phase 5: Delete native `.siesta` write/update paths

Goal:

- remove `.siesta` as a first-class product/storage concept

Exit criteria:

- workspace-native storage handles labels, predictions, segmentation, metadata,
  and portable export generation
- store commits no longer point at archive blobs
- public docs no longer frame `.siesta` as part of the normal user workflow

## Known Hotspots

- `src/posetta/io/project_workspace.py`
- `src/posetta/io/siesta_store/store.py`
- `src/posetta/io/siesta_format/writer_core.py`
- `src/posetta/io/siesta_format/reader_core.py`
- `src/posetta/io/labels/serialization.py`

## Verification Expectations

For each phase:

- update docs in the same change as code
- keep targeted pytest coverage passing for touched surfaces
- prefer cutover over compatibility aliases unless there is a concrete migration
  reason

## Notes

- `docs/architecture/storage-direction.md` holds the storage rationale
- this file is the execution tracker for the code cutover
