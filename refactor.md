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

The public contract has moved ahead of the implementation.

Today the code still relies on HDF5 bundle staging for the actual durable
payload path:

- workspace saves still stage and commit `.sta` bundles in
  `src/posetta/io/project_workspace.py`
- the durable store still commits immutable archive objects in
  `src/posetta/io/siesta_store/store.py`
- the only full round-trip serializer still lives in
  `src/posetta/io/siesta_format/`

That means the product story is workspace-first, but the runtime storage engine
is still archive-first underneath.

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

Remaining:

- isolate the workspace save path from the archive serializer
- move the current `.sta` staging flow behind a backend-neutral save contract

### Phase 3: Stop using `.siesta` as the workspace save engine

Goal:

- make workspace-native state the source of truth

Required outcome:

- workspace save no longer writes a staged `.sta` bundle first
- `.posetta/` stores canonical state directly
- save/update/load behavior no longer depends on `write_siesta(...)` and
  `update_labels_siesta(...)`

### Phase 4: Demote `.siesta` to one-way compatibility

Goal:

- keep `.siesta` only for import/migration during the transition

Allowed uses:

- explicit legacy import
- fixtures needed during cutover
- migration tooling

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
