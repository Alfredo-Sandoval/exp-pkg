# Changelog

Changes for `exp-pkg`.

The Python API and CLI are still pre-1.0. Minor releases may change those
surfaces. The `.expkg` file layout is treated as the v1 durability contract: a
zip container with `EXPKG.json`, a project descriptor, private `.xpkg/` state,
and managed media metadata. The Python API and embedded experiment/session
schemas remain pre-1.0 and use explicit breaking schema bumps.

## Unreleased

### Added

- Added project, service, and CLI imports for generic event CSV, generic
  behavior CSV/JSON, BORIS, B-SOiD, SimBA, Keypoint-MoSeq, and paired
  synchronization CSV.
- Added `TimebaseCorrespondence` for paired clock observations and
  `fit_timebase_alignment` for fitted offset and affine mappings.
- Added generated ontology, recording-session, and experiment schema documents
  to the source tree and built wheel.
- Added first-class behavior-to-subject links, bounded subject-to-track
  assignments, and typed relationships between identified events.
- Added EMG and force-plate data to the canonical recording-session signal
  contract.

### Changed

- Bumped `xpkg.experiment` and `xpkg.recording-session` to schema version 4.
  Recording sessions now persist a named timebase registry, named event
  streams, stable event identifiers, plural behavior media links, and typed
  source provenance. Older embedded state documents are rejected.
- Split alignment transform shape (`AlignmentModel`) from observation method
  (`SynchronizationMethod`). Alignment residuals are derived from stored
  correspondence evidence.
- Consolidated bounded CSV loading into one internal reader boundary.
- Moved label objects, query operations, merge operations, caches, and export
  operations into the semantic model layer. The IO layer now contains only
  explicit serializers and source-format adapters.

### Removed

- Removed the `ArchiveStore` and `ArchiveStoreError` Python aliases. Import
  `ProjectDurableStore` and `ProjectDurableStoreError` from
  `xpkg.project.durable_store`.
- Removed the `read_project_state_payload` alias. Use `read_project_state`.
- Removed the `load_skeleton_archive_json` alias. Use
  `load_skeleton_xpkg_json`.
- Removed segmentation re-exports from `xpkg.pose.annotations` and deleted
  `xpkg.pose.annotations.regions`. Import segmentation objects from
  `xpkg.segmentation`.
- Removed the reverse `xpkg.segmentation.project` re-export. Project mask
  storage remains in `xpkg.project.segmentation` and `ProjectService`.
- Removed unused discovery helpers for obsolete single-file `.xpkg`
  annotation archives.
- Removed the duplicate inspection `warnings` string list. Use typed
  `warning_records` in the Python API and JSON output.
- Removed the `xpkg.archive-store` discriminator parser and the singular
  dataset-share `funder` parser. Use `xpkg.project-durable-store` and `funders`.
- Removed model-owned label and skeleton file methods. Use project actions or
  explicit IO serializers at the file boundary.
- Removed the singular recording-session event table and its replacement
  action. Use named `SessionEventStream` objects and the event-stream actions.

## 0.1.0 - 2026-06-10

Initial release.

- `ProjectService` for creating, opening, validating, packing, unpacking, and
  importing projects, with generated summary indexes and fast `project
  describe` reads.
- Project imports for DeepLabCut, Lightning Pose, SLEAP, MMPose, MediaPipe,
  Anipose calibration, and OpenCV stereo calibration.
- Direct, experimental readers for photometry, event, and behavior files that
  parse source files into typed in-memory objects.
- Portable `.expkg` artifacts with private `.xpkg/` project state and explicit
  media modes.
- A Typer CLI with JSON output for command discovery, project operations,
  imports, inspection, completion, and packaging.
- Pose, label, skeleton, media, calibration, behavior-label, segmentation-mask,
  and HDF5 table data models.
- Segmentation overlay rendering via the optional `segmentation-overlays` extra.
- Project inspection that reports descriptor, shallow state summary, metadata
  slots, associated media, and QC warnings without hydrating full payloads.
- Descriptor validation, SHA-256 checks, and atomic JSON writes for project
  state.
- Docs for the project service, CLI, media IO, storage layout, direct readers,
  and the `.expkg` file contract.
