# Changelog

Changes for `exp-pkg`.

The Python API and CLI are still pre-1.0. Minor releases may change those
surfaces. The `.expkg` file layout is treated as the v1 durability contract: a
zip container with `EXPKG.json`, a project descriptor, private `.xpkg/` state,
and managed media metadata. Files written by 0.x releases should remain readable
by later 0.x and 1.0 releases.

## Unreleased

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

### Compatibility policy

- Persisted 0.x data remains readable at boundary loaders. The durable store
  still parses the older `xpkg.archive-store` discriminator, and dataset-share
  metadata still parses the older singular `funder` key into canonical
  `funders`.
- Rejected deleting those persisted-data parsers because the artifact contract
  promises that files written by 0.x releases remain readable. Interior code
  receives only the canonical objects and field names.

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
