# Changelog

Changes for `exp-pkg`.

The Python API and CLI are still pre-1.0. Minor releases may change those
surfaces. The `.expkg` file layout is treated as the v1 durability contract: a
zip container with `EXPKG.json`, a project descriptor, private `.xpkg/` state,
and managed media metadata. Files written by 0.x releases should remain readable
by later 0.x and 1.0 releases.

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
