# Changelog

All notable changes to `exp-pkg` will be documented here.

This project follows semantic versioning after the public API stabilizes. Before
`1.0`, minor versions may still refine the session, project, and importer
contracts.

The `.expkg` on-disk artifact format is the exception. It is the frozen v1
durability contract (zip container, `EXPKG.json` manifest, and the project /
`.xpkg/` / `.expkg` three-class layout). Files written by 0.x releases should
remain readable by later 0.x and 1.0 releases. The Python API and CLI command
surface remain 0.x and may change before 1.0.

## 0.1.0 - 2026-05-20

Initial public package line.

- Defines `exp-pkg` as the distribution name and `xpkg` as the Python import and
  CLI command.
- Provides project-first lifecycle APIs through `xpkg.services.ProjectService`.
- Supports project imports for Vicon, Anipose calibration, DeepLabCut,
  Lightning Pose, SLEAP, MMPose, and MediaPipe pose-landmark data.
- Provides portable `.expkg` artifacts and private `.xpkg/` project state.
- Ships an agent-friendly Typer CLI with JSON output for canonical commands.
- Focuses on pose estimation, motion capture (Vicon/C3D), synchronized video,
  segmentation masks, and portable project packaging.
- Frames fiber photometry and patch-clamp electrophysiology as planned future
  direction, not current capability. Their readers exist as experimental direct
  readers in `xpkg.readers` but are not importable through `ProjectService` or
  the CLI.
