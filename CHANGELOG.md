# Changelog

All notable changes to `exp-pkg` will be documented here.

This project follows semantic versioning after the public API stabilizes. Before
`1.0`, minor versions may still refine the session, project, and importer
contracts.

## 0.1.0 - 2026-05-20

Initial public package line.

- Defines `exp-pkg` as the distribution name and `xpkg` as the Python import and
  CLI command.
- Provides project-first lifecycle APIs through `xpkg.services.ProjectService`.
- Supports project imports for Vicon, Anipose calibration, DeepLabCut,
  Lightning Pose, SLEAP, MMPose, and MediaPipe pose-landmark data.
- Provides portable `.expkg` artifacts and private `.xpkg/` project state.
- Ships an agent-friendly Typer CLI with JSON output for canonical commands.
- Frames the package direction as multimodal neuroscience IO for pose, video,
  signals such as fiber photometry, behavioral events, synchronization, and
  portable artifacts.
