---
hide:
  - toc
---

<div class="manual-head" markdown="1">

<div class="manual-kicker">MULTIMODAL NEUROSCIENCE IO</div>

# xpkg

<p class="manual-deck">
xpkg is the canonical IO and artifact layer for multimodal neuroscience
experiment data, built around an editable project folder, a private
<code>.xpkg/</code> store, and portable <code>.expkg</code> exports.
The repo and distribution name are <code>exp-pkg</code>; the Python import and CLI name are
<code>xpkg</code>.
</p>

</div>

<div class="spec-grid spec-grid-2" markdown="1">

<div class="spec-panel" markdown="1">
### At a Glance

| Item | Value |
| --- | --- |
| Mission | multimodal neuroscience IO and artifact contracts |
| Public project contract | project folder + private `.xpkg/` + `.expkg` |
| Primary lifecycle API | `xpkg.services.ProjectService` |
| Service-bound project imports | `project.import_pose(...)` / `import_calibration(...)` / `import_motion(...)` |
| Output artifact registry | `project.artifacts.*` and `.xpkg/artifacts/index.json` |
| Function-level project imports | `xpkg.project.import_*_project(...)` |
| External import ecosystems | Vicon, DeepLabCut, Lightning Pose, SLEAP, MMPose, MediaPipe |
| Core objects | `xpkg.model` |
| In-memory exchange | `xpkg.adapters` |
</div>

<div class="spec-panel" markdown="1">
### Choose by Task

- Use `xpkg.services.ProjectService` when you need to create, open, import
  into, validate, pack, or unpack a project.
- Use `project.import_pose(format, ...)`, `project.import_calibration(format, ...)`,
  and `project.import_motion(format, ...)` for the normal project-first import
  flow from DeepLabCut, Lightning Pose, SLEAP, MMPose, MediaPipe, Anipose, and
  Vicon.
- Use `project.artifacts.*` to register figures, tables, analyses, reports,
  stats, and other output files with provenance and checksums.
- Use `xpkg.project.import_*_project(...)` when you want the same
  project-first importers as explicit free functions.
- Use `xpkg.model` when you need `Labels`, `Skeleton`, `Instance`, or `Video`.
- Use `xpkg.adapters` when you need arrays, tables, or JSON payloads without
  touching project internals.
- Read [Artifact Contract v1](artifact_contract_v1.md) for the public project
  and `.expkg` contract.
- Read [Roadmap](roadmap.md) for the current baseline and the multimodal work
  still ahead.
- Read [Multimodal Session Model](architecture/multimodal-session.md) for the
  timing, event, signal, photometry, and session primitives.
- Read [CLI Command Spec v1](cli_command_spec_v1.md) for the locked command
  surface.
- Read [Services](api/services.md) for the normal downstream
  create/open/import/validate/pack/unpack flow.
- Read [Storage Direction](architecture/storage-direction.md) when you want the
  architectural explanation for how the private store fits together.
</div>

</div>

## Current Coverage

<div class="spec-grid spec-grid-3" markdown="1">

<div class="spec-panel" markdown="1">
### Core Experiment Layer

- `Labels`
- `LabeledFrame`
- `Instance`, `PredictedInstance`
- `Point`, `PredictedPoint`
- `Skeleton`, `Keypoint`, `Track`
- `Video`
- project descriptors and managed media roots
</div>

<div class="spec-panel" markdown="1">
### Project-First Project APIs

- `ProjectService`
- `ProjectService.imports`
- `ProjectService.artifacts`
- `ProjectService.figures`
- `ProjectService.segmentation`
- `init_project`, `pack_project`, `unpack_project`
- `validate_project`
- `import_dlc_*_project`, `import_sleap_*_project`
- `import_lightning_pose_csv_project`
- `import_mmpose_topdown_json_project`
- `import_mediapipe_pose_landmarks_json_project`
</div>

</div>

## Public Artifact Layout

```text
My Project/
  PROJECT.json
  .xpkg/
  Media/
  Exports/
    My Project.expkg
```

The project contract exists because experiment state usually extends beyond a
single converter output.

## Navigation

<div class="quick-links" markdown="1">

- Start with [Getting Started](getting-started.md) for install and first-use examples.
- Read [Roadmap](roadmap.md) for what is implemented now versus planned next.
- Read [Artifact Contract v1](artifact_contract_v1.md) for the locked public
  project and portable artifact semantics.
- Read [CLI Command Spec v1](cli_command_spec_v1.md) for `init`, `import`,
  `pack`, `unpack`, and `validate`.
- Read [Media IO Stack](architecture/media-io.md) for the target ownership split between xpkg and downstream GUI apps.
- Read [Multimodal Session Model](architecture/multimodal-session.md) for the
  shared timing/events/signals layer.
- Read [Storage Direction](architecture/storage-direction.md) for the current
  rationale around `.xpkg/` and `.expkg`.
- Read [Project Durability](architecture/experimental-store.md) for the commit-oriented recovery workflow.
- Read [Model](api/model.md) for the pose object graph.
- Read [Adapters](api/adapters.md) for in-memory JSON / dataframe / numpy
  conversions.
- Read [Services](api/services.md) for the primary consumer-facing project
  API.
- Read [Project](api/project.md) for project-first lifecycle and import APIs.
- Use the reference pages when you need exact signatures and docstrings.

</div>
