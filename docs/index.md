---
hide:
  - toc
---

<div class="manual-head" markdown="1">

<div class="manual-kicker">MULTIMODAL NEUROSCIENCE IO</div>

# xpkg

<p class="manual-deck">
xpkg is the canonical IO and artifact layer for multimodal neuroscience
experiment data, built around an editable workspace folder, a private
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
| Public project contract | workspace folder + private `.xpkg/` + `.expkg` |
| Primary lifecycle API | `xpkg.services.WorkspaceService` |
| Service-bound workspace imports | `workspace.imports.*` from `xpkg.services.WorkspaceService` |
| Output artifact registry | `workspace.artifacts.*` and `.xpkg/artifacts/index.json` |
| Function-level workspace imports | `xpkg.workspace.import_*_workspace(...)` |
| External import ecosystems | Vicon, DeepLabCut, Lightning Pose, SLEAP, MMPose, MediaPipe |
| Core objects | `xpkg.model` |
| In-memory exchange | `xpkg.adapters` |
</div>

<div class="spec-panel" markdown="1">
### Choose by Task

- Use `xpkg.services.WorkspaceService` when you need to create, open, import
  into, validate, pack, or unpack a project.
- Use `workspace.imports.*` for the normal workspace-first import flow from
  DeepLabCut, Lightning Pose, SLEAP, MMPose, or MediaPipe.
- Use `workspace.artifacts.*` to register figures, tables, analyses, reports,
  stats, and other output files with provenance and checksums.
- Use `xpkg.workspace.import_*_workspace(...)` when you want the same
  workspace-first importers as explicit free functions.
- Use `xpkg.model` when you need `Labels`, `Skeleton`, `Instance`, or `Video`.
- Use `xpkg.adapters` when you need arrays, tables, or JSON payloads without
  touching workspace internals.
- Read [Artifact Contract v1](artifact_contract_v1.md) for the public workspace
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
- workspace descriptors and managed media roots
</div>

<div class="spec-panel" markdown="1">
### Workspace-First Project APIs

- `WorkspaceService`
- `WorkspaceService.imports`
- `WorkspaceService.artifacts`
- `WorkspaceService.figures`
- `WorkspaceService.segmentation`
- `init_project`, `pack_project`, `unpack_project`
- `validate_workspace`
- `import_dlc_*_workspace`, `import_sleap_*_workspace`
- `import_lightning_pose_csv_workspace`
- `import_mmpose_topdown_json_workspace`
- `import_mediapipe_pose_landmarks_json_workspace`
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

The workspace contract exists because experiment state usually extends beyond a
single converter output.

## Navigation

<div class="quick-links" markdown="1">

- Start with [Getting Started](getting-started.md) for install and first-use examples.
- Read [Roadmap](roadmap.md) for what is implemented now versus planned next.
- Read [Artifact Contract v1](artifact_contract_v1.md) for the locked public
  workspace and portable artifact semantics.
- Read [CLI Command Spec v1](cli_command_spec_v1.md) for `init`, `import`,
  `pack`, `unpack`, and `validate`.
- Read [Media IO Stack](architecture/media-io.md) for the target ownership split between xpkg and downstream GUI apps.
- Read [Multimodal Session Model](architecture/multimodal-session.md) for the
  shared timing/events/signals layer.
- Read [Storage Direction](architecture/storage-direction.md) for the current
  rationale around `.xpkg/` and `.expkg`.
- Read [Experimental Durable Store](architecture/experimental-store.md) for the commit-oriented recovery workflow.
- Read [Model](api/model.md) for the pose object graph.
- Read [Adapters](api/adapters.md) for in-memory JSON / dataframe / numpy
  conversions.
- Read [Services](api/services.md) for the primary consumer-facing workspace
  API.
- Read [Workspace](api/workspace.md) for workspace-first lifecycle and import APIs.
- Use the reference pages when you need exact signatures and docstrings.

</div>
