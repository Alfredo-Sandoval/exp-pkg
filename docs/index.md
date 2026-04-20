---
hide:
  - toc
---

<div class="manual-head" markdown="1">

<div class="manual-kicker">BEHAVIOR WORKSPACE</div>

# xpkg

<p class="manual-deck">
xpkg is the canonical IO and artifact layer for experiment data, built around
an editable workspace folder, a private <code>.xpkg/</code> store, and portable
<code>.expkg</code> exports. Legacy <code>.xpkg</code> archives now enter only
through the explicit migration path instead of acting as a first-class project
surface.
The repo and distribution name are <code>exp-pkg</code>; the Python import and CLI name are
<code>xpkg</code>.
</p>

</div>

<div class="spec-grid spec-grid-2" markdown="1">

<div class="spec-panel" markdown="1">
### At a Glance

| Item | Value |
| --- | --- |
| Mission | experiment-data IO and artifact contracts |
| Public project contract | workspace folder + private `.xpkg/` + `.expkg` |
| Primary lifecycle API | `xpkg.services.WorkspaceService` |
| Service-bound workspace imports | `workspace.imports.*` from `xpkg.services.WorkspaceService` |
| Function-level workspace imports | `xpkg.formats.import_*_workspace(...)` |
| Legacy migration seam | `xpkg migrate` or `xpkg.formats.migrate_legacy_archive(...)` |
| External import ecosystems | DeepLabCut, SLEAP, MMPose, MediaPipe, OpenPose, Detectron2 |
| Core objects | `xpkg.model` |
| In-memory codecs | `xpkg.codecs` |
</div>

<div class="spec-panel" markdown="1">
### Choose by Task

- Use `xpkg.services.WorkspaceService` when you need to create, open, import
  into, validate, pack, or unpack a project.
- Use `workspace.imports.*` for the normal workspace-first import flow from
  DeepLabCut, SLEAP, MMPose, MediaPipe, OpenPose, or Detectron2.
- Use `xpkg.formats.import_*_workspace(...)` when you want the same
  workspace-first importers as explicit free functions.
- Use `xpkg.formats.migrate_legacy_archive(...)` or `xpkg migrate` only when
  you are cutting over an older `.xpkg` archive into the workspace contract.
- Use `xpkg.model` when you need `Labels`, `Skeleton`, `Instance`, or `Video`.
- Use `xpkg.codecs` when you need arrays, tables, or JSON payloads without
  touching workspace internals.
- Read [Artifact Contract v1](artifact_contract_v1.md) for the public workspace
  and `.expkg` contract.
- Read [CLI Command Spec v1](cli_command_spec_v1.md) for the locked command
  surface.
- Read [Services](api/services.md) for the normal downstream
  create/open/import/validate/pack/unpack flow.
- Read [Storage Direction](architecture/storage-direction.md) when you want the
  architectural explanation for how the private store and legacy migration seam
  fit together.
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
- `init_project`, `pack_project`, `unpack_project`
- `validate_workspace`
- `import_dlc_*_workspace`, `import_sleap_*_workspace`
- `import_mmpose_topdown_json_workspace`
- `import_mediapipe_pose_landmarks_json_workspace`
- `import_openpose_json_workspace`
- `import_detectron2_coco_workspace`
</div>

<div class="spec-panel" markdown="1">
### Legacy Bridge

- `migrate_legacy_archive`
- `xpkg migrate`
- explicit cutover from older `.xpkg` archives into a workspace
- no public `xpkg convert` or package-level `xpkg.adapters` facade
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
- Read [Artifact Contract v1](artifact_contract_v1.md) for the locked public
  workspace and portable artifact semantics.
- Read [CLI Command Spec v1](cli_command_spec_v1.md) for `init`, `import`,
  `pack`, `unpack`, `validate`, and `migrate`.
- Read [Media IO Stack](architecture/media-io.md) for the target ownership split between xpkg and the GUI app.
- Read [Storage Direction](architecture/storage-direction.md) for the current
  rationale and cutover status around `.xpkg/`, legacy `.xpkg`, and
  `.expkg`.
- Read [Experimental Durable Store](architecture/experimental-store.md) for the commit-oriented recovery workflow.
- Read [Model](api/model.md) for the pose object graph.
- Read [Codecs](api/codecs.md) for in-memory JSON / dataframe / numpy
  conversions.
- Read [Services](api/services.md) for the primary consumer-facing workspace
  API.
- Read [Formats](api/formats.md) for workspace-first lifecycle, import APIs,
  and the explicit legacy migration helper.
- Use the reference pages when you need exact signatures and docstrings.

</div>
