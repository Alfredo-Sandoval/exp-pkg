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
<code>.expkg</code> exports. Low-level archive IO lives in
<code>xpkg.compat</code>, with <code>.xpkg</code> as the canonical direct-archive
suffix.
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
| Public project contract | workspace folder + `.expkg` |
| Authoritative mutable state | `.xpkg/` inside the workspace |
| Edge compatibility layer | `xpkg.compat` for direct `.xpkg` archive IO |
| External import ecosystems | DeepLabCut, SLEAP, MMPose, MediaPipe, OpenPose, Detectron2 |
| Workspace-first imports | `xpkg.formats.import_*_workspace(...)` |
| Core objects | `xpkg.model` |
| In-memory codecs | `xpkg.codecs` |
| Low-level compatibility IO | `xpkg.compat` |
| Compatibility adapters | `xpkg.adapters` |
</div>

<div class="spec-panel" markdown="1">
### Choose by Task

- Use `xpkg.model` when you need `Labels`, `Skeleton`, `Instance`, or `Video`.
- Use `xpkg.codecs` when you need arrays, tables, or JSON payloads without
  touching workspace or archive internals.
- Use xpkg when you need a coherent experiment workspace with managed
  artifacts and compatibility import surfaces.
- Use `xpkg.formats.import_*_workspace(...)` for workspace-first imports from
  DeepLabCut, SLEAP, MMPose, MediaPipe, OpenPose, or Detectron2.
- Read [Artifact Contract v1](artifact_contract_v1.md) for the public workspace
  and `.expkg` contract.
- Read [CLI Command Spec v1](cli_command_spec_v1.md) for the locked command
  surface.
- Read [Storage Direction](architecture/storage-direction.md) when you want the
  architectural explanation for how direct archive IO relates to the
  workspace/store model.
- Use `xpkg.compat` when you need low-level `.xpkg` archive IO.
- Use `xpkg.adapters` only when you explicitly need a direct compatibility
  `.xpkg` archive from one of the supported import ecosystems.
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
### Archive and Compatibility IO

- `read_xpkg`
- `write_xpkg`
- `update_labels_xpkg`
- prediction append and merge
- metrics table IO
- validation and summary
- experimental durable store
</div>

<div class="spec-panel" markdown="1">
### Imports and Migration

- `import_dlc_*_workspace`, `import_sleap_*_workspace`
- `import_mmpose_topdown_json_workspace`
- `import_mediapipe_pose_landmarks_json_workspace`
- `import_openpose_json_workspace`
- `import_detectron2_coco_workspace`
- compatibility `convert_*` adapter helpers
- `ConversionResult`
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

`.xpkg` is the canonical direct-archive suffix, but it is not the portable
public project artifact.

The workspace contract exists because experiment state usually extends beyond a
single archive or converter output.

## Navigation

<div class="quick-links" markdown="1">

- Start with [Getting Started](getting-started.md) for install and first-use examples.
- Read [Artifact Contract v1](artifact_contract_v1.md) for the locked public
  workspace and portable artifact semantics.
- Read [CLI Command Spec v1](cli_command_spec_v1.md) for `init`, `import`,
  `pack`, `unpack`, `validate`, `migrate`, and the legacy `convert` helper.
- Read [Media IO Stack](architecture/media-io.md) for the target ownership split between xpkg and the GUI app.
- Read [Storage Direction](architecture/storage-direction.md) for the current
  rationale and cutover pressure around `.xpkg`, `.xpkg/`, and
  `.expkg`.
- Read [Experimental Durable Store](architecture/experimental-store.md) for the new
  commit-oriented recovery workflow.
- Read [Model](api/model.md) for the pose object graph.
- Read [Codecs](api/codecs.md) for in-memory JSON / dataframe / numpy
  conversions.
- Read [Formats](api/formats.md) for workspace-first lifecycle and import APIs.
- Read [Compatibility](api/compat.md) for the edge `.xpkg` archive APIs.
- Read [Adapters](api/adapters.md) for compatibility adapters across the
  shipped importer surface.
- Use the reference pages when you need exact signatures and docstrings.

</div>
