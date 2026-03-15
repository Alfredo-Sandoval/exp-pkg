---
hide:
  - toc
---

<div class="manual-head" markdown="1">

<div class="manual-kicker">POSE IO LIBRARY</div>

# Posetta

<p class="manual-deck">
Posetta uses a locked workspace-first public artifact contract:
editable workspace folder, private <code>.posetta/</code> store, and portable
<code>.poseproj</code> export. The current low-level <code>.siesta</code> APIs
remain as legacy compatibility surfaces during the transition.
</p>

</div>

<div class="spec-grid spec-grid-2" markdown="1">

<div class="spec-panel" markdown="1">
### At a Glance

| Item | Value |
| --- | --- |
| Public project contract | workspace folder + `.poseproj` |
| Authoritative mutable state | `.posetta/` inside the workspace |
| Legacy compatibility format | `.siesta` import/read APIs |
| External adapters | DLC, SLEAP |
| Pose objects | `posetta.model` |
| Low-level compatibility IO | `posetta.formats` |
| Import tools | `posetta.adapters` |
</div>

<div class="spec-panel" markdown="1">
### Choose by Task

- Use `posetta.model` when you need `Labels`, `Skeleton`, `Instance`, or `Video`.
- Read [Artifact Contract v1](artifact_contract_v1.md) for the public workspace
  and `.poseproj` contract.
- Read [CLI Command Spec v1](cli_command_spec_v1.md) for the locked command
  surface.
- Use `posetta.formats` when you need low-level legacy `.siesta`
  compatibility IO.
- Use `posetta.adapters` when you need to import DLC or SLEAP.
</div>

</div>

## Current Coverage

<div class="spec-grid spec-grid-3" markdown="1">

<div class="spec-panel" markdown="1">
### Model

- `Labels`
- `LabeledFrame`
- `Instance`, `PredictedInstance`
- `Point`, `PredictedPoint`
- `Skeleton`, `Keypoint`, `Track`
- `Video`
</div>

<div class="spec-panel" markdown="1">
### Compatibility IO

- `read_siesta`
- `write_siesta`
- `update_labels_siesta`
- prediction append and merge
- metrics table IO
- validation and summary
- experimental durable store
</div>

<div class="spec-panel" markdown="1">
### Adapters

- `convert_dlc_csv`
- `convert_dlc_h5`
- `convert_dlc_project`
- `convert_sleap_package`
- `ConversionResult`
</div>

</div>

## Public Artifact Layout

```text
My Project/
  PROJECT.json
  .posetta/
  Media/
  Exports/
    My Project.poseproj
```

`.siesta` remains available as a legacy compatibility layer, but it is no
longer the native public project artifact.

## Navigation

<div class="quick-links" markdown="1">

- Start with [Getting Started](getting-started.md) for install and first-use examples.
- Read [Artifact Contract v1](artifact_contract_v1.md) for the locked public
  workspace and portable artifact semantics.
- Read [CLI Command Spec v1](cli_command_spec_v1.md) for `init`, `import`,
  `pack`, `unpack`, and `migrate`.
- Read [Media IO Stack](architecture/media-io.md) for the target ownership split between Posetta and Siesta.
- Read [Experimental Durable Store](architecture/experimental-store.md) for the new
  commit-oriented recovery workflow.
- Read [Model](api/model.md) for the pose object graph.
- Read [Formats](api/formats.md) for the legacy `.siesta` compatibility APIs.
- Read [Adapters](api/adapters.md) for DLC and SLEAP conversion.
- Use the reference pages when you need exact signatures and docstrings.

</div>
