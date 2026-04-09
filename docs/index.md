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
<code>.expkg</code> exports. Edge archive compatibility lives in
<code>xpkg.compat</code>, with <code>.xpkg</code> as the canonical archive suffix
and <code>.sta</code> / <code>.sta</code> kept as legacy aliases during the transition.
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
| Edge compatibility layer | `xpkg.compat` for `.xpkg` / legacy `.sta` / `.sta` |
| External adapters | DLC, SLEAP |
| Core objects | `xpkg.model` |
| Low-level compatibility IO | `xpkg.compat` |
| Import and migration tools | `xpkg.adapters` |
</div>

<div class="spec-panel" markdown="1">
### Choose by Task

- Use `xpkg.model` when you need `Labels`, `Skeleton`, `Instance`, or `Video`.
- Use xpkg when you need a coherent experiment workspace with managed
  artifacts and compatibility import surfaces.
- Read [Artifact Contract v1](artifact_contract_v1.md) for the public workspace
  and `.expkg` contract.
- Read [CLI Command Spec v1](cli_command_spec_v1.md) for the locked command
  surface.
- Read [Storage Direction](architecture/storage-direction.md) when you want the
  blunt explanation for why `.sta` still exists in the implementation.
- Use `xpkg.compat` when you need low-level `.xpkg` archive IO or legacy
  `.sta` / `.sta` compatibility.
- Use `xpkg.adapters` when you need to import DLC or SLEAP.
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
  .xpkg/
  Media/
  Exports/
    My Project.expkg
```

`.xpkg` is the canonical edge archive suffix. `.sta` and `.sta` remain
available as legacy compatibility aliases, but none of them are the portable
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
- Read [Media IO Stack](architecture/media-io.md) for the target ownership split between xpkg and Siesta.
- Read [Storage Direction](architecture/storage-direction.md) for the current
  rationale and cutover pressure around `.sta`, `.xpkg/`, and
  `.expkg`.
- Read [Experimental Durable Store](architecture/experimental-store.md) for the new
  commit-oriented recovery workflow.
- Read [Model](api/model.md) for the pose object graph.
- Read [Compatibility](api/compat.md) for the edge `.xpkg` / legacy `.sta` /
  `.sta` APIs.
- Read [Adapters](api/adapters.md) for DLC and SLEAP conversion.
- Use the reference pages when you need exact signatures and docstrings.

</div>
