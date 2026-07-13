# Project

<div class="page-intro">
<p>
<code>xpkg.project</code> is the core project format surface. It
defines the public artifact contract around projects, private
<code>.xpkg/</code> state, portable <code>.expkg</code> artifacts, and the
project storage helpers used by the service layer.
If you are starting a new downstream integration, read
<a href="../services/"><code>xpkg.services</code></a> first and use this module
for lower-level project layout, artifact, validation, and payload operations.
</p>
</div>

!!! note
    <code>xpkg.project</code> intentionally exposes the project contract, not
    direct HDF5 archive convenience wrappers.

## Start Here

- Use <code>xpkg.services.ProjectService</code> for the normal create/open/import/validate/pack/unpack lifecycle.
- Use <code>ProjectService.import_pose</code>, <code>import_calibration</code>, or <code>import_signals</code> for the preferred service-bound import flow.

## Project Contract

### `ProjectDescriptor`

The public descriptor object for `PROJECT.json`. It carries the stable
project metadata and locator fields for the xpkg v1 artifact contract.

### `PROJECT_DESCRIPTOR_FILENAME`

Always `"PROJECT.json"`.

### `EXPKG_SUFFIX`

Always `".expkg"`.

## Project Lifecycle

### `init_project(project, *, title=None, project_id=None, force=False)`

Create a new project root with the canonical public layout.

### `load_project_descriptor(path)`

Load and validate `PROJECT.json` from a project.

### `write_project_descriptor(path, descriptor)`

Write a normalized `PROJECT.json` back to disk.

### `load_project_summary(path)`

Load `.xpkg/indexes/project_summary.json`, the generated shallow inventory for
project pickers, catalogs, and agent describe paths.

### `refresh_project_summary(path)`

Refresh the generated project summary index from descriptor, state stats,
typed metadata slot files, and the artifact index without loading full labels,
predictions, dense masks, or media.

### `resolve_project_root(path)`

Resolve a path into the owning project root when possible.

### `is_project_root(path)`

Return whether a path points at a valid project root.

## Paths and Managed Roots

### `project_descriptor_path(path)`

Resolve the `PROJECT.json` path for a project or project-adjacent input.

### `project_store_root(path)`

Resolve the internal `.xpkg/` directory for a project.

### `project_state_root(path)`

Resolve the private state directory under `.xpkg/`.

### `project_indexes_root(path)`

Resolve the generated indexes directory under `.xpkg/`.

### `project_summary_path(path)`

Resolve the generated project summary index path.

### `project_media_root(path)`

Resolve the managed `Media/` directory.

### `project_exports_root(path)`

Resolve the standard `Exports/` directory.

### `default_expkg_path(path)`

Resolve the default packed artifact destination:

```text
<project>/Exports/<project-name>.expkg
```

## Pack / Unpack / Validate

### `pack_project(project, *, out=None, media=None, overwrite=False)`

Pack a project into a portable `.expkg` artifact.

`media` accepts:

- `"full"` or `None`: include every managed file under `Media/`.
- `"package"`: include package-sized media such as image sequences, while
  manifesting video containers without storing their bytes.
- `"manifest"`: store no media bytes and record managed media paths, sizes,
  and SHA-256 digests in `EXPKG.json`.

### `unpack_project(artifact, out, *, force=False, rename_title=None)`

Unpack a `.expkg` artifact into a project.

### `validate_project(path)`

Validate a project root.

### `validate_expkg(path)`

Validate a packed `.expkg` artifact.

### `validate_artifact(path)`

Validate either a project or a packed artifact, dispatching by path type.

## Save Current Project State

### `save_project_session(project, session, *, reason="project.save.recording")`

Add or replace one typed `RecordingSession` in the project's canonical
`Experiment` aggregate.

### `load_project_session(project)`

Load one recording session from the canonical experiment.

### `save_project_experiment(project, experiment, *, reason=...)`

Commit the complete canonical `Experiment` aggregate.

### `load_project_experiment(project)`

Load the complete experiment with subjects, protocols, conditions, and all
recording sessions.

### `save_project_labels(...)`

Add or replace a `SessionPose` containing `Labels` in a recording session and
refresh managed experiment state. The committed durable state head remains authoritative;
`.xpkg/state/current.json` is refreshed as a rebuildable cache.

## Generic Artifact Registry

### `save_project_artifact(...)`

Copy output files into `.xpkg/artifacts/<kind>/<artifact_id>/` and write a
portable `manifest.json` with artifact type, title, inputs, producer metadata,
outputs, stats reports, optional metadata, and checksum-bearing file records.

Pass `namespace="neuro-analysis"` or any other caller-owned namespace to save
under `.xpkg/<namespace>/<kind>/<artifact_id>/`. xpkg normalizes the namespace
into a path-safe slug but does not interpret it.

### `load_project_artifact(...)`

Load one saved `ArtifactManifest` by artifact id, with optional
`artifact_type=...` and `namespace=...` disambiguation.

### `list_project_artifacts(...)`

Return saved artifact manifests, optionally filtered by artifact type or
namespace.

### `list_project_artifact_index(...)`

Return compact `ArtifactIndexEntry` records from `.xpkg/artifacts/index.json`,
rebuilding the index when it is missing.

### `validate_project_artifact(...)`

Validate one artifact manifest, ensure every referenced input/output/stat file
exists, and verify recorded file checksums and sizes when present.

### `validate_project_artifacts(...)`

Validate every saved artifact manifest in the project, optionally filtered by
artifact type or namespace.

### `rebuild_project_artifact_index(...)`

Rebuild `.xpkg/artifacts/index.json` from artifact manifests. This is useful
after manual repair or when importing a project created by an older xpkg
version.

### `project_artifact_type_root(...)` / `project_artifact_root(...)`

Resolve private artifact directories for a type or one artifact instance.

### `project_artifact_index_path(...)`

Resolve `.xpkg/artifacts/index.json`.

## Figure Artifacts

### `save_project_figure(...)`

Copy figure outputs into `.xpkg/artifacts/figures/<figure_id>/` and write a
portable `manifest.json` with the figure title, inputs, producer metadata,
outputs, stats reports, and optional metadata.

Pass any caller-owned namespace, such as `namespace="neuro-analysis"`, to save
under `.xpkg/<namespace>/figures/<figure_id>/` instead of the generic registry.
`xpkg` does not reserve or hard-code downstream package names.

The figure helpers are the figure-specific convenience layer over
`save_project_artifact(..., artifact_type="figure", ...)`.

### `load_project_figure(...)`

Load one saved `FigureArtifact` manifest by figure id.

### `list_project_figures(...)`

Return all saved figure manifests in a project.

### `validate_project_figure(...)`

Validate one figure manifest and every input/output/stat path it references.

### `validate_project_figures(...)`

Validate every saved figure artifact in the project.

### `project_figures_root(...)` / `project_figure_root(...)`

Resolve the private figure artifact directories under `.xpkg/artifacts/`.

### `save_project_segmentation_masks(...)`

Save the segmentation masks for one project video frame. This is the
function-level form of `project.segmentation.save_masks(...)`; pass
`mode="append"` to keep existing masks on the frame.

### `load_project_segmentation_masks(...)`

Load masks for one project video frame. The `video` selector may be omitted
when the project has exactly one video, or provided as a video index, id,
label, path, or video object.

### `load_project_segmentation_frames(...)`

Load every frame that currently has segmentation masks, optionally filtered by
video, frame index, prediction status, or class name.

### `clear_project_segmentation_masks(...)`

Remove all segmentation masks from one project video frame.

## JSON Label Interchange

These helpers remain file-oriented interchange helpers on
<code>xpkg.project</code>. The pure in-memory payload conversions now live in
<code>xpkg.adapters</code>.

### `read_labels_json_payload(path)`

Load the JSON interchange payload for labels.

### `write_labels_json(labels, path, *, indent=2)`

Write labels as JSON interchange rather than a managed project artifact.
