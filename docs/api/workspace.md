# Workspace

<div class="page-intro">
<p>
<code>xpkg.workspace</code> is the core project/workspace format surface. It
defines the public artifact contract around workspaces, private
<code>.xpkg/</code> state, portable <code>.expkg</code> artifacts, and the
workspace-first import APIs for DeepLabCut, SLEAP, MMPose, MediaPipe,
and Lightning Pose.
If you are starting a new downstream integration, read
<a href="../services/"><code>xpkg.services</code></a> first and use this module
when you want the explicit function-level form.
</p>
</div>

!!! note
    <code>xpkg.workspace</code> intentionally exposes the workspace contract, not
    direct HDF5 archive convenience wrappers.

## Start Here

- Use <code>xpkg.services.WorkspaceService</code> for the normal create/open/import/validate/pack/unpack lifecycle.
- Use <code>WorkspaceService.imports.*</code> for the preferred service-bound import flow.
- Use the <code>import_*_workspace(...)</code> helpers below when you want the same importers as explicit free functions.

## Project Contract

### `ProjectDescriptor`

The public descriptor object for `PROJECT.json`. It carries the stable
workspace metadata and locator fields for the xpkg v1 artifact contract.

### `PROJECT_DESCRIPTOR_FILENAME`

Always `"PROJECT.json"`.

### `EXPKG_SUFFIX`

Always `".expkg"`.

## Workspace Lifecycle

### `init_project(workspace, *, title=None, project_id=None, default_pack_mode="portable", force=False)`

Create a new workspace root with the canonical public layout.

### `load_project_descriptor(path)`

Load and validate `PROJECT.json` from a workspace.

### `write_project_descriptor(path, descriptor)`

Write a normalized `PROJECT.json` back to disk.

### `resolve_workspace_root(path)`

Resolve a path into the owning workspace root when possible.

### `is_workspace_root(path)`

Return whether a path points at a valid workspace root.

## Paths and Managed Roots

### `project_descriptor_path(path)`

Resolve the `PROJECT.json` path for a workspace or workspace-adjacent input.

### `workspace_store_root(path)`

Resolve the internal `.xpkg/` directory for a workspace.

### `workspace_state_root(path)`

Resolve the private state directory under `.xpkg/`.

### `workspace_media_root(path)`

Resolve the managed `Media/` directory.

### `workspace_exports_root(path)`

Resolve the standard `Exports/` directory.

### `default_expkg_path(path)`

Resolve the default packed artifact destination:

```text
<workspace>/Exports/<workspace-name>.expkg
```

## Pack / Unpack / Validate

### `pack_project(workspace, *, out=None, mode=None, overwrite=False)`

Pack a workspace into a portable `.expkg` artifact.

### `unpack_project(artifact, out, *, force=False, rename_title=None)`

Unpack a `.expkg` artifact into a workspace.

### `validate_workspace(path)`

Validate a workspace root.

### `validate_expkg(path)`

Validate a packed `.expkg` artifact.

### `validate_artifact(path)`

Validate either a workspace or a packed artifact, dispatching by path type.

## Import Into Workspaces

These free functions are the reusable workspace import implementation. New
service-based integrations should usually call them through
<code>WorkspaceService.imports.*</code>; use the explicit functions here when
you want function-level imports.

### `import_dlc_csv_workspace(...)`

Import a DeepLabCut CSV into a workspace.

### `import_dlc_h5_workspace(...)`

Import a DeepLabCut H5 into a workspace.

### `import_dlc_project_workspace(...)`

Import a whole DeepLabCut project into one workspace, skipping incomplete
project entries and preserving all imported items in the same managed state.

### `import_lightning_pose_csv_workspace(...)`

Import a Lightning Pose prediction CSV plus its matching video into a
workspace.

### `import_sleap_h5_workspace(...)`

Import a SLEAP analysis H5 export into a workspace.

### `import_sleap_package_workspace(...)`

Import a SLEAP package into a workspace.

### `import_mmpose_topdown_json_workspace(...)`

Import an official MMPose top-down demo JSON export plus its matching video
into a workspace.

### `import_mediapipe_pose_landmarks_json_workspace(...)`

Import the supported MediaPipe pose-landmarks JSON contract plus its matching
video into a workspace.

## Save Current Workspace State

### `save_workspace_labels(...)`

Persist the current `Labels` state into a workspace and refresh the managed
project state. The committed durable snapshot head remains authoritative;
`.xpkg/state/current.json` is refreshed as a rebuildable cache.

## Generic Artifact Registry

### `save_workspace_artifact(...)`

Copy output files into `.xpkg/artifacts/<kind>/<artifact_id>/` and write a
portable `manifest.json` with artifact type, title, inputs, producer metadata,
outputs, stats reports, optional metadata, and checksum-bearing file records.

Pass `namespace="neuro-analysis"` or any other caller-owned namespace to save
under `.xpkg/<namespace>/<kind>/<artifact_id>/`. xpkg normalizes the namespace
into a path-safe slug but does not interpret it.

### `load_workspace_artifact(...)`

Load one saved `ArtifactManifest` by artifact id, with optional
`artifact_type=...` and `namespace=...` disambiguation.

### `list_workspace_artifacts(...)`

Return saved artifact manifests, optionally filtered by artifact type or
namespace.

### `list_workspace_artifact_index(...)`

Return compact `ArtifactIndexEntry` records from `.xpkg/artifacts/index.json`,
rebuilding the index when it is missing.

### `validate_workspace_artifact(...)`

Validate one artifact manifest, ensure every referenced input/output/stat file
exists, and verify recorded file checksums and sizes when present.

### `validate_workspace_artifacts(...)`

Validate every saved artifact manifest in the workspace, optionally filtered by
artifact type or namespace.

### `rebuild_workspace_artifact_index(...)`

Rebuild `.xpkg/artifacts/index.json` from artifact manifests. This is useful
after manual repair or when importing a workspace created by an older xpkg
version.

### `workspace_artifact_type_root(...)` / `workspace_artifact_root(...)`

Resolve private artifact directories for a type or one artifact instance.

### `workspace_artifact_index_path(...)`

Resolve `.xpkg/artifacts/index.json`.

## Figure Artifacts

### `save_workspace_figure(...)`

Copy figure outputs into `.xpkg/artifacts/figures/<figure_id>/` and write a
portable `manifest.json` with the figure title, inputs, producer metadata,
outputs, stats reports, and optional metadata.

Pass any caller-owned namespace, such as `namespace="neuro-analysis"`, to save
under `.xpkg/<namespace>/figures/<figure_id>/` instead of the generic registry.
`xpkg` does not reserve or hard-code downstream package names.

The figure helpers are the figure-specific convenience layer over
`save_workspace_artifact(..., artifact_type="figure", ...)`.

### `load_workspace_figure(...)`

Load one saved `FigureArtifact` manifest by figure id.

### `list_workspace_figures(...)`

Return all saved figure manifests in a workspace.

### `validate_workspace_figure(...)`

Validate one figure manifest and every input/output/stat path it references.

### `validate_workspace_figures(...)`

Validate every saved figure artifact in the workspace.

### `workspace_figures_root(...)` / `workspace_figure_root(...)`

Resolve the private figure artifact directories under `.xpkg/artifacts/`.

### `save_workspace_segmentation_masks(...)`

Save the segmentation masks for one workspace video frame. This is the
function-level form of `workspace.segmentation.save_masks(...)`; pass
`mode="append"` to keep existing masks on the frame.

### `load_workspace_segmentation_masks(...)`

Load masks for one workspace video frame. The `video` selector may be omitted
when the workspace has exactly one video, or provided as a video index, id,
label, path, or video object.

### `load_workspace_segmentation_frames(...)`

Load every frame that currently has segmentation masks, optionally filtered by
video, frame index, prediction status, or class name.

### `clear_workspace_segmentation_masks(...)`

Remove all segmentation masks from one workspace video frame.

## JSON Label Interchange

These helpers remain file-oriented interchange helpers on
<code>xpkg.workspace</code>. The pure in-memory payload conversions now live in
<code>xpkg.adapters</code>.

### `read_labels_json_payload(path)`

Load the JSON interchange payload for labels.

### `write_labels_json(labels, path, *, indent=2)`

Write labels as JSON interchange rather than a managed workspace artifact.
