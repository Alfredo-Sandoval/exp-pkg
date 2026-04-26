# Formats

<div class="page-intro">
<p>
<code>xpkg.formats</code> is the core project/workspace format surface. It
defines the public artifact contract around workspaces, private
<code>.xpkg/</code> state, portable <code>.expkg</code> artifacts, and the
workspace-first import APIs for DeepLabCut, SLEAP, MMPose, MediaPipe,
OpenPose, and Detectron2.
If you are starting a new downstream integration, read
<a href="../services/"><code>xpkg.services</code></a> first and use this module
when you want the explicit function-level form.
</p>
</div>

!!! note
    <code>xpkg.formats</code> intentionally exposes the workspace contract, not
    direct archive convenience wrappers. The only retained legacy seam here is
    <code>migrate_legacy_archive(...)</code> for cutting older
    <code>.xpkg</code> archives over to the workspace path.

## Start Here

- Use <code>xpkg.services.WorkspaceService</code> for the normal create/open/import/validate/pack/unpack lifecycle.
- Use <code>WorkspaceService.imports.*</code> for the preferred service-bound import flow.
- Use the <code>import_*_workspace(...)</code> helpers below when you want the same importers as explicit free functions.
- Use <code>migrate_legacy_archive(...)</code> only when you are cutting over an older direct archive into a workspace.

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

### `import_openpose_json_workspace(...)`

Import an OpenPose `--write_json` BODY_25 directory plus its matching video
into a workspace.

### `import_detectron2_coco_workspace(...)`

Import Detectron2 COCO keypoint predictions
(`coco_instances_results.json`) plus the paired dataset JSON and `image_root`
into a workspace.

## Legacy Migration

### `migrate_legacy_archive(...)`

Cut a canonical legacy `.xpkg` archive over into the workspace-first xpkg
contract.

This is intentionally the one retained legacy bridge on
<code>xpkg.formats</code>. Direct archive conversion/export convenience wrappers
were removed from this public facade during the workspace-first cutover.

## Save Current Workspace State

### `save_workspace_labels(...)`

Persist the current `Labels` state into a workspace and refresh the managed
project state. The committed durable snapshot head remains authoritative;
`.xpkg/state/current.json` is refreshed as a rebuildable cache.

## Figure Artifacts

### `save_workspace_figure(...)`

Copy figure outputs into `.xpkg/artifacts/figures/<figure_id>/` and write a
portable `manifest.json` with the figure title, inputs, producer metadata,
outputs, stats reports, and optional metadata.

Pass `namespace="openoperant"` or another app name to save under
`.xpkg/<namespace>/figures/<figure_id>/` instead of the generic registry.

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
<code>xpkg.formats</code>. The pure in-memory payload conversions now live in
<code>xpkg.codecs</code>.

### `read_labels_json_payload(path)`

Load the JSON interchange payload for labels.

### `write_labels_json(labels, path, *, indent=2)`

Write labels as JSON interchange rather than a managed workspace artifact.
