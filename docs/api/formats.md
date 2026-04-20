# Formats

<div class="page-intro">
<p>
<code>xpkg.formats</code> is the core project/workspace format surface. It
defines the public artifact contract around workspaces, <code>.xpkg/</code>,
and <code>.expkg</code>, including the primary workspace-first import APIs for
DeepLabCut, SLEAP, MMPose, MediaPipe, OpenPose, and Detectron2.
</p>
</div>

!!! note
    Use <code>xpkg.compat</code> when you need low-level direct archive access
    for <code>.xpkg</code> files. Use
    <code>xpkg.formats</code> for the stable project/workspace boundary.

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

## Explicit Archive Compatibility Export

### `export_project_archive(path, *, out=None)`

Materialize a direct `.xpkg` archive from the current committed workspace head
when you explicitly need archive interop.

### `current_project_archive_path(path)`

Legacy compatibility alias for `export_project_archive(...)`. Prefer the
explicit export helper in new code.

## Import Into Workspaces

These are the preferred import entrypoints for new integrations. The similarly
named <code>xpkg.adapters.convert_*</code> helpers remain compatibility-only
direct <code>.xpkg</code> emitters.

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

### `import_legacy_archive(...)`

Import a legacy archive into a workspace.

### `migrate_legacy_archive(...)`

Migrate a legacy archive into the workspace-first xpkg contract.

## Save Current Workspace State

### `save_workspace_labels(...)`

Persist the current `Labels` state into a workspace and refresh the managed
project state. The committed durable snapshot head remains authoritative;
`.xpkg/state/current.json` is refreshed as a rebuildable cache.

## JSON Label Interchange

These helpers remain file-oriented interchange helpers on
<code>xpkg.formats</code>. The pure in-memory payload conversions now live in
<code>xpkg.codecs</code>.

### `read_labels_json_payload(path)`

Load the JSON interchange payload for labels.

### `write_labels_json(labels, path, *, indent=2)`

Write labels as JSON interchange rather than a managed workspace artifact.
