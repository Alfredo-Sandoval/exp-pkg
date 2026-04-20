"""Workspace-first project helpers for the public artifact contract.

These free functions stay public for integrations that want an explicit
function-level API. New code should usually drive lifecycle and ingestion
through ``WorkspaceService`` and ``WorkspaceService.imports`` rather than build
around archive-first helpers.
"""

from __future__ import annotations

from xpkg.io.project_artifact import (
    pack_project,
    unpack_project,
    validate_artifact,
    validate_expkg,
    validate_workspace,
)
from xpkg.io.project_layout import (
    EXPKG_SUFFIX,
    PROJECT_DESCRIPTOR_FILENAME,
    ProjectDescriptor,
    default_expkg_path,
    is_workspace_root,
    load_project_descriptor,
    project_descriptor_path,
    resolve_workspace_root,
    workspace_exports_root,
    workspace_media_root,
    workspace_state_root,
    workspace_store_root,
    write_project_descriptor,
)
from xpkg.io.project_workspace import (
    current_project_archive_path,
    current_project_snapshot_path,
    current_project_state_path,
    export_project_archive,
    import_detectron2_coco_workspace,
    import_dlc_csv_workspace,
    import_dlc_h5_workspace,
    import_dlc_project_workspace,
    import_legacy_archive,
    import_mediapipe_pose_landmarks_json_workspace,
    import_mmpose_topdown_json_workspace,
    import_openpose_json_workspace,
    import_sleap_h5_workspace,
    import_sleap_package_workspace,
    init_project,
    migrate_legacy_archive,
    save_workspace_labels,
)

__all__ = [
    "EXPKG_SUFFIX",
    "PROJECT_DESCRIPTOR_FILENAME",
    "ProjectDescriptor",
    "init_project",
    "load_project_descriptor",
    "write_project_descriptor",
    "resolve_workspace_root",
    "is_workspace_root",
    "project_descriptor_path",
    "workspace_exports_root",
    "workspace_media_root",
    "workspace_state_root",
    "workspace_store_root",
    "default_expkg_path",
    "pack_project",
    "unpack_project",
    "validate_workspace",
    "validate_expkg",
    "validate_artifact",
    "save_workspace_labels",
    "current_project_state_path",
    "current_project_snapshot_path",
    "import_dlc_csv_workspace",
    "import_dlc_h5_workspace",
    "import_dlc_project_workspace",
    "import_sleap_h5_workspace",
    "import_sleap_package_workspace",
    "import_mmpose_topdown_json_workspace",
    "import_mediapipe_pose_landmarks_json_workspace",
    "import_openpose_json_workspace",
    "import_detectron2_coco_workspace",
    "migrate_legacy_archive",
    "import_legacy_archive",
    "export_project_archive",
    "current_project_archive_path",
]
