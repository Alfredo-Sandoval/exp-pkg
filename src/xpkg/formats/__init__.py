"""Public workspace-first format entry points.

This module defines the stable workspace/project boundary. New integrations
should prefer ``WorkspaceService`` for lifecycle work and the
``import_*_workspace(...)`` helpers for foreign inputs. Direct archive helpers
remain available only for explicit compatibility interop.
"""

from __future__ import annotations

import importlib
import warnings
from typing import Any

from xpkg.formats.project import (
    EXPKG_SUFFIX,
    PROJECT_DESCRIPTOR_FILENAME,
    ProjectDescriptor,
    current_project_archive_path,
    current_project_snapshot_path,
    current_project_state_path,
    default_expkg_path,
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
    is_workspace_root,
    load_project_descriptor,
    migrate_legacy_archive,
    pack_project,
    project_descriptor_path,
    resolve_workspace_root,
    save_workspace_labels,
    unpack_project,
    validate_artifact,
    validate_expkg,
    validate_workspace,
    workspace_exports_root,
    workspace_media_root,
    workspace_state_root,
    workspace_store_root,
    write_project_descriptor,
)
from xpkg.io.labels.json_format import read_labels_json_payload, write_labels_json

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
    "write_labels_json",
    "read_labels_json_payload",
]

_COMPAT_EXPORTS: dict[str, str] = {
    "LazyDatasetHandle": "LazyDatasetHandle",
    "MaxInstancesExceededError": "MaxInstancesExceededError",
    "PredictionAppendItem": "PredictionAppendItem",
    "SerializerPredictedInstance": "SerializerPredictedInstance",
    "ArchiveStore": "ArchiveStore",
    "append_predictions_archive": "append_predictions_xpkg",
    "create_store_from_archive": "create_store_from_xpkg",
    "merge_predictions_archive": "merge_predictions_xpkg",
    "open_store": "open_store",
    "read_metrics_table": "read_metrics_table",
    "read_archive": "read_xpkg",
    "summarize_project": "summarize_xpkg",
    "update_labels_archive": "update_labels_xpkg",
    "validate_project": "validate_xpkg",
    "write_metrics_table": "write_metrics_table",
    "write_archive": "write_xpkg",
}


def __getattr__(name: str) -> Any:
    canonical_name = _COMPAT_EXPORTS.get(name)
    if canonical_name is not None:
        warnings.warn(
            "xpkg.formats."
            f"{name} is a compatibility-only archive API; use xpkg.compat."
            f"{canonical_name} for explicit archive interop.",
            DeprecationWarning,
            stacklevel=2,
        )
        compat = importlib.import_module("xpkg.compat")
        value = getattr(compat, canonical_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
