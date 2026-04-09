"""Public format entry points for workspace and project artifacts."""

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
    import_dlc_csv_workspace,
    import_dlc_h5_workspace,
    import_legacy_archive,
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
    "current_project_archive_path",
    "current_project_snapshot_path",
    "current_project_state_path",
    "default_expkg_path",
    "import_dlc_csv_workspace",
    "import_dlc_h5_workspace",
    "import_legacy_archive",
    "import_sleap_package_workspace",
    "init_project",
    "is_workspace_root",
    "load_project_descriptor",
    "migrate_legacy_archive",
    "pack_project",
    "project_descriptor_path",
    "read_labels_json_payload",
    "resolve_workspace_root",
    "save_workspace_labels",
    "unpack_project",
    "validate_artifact",
    "validate_expkg",
    "validate_workspace",
    "workspace_exports_root",
    "workspace_media_root",
    "workspace_state_root",
    "workspace_store_root",
    "write_labels_json",
    "write_project_descriptor",
]

_COMPAT_EXPORTS = frozenset(
    {
        "LazyDatasetHandle",
        "MaxInstancesExceededError",
        "PredictionAppendItem",
        "SerializerPredictedInstance",
        "ArchiveStore",
        "append_predictions_archive",
        "create_store_from_archive",
        "merge_predictions_archive",
        "open_store",
        "read_metrics_table",
        "read_archive",
        "summarize_project",
        "update_labels_archive",
        "validate_project",
        "write_metrics_table",
        "write_archive",
    }
)


def __getattr__(name: str) -> Any:
    if name in _COMPAT_EXPORTS:
        warnings.warn(
            f"xpkg.formats.{name} is compatibility-only and has moved to xpkg.compat.{name}",
            DeprecationWarning,
            stacklevel=2,
        )
        compat = importlib.import_module("xpkg.compat")
        value = getattr(compat, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | set(_COMPAT_EXPORTS))
