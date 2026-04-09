"""Stable public API for Posetta integrations and workspace services."""

from __future__ import annotations

import importlib
from typing import Any

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "ConversionResult": (".adapters", "ConversionResult"),
    "Instance": (".model", "Instance"),
    "Keypoint": (".model", "Keypoint"),
    "LabeledFrame": (".model", "LabeledFrame"),
    "Labels": (".model", "Labels"),
    "ProjectDescriptor": (".formats.project", "ProjectDescriptor"),
    "ROI": (".model", "ROI"),
    "SegmentationMask": (".model", "SegmentationMask"),
    "Skeleton": (".model", "Skeleton"),
    "PoseTrack": (".io.readers", "PoseTrack"),
    "Track": (".model", "Track"),
    "Video": (".model", "Video"),
    "WorkspaceLayout": (".services", "WorkspaceLayout"),
    "WorkspaceService": (".services", "WorkspaceService"),
    "convert_dlc_csv": (".adapters", "convert_dlc_csv"),
    "convert_dlc_h5": (".adapters", "convert_dlc_h5"),
    "convert_dlc_project": (".adapters", "convert_dlc_project"),
    "convert_sleap_package": (".adapters", "convert_sleap_package"),
    "import_dlc_csv_workspace": (".formats.project", "import_dlc_csv_workspace"),
    "import_dlc_h5_workspace": (".formats.project", "import_dlc_h5_workspace"),
    "import_legacy_archive": (".formats.project", "import_legacy_archive"),
    "import_sleap_package_workspace": (".formats.project", "import_sleap_package_workspace"),
    "init_project": (".formats.project", "init_project"),
    "load_project_descriptor": (".formats.project", "load_project_descriptor"),
    "pack_project": (".formats.project", "pack_project"),
    "read_pose_node_names": (".io.readers", "read_pose_node_names"),
    "read_pose_track": (".io.readers", "read_pose_track"),
    "resolve_pose_node_indices": (".io.readers", "resolve_pose_node_indices"),
    "save_workspace_labels": (".formats.project", "save_workspace_labels"),
    "unpack_project": (".formats.project", "unpack_project"),
    "validate_workspace": (".formats.project", "validate_workspace"),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    module_ref = _LAZY_EXPORTS.get(name)
    if module_ref is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = module_ref
    module = importlib.import_module(module_name, __package__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
