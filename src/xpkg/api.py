"""Stable public API for xpkg integrations.

New integrations should start with ``WorkspaceService`` and
``WorkspaceService.imports``. The explicit ``import_*_workspace(...)`` helpers
remain public for function-level callers. Compatibility adapters remain public,
but they are grouped later in this facade so the workspace-first path is easier
to discover.
"""

from __future__ import annotations

import importlib
from typing import Any

_WORKSPACE_EXPORTS: dict[str, tuple[str, str]] = {
    "WorkspaceLayout": (".services", "WorkspaceLayout"),
    "WorkspaceImports": (".services", "WorkspaceImports"),
    "WorkspaceService": (".services", "WorkspaceService"),
    "ProjectDescriptor": (".formats.project", "ProjectDescriptor"),
    "init_project": (".formats.project", "init_project"),
    "load_project_descriptor": (".formats.project", "load_project_descriptor"),
    "save_workspace_labels": (".formats.project", "save_workspace_labels"),
    "current_project_state_path": (".formats.project", "current_project_state_path"),
    "current_project_snapshot_path": (".formats.project", "current_project_snapshot_path"),
    "pack_project": (".formats.project", "pack_project"),
    "unpack_project": (".formats.project", "unpack_project"),
    "validate_workspace": (".formats.project", "validate_workspace"),
    "default_expkg_path": (".formats.project", "default_expkg_path"),
    "import_detectron2_coco_workspace": (".formats.project", "import_detectron2_coco_workspace"),
    "import_dlc_csv_workspace": (".formats.project", "import_dlc_csv_workspace"),
    "import_dlc_h5_workspace": (".formats.project", "import_dlc_h5_workspace"),
    "import_dlc_project_workspace": (".formats.project", "import_dlc_project_workspace"),
    "import_legacy_archive": (".formats.project", "import_legacy_archive"),
    "import_mediapipe_pose_landmarks_json_workspace": (
        ".formats.project",
        "import_mediapipe_pose_landmarks_json_workspace",
    ),
    "import_mmpose_topdown_json_workspace": (
        ".formats.project",
        "import_mmpose_topdown_json_workspace",
    ),
    "import_openpose_json_workspace": (".formats.project", "import_openpose_json_workspace"),
    "import_sleap_h5_workspace": (".formats.project", "import_sleap_h5_workspace"),
    "import_sleap_package_workspace": (".formats.project", "import_sleap_package_workspace"),
    "export_project_archive": (".formats.project", "export_project_archive"),
    "current_project_archive_path": (".formats.project", "current_project_archive_path"),
}

_MODEL_EXPORTS: dict[str, tuple[str, str]] = {
    "Instance": (".model", "Instance"),
    "Keypoint": (".model", "Keypoint"),
    "LabeledFrame": (".model", "LabeledFrame"),
    "Labels": (".model", "Labels"),
    "ROI": (".model", "ROI"),
    "SegmentationMask": (".model", "SegmentationMask"),
    "Skeleton": (".model", "Skeleton"),
    "Track": (".model", "Track"),
    "Video": (".model", "Video"),
}

_CODEC_AND_READER_EXPORTS: dict[str, tuple[str, str]] = {
    "PoseTrack": (".io.readers", "PoseTrack"),
    "labels_from_json_payload": (".codecs", "labels_from_json_payload"),
    "labels_numpy": (".codecs", "labels_numpy"),
    "labels_to_dataframe": (".codecs", "labels_to_dataframe"),
    "labels_to_json_payload": (".codecs", "labels_to_json_payload"),
    "read_pose_node_names": (".io.readers", "read_pose_node_names"),
    "read_pose_track": (".io.readers", "read_pose_track"),
    "resolve_pose_node_indices": (".io.readers", "resolve_pose_node_indices"),
}

_ADAPTER_EXPORTS: dict[str, tuple[str, str]] = {
    "ConversionResult": (".adapters", "ConversionResult"),
    "convert_dlc_csv": (".adapters", "convert_dlc_csv"),
    "convert_dlc_h5": (".adapters", "convert_dlc_h5"),
    "convert_dlc_project": (".adapters", "convert_dlc_project"),
    "convert_detectron2_coco": (".adapters", "convert_detectron2_coco"),
    "convert_mediapipe_pose_landmarks_json": (
        ".adapters",
        "convert_mediapipe_pose_landmarks_json",
    ),
    "convert_mmpose_topdown_json": (".adapters", "convert_mmpose_topdown_json"),
    "convert_openpose_json": (".adapters", "convert_openpose_json"),
    "convert_sleap_h5": (".adapters", "convert_sleap_h5"),
    "convert_sleap_package": (".adapters", "convert_sleap_package"),
}

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    **_WORKSPACE_EXPORTS,
    **_MODEL_EXPORTS,
    **_CODEC_AND_READER_EXPORTS,
    **_ADAPTER_EXPORTS,
}

__all__ = list(_LAZY_EXPORTS)


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
