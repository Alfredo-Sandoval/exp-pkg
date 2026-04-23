"""Stable public API for workspace-first xpkg integrations.

New integrations should start with ``WorkspaceService`` and
``WorkspaceService.imports``. The explicit ``import_*_workspace(...)`` helpers
remain public for function-level callers, and ``migrate_legacy_archive(...)``
is the one retained bridge for cutting older ``.xpkg`` archives over to the
workspace contract.
"""

from __future__ import annotations

import importlib
from typing import Any

_WORKSPACE_EXPORTS: dict[str, tuple[str, str]] = {
    "WorkspaceService": (".services", "WorkspaceService"),
    "WorkspaceImports": (".services", "WorkspaceImports"),
    "WorkspaceLayout": (".services", "WorkspaceLayout"),
    "WorkspaceInspection": (".formats.project", "WorkspaceInspection"),
    "ProjectDescriptor": (".formats.project", "ProjectDescriptor"),
    "init_project": (".formats.project", "init_project"),
    "load_project_descriptor": (".formats.project", "load_project_descriptor"),
    "load_workspace_vicon_recording": (".formats.project", "load_workspace_vicon_recording"),
    "inspect_workspace": (".formats.project", "inspect_workspace"),
    "save_workspace_labels": (".formats.project", "save_workspace_labels"),
    "load_workspace_metadata_field": (".formats.project", "load_workspace_metadata_field"),
    "save_workspace_metadata_field": (".formats.project", "save_workspace_metadata_field"),
    "current_project_state_path": (".formats.project", "current_project_state_path"),
    "current_project_snapshot_path": (".formats.project", "current_project_snapshot_path"),
    "pack_project": (".formats.project", "pack_project"),
    "unpack_project": (".formats.project", "unpack_project"),
    "validate_workspace": (".formats.project", "validate_workspace"),
    "default_expkg_path": (".formats.project", "default_expkg_path"),
    "import_vicon_workspace": (".formats.project", "import_vicon_workspace"),
    "import_vicon_csv_workspace": (".formats.project", "import_vicon_csv_workspace"),
    "import_vicon_c3d_workspace": (".formats.project", "import_vicon_c3d_workspace"),
    "import_detectron2_coco_workspace": (".formats.project", "import_detectron2_coco_workspace"),
    "import_dlc_csv_workspace": (".formats.project", "import_dlc_csv_workspace"),
    "import_dlc_h5_workspace": (".formats.project", "import_dlc_h5_workspace"),
    "import_dlc_project_workspace": (".formats.project", "import_dlc_project_workspace"),
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
    "migrate_legacy_archive": (".formats.project", "migrate_legacy_archive"),
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
    "VideoStub": (".model", "VideoStub"),
    "ViconAdditionalPointData": (".model", "ViconAdditionalPointData"),
    "ViconAnalogData": (".model", "ViconAnalogData"),
    "ViconCamera": (".model", "ViconCamera"),
    "ViconMarkerModel": (".model", "ViconMarkerModel"),
    "ViconRecording": (".model", "ViconRecording"),
}

_CODEC_AND_READER_EXPORTS: dict[str, tuple[str, str]] = {
    "PoseTrack": (".io.readers", "PoseTrack"),
    "build_prediction_stub": (".model", "build_prediction_stub"),
    "labels_from_json_payload": (".codecs", "labels_from_json_payload"),
    "labels_numpy": (".codecs", "labels_numpy"),
    "labels_to_dataframe": (".codecs", "labels_to_dataframe"),
    "labels_to_json_payload": (".codecs", "labels_to_json_payload"),
    "read_vicon_json_payload": (".codecs", "read_vicon_json_payload"),
    "vicon_recording_from_json_payload": (".codecs", "vicon_recording_from_json_payload"),
    "vicon_recording_to_json_payload": (".codecs", "vicon_recording_to_json_payload"),
    "read_pose_node_names": (".io.readers", "read_pose_node_names"),
    "read_pose_track": (".io.readers", "read_pose_track"),
    "read_vicon_c3d": (".io.readers", "read_vicon_c3d"),
    "read_vicon_csv": (".io.readers", "read_vicon_csv"),
    "read_vicon_recording": (".io.readers", "read_vicon_recording"),
    "resolve_pose_node_indices": (".io.readers", "resolve_pose_node_indices"),
}

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    **_WORKSPACE_EXPORTS,
    **_MODEL_EXPORTS,
    **_CODEC_AND_READER_EXPORTS,
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
