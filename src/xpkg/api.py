"""Stable public API for project-first xpkg integrations.

New integrations should start with ``ProjectService`` and
``ProjectService.imports``. The explicit ``import_*_project(...)`` helpers
remain public for function-level callers.
"""

from __future__ import annotations

import importlib
from typing import Any

_PROJECT_EXPORTS: dict[str, tuple[str, str]] = {
    "ProjectService": (".services", "ProjectService"),
    "ProjectImports": (".services", "ProjectImports"),
    "ProjectLayout": (".services", "ProjectLayout"),
    "ProjectArtifacts": (".services", "ProjectArtifacts"),
    "ProjectFigures": (".services", "ProjectFigures"),
    "ProjectSegmentation": (".services", "ProjectSegmentation"),
    "ProjectInspection": (".project.inspection", "ProjectInspection"),
    "ProjectDescriptor": (".project.layout", "ProjectDescriptor"),
    "ArtifactFile": (".project.artifacts", "ArtifactFile"),
    "ArtifactIndexEntry": (".project.artifacts", "ArtifactIndexEntry"),
    "ArtifactManifest": (".project.artifacts", "ArtifactManifest"),
    "FigureArtifact": (".project.figures", "FigureArtifact"),
    "SegmentationFrame": (".project", "SegmentationFrame"),
    "init_project": (".project", "init_project"),
    "load_project_descriptor": (".project.layout", "load_project_descriptor"),
    "load_project_vicon_recording": (".project", "load_project_vicon_recording"),
    "inspect_project": (".project.inspection", "inspect_project"),
    "save_project_labels": (".project", "save_project_labels"),
    "load_project_metadata_field": (".project.metadata", "load_project_metadata_field"),
    "save_project_metadata_field": (".project.metadata", "save_project_metadata_field"),
    "save_project_metadata": (".project", "save_project_metadata"),
    "load_project_metadata": (".project", "load_project_metadata"),
    "load_project_payload": (".project", "load_project_payload"),
    "list_project_artifacts": (".project.artifacts", "list_project_artifacts"),
    "list_project_artifact_index": (".project.artifacts", "list_project_artifact_index"),
    "load_project_artifact": (".project.artifacts", "load_project_artifact"),
    "save_project_artifact": (".project.artifacts", "save_project_artifact"),
    "validate_project_artifact": (".project.artifacts", "validate_project_artifact"),
    "validate_project_artifacts": (".project.artifacts", "validate_project_artifacts"),
    "rebuild_project_artifact_index": (
        ".project.artifacts",
        "rebuild_project_artifact_index",
    ),
    "list_project_figures": (".project.figures", "list_project_figures"),
    "load_project_figure": (".project.figures", "load_project_figure"),
    "save_project_figure": (".project.figures", "save_project_figure"),
    "validate_project_figure": (".project.figures", "validate_project_figure"),
    "validate_project_figures": (".project.figures", "validate_project_figures"),
    "load_project_segmentation_frames": (
        ".project",
        "load_project_segmentation_frames",
    ),
    "load_project_segmentation_masks": (
        ".project",
        "load_project_segmentation_masks",
    ),
    "save_project_segmentation_masks": (
        ".project",
        "save_project_segmentation_masks",
    ),
    "clear_project_segmentation_masks": (
        ".project",
        "clear_project_segmentation_masks",
    ),
    "current_project_state_path": (".project", "current_project_state_path"),
    "current_project_snapshot_path": (".project", "current_project_snapshot_path"),
    "pack_project": (".project", "pack_project"),
    "unpack_project": (".project", "unpack_project"),
    "validate_project": (".project", "validate_project"),
    "default_expkg_path": (".project.layout", "default_expkg_path"),
    "import_vicon_project": (".project", "import_vicon_project"),
    "import_vicon_csv_project": (".project", "import_vicon_csv_project"),
    "import_vicon_c3d_project": (".project", "import_vicon_c3d_project"),
    "import_dlc_csv_project": (".project", "import_dlc_csv_project"),
    "import_dlc_h5_project": (".project", "import_dlc_h5_project"),
    "import_dlc_project_directory": (".project", "import_dlc_project_directory"),
    "import_lightning_pose_csv_project": (
        ".project",
        "import_lightning_pose_csv_project",
    ),
    "import_mediapipe_pose_landmarks_json_project": (
        ".project",
        "import_mediapipe_pose_landmarks_json_project",
    ),
    "import_mmpose_topdown_json_project": (
        ".project",
        "import_mmpose_topdown_json_project",
    ),
    "import_sleap_h5_project": (".project", "import_sleap_h5_project"),
    "import_sleap_package_project": (".project", "import_sleap_package_project"),
}

_MODEL_EXPORTS: dict[str, tuple[str, str]] = {
    "EMGSignalData": (".model", "EMGSignalData"),
    "Event": (".model", "Event"),
    "EventTable": (".model", "EventTable"),
    "ForcePlateData": (".model", "ForcePlateData"),
    "Instance": (".model", "Instance"),
    "Keypoint": (".model", "Keypoint"),
    "LabeledFrame": (".model", "LabeledFrame"),
    "Labels": (".model", "Labels"),
    "PhotometryChannel": (".model", "PhotometryChannel"),
    "PhotometryRecording": (".model", "PhotometryRecording"),
    "RecordingSession": (".model", "RecordingSession"),
    "ROI": (".model", "ROI"),
    "SegmentationMask": (".model", "SegmentationMask"),
    "SignalChannel": (".model", "SignalChannel"),
    "Skeleton": (".model", "Skeleton"),
    "SyncEvent": (".model", "SyncEvent"),
    "Timeline": (".model", "Timeline"),
    "TimeRange": (".model", "TimeRange"),
    "TimeSeries": (".model", "TimeSeries"),
    "Timebase": (".model", "Timebase"),
    "Track": (".model", "Track"),
    "Video": (".model", "Video"),
    "VideoStub": (".model", "VideoStub"),
    "ViconAdditionalPointData": (".model", "ViconAdditionalPointData"),
    "ViconAnalogData": (".model", "ViconAnalogData"),
    "ViconCamera": (".model", "ViconCamera"),
    "ViconEvent": (".model", "ViconEvent"),
    "ViconForcePlatformMetadata": (".model", "ViconForcePlatformMetadata"),
    "ViconMarkerModel": (".model", "ViconMarkerModel"),
    "ViconRecording": (".model", "ViconRecording"),
}

_ADAPTER_AND_READER_EXPORTS: dict[str, tuple[str, str]] = {
    "PoseTrack": (".io.readers", "PoseTrack"),
    "build_force_plate_data_from_vicon_recording": (
        ".io.readers",
        "build_force_plate_data_from_vicon_recording",
    ),
    "build_prediction_stub": (".model", "build_prediction_stub"),
    "candidate_vicon_emg_channels": (".io.readers", "candidate_vicon_emg_channels"),
    "extract_vicon_emg": (".io.readers", "extract_vicon_emg"),
    "labels_from_json_payload": (".adapters", "labels_from_json_payload"),
    "labels_numpy": (".adapters", "labels_numpy"),
    "labels_to_dataframe": (".adapters", "labels_to_dataframe"),
    "labels_to_json_payload": (".adapters", "labels_to_json_payload"),
    "read_doric_photometry": (".io.readers", "read_doric_photometry"),
    "read_events_csv": (".io.readers", "read_events_csv"),
    "read_neurophotometrics_csv": (".io.readers", "read_neurophotometrics_csv"),
    "read_pmat_events_csv": (".io.readers", "read_pmat_events_csv"),
    "read_pmat_photometry_csv": (".io.readers", "read_pmat_photometry_csv"),
    "read_vicon_json_payload": (".adapters", "read_vicon_json_payload"),
    "vicon_recording_from_json_payload": (".adapters", "vicon_recording_from_json_payload"),
    "vicon_recording_to_json_payload": (".adapters", "vicon_recording_to_json_payload"),
    "read_pose_node_names": (".io.readers", "read_pose_node_names"),
    "read_pose_track": (".io.readers", "read_pose_track"),
    "read_pyphotometry_csv": (".io.readers", "read_pyphotometry_csv"),
    "read_pyphotometry_ppd": (".io.readers", "read_pyphotometry_ppd"),
    "read_rwd_ofrs_session": (".io.readers", "read_rwd_ofrs_session"),
    "read_tdt_photometry_block": (".io.readers", "read_tdt_photometry_block"),
    "read_teleopto_h5": (".io.readers", "read_teleopto_h5"),
    "read_vicon_c3d": (".io.readers", "read_vicon_c3d"),
    "read_vicon_csv": (".io.readers", "read_vicon_csv"),
    "read_vicon_recording": (".io.readers", "read_vicon_recording"),
    "resolve_pose_node_indices": (".io.readers", "resolve_pose_node_indices"),
    "read_photometry_csv": (".io.readers", "read_photometry_csv"),
}

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    **_PROJECT_EXPORTS,
    **_MODEL_EXPORTS,
    **_ADAPTER_AND_READER_EXPORTS,
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
