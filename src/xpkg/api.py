"""Stable public API for workspace-first xpkg integrations.

New integrations should start with ``WorkspaceService`` and
``WorkspaceService.imports``. The explicit ``import_*_workspace(...)`` helpers
remain public for function-level callers.
"""

from __future__ import annotations

import importlib
from typing import Any

_WORKSPACE_EXPORTS: dict[str, tuple[str, str]] = {
    "WorkspaceService": (".services", "WorkspaceService"),
    "WorkspaceImports": (".services", "WorkspaceImports"),
    "WorkspaceLayout": (".services", "WorkspaceLayout"),
    "WorkspaceArtifacts": (".services", "WorkspaceArtifacts"),
    "WorkspaceFigures": (".services", "WorkspaceFigures"),
    "WorkspaceSegmentation": (".services", "WorkspaceSegmentation"),
    "WorkspaceInspection": (".workspace.project", "WorkspaceInspection"),
    "ProjectDescriptor": (".workspace.project", "ProjectDescriptor"),
    "ArtifactFile": (".workspace.project", "ArtifactFile"),
    "ArtifactIndexEntry": (".workspace.project", "ArtifactIndexEntry"),
    "ArtifactManifest": (".workspace.project", "ArtifactManifest"),
    "FigureArtifact": (".workspace.project", "FigureArtifact"),
    "SegmentationFrame": (".workspace.project", "SegmentationFrame"),
    "init_project": (".workspace.project", "init_project"),
    "load_project_descriptor": (".workspace.project", "load_project_descriptor"),
    "load_workspace_vicon_recording": (".workspace.project", "load_workspace_vicon_recording"),
    "inspect_workspace": (".workspace.project", "inspect_workspace"),
    "save_workspace_labels": (".workspace.project", "save_workspace_labels"),
    "load_workspace_metadata_field": (".workspace.project", "load_workspace_metadata_field"),
    "save_workspace_metadata_field": (".workspace.project", "save_workspace_metadata_field"),
    "save_workspace_metadata": (".workspace.project", "save_workspace_metadata"),
    "load_workspace_metadata": (".workspace.project", "load_workspace_metadata"),
    "load_workspace_payload": (".workspace.project", "load_workspace_payload"),
    "list_workspace_artifacts": (".workspace.project", "list_workspace_artifacts"),
    "list_workspace_artifact_index": (".workspace.project", "list_workspace_artifact_index"),
    "load_workspace_artifact": (".workspace.project", "load_workspace_artifact"),
    "save_workspace_artifact": (".workspace.project", "save_workspace_artifact"),
    "validate_workspace_artifact": (".workspace.project", "validate_workspace_artifact"),
    "validate_workspace_artifacts": (".workspace.project", "validate_workspace_artifacts"),
    "rebuild_workspace_artifact_index": (
        ".workspace.project",
        "rebuild_workspace_artifact_index",
    ),
    "list_workspace_figures": (".workspace.project", "list_workspace_figures"),
    "load_workspace_figure": (".workspace.project", "load_workspace_figure"),
    "save_workspace_figure": (".workspace.project", "save_workspace_figure"),
    "validate_workspace_figure": (".workspace.project", "validate_workspace_figure"),
    "validate_workspace_figures": (".workspace.project", "validate_workspace_figures"),
    "load_workspace_segmentation_frames": (
        ".workspace.project",
        "load_workspace_segmentation_frames",
    ),
    "load_workspace_segmentation_masks": (
        ".workspace.project",
        "load_workspace_segmentation_masks",
    ),
    "save_workspace_segmentation_masks": (
        ".workspace.project",
        "save_workspace_segmentation_masks",
    ),
    "clear_workspace_segmentation_masks": (
        ".workspace.project",
        "clear_workspace_segmentation_masks",
    ),
    "current_project_state_path": (".workspace.project", "current_project_state_path"),
    "current_project_snapshot_path": (".workspace.project", "current_project_snapshot_path"),
    "pack_project": (".workspace.project", "pack_project"),
    "unpack_project": (".workspace.project", "unpack_project"),
    "validate_workspace": (".workspace.project", "validate_workspace"),
    "default_expkg_path": (".workspace.project", "default_expkg_path"),
    "import_vicon_workspace": (".workspace.project", "import_vicon_workspace"),
    "import_vicon_csv_workspace": (".workspace.project", "import_vicon_csv_workspace"),
    "import_vicon_c3d_workspace": (".workspace.project", "import_vicon_c3d_workspace"),
    "import_dlc_csv_workspace": (".workspace.project", "import_dlc_csv_workspace"),
    "import_dlc_h5_workspace": (".workspace.project", "import_dlc_h5_workspace"),
    "import_dlc_project_workspace": (".workspace.project", "import_dlc_project_workspace"),
    "import_lightning_pose_csv_workspace": (
        ".workspace.project",
        "import_lightning_pose_csv_workspace",
    ),
    "import_mediapipe_pose_landmarks_json_workspace": (
        ".workspace.project",
        "import_mediapipe_pose_landmarks_json_workspace",
    ),
    "import_mmpose_topdown_json_workspace": (
        ".workspace.project",
        "import_mmpose_topdown_json_workspace",
    ),
    "import_sleap_h5_workspace": (".workspace.project", "import_sleap_h5_workspace"),
    "import_sleap_package_workspace": (".workspace.project", "import_sleap_package_workspace"),
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
    "decode_hdf5_string": (".io.archive_format.tables_hdf5", "decode_hdf5_string"),
    "read_hdf5_table": (".io.archive_format.tables_hdf5", "read_hdf5_table"),
    "read_hdf5_table_group": (".io.archive_format.tables_hdf5", "read_hdf5_table_group"),
    "read_photometry_csv": (".io.readers", "read_photometry_csv"),
    "write_hdf5_table": (".io.archive_format.tables_hdf5", "write_hdf5_table"),
    "write_hdf5_table_group": (".io.archive_format.tables_hdf5", "write_hdf5_table_group"),
}

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    **_WORKSPACE_EXPORTS,
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
