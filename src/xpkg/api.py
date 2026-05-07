"""Stable public API for project-first xpkg integrations.

New integrations should start with ``ProjectService`` and
``ProjectService.imports``. The explicit ``import_*_project(...)`` functions
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
    "ProjectCalibrations": (".services", "ProjectCalibrations"),
    "ProjectFigures": (".services", "ProjectFigures"),
    "ProjectSegmentation": (".services", "ProjectSegmentation"),
    "ProjectInspection": (".project.inspection", "ProjectInspection"),
    "ProjectDescriptor": (".project.layout", "ProjectDescriptor"),
    "ACQUISITION_METADATA_FILENAME": (
        ".project.metadata",
        "ACQUISITION_METADATA_FILENAME",
    ),
    "CALIBRATION_FILENAME": (".project.calibration", "CALIBRATION_FILENAME"),
    "CALIBRATION_SOURCE_DIRNAME": (".project.calibration", "CALIBRATION_SOURCE_DIRNAME"),
    "CALIBRATIONS_DIRNAME": (".project.calibration", "CALIBRATIONS_DIRNAME"),
    "DATASET_SHARE_METADATA_FILENAME": (
        ".project.metadata",
        "DATASET_SHARE_METADATA_FILENAME",
    ),
    "PROJECT_METADATA_DIRNAME": (".project.metadata", "PROJECT_METADATA_DIRNAME"),
    "ArtifactFile": (".project.artifacts", "ArtifactFile"),
    "ArtifactIndexEntry": (".project.artifacts", "ArtifactIndexEntry"),
    "ArtifactManifest": (".project.artifacts", "ArtifactManifest"),
    "FigureArtifact": (".project.artifacts", "FigureArtifact"),
    "SegmentationFrame": (".project", "SegmentationFrame"),
    "init_project": (".project", "init_project"),
    "load_project_descriptor": (".project.layout", "load_project_descriptor"),
    "load_project_vicon_recording": (".project", "load_project_vicon_recording"),
    "inspect_path": (".inspection", "inspect_path"),
    "inspect_project": (".project.inspection", "inspect_project"),
    "save_project_labels": (".project", "save_project_labels"),
    "load_project_acquisition_metadata": (
        ".project.metadata",
        "load_project_acquisition_metadata",
    ),
    "load_project_dataset_share_metadata": (
        ".project.metadata",
        "load_project_dataset_share_metadata",
    ),
    "load_project_pose_provenance": (
        ".project.metadata",
        "load_project_pose_provenance",
    ),
    "load_project_metadata_field": (".project.metadata", "load_project_metadata_field"),
    "project_acquisition_metadata_path": (
        ".project.metadata",
        "project_acquisition_metadata_path",
    ),
    "project_dataset_share_metadata_path": (
        ".project.metadata",
        "project_dataset_share_metadata_path",
    ),
    "project_pose_provenance_path": (
        ".project.metadata",
        "project_pose_provenance_path",
    ),
    "project_metadata_root": (".project.metadata", "project_metadata_root"),
    "save_project_acquisition_metadata": (
        ".project.metadata",
        "save_project_acquisition_metadata",
    ),
    "save_project_dataset_share_metadata": (
        ".project.metadata",
        "save_project_dataset_share_metadata",
    ),
    "save_project_pose_provenance": (
        ".project.metadata",
        "save_project_pose_provenance",
    ),
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
    "list_project_figures": (".project.artifacts", "list_project_figures"),
    "load_project_figure": (".project.artifacts", "load_project_figure"),
    "save_project_figure": (".project.artifacts", "save_project_figure"),
    "validate_project_figure": (".project.artifacts", "validate_project_figure"),
    "validate_project_figures": (".project.artifacts", "validate_project_figures"),
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
    "import_anipose_calibration_project": (
        ".project",
        "import_anipose_calibration_project",
    ),
    "list_project_calibrations": (".project", "list_project_calibrations"),
    "load_project_calibration": (".project", "load_project_calibration"),
    "project_calibration_path": (".project", "project_calibration_path"),
    "project_calibration_root": (".project", "project_calibration_root"),
    "project_calibration_source_root": (".project", "project_calibration_source_root"),
    "project_calibrations_root": (".project", "project_calibrations_root"),
    "save_project_calibration": (".project", "save_project_calibration"),
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
    "AcquisitionMetadata": (".model", "AcquisitionMetadata"),
    "CALIBRATION_SCHEMA_VERSION": (".model", "CALIBRATION_SCHEMA_VERSION"),
    "Calibration": (".model", "Calibration"),
    "CalibrationQuality": (".model", "CalibrationQuality"),
    "CalibrationSource": (".model", "CalibrationSource"),
    "Camera": (".model", "Camera"),
    "CameraDistortion": (".model", "CameraDistortion"),
    "CameraExtrinsics": (".model", "CameraExtrinsics"),
    "CameraIntrinsics": (".model", "CameraIntrinsics"),
    "CameraMetadata": (".model", "CameraMetadata"),
    "CameraRotation": (".model", "CameraRotation"),
    "DatasetShareMetadata": (".model", "DatasetShareMetadata"),
    "PoseModelProvenance": (".model", "PoseModelProvenance"),
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
    "WorldFrame": (".model", "WorldFrame"),
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
    "read_anipose_calibration": (".io.readers", "read_anipose_calibration"),
    "read_calibration_json": (".io.calibration", "read_calibration_json"),
    "write_calibration_json": (".io.calibration", "write_calibration_json"),
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
    "write_anipose_calibration": (".io.readers", "write_anipose_calibration"),
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
