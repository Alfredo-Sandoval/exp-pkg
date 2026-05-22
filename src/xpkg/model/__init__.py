"""Canonical in-memory model surface for xpkg."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "AcquisitionMetadata": ("xpkg.model.metadata", "AcquisitionMetadata"),
    "BEHAVIOR_LABELS_SCHEMA_VERSION": (
        "xpkg.model.behavior",
        "BEHAVIOR_LABELS_SCHEMA_VERSION",
    ),
    "BehaviorEmbedding": ("xpkg.model.behavior", "BehaviorEmbedding"),
    "BehaviorFrameLabel": ("xpkg.model.behavior", "BehaviorFrameLabel"),
    "BehaviorInterval": ("xpkg.model.behavior", "BehaviorInterval"),
    "BehaviorLabels": ("xpkg.model.behavior", "BehaviorLabels"),
    "build_keypoint_skeleton": ("xpkg.pose.skeleton", "build_keypoint_skeleton"),
    "CALIBRATION_SCHEMA_VERSION": (
        "xpkg.model.calibration",
        "CALIBRATION_SCHEMA_VERSION",
    ),
    "Calibration": ("xpkg.model.calibration", "Calibration"),
    "CalibrationQuality": ("xpkg.model.calibration", "CalibrationQuality"),
    "CalibrationSource": ("xpkg.model.calibration", "CalibrationSource"),
    "Camera": ("xpkg.model.calibration", "Camera"),
    "CameraDistortion": ("xpkg.model.calibration", "CameraDistortion"),
    "CameraExtrinsics": ("xpkg.model.calibration", "CameraExtrinsics"),
    "CameraIntrinsics": ("xpkg.model.calibration", "CameraIntrinsics"),
    "CameraMetadata": ("xpkg.model.metadata", "CameraMetadata"),
    "CameraRotation": ("xpkg.model.calibration", "CameraRotation"),
    "ChannelRole": ("xpkg.model.ephys", "ChannelRole"),
    "DatasetDatasheet": ("xpkg.model.reporting", "DatasetDatasheet"),
    "DatasetShareMetadata": ("xpkg.model.metadata", "DatasetShareMetadata"),
    "DatasheetCollection": ("xpkg.model.reporting", "DatasheetCollection"),
    "DatasheetComposition": ("xpkg.model.reporting", "DatasheetComposition"),
    "DatasheetDistribution": ("xpkg.model.reporting", "DatasheetDistribution"),
    "DatasheetMaintenance": ("xpkg.model.reporting", "DatasheetMaintenance"),
    "DatasheetMotivation": ("xpkg.model.reporting", "DatasheetMotivation"),
    "DatasheetPreprocessing": ("xpkg.model.reporting", "DatasheetPreprocessing"),
    "DatasheetUses": ("xpkg.model.reporting", "DatasheetUses"),
    "EMGSignalData": ("xpkg.model.emg", "EMGSignalData"),
    "EphysRecording": ("xpkg.model.ephys", "EphysRecording"),
    "Event": ("xpkg.model.events", "Event"),
    "EventTable": ("xpkg.model.events", "EventTable"),
    "ForcePlateData": ("xpkg.model.force", "ForcePlateData"),
    "IDENTITY_PROVENANCE_SCHEMA_VERSION": (
        "xpkg.model.identity",
        "IDENTITY_PROVENANCE_SCHEMA_VERSION",
    ),
    "IDENTITY_SOURCES": ("xpkg.model.identity", "IDENTITY_SOURCES"),
    "Instance": ("xpkg.pose.annotations", "Instance"),
    "IdentityConfidenceSpan": ("xpkg.model.identity", "IdentityConfidenceSpan"),
    "IdentityEvent": ("xpkg.model.identity", "IdentityEvent"),
    "IdentityProofreadingSpan": ("xpkg.model.identity", "IdentityProofreadingSpan"),
    "IdentityProvenanceRecord": ("xpkg.model.identity", "IdentityProvenanceRecord"),
    "is_predicted_instance": ("xpkg.pose.annotations", "is_predicted_instance"),
    "Keypoint": ("xpkg.pose.skeleton", "Keypoint"),
    "KPFlag": ("xpkg.pose.annotations", "KPFlag"),
    "Labels": ("xpkg.io.labels.model", "Labels"),
    "LabeledFrame": ("xpkg.pose.annotations", "LabeledFrame"),
    "load_skeleton": ("xpkg.io.skeleton_loaders", "load_skeleton"),
    "load_skeleton_dlc": ("xpkg.io.skeleton_loaders", "load_skeleton_dlc"),
    "load_skeleton_sleap": ("xpkg.io.skeleton_loaders", "load_skeleton_sleap"),
    "load_skeleton_ultralytics": (
        "xpkg.io.skeleton_loaders",
        "load_skeleton_ultralytics",
    ),
    "load_skeleton_xpkg_json": ("xpkg.io.skeleton_loaders", "load_skeleton_xpkg_json"),
    "MaskType": ("xpkg.segmentation", "MaskType"),
    "ModelCard": ("xpkg.model.reporting", "ModelCard"),
    "ModelCardAnalysis": ("xpkg.model.reporting", "ModelCardAnalysis"),
    "ModelCardData": ("xpkg.model.reporting", "ModelCardData"),
    "ModelCardDetails": ("xpkg.model.reporting", "ModelCardDetails"),
    "ModelCardFactors": ("xpkg.model.reporting", "ModelCardFactors"),
    "ModelCardIntendedUse": ("xpkg.model.reporting", "ModelCardIntendedUse"),
    "ModelCardMetrics": ("xpkg.model.reporting", "ModelCardMetrics"),
    "PhotometryChannel": ("xpkg.model.signals", "PhotometryChannel"),
    "PhotometryRecording": ("xpkg.model.signals", "PhotometryRecording"),
    "Point": ("xpkg.pose.annotations", "Point"),
    "PointArray": ("xpkg.pose.annotations", "PointArray"),
    "PoseModelProvenance": ("xpkg.model.metadata", "PoseModelProvenance"),
    "PoseTrajectory": ("xpkg.pose.trajectory", "PoseTrajectory"),
    "pose_trajectory_from_vicon_recording": (
        "xpkg.pose.trajectory",
        "pose_trajectory_from_vicon_recording",
    ),
    "PredictedInstance": ("xpkg.pose.annotations", "PredictedInstance"),
    "PredictedPoint": ("xpkg.pose.annotations", "PredictedPoint"),
    "PredictedPointArray": ("xpkg.pose.annotations", "PredictedPointArray"),
    "PromptType": ("xpkg.segmentation", "PromptType"),
    "RecordingMode": ("xpkg.model.ephys", "RecordingMode"),
    "RecordingSession": ("xpkg.model.session", "RecordingSession"),
    "ROI": ("xpkg.segmentation", "ROI"),
    "rle_decode": ("xpkg.segmentation", "rle_decode"),
    "rle_encode": ("xpkg.segmentation", "rle_encode"),
    "SegmentationMask": ("xpkg.segmentation", "SegmentationMask"),
    "SegmentationPrompt": ("xpkg.segmentation", "SegmentationPrompt"),
    "SignalChannel": ("xpkg.model.signals", "SignalChannel"),
    "Skeleton": ("xpkg.pose.skeleton", "Skeleton"),
    "skeleton_to_vicon_marker_model": (
        "xpkg.pose.adapters",
        "skeleton_to_vicon_marker_model",
    ),
    "StimulusEpoch": ("xpkg.model.ephys", "StimulusEpoch"),
    "SuggestionFrame": ("xpkg.io.labels.model", "SuggestionFrame"),
    "Sweep": ("xpkg.model.ephys", "Sweep"),
    "SweepSet": ("xpkg.model.ephys", "SweepSet"),
    "SyncEvent": ("xpkg.model.events", "SyncEvent"),
    "Timebase": ("xpkg.model.time", "Timebase"),
    "Timeline": ("xpkg.model.time", "Timeline"),
    "TimeRange": ("xpkg.model.time", "TimeRange"),
    "TimeSeries": ("xpkg.model.signals", "TimeSeries"),
    "Track": ("xpkg.pose.annotations", "Track"),
    "ViconAdditionalPointData": ("xpkg.model.vicon", "ViconAdditionalPointData"),
    "ViconAnalogData": ("xpkg.model.vicon", "ViconAnalogData"),
    "ViconCamera": ("xpkg.model.vicon", "ViconCamera"),
    "ViconEvent": ("xpkg.model.vicon", "ViconEvent"),
    "ViconForcePlatformMetadata": ("xpkg.model.vicon", "ViconForcePlatformMetadata"),
    "ViconMarkerModel": ("xpkg.model.vicon", "ViconMarkerModel"),
    "ViconRecording": ("xpkg.model.vicon", "ViconRecording"),
    "vicon_marker_model_to_skeleton": (
        "xpkg.pose.adapters",
        "vicon_marker_model_to_skeleton",
    ),
    "Video": ("xpkg.media.video", "Video"),
    "VideoStub": ("xpkg.model.stubs", "VideoStub"),
    "WorldFrame": ("xpkg.model.calibration", "WorldFrame"),
    "build_prediction_stub": ("xpkg.model.stubs", "build_prediction_stub"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
