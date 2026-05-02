"""Canonical in-memory pose model for xpkg."""

from __future__ import annotations

from xpkg.io.labels.model import Labels, SuggestionFrame
from xpkg.io.skeleton_loaders import (
    load_skeleton,
    load_skeleton_dlc,
    load_skeleton_sleap,
    load_skeleton_ultralytics,
    load_skeleton_xpkg_json,
)
from xpkg.media.video import Video
from xpkg.model.emg import EMGSignalData
from xpkg.model.events import Event, EventTable, SyncEvent
from xpkg.model.force import ForcePlateData
from xpkg.model.session import RecordingSession
from xpkg.model.signals import (
    PhotometryChannel,
    PhotometryRecording,
    SignalChannel,
    TimeSeries,
)
from xpkg.model.stubs import VideoStub, build_prediction_stub
from xpkg.model.time import Timebase, Timeline, TimeRange
from xpkg.model.vicon import (
    ViconAdditionalPointData,
    ViconAnalogData,
    ViconCamera,
    ViconEvent,
    ViconForcePlatformMetadata,
    ViconMarkerModel,
    ViconRecording,
)
from xpkg.pose.adapters import (
    skeleton_to_vicon_marker_model,
    vicon_marker_model_to_skeleton,
)
from xpkg.pose.annotations import (
    ROI,
    Instance,
    KPFlag,
    LabeledFrame,
    MaskType,
    Point,
    PointArray,
    PredictedInstance,
    PredictedPoint,
    PredictedPointArray,
    PromptType,
    SegmentationMask,
    SegmentationPrompt,
    Track,
    is_predicted_instance,
    rle_decode,
    rle_encode,
)
from xpkg.pose.skeleton import Keypoint, Skeleton, build_keypoint_skeleton
from xpkg.pose.trajectory import (
    PoseTrajectory,
    pose_trajectory_from_vicon_recording,
)

__all__ = [
    "build_keypoint_skeleton",
    "EMGSignalData",
    "Event",
    "EventTable",
    "ForcePlateData",
    "Instance",
    "KPFlag",
    "Keypoint",
    "Labels",
    "LabeledFrame",
    "MaskType",
    "Point",
    "PointArray",
    "PredictedInstance",
    "PredictedPoint",
    "PredictedPointArray",
    "PhotometryChannel",
    "PhotometryRecording",
    "PoseTrajectory",
    "PromptType",
    "RecordingSession",
    "ROI",
    "SegmentationMask",
    "SegmentationPrompt",
    "SignalChannel",
    "Skeleton",
    "SyncEvent",
    "Timeline",
    "TimeRange",
    "TimeSeries",
    "Timebase",
    "SuggestionFrame",
    "Track",
    "Video",
    "VideoStub",
    "ViconAdditionalPointData",
    "ViconAnalogData",
    "ViconCamera",
    "ViconEvent",
    "ViconForcePlatformMetadata",
    "ViconMarkerModel",
    "ViconRecording",
    "build_prediction_stub",
    "pose_trajectory_from_vicon_recording",
    "skeleton_to_vicon_marker_model",
    "vicon_marker_model_to_skeleton",
    "is_predicted_instance",
    "load_skeleton",
    "load_skeleton_dlc",
    "load_skeleton_xpkg_json",
    "load_skeleton_sleap",
    "load_skeleton_ultralytics",
    "rle_decode",
    "rle_encode",
]
