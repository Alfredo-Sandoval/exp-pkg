"""Canonical in-memory pose model for xpkg."""

from __future__ import annotations

from xpkg.core.annotations import (
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
from xpkg.core.skeleton import Keypoint, Skeleton, build_keypoint_skeleton
from xpkg.io.labels.model import Labels, SuggestionFrame
from xpkg.io.skeleton_loaders import (
    load_skeleton,
    load_skeleton_dlc,
    load_skeleton_sleap,
    load_skeleton_ultralytics,
    load_skeleton_xpkg_json,
)
from xpkg.io.video import Video
from xpkg.model.stubs import VideoStub, build_prediction_stub
from xpkg.model.vicon import (
    ViconAdditionalPointData,
    ViconAnalogData,
    ViconCamera,
    ViconMarkerModel,
    ViconRecording,
)

__all__ = [
    "build_keypoint_skeleton",
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
    "PromptType",
    "ROI",
    "SegmentationMask",
    "SegmentationPrompt",
    "Skeleton",
    "SuggestionFrame",
    "Track",
    "Video",
    "VideoStub",
    "ViconAdditionalPointData",
    "ViconAnalogData",
    "ViconCamera",
    "ViconMarkerModel",
    "ViconRecording",
    "build_prediction_stub",
    "is_predicted_instance",
    "load_skeleton",
    "load_skeleton_dlc",
    "load_skeleton_xpkg_json",
    "load_skeleton_sleap",
    "load_skeleton_ultralytics",
    "rle_decode",
    "rle_encode",
]
