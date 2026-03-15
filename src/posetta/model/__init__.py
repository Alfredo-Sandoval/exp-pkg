"""Canonical in-memory pose model for Posetta."""

from __future__ import annotations

from posetta.core.annotations import (
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
from posetta.core.skeleton import Keypoint, Skeleton, build_keypoint_skeleton
from posetta.io.labels.model import Labels, SuggestionFrame
from posetta.io.skeleton_loaders import (
    load_skeleton,
    load_skeleton_dlc,
    load_skeleton_siesta_json,
    load_skeleton_sleap,
    load_skeleton_ultralytics,
)
from posetta.io.video import Video

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
    "is_predicted_instance",
    "load_skeleton",
    "load_skeleton_dlc",
    "load_skeleton_siesta_json",
    "load_skeleton_sleap",
    "load_skeleton_ultralytics",
    "rle_decode",
    "rle_encode",
]
