"""
Data structures for all labeled data contained with a project.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from xpkg.pose.annotations.frames import InstancesList, LabeledFrame
from xpkg.pose.annotations.instances import (
    Instance,
    PredictedInstance,
    Track,
    is_predicted_instance,
)
from xpkg.pose.annotations.normalize import normalize_point_like, normalize_points_sequence
from xpkg.pose.annotations.points import (
    KPFlag,
    Point,
    PointArray,
    PredictedPoint,
    PredictedPointArray,
)
from xpkg.pose.annotations.regions import (
    ROI,
    MaskType,
    PromptType,
    SegmentationMask,
    SegmentationPrompt,
    rle_decode,
    rle_encode,
)
from xpkg.pose.annotations.serde import make_instance_cattr
from xpkg.pose.skeleton import Keypoint, Skeleton

from ..._core.logging_utils import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from xpkg.media.video import Video
else:
    from typing import Any as Video

__all__ = [
    "Instance",
    "InstancesList",
    "KPFlag",
    "Keypoint",
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
    "Track",
    "Video",
    "is_predicted_instance",
    "logger",
    "make_instance_cattr",
    "normalize_point_like",
    "normalize_points_sequence",
    "rle_decode",
    "rle_encode",
]
