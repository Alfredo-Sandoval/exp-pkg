"""
Data structures for all labeled data contained with a project.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from xpkg.core.annotations.frames import InstancesList, LabeledFrame
from xpkg.core.annotations.instances import (
    Instance,
    PredictedInstance,
    Track,
    is_predicted_instance,
)
from xpkg.core.annotations.normalize import normalize_point_like, normalize_points_sequence
from xpkg.core.annotations.points import (
    KPFlag,
    Point,
    PointArray,
    PredictedPoint,
    PredictedPointArray,
)
from xpkg.core.annotations.regions import (
    ROI,
    MaskType,
    PromptType,
    SegmentationMask,
    SegmentationPrompt,
    rle_decode,
    rle_encode,
)
from xpkg.core.annotations.serde import make_instance_cattr
from xpkg.core.logging_utils import get_logger
from xpkg.core.skeleton import Keypoint, Skeleton

logger = get_logger(__name__)

if TYPE_CHECKING:
    from xpkg.io.video import Video
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
