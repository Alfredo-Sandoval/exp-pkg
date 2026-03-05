"""
Data structures for all labeled data contained with a project.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from posetta.core.annotations.frames import InstancesList, LabeledFrame
from posetta.core.annotations.instances import (
    Instance,
    PredictedInstance,
    Track,
    is_predicted_instance,
)
from posetta.core.annotations.normalize import normalize_point_like, normalize_points_sequence
from posetta.core.annotations.points import (
    KPFlag,
    Point,
    PointArray,
    PredictedPoint,
    PredictedPointArray,
)
from posetta.core.annotations.serde import make_instance_cattr
from posetta.core.logging_utils import get_logger
from posetta.core.skeleton import Keypoint, Skeleton

logger = get_logger(__name__)

if TYPE_CHECKING:
    from posetta.io.video import Video
else:
    from typing import Any as Video

__all__ = [
    "Instance",
    "InstancesList",
    "KPFlag",
    "Keypoint",
    "LabeledFrame",
    "Point",
    "PointArray",
    "PredictedInstance",
    "PredictedPoint",
    "PredictedPointArray",
    "Skeleton",
    "Track",
    "Video",
    "is_predicted_instance",
    "logger",
    "make_instance_cattr",
    "normalize_point_like",
    "normalize_points_sequence",
]
