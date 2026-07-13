"""Canonical lazy-export registry for :mod:`xpkg.pose`."""

from __future__ import annotations

EXPORTS: dict[str, tuple[str, str]] = {
    "CoordinateFrameKind": ("xpkg.pose.trajectory", "CoordinateFrameKind"),
    "Instance": (".annotations", "Instance"),
    "KPFlag": (".annotations", "KPFlag"),
    "Keypoint": (".skeleton", "Keypoint"),
    "LabeledFrame": (".annotations", "LabeledFrame"),
    "Point": (".annotations", "Point"),
    "PointArray": (".annotations", "PointArray"),
    "PoseCoordinateFrame": ("xpkg.pose.trajectory", "PoseCoordinateFrame"),
    "PoseTrack": (".trajectory", "PoseTrack"),
    "PoseTrajectory": (".trajectory", "PoseTrajectory"),
    "PredictedInstance": (".annotations", "PredictedInstance"),
    "PredictedPoint": (".annotations", "PredictedPoint"),
    "PredictedPointArray": (".annotations", "PredictedPointArray"),
    "Skeleton": (".skeleton", "Skeleton"),
    "Track": (".annotations", "Track"),
    "build_keypoint_skeleton": (".skeleton", "build_keypoint_skeleton"),
    "is_predicted_instance": (".annotations", "is_predicted_instance"),
}
