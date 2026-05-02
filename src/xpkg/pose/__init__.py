"""Pose primitives, skeletons, and trajectory adapters."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "Instance": (".annotations", "Instance"),
    "KPFlag": (".annotations", "KPFlag"),
    "Keypoint": (".skeleton", "Keypoint"),
    "LabeledFrame": (".annotations", "LabeledFrame"),
    "MaskType": (".annotations", "MaskType"),
    "Point": (".annotations", "Point"),
    "PointArray": (".annotations", "PointArray"),
    "PoseTrajectory": (".trajectory", "PoseTrajectory"),
    "PredictedInstance": (".annotations", "PredictedInstance"),
    "PredictedPoint": (".annotations", "PredictedPoint"),
    "PredictedPointArray": (".annotations", "PredictedPointArray"),
    "PromptType": (".annotations", "PromptType"),
    "ROI": (".annotations", "ROI"),
    "SegmentationMask": (".annotations", "SegmentationMask"),
    "SegmentationPrompt": (".annotations", "SegmentationPrompt"),
    "Skeleton": (".skeleton", "Skeleton"),
    "Track": (".annotations", "Track"),
    "build_keypoint_skeleton": (".skeleton", "build_keypoint_skeleton"),
    "is_predicted_instance": (".annotations", "is_predicted_instance"),
    "pose_trajectory_from_vicon_recording": (
        ".trajectory",
        "pose_trajectory_from_vicon_recording",
    ),
    "rle_decode": (".annotations", "rle_decode"),
    "rle_encode": (".annotations", "rle_encode"),
    "skeleton_to_vicon_marker_model": (".adapters", "skeleton_to_vicon_marker_model"),
    "vicon_marker_model_to_skeleton": (".adapters", "vicon_marker_model_to_skeleton"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
