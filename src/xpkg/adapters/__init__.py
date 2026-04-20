"""Public adapter entry points for external pose ecosystems."""

from __future__ import annotations

from xpkg.adapters.detectron2 import convert_detectron2_coco
from xpkg.adapters.dlc import (
    convert_dlc_csv,
    convert_dlc_h5,
    convert_dlc_h5_project,
    convert_dlc_project,
)
from xpkg.adapters.mediapipe import convert_mediapipe_pose_landmarks_json
from xpkg.adapters.mmpose import convert_mmpose_topdown_json
from xpkg.adapters.openpose import convert_openpose_json
from xpkg.adapters.sleap import convert_sleap_h5, convert_sleap_package
from xpkg.io.converters.converter_helpers import ConversionResult

__all__ = [
    "ConversionResult",
    "convert_dlc_csv",
    "convert_dlc_h5",
    "convert_dlc_h5_project",
    "convert_dlc_project",
    "convert_detectron2_coco",
    "convert_mediapipe_pose_landmarks_json",
    "convert_mmpose_topdown_json",
    "convert_openpose_json",
    "convert_sleap_h5",
    "convert_sleap_package",
]
