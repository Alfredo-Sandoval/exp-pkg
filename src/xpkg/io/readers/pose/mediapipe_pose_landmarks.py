"""Low-level readers for serialized MediaPipe Pose Landmarker exports."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from xpkg._core.json_utils import load_json_dict
from xpkg.io.readers.pose._common import (
    PoseTrack,
    build_pose_track,
    resolve_node_indices_from_names,
)

MEDIAPIPE_POSE_LANDMARKS_JSON_SCHEMA = "mediapipe_pose_landmarks/v1"

MEDIAPIPE_POSE_LANDMARK_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)

MEDIAPIPE_POSE_CONNECTIONS: tuple[tuple[str, str], ...] = (
    ("nose", "left_eye_inner"),
    ("left_eye_inner", "left_eye"),
    ("left_eye", "left_eye_outer"),
    ("left_eye_outer", "left_ear"),
    ("nose", "right_eye_inner"),
    ("right_eye_inner", "right_eye"),
    ("right_eye", "right_eye_outer"),
    ("right_eye_outer", "right_ear"),
    ("mouth_left", "mouth_right"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("left_wrist", "left_pinky"),
    ("left_wrist", "left_index"),
    ("left_wrist", "left_thumb"),
    ("left_pinky", "left_index"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("right_wrist", "right_pinky"),
    ("right_wrist", "right_index"),
    ("right_wrist", "right_thumb"),
    ("right_pinky", "right_index"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("right_hip", "right_knee"),
    ("left_knee", "left_ankle"),
    ("right_knee", "right_ankle"),
    ("left_ankle", "left_heel"),
    ("right_ankle", "right_heel"),
    ("left_heel", "left_foot_index"),
    ("right_heel", "right_foot_index"),
    ("left_ankle", "left_foot_index"),
    ("right_ankle", "right_foot_index"),
)


@dataclass(frozen=True)
class _SerializedLandmark:
    x: float
    y: float
    z: float
    visibility: float
    presence: float | None


@dataclass(frozen=True)
class _SerializedFrame:
    frame_index: int
    pose_landmarks: tuple[_SerializedLandmark, ...] | None


@dataclass(frozen=True)
class _MediaPipePoseExport:
    image_width: int
    image_height: int
    frames: tuple[_SerializedFrame, ...]


def _require_mapping(value: Any, *, field_name: str, path: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a JSON object in {path}")
    return value


def _require_sequence(value: Any, *, field_name: str, path: Path) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a JSON array in {path}")
    return value


def _require_nonnegative_int(value: Any, *, field_name: str, path: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer in {path}")
    if value < 0:
        raise ValueError(f"{field_name} must be >= 0 in {path}")
    return int(value)


def _require_positive_int(value: Any, *, field_name: str, path: Path) -> int:
    parsed = _require_nonnegative_int(value, field_name=field_name, path=path)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be > 0 in {path}")
    return parsed


def _require_finite_float(value: Any, *, field_name: str, path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field_name} must be numeric in {path}")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite in {path}")
    return parsed


def _parse_landmark(value: Any, *, path: Path, index: int) -> _SerializedLandmark:
    payload = _require_mapping(
        value,
        field_name=f"frames[].pose_landmarks[{index}]",
        path=path,
    )
    presence_raw = payload.get("presence")
    presence = (
        None
        if presence_raw is None
        else _require_finite_float(
            presence_raw,
            field_name=f"frames[].pose_landmarks[{index}].presence",
            path=path,
        )
    )
    return _SerializedLandmark(
        x=_require_finite_float(
            payload.get("x"),
            field_name="frames[].pose_landmarks[].x",
            path=path,
        ),
        y=_require_finite_float(
            payload.get("y"),
            field_name="frames[].pose_landmarks[].y",
            path=path,
        ),
        z=_require_finite_float(
            payload.get("z"),
            field_name="frames[].pose_landmarks[].z",
            path=path,
        ),
        visibility=_require_finite_float(
            payload.get("visibility"),
            field_name="frames[].pose_landmarks[].visibility",
            path=path,
        ),
        presence=presence,
    )


def _parse_frame(value: Any, *, path: Path) -> _SerializedFrame:
    payload = _require_mapping(value, field_name="frames[]", path=path)
    frame_index = _require_nonnegative_int(
        payload.get("frame_index"),
        field_name="frame_index",
        path=path,
    )
    raw_landmarks = payload.get("pose_landmarks")

    if raw_landmarks is None:
        return _SerializedFrame(frame_index=frame_index, pose_landmarks=None)

    landmarks = _require_sequence(raw_landmarks, field_name="pose_landmarks", path=path)
    if not landmarks:
        return _SerializedFrame(frame_index=frame_index, pose_landmarks=None)
    if len(landmarks) != len(MEDIAPIPE_POSE_LANDMARK_NAMES):
        raise ValueError(
            "pose_landmarks must contain exactly "
            f"{len(MEDIAPIPE_POSE_LANDMARK_NAMES)} landmarks in {path}"
        )
    return _SerializedFrame(
        frame_index=frame_index,
        pose_landmarks=tuple(
            _parse_landmark(entry, path=path, index=landmark_index)
            for landmark_index, entry in enumerate(landmarks)
        ),
    )


def _load_export(path: Path) -> _MediaPipePoseExport:
    payload = load_json_dict(path)
    schema = str(payload.get("schema", "")).strip()
    if schema != MEDIAPIPE_POSE_LANDMARKS_JSON_SCHEMA:
        raise ValueError(
            "Unsupported MediaPipe pose-landmarks schema "
            f"{schema!r} in {path}. Expected {MEDIAPIPE_POSE_LANDMARKS_JSON_SCHEMA!r}."
        )

    frames = tuple(
        _parse_frame(frame_payload, path=path)
        for frame_payload in _require_sequence(
            payload.get("frames"),
            field_name="frames",
            path=path,
        )
    )
    frame_indices = [frame.frame_index for frame in frames]
    if len(frame_indices) != len(set(frame_indices)):
        raise ValueError(f"Duplicate frame_index values are not allowed in {path}")

    return _MediaPipePoseExport(
        image_width=_require_positive_int(
            payload.get("image_width"),
            field_name="image_width",
            path=path,
        ),
        image_height=_require_positive_int(
            payload.get("image_height"),
            field_name="image_height",
            path=path,
        ),
        frames=frames,
    )


def _landmark_score(landmark: _SerializedLandmark) -> float:
    if landmark.presence is None:
        return landmark.visibility
    return min(landmark.visibility, landmark.presence)


def read_image_size(path: Path) -> tuple[int, int]:
    """Return the serialized source image size for a MediaPipe pose export."""

    export = _load_export(Path(path))
    return export.image_width, export.image_height


def read_node_names(path: Path) -> list[str]:
    """Return the canonical MediaPipe pose-landmark names."""

    _load_export(Path(path))
    return list(MEDIAPIPE_POSE_LANDMARK_NAMES)


def read_track(path: Path, *, track_index: int) -> PoseTrack:
    """Read the single supported MediaPipe pose track as a PoseTrack."""

    idx = int(track_index)
    if idx != 0:
        raise ValueError(
            "MediaPipe pose-landmarks JSON currently supports only a single pose track; "
            f"track_index must be 0, got {track_index!r}."
        )

    resolved_path = Path(path)
    export = _load_export(resolved_path)
    frame_count = max((frame.frame_index for frame in export.frames), default=-1) + 1
    node_count = len(MEDIAPIPE_POSE_LANDMARK_NAMES)

    coords = np.full((frame_count, node_count, 2), np.nan, dtype=np.float64)
    scores = np.full((frame_count, node_count), np.nan, dtype=np.float64)

    for frame in export.frames:
        if frame.pose_landmarks is None:
            continue
        for node_index, landmark in enumerate(frame.pose_landmarks):
            coords[frame.frame_index, node_index, 0] = landmark.x * export.image_width
            coords[frame.frame_index, node_index, 1] = landmark.y * export.image_height
            scores[frame.frame_index, node_index] = _landmark_score(landmark)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        instance_score = np.nanmean(scores, axis=1)

    return build_pose_track(
        coords=coords,
        scores=scores,
        node_names=MEDIAPIPE_POSE_LANDMARK_NAMES,
        instance_score=instance_score,
        source_label=f"MediaPipe pose-landmarks file {resolved_path}",
    )


def resolve_node_indices(path: Path, target_names: Sequence[str]) -> list[int]:
    """Map target node names to their MediaPipe pose-landmark indices."""

    return resolve_node_indices_from_names(read_node_names(path), target_names)


__all__ = [
    "MEDIAPIPE_POSE_CONNECTIONS",
    "MEDIAPIPE_POSE_LANDMARKS_JSON_SCHEMA",
    "MEDIAPIPE_POSE_LANDMARK_NAMES",
    "PoseTrack",
    "read_image_size",
    "read_node_names",
    "read_track",
    "resolve_node_indices",
]
