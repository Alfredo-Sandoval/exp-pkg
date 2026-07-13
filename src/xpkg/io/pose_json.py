"""JSON payload conversion for typed session-pose links."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from xpkg.io.labels.json_format import labels_from_json_payload, labels_to_json_payload
from xpkg.io.predictions import (
    PredictionAppendItem,
    PredictionLabelsView,
    coerce_predictions_from_labels,
)


def pose_labels_payload(
    labels: PredictionLabelsView, *, project_root: Path | None
) -> dict[str, Any]:
    """Serialize pose labels and predictions for a session-pose link."""
    document = labels_to_json_payload(cast(Any, labels))
    payload = cast("dict[str, Any]", document["payload"])
    payload["predictions"] = predictions_payload_from_labels(labels)
    if project_root is not None:
        _relativize_video_paths(payload, project_root)
    return payload


def pose_labels_from_payload(payload: Mapping[str, Any], *, project_root: Path | None):
    """Parse pose labels from a session-pose payload."""
    normalized = dict(payload)
    if project_root is not None:
        _rebase_video_paths(normalized, project_root)
    return labels_from_json_payload(normalized)


def predictions_payload_from_labels(labels: PredictionLabelsView) -> dict[str, Any]:
    """Serialize predicted instances exposed by a Labels-like object."""
    items = coerce_predictions_from_labels(labels)
    skeletons = getattr(labels, "skeletons", ())
    skeleton = getattr(labels, "skeleton", None)
    keypoint_count = len(skeleton.keypoints) if skeletons and skeleton is not None else 0
    return _prediction_payload(items, keypoint_count=keypoint_count)


def _prediction_payload(
    items: Sequence[PredictionAppendItem], *, keypoint_count: int
) -> dict[str, Any]:
    if not items:
        return _empty_predictions(keypoint_count=keypoint_count)
    row_count = len(items)
    max_instances = max(len(item.instances or ()) for item in items)
    heatmap_shape = _prediction_heatmap_shape(items)
    arrays = _prediction_arrays(
        row_count=row_count,
        max_instances=max_instances,
        keypoint_count=keypoint_count,
        heatmap_shape=heatmap_shape,
    )
    _fill_prediction_arrays(items, arrays)
    return _prediction_arrays_payload(
        arrays,
        row_count=row_count,
        max_instances=max_instances,
        keypoint_count=keypoint_count,
        heatmap_shape=heatmap_shape,
    )


def _prediction_arrays(
    *,
    row_count: int,
    max_instances: int,
    keypoint_count: int,
    heatmap_shape: tuple[int, int] | None,
) -> dict[str, np.ndarray | None]:
    shape = (row_count, max_instances, keypoint_count)
    arrays: dict[str, np.ndarray | None] = {
        "video_index": np.zeros(row_count, dtype=np.int32),
        "frame_index": np.zeros(row_count, dtype=np.int32),
        "num_instances": np.zeros(row_count, dtype=np.int32),
        "keypoints": np.full((*shape, 3), np.nan, dtype=np.float32),
        "keypoint_score": np.zeros(shape, dtype=np.float32),
        "instance_score": np.zeros((row_count, max_instances), dtype=np.float32),
        "track_id": np.full((row_count, max_instances), -1, dtype=np.int32),
        "deleted": np.zeros((row_count, max_instances), dtype=np.uint8),
        "heatmaps": None,
    }
    if heatmap_shape is not None:
        arrays["heatmaps"] = np.zeros(
            (row_count, keypoint_count, *heatmap_shape), dtype=np.float16
        )
    return arrays


def _fill_prediction_arrays(
    items: Sequence[PredictionAppendItem], arrays: dict[str, np.ndarray | None]
) -> None:
    for row_index, item in enumerate(items):
        _required_array(arrays, "video_index")[row_index] = int(item.video_index)
        _required_array(arrays, "frame_index")[row_index] = int(item.frame_index)
        _required_array(arrays, "num_instances")[row_index] = len(item.instances or ())
        heatmaps = arrays["heatmaps"]
        if heatmaps is not None and item.heatmaps is not None:
            heatmaps[row_index] = np.asarray(item.heatmaps, dtype=np.float16)
        for instance_index, instance in enumerate(item.instances or ()):
            _fill_prediction_instance(arrays, row_index, instance_index, instance)


def _fill_prediction_instance(
    arrays: dict[str, np.ndarray | None], row: int, column: int, instance: Any
) -> None:
    points = instance.point_records(copy=False)
    scores = (
        np.asarray(points["score"], dtype=np.float32)
        if "score" in points.dtype.names
        else np.asarray(points["visible"], dtype=np.float32)
    )
    keypoints = _required_array(arrays, "keypoints")
    keypoints[row, column, :, 0] = np.asarray(points["x"], dtype=np.float32)
    keypoints[row, column, :, 1] = np.asarray(points["y"], dtype=np.float32)
    keypoints[row, column, :, 2] = scores
    _required_array(arrays, "keypoint_score")[row, column] = scores
    _required_array(arrays, "instance_score")[row, column] = float(
        getattr(instance, "score", 0.0)
    )
    track = getattr(instance, "track", None)
    if track is not None:
        _required_array(arrays, "track_id")[row, column] = int(track.id)


def _prediction_arrays_payload(
    arrays: dict[str, np.ndarray | None],
    *,
    row_count: int,
    max_instances: int,
    keypoint_count: int,
    heatmap_shape: tuple[int, int] | None,
) -> dict[str, Any]:
    materialized = {
        key: None if value is None else value.tolist() for key, value in arrays.items()
    }
    return {
        "frames": {
            key: materialized.pop(key)
            for key in ("video_index", "frame_index", "num_instances")
        },
        "data": materialized,
        "attrs": {"committed_length": row_count},
        "metadata": {
            "num_frames": row_count,
            "max_instances": max_instances,
            "num_keypoints": keypoint_count,
            "heatmap_height": heatmap_shape[0] if heatmap_shape else 0,
            "heatmap_width": heatmap_shape[1] if heatmap_shape else 0,
        },
    }


def _empty_predictions(*, keypoint_count: int) -> dict[str, Any]:
    return {
        "frames": {"video_index": [], "frame_index": [], "num_instances": []},
        "data": {
            "keypoints": [], "keypoint_score": [], "instance_score": [],
            "track_id": [], "deleted": [], "heatmaps": None,
        },
        "attrs": {"committed_length": 0},
        "metadata": {
            "num_frames": 0, "max_instances": 0, "num_keypoints": keypoint_count,
            "heatmap_height": 0, "heatmap_width": 0,
        },
    }


def _prediction_heatmap_shape(
    items: Sequence[PredictionAppendItem],
) -> tuple[int, int] | None:
    expected: tuple[int, int] | None = None
    for item in items:
        if item.heatmaps is None:
            continue
        heatmaps = np.asarray(item.heatmaps)
        if heatmaps.ndim != 3:
            raise ValueError("Prediction heatmaps must be a (K, H, W) array")
        shape = (int(heatmaps.shape[1]), int(heatmaps.shape[2]))
        if expected is not None and shape != expected:
            raise ValueError("Prediction heatmaps must share a consistent spatial shape")
        expected = shape
    return expected


def _required_array(
    arrays: Mapping[str, np.ndarray | None], name: str
) -> np.ndarray:
    value = arrays[name]
    if value is None:
        raise AssertionError(f"Required prediction array {name!r} is absent")
    return value


def _relativize_video_paths(payload: dict[str, Any], project_root: Path) -> None:
    videos = cast("dict[str, Any]", payload["videos"])
    root = project_root.resolve()
    videos["filenames"] = [_relative_path(value, root) for value in videos.get("filenames", [])]
    videos["resolved_paths"] = list(videos["filenames"])
    videos["image_filenames"] = [
        [_relative_path(frame, root) for frame in sequence]
        for sequence in videos.get("image_filenames", [])
    ]


def _relative_path(value: object, project_root: Path) -> str:
    raw = str(value).strip()
    if not raw:
        return ""
    return Path(raw).resolve().relative_to(project_root).as_posix()


def _rebase_video_paths(payload: dict[str, Any], project_root: Path) -> None:
    videos = cast("dict[str, Any]", payload["videos"])
    root = project_root.resolve()
    filenames = [_absolute_path(value, root) for value in videos.get("filenames", [])]
    videos["filenames"] = filenames
    videos["resolved_paths"] = filenames
    videos["resolved_exists"] = [bool(value) and Path(value).exists() for value in filenames]
    videos["image_filenames"] = [
        [_absolute_path(frame, root) for frame in sequence]
        for sequence in videos.get("image_filenames", [])
    ]


def _absolute_path(value: object, project_root: Path) -> str:
    raw = str(value).strip()
    if not raw:
        return ""
    path = Path(raw)
    return (path if path.is_absolute() else project_root / path).resolve().as_posix()


__all__ = [
    "pose_labels_from_payload",
    "pose_labels_payload",
    "predictions_payload_from_labels",
]
