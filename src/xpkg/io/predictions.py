"""Prediction payload extraction and coercion for project state documents."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Protocol, TypedDict

import numpy as np

from xpkg.payloads import mapping_or_empty


class BBox(TypedDict):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


type KeypointXY = tuple[float, float] | list[float]
type KeypointXYC = tuple[float, float, float] | list[float]


class CoordinateOnlyInstancePrediction(TypedDict, total=False):
    keypoints: list[KeypointXY]
    score: float
    bbox: BBox | None
    track_id: int | None
    keypoint_score: list[float]
    keypoint_scores: list[float]
    deleted: bool


class CoordinateConfidenceInstancePrediction(TypedDict, total=False):
    keypoints: list[KeypointXYC]
    score: float
    bbox: BBox | None
    track_id: int | None
    keypoint_score: list[float]
    keypoint_scores: list[float]
    deleted: bool


type InstancePrediction = CoordinateOnlyInstancePrediction | CoordinateConfidenceInstancePrediction

import numpy as np

_POINT_DTYPE = np.dtype(
    [
        ("x", "f4"),
        ("y", "f4"),
        ("visible", "?"),
        ("complete", "?"),
        ("score", "f4"),
        ("flags", "u1"),
    ]
)


class _SerializerTrack:
    __slots__ = ("id",)

    def __init__(self, track_id: int) -> None:
        self.id = int(track_id)


class PredictionLabelsView(Protocol):
    @property
    def videos(self) -> Sequence[object]: ...

    @property
    def labeled_frames(self) -> Sequence[Any]: ...


class SerializerPredictedInstance:
    """Small predicted-instance DTO for xpkg state serialization.

    The object intentionally exposes the same attributes the project state writer
    consumes from model-level predicted instances while remaining cheap to build
    from external inference code.
    """

    __slots__ = (
        "_points",
        "deleted",
        "keypoint_scores",
        "keypoints",
        "score",
        "track",
        "track_id",
    )

    def __init__(
        self,
        keypoints: Sequence[Sequence[float]] | np.ndarray,
        *,
        score: float = 0.0,
        track_id: int | None = None,
        deleted: bool = False,
        keypoint_scores: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        point_array = np.asarray(keypoints, dtype=np.float32)
        if point_array.ndim != 2 or point_array.shape[1] < 2:
            raise ValueError("Predicted instance keypoints must be a two-dimensional array")

        self.keypoints = point_array[:, :2].copy()
        self.score = float(score)
        self.track_id = None if track_id is None else int(track_id)
        self.track = None if self.track_id is None else _SerializerTrack(self.track_id)
        self.deleted = bool(deleted)

        if keypoint_scores is None:
            if point_array.shape[1] >= 3:
                scores = point_array[:, 2].astype(np.float32, copy=True)
            else:
                scores = np.ones((point_array.shape[0],), dtype=np.float32)
        else:
            scores = np.asarray(keypoint_scores, dtype=np.float32)
            if scores.ndim != 1 or scores.shape[0] != point_array.shape[0]:
                raise ValueError("keypoint_scores must have one value per keypoint")

        self.keypoint_scores = scores.copy()
        self._points = np.zeros((point_array.shape[0],), dtype=_POINT_DTYPE)
        self._points["x"] = self.keypoints[:, 0]
        self._points["y"] = self.keypoints[:, 1]
        self._points["visible"] = np.isfinite(self.keypoints).all(axis=1)
        self._points["complete"] = self._points["visible"]
        self._points["score"] = self.keypoint_scores
        self._points["flags"] = 0

    def get_points_array(self, *, copy: bool = False, full: bool = True) -> np.ndarray:
        """Return points in the structured array shape expected by xpkg writers."""

        if full:
            return self._points.copy() if copy else self._points

        compact = self._points[["x", "y", "visible", "complete"]]
        return compact.copy() if copy else compact


class PredictionAppendItem:
    """Linear prediction row used by project state-document serialization."""

    __slots__ = ("frame_index", "heatmaps", "instances", "video_index")

    def __init__(
        self,
        video_index: int,
        frame_index: int,
        instances: Sequence[Any],
        *,
        heatmaps: Any | None = None,
    ) -> None:
        self.video_index = int(video_index)
        self.frame_index = int(frame_index)
        self.instances = list(instances)
        self.heatmaps = heatmaps


def coerce_predictions_from_labels(labels: PredictionLabelsView) -> list[PredictionAppendItem]:
    """Extract predicted instances from a Labels-like object."""

    items: list[PredictionAppendItem] = []
    video_to_index = {video: idx for idx, video in enumerate(labels.videos)}

    for labeled_frame in labels.labeled_frames:
        predicted_instances = list(labeled_frame.predicted_instances)
        if not predicted_instances:
            continue
        if labeled_frame.video not in video_to_index:
            continue
        items.append(
            PredictionAppendItem(
                video_index=video_to_index[labeled_frame.video],
                frame_index=labeled_frame.frame_idx,
                instances=predicted_instances,
                heatmaps=labeled_frame.heatmaps,
            )
        )
    return items


def _prediction_sections(
    payload: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    attrs = mapping_or_empty(payload.get("attrs"), label="predictions.attrs")
    frames = mapping_or_empty(payload.get("frames"), label="predictions.frames")
    data = mapping_or_empty(payload.get("data"), label="predictions.data")
    return attrs, frames, data


def _frame_series(frames: Mapping[str, object], *, key: str, dtype: Any) -> np.ndarray:
    return np.asarray(frames.get(key, []), dtype=dtype)


def _optional_series(data: Mapping[str, object], *, key: str, dtype: Any) -> np.ndarray | None:
    raw = data.get(key)
    return None if raw is None else np.asarray(raw, dtype=dtype)


def _normalize_num_instances(num_instances: np.ndarray, *, frame_count: int) -> np.ndarray:
    if num_instances.shape[0] >= frame_count:
        return num_instances[:frame_count]
    padded = np.zeros((frame_count,), dtype=np.int32)
    padded[: num_instances.shape[0]] = num_instances.astype(np.int32, copy=False)
    return padded


def _validate_heatmaps(data: Mapping[str, object], *, frame_count: int) -> None:
    raw = data.get("heatmaps")
    if raw is None:
        return
    heatmaps = np.asarray(raw)
    if heatmaps.size == 0:
        return
    if heatmaps.ndim == 3 and frame_count == 1:
        return
    if heatmaps.ndim != 4:
        raise ValueError(f"Heatmaps must have shape (N, K, H, W), got ndim={heatmaps.ndim}")
    if heatmaps.shape[0] < frame_count:
        raise ValueError(
            f"Heatmaps contain {heatmaps.shape[0]} frames but payload declares {frame_count}"
        )


def _base_instance_prediction(
    *,
    keypoints: list[KeypointXY],
    score: float,
    track_id: int | None,
) -> CoordinateOnlyInstancePrediction:
    return {
        "keypoints": keypoints,
        "score": float(score),
        "track_id": track_id,
        "bbox": None,
    }


def prediction_frame_payloads_from_payload(
    payload: dict[str, Any],
) -> dict[int, list[InstancePrediction]]:
    """Convert serialized xpkg prediction arrays into frame-indexed instances."""
    _, frames, data = _prediction_sections(payload)

    frame_index = _frame_series(frames, key="frame_index", dtype=np.int64)
    num_instances = _frame_series(frames, key="num_instances", dtype=np.int32)

    keypoints = data.get("keypoints")
    if keypoints is None:
        return {}
    keypoints_arr = np.asarray(keypoints, dtype=np.float32)
    if keypoints_arr.size == 0:
        return {}

    kp_scores_arr = _optional_series(data, key="keypoint_score", dtype=np.float32)
    inst_scores_arr = _optional_series(data, key="instance_score", dtype=np.float32)
    track_ids_arr = _optional_series(data, key="track_id", dtype=np.int32)
    deleted_arr = _optional_series(data, key="deleted", dtype=np.uint8)

    n_frames = int(frame_index.shape[0])
    if n_frames == 0 or keypoints_arr.shape[0] == 0:
        return {}
    _validate_heatmaps(data, frame_count=n_frames)

    if keypoints_arr.ndim == 3:
        keypoints_arr = keypoints_arr[:, np.newaxis, :, :]
    elif keypoints_arr.ndim == 2:
        keypoints_arr = keypoints_arr[np.newaxis, np.newaxis, :, :]
    elif keypoints_arr.ndim < 2:
        return {}

    max_instances = keypoints_arr.shape[1]
    num_keypoints = int(keypoints_arr.shape[2])

    num_instances = _normalize_num_instances(num_instances, frame_count=n_frames)

    def _iter_keypoints(points: np.ndarray) -> Iterable[tuple[float, float, float]]:
        for row in points:
            if row.shape[-1] >= 3:
                yield (float(row[0]), float(row[1]), float(row[2]))
            elif row.shape[-1] == 2:
                yield (float(row[0]), float(row[1]), 0.0)
            elif row.shape[-1] == 1:
                yield (float(row[0]), 0.0, 0.0)
            else:
                yield (0.0, 0.0, 0.0)

    def _extract_scores(arr: np.ndarray | None, frame_idx: int, inst_idx: int):
        if arr is None:
            return None
        if arr.ndim >= 3:
            if frame_idx >= arr.shape[0] or inst_idx >= arr.shape[1]:
                return None
            return arr[frame_idx, inst_idx]
        if arr.ndim == 2:
            if frame_idx >= arr.shape[0]:
                return None
            if arr.shape[1] == max_instances and inst_idx < arr.shape[1]:
                return arr[frame_idx, inst_idx]
            if inst_idx == 0:
                return arr[frame_idx]
            return None
        if arr.ndim == 1:
            if arr.shape[0] == n_frames and inst_idx == 0:
                return arr[frame_idx]
            if arr.shape[0] == max_instances and frame_idx == 0:
                return arr[inst_idx]
        return None

    def _has_payload(frame_idx: int, inst_idx: int) -> bool:
        if inst_idx >= max_instances:
            return False
        pts = keypoints_arr[frame_idx, inst_idx]
        if pts.size and not np.isnan(pts).all() and not np.allclose(pts, 0.0):
            return True
        scores = _extract_scores(kp_scores_arr, frame_idx, inst_idx)
        if scores is not None:
            scores_arr = np.asarray(scores, dtype=np.float32)
            if (
                scores_arr.size
                and not np.isnan(scores_arr).all()
                and not np.allclose(scores_arr, 0.0)
            ):
                return True
        score_val = _extract_scores(inst_scores_arr, frame_idx, inst_idx)
        if score_val is not None:
            score_arr = np.asarray(score_val, dtype=np.float32).ravel()
            if score_arr.size and not np.isnan(score_arr).all() and not np.allclose(score_arr, 0.0):
                return True
        track_val = _extract_scores(track_ids_arr, frame_idx, inst_idx)
        if track_val is not None:
            tid_val = int(np.asarray(track_val).ravel()[0])
            if tid_val not in (-1, 0):
                return True
        if deleted_arr is not None:
            if deleted_arr.ndim >= 2:
                if bool(deleted_arr[frame_idx, inst_idx]):
                    return True
            else:
                deleted_val = _extract_scores(deleted_arr, frame_idx, inst_idx)
                if deleted_val is not None and bool(int(deleted_val)):
                    return True
        return False

    frames_map: dict[int, list[InstancePrediction]] = {}
    for idx in range(n_frames):
        frame_id = int(frame_index[idx])
        declared = int(num_instances[idx]) if num_instances.size else 0
        declared = max(0, min(declared, max_instances))
        detected_indices = (
            [inst_idx for inst_idx in range(max_instances) if _has_payload(idx, inst_idx)]
            if max_instances > 0
            else []
        )

        inst_indices: list[int] = []
        if declared:
            inst_indices.extend(range(declared))
        for inst_idx in detected_indices:
            if inst_idx not in inst_indices:
                inst_indices.append(inst_idx)

        inst_indices = [i for i in inst_indices if i < max_instances]

        instances: list[InstancePrediction] = []
        if not inst_indices:
            frames_map[frame_id] = instances
            continue

        for inst_idx in inst_indices:
            pts = keypoints_arr[idx, inst_idx]
            pts_list = [
                [float(x), float(y), float(c)]
                for x, y, c in _iter_keypoints(pts[:num_keypoints] if num_keypoints else pts)
            ]

            score = 0.0
            score_val = _extract_scores(inst_scores_arr, idx, inst_idx)
            if score_val is not None:
                score_arr = np.asarray(score_val, dtype=np.float32).ravel()
                if score_arr.size:
                    score = float(score_arr[0])

            track_id = -1
            track_val = _extract_scores(track_ids_arr, idx, inst_idx)
            if track_val is not None:
                track_arr = np.asarray(track_val).ravel()
                if track_arr.size:
                    track_id = int(track_arr[0])

            inst_payload: CoordinateOnlyInstancePrediction = _base_instance_prediction(
                keypoints=pts_list,
                score=float(score),
                track_id=None if track_id < 0 else int(track_id),
            )

            kp_scores = _extract_scores(kp_scores_arr, idx, inst_idx)
            if kp_scores is not None:
                kp_scores_arr_local = np.asarray(kp_scores, dtype=np.float32)
                inst_payload["keypoint_score"] = [
                    float(v) for v in kp_scores_arr_local[: len(pts_list)]
                ]

            deleted_val = None
            if deleted_arr is not None:
                if deleted_arr.ndim >= 2:
                    deleted_val = deleted_arr[idx, inst_idx]
                else:
                    deleted_val = _extract_scores(deleted_arr, idx, inst_idx)
            if deleted_val is not None:
                inst_payload["deleted"] = bool(int(deleted_val))

            instances.append(inst_payload)

        frames_map[frame_id] = instances

    return frames_map


__all__ = [
    "BBox",
    "CoordinateConfidenceInstancePrediction",
    "CoordinateOnlyInstancePrediction",
    "InstancePrediction",
    "KeypointXY",
    "KeypointXYC",
    "PredictionAppendItem",
    "PredictionLabelsView",
    "SerializerPredictedInstance",
    "coerce_predictions_from_labels",
    "prediction_frame_payloads_from_payload",
]
