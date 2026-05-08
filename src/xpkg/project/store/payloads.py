"""State ↔ public payload converters and prediction-instance helpers."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np

from xpkg.project.state_io import (
    PROJECT_COMMIT_ID_KEY,
    normalize_predictions_payload,
)

if TYPE_CHECKING:
    pass


def _public_payload_from_state_labels(
    state_payload: dict[str, Any],
    *,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    labels_payload = _public_labels_payload_from_state(state_payload, metadata=metadata)
    keypoint_count = int(labels_payload["metadata"]["num_keypoints"])
    return {
        "labels": labels_payload,
        "predictions": _public_predictions_payload_from_state(
            state_payload,
            keypoint_count=keypoint_count,
        ),
        "metrics": {
            "schema_version": 0,
            "tables": {},
            "metadata": {},
        },
        "suggestions": _public_suggestions_payload_from_state(state_payload),
        "runs": _empty_runs_payload(),
        "metadata": dict(metadata),
        "provenance": deepcopy(state_payload.get("provenance") or {"events": []}),
        "session": _public_session_payload_from_state(state_payload, metadata=metadata),
        "segmentation": _public_segmentation_payload_from_state(state_payload),
    }


def _public_labels_payload_from_state(
    state_payload: dict[str, Any],
    *,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    labels_state = _strip_prediction_instances_from_state_payload(state_payload)
    frames_info = labels_state.get("frames")
    data_info = labels_state.get("data")
    if not isinstance(frames_info, dict) or not isinstance(data_info, dict):
        raise TypeError("Project state labels payload must contain frames/data mappings")

    skeleton_info = deepcopy(labels_state.get("skeleton") or {})
    skeleton_names = list(skeleton_info.get("names") or [])
    video_index = np.asarray(frames_info.get("video_index", []), dtype=np.int32)
    frame_index = np.asarray(frames_info.get("frame_index", []), dtype=np.int32)
    num_instances = np.asarray(frames_info.get("num_instances", []), dtype=np.int32)
    raw_keypoints = data_info.get("keypoints")
    raw_keypoints_array = np.asarray(
        [] if raw_keypoints is None else raw_keypoints,
        dtype=np.float32,
    )

    frame_count = max(
        video_index.shape[0],
        frame_index.shape[0],
        num_instances.shape[0],
        raw_keypoints_array.shape[0] if raw_keypoints_array.ndim >= 1 else 0,
    )
    keypoint_count = (
        int(raw_keypoints_array.shape[2])
        if raw_keypoints_array.ndim >= 3
        else len(skeleton_names)
    )
    max_instances = (
        int(raw_keypoints_array.shape[1])
        if raw_keypoints_array.ndim >= 2
        else int(num_instances.max())
        if num_instances.size
        else 1
    )
    max_instances = max(max_instances, 1)

    keypoints = _coerce_array(
        raw_keypoints,
        dtype=np.float32,
        default=np.full(
            (frame_count, max_instances, keypoint_count, 3),
            np.nan,
            dtype=np.float32,
        ),
    )
    flags = _coerce_array(
        data_info.get("flags"),
        dtype=np.uint8,
        default=np.zeros((frame_count, max_instances, keypoint_count), dtype=np.uint8),
    )
    track_ids = _label_track_ids_array(
        data_info,
        row_count=frame_count,
        max_instances=max_instances,
    )
    visibility = _public_visibility_array(
        data_info.get("visibility"),
        keypoints=keypoints,
        frame_count=frame_count,
        max_instances=max_instances,
        keypoint_count=keypoint_count,
    )

    return {
        "frames": {
            "video_index": _coerce_array(
                frames_info.get("video_index"),
                dtype=np.int32,
                default=np.zeros((frame_count,), dtype=np.int32),
            ),
            "frame_index": _coerce_array(
                frames_info.get("frame_index"),
                dtype=np.int32,
                default=np.arange(frame_count, dtype=np.int32),
            ),
            "num_instances": _coerce_array(
                frames_info.get("num_instances"),
                dtype=np.int32,
                default=np.zeros((frame_count,), dtype=np.int32),
            ),
        },
        "data": {
            "keypoints": keypoints,
            "flags": flags,
            "track_id": track_ids,
            "visibility": visibility,
        },
        "metadata": {
            "num_frames": int(frame_count),
            "max_instances": int(max_instances),
            "num_keypoints": int(keypoint_count),
            "preferences": dict(metadata.get("preferences") or {}),
        },
        "skeleton": skeleton_info,
        "videos": _public_videos_payload_from_state(labels_state),
        "tracks": deepcopy(labels_state.get("tracks") or {}),
        "provenance": deepcopy(labels_state.get("provenance") or {"events": []}),
    }


def _coerce_array(value: Any, *, dtype: Any, default: np.ndarray) -> np.ndarray:
    if value is None:
        return default
    array = np.asarray(value, dtype=dtype)
    if array.size == 0 and default.shape:
        return default
    return array


def _public_visibility_array(
    value: Any,
    *,
    keypoints: np.ndarray,
    frame_count: int,
    max_instances: int,
    keypoint_count: int,
) -> np.ndarray:
    default = np.zeros((frame_count, max_instances, keypoint_count), dtype=np.uint8)
    if value is not None:
        return _coerce_array(value, dtype=np.uint8, default=default)
    if keypoints.ndim < 4 or keypoints.shape[-1] < 2:
        return default
    return (np.isfinite(keypoints[..., 0]) & np.isfinite(keypoints[..., 1])).astype(np.uint8)


def _public_videos_payload_from_state(state_payload: dict[str, Any]) -> dict[str, Any]:
    videos_info = deepcopy(state_payload.get("videos") or {})
    shapes = np.asarray(videos_info.get("shapes", []), dtype=np.int32)
    video_count = max(
        len(videos_info.get("filenames") or []),
        len(videos_info.get("image_filenames") or []),
        len(videos_info.get("resolved_paths") or []),
        shapes.shape[0] if shapes.ndim >= 2 else 0,
    )
    videos_info.setdefault("base_dir", "")
    videos_info["filenames"] = list(videos_info.get("filenames") or [""] * video_count)
    videos_info["image_filenames"] = list(videos_info.get("image_filenames") or [[]] * video_count)
    videos_info["backends"] = list(videos_info.get("backends") or ["opencv"] * video_count)
    videos_info["sha256"] = list(videos_info.get("sha256") or [""] * video_count)
    videos_info["video_ids"] = list(videos_info.get("video_ids") or [""] * video_count)
    videos_info["video_labels"] = list(videos_info.get("video_labels") or [""] * video_count)
    videos_info["shapes"] = (
        shapes if shapes.size else np.zeros((video_count, 4), dtype=np.int32)
    )
    return videos_info


def _public_predictions_payload_from_state(
    state_payload: dict[str, Any],
    *,
    keypoint_count: int,
) -> dict[str, Any]:
    predictions = normalize_predictions_payload(
        _predictions_payload_from_state_payload(state_payload)
    )
    metadata = dict(predictions.get("metadata") or {})
    attrs = dict(predictions.get("attrs") or {})
    frame_count = int(attrs.get("committed_length") or metadata.get("num_frames") or 0)
    max_instances = max(int(metadata.get("max_instances") or 0), 1)
    prediction_keypoints = int(metadata.get("num_keypoints") or keypoint_count)

    raw_frames_info = predictions.get("frames")
    frames_info: dict[str, Any] = raw_frames_info if isinstance(raw_frames_info, dict) else {}
    raw_data_info = predictions.get("data")
    data_info: dict[str, Any] = raw_data_info if isinstance(raw_data_info, dict) else {}
    data = {
        "keypoints": _coerce_array(
            data_info.get("keypoints"),
            dtype=np.float32,
            default=np.zeros(
                (frame_count, max_instances, prediction_keypoints, 3),
                dtype=np.float32,
            ),
        ),
        "keypoint_score": _coerce_array(
            data_info.get("keypoint_score"),
            dtype=np.float32,
            default=np.zeros(
                (frame_count, max_instances, prediction_keypoints),
                dtype=np.float32,
            ),
        ),
        "instance_score": _coerce_array(
            data_info.get("instance_score"),
            dtype=np.float32,
            default=np.zeros((frame_count, max_instances), dtype=np.float32),
        ),
        "track_id": _coerce_array(
            data_info.get("track_id"),
            dtype=np.int32,
            default=np.full((frame_count, max_instances), -1, dtype=np.int32),
        ),
        "deleted": _coerce_array(
            data_info.get("deleted"),
            dtype=np.uint8,
            default=np.zeros((frame_count, max_instances), dtype=np.uint8),
        ),
        "heatmaps": None,
    }
    heatmaps = data_info.get("heatmaps")
    if heatmaps is not None:
        data["heatmaps"] = np.asarray(heatmaps, dtype=np.float16)

    return {
        "frames": {
            "video_index": _coerce_array(
                frames_info.get("video_index") if isinstance(frames_info, dict) else None,
                dtype=np.int32,
                default=np.zeros((frame_count,), dtype=np.int32),
            ),
            "frame_index": _coerce_array(
                frames_info.get("frame_index") if isinstance(frames_info, dict) else None,
                dtype=np.int32,
                default=np.arange(frame_count, dtype=np.int32),
            ),
            "num_instances": _coerce_array(
                frames_info.get("num_instances") if isinstance(frames_info, dict) else None,
                dtype=np.int32,
                default=np.zeros((frame_count,), dtype=np.int32),
            ),
        },
        "data": data,
        "attrs": {"committed_length": frame_count},
        "metadata": {
            "num_frames": frame_count,
            "max_instances": max_instances,
            "num_keypoints": prediction_keypoints,
            "heatmap_height": int(metadata.get("heatmap_height") or 0),
            "heatmap_width": int(metadata.get("heatmap_width") or 0),
        },
    }


def _public_suggestions_payload_from_state(state_payload: dict[str, Any]) -> dict[str, Any]:
    suggestions = state_payload.get("suggestions")
    if not isinstance(suggestions, dict):
        return {
            "video_indices": np.zeros((0,), dtype=np.int32),
            "frame_indices": np.zeros((0,), dtype=np.int32),
            "scores": None,
        }
    return {
        "video_indices": _coerce_array(
            suggestions.get("video_indices"),
            dtype=np.int32,
            default=np.zeros((0,), dtype=np.int32),
        ),
        "frame_indices": _coerce_array(
            suggestions.get("frame_indices"),
            dtype=np.int32,
            default=np.zeros((0,), dtype=np.int32),
        ),
        "scores": (
            None
            if suggestions.get("scores") is None
            else np.asarray(suggestions.get("scores"), dtype=np.float32)
        ),
    }


def _empty_runs_payload() -> dict[str, Any]:
    return {
        "table": {
            "run_id": np.zeros((0,), dtype=np.int32),
            "created_ns": np.zeros((0,), dtype=np.int64),
            "config_json": np.zeros((0,), dtype=object),
        },
        "entries": [],
    }


def _public_session_payload_from_state(
    state_payload: dict[str, Any],
    *,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    session_json = metadata.get("session_json")
    if isinstance(session_json, dict):
        return deepcopy(session_json)
    session_payload = state_payload.get("session")
    return deepcopy(session_payload) if isinstance(session_payload, dict) else {}


def _public_segmentation_payload_from_state(state_payload: dict[str, Any]) -> dict[str, Any]:
    segmentation = state_payload.get("segmentation")
    if isinstance(segmentation, dict):
        return deepcopy(segmentation)
    return {"masks": [], "rois": [], "schema_version": ""}


def _state_metadata_from_state_payload(
    state_payload: dict[str, Any],
) -> dict[str, Any] | None:
    metadata = state_payload.get("metadata")
    if not isinstance(metadata, dict):
        return None
    normalized = dict(metadata)
    normalized.pop(PROJECT_COMMIT_ID_KEY, None)
    return normalized


def _predictions_payload_from_state_payload(
    state_payload: dict[str, Any],
) -> dict[str, Any] | None:
    raw_predictions = state_payload.get("predictions")
    if not isinstance(raw_predictions, dict):
        return None
    return normalize_predictions_payload(raw_predictions)


def _predictions_committed_length(predictions: dict[str, Any] | None) -> int:
    normalized = normalize_predictions_payload(predictions)
    attrs = normalized.get("attrs")
    if not isinstance(attrs, dict):
        return 0
    return int(attrs.get("committed_length", 0) or 0)


def _label_track_ids_array(
    data_info: dict[str, Any],
    *,
    row_count: int,
    max_instances: int,
) -> np.ndarray:
    raw_track_ids = data_info.get("track_ids")
    if raw_track_ids is None:
        raw_track_ids = data_info.get("track_id")
    if raw_track_ids is None:
        return np.full((row_count, max_instances), -1, dtype=np.int32)
    return np.asarray(raw_track_ids, dtype=np.int32)


def _prediction_instance_signatures(
    predictions_payload: dict[str, Any] | None,
) -> dict[tuple[int, int], list[tuple[np.ndarray, int]]]:
    if _predictions_committed_length(predictions_payload) <= 0:
        return {}
    assert predictions_payload is not None

    frames = predictions_payload.get("frames")
    data = predictions_payload.get("data")
    if not isinstance(frames, dict) or not isinstance(data, dict):
        raise TypeError("Predictions payload must contain frames/data mappings")

    frame_index = np.asarray(frames.get("frame_index", []), dtype=np.int32)
    video_index = np.asarray(frames.get("video_index", []), dtype=np.int32)
    num_instances = np.asarray(frames.get("num_instances", []), dtype=np.int32)
    keypoints = np.asarray(data.get("keypoints", []), dtype=np.float32)
    max_instances = keypoints.shape[1] if keypoints.ndim >= 2 else 0
    track_ids = _label_track_ids_array(
        data,
        row_count=keypoints.shape[0] if keypoints.ndim >= 1 else 0,
        max_instances=max_instances,
    )

    row_count = min(
        frame_index.shape[0],
        video_index.shape[0],
        num_instances.shape[0],
        keypoints.shape[0] if keypoints.ndim >= 1 else 0,
    )
    grouped: dict[tuple[int, int], list[tuple[np.ndarray, int]]] = {}
    for row in range(row_count):
        key = (int(video_index[row]), int(frame_index[row]))
        row_signatures = grouped.setdefault(key, [])
        inst_count = min(int(num_instances[row]), max_instances)
        for inst_idx in range(inst_count):
            row_signatures.append(
                (
                    np.asarray(keypoints[row, inst_idx], dtype=np.float32),
                    int(track_ids[row, inst_idx]),
                )
            )
    return grouped


def _strip_prediction_instances_from_state_payload(
    state_payload: dict[str, Any],
) -> dict[str, Any]:
    stripped_payload = deepcopy(state_payload)
    predictions_payload = _predictions_payload_from_state_payload(state_payload)
    prediction_map = _prediction_instance_signatures(predictions_payload)
    if not prediction_map:
        return stripped_payload

    frames_info = stripped_payload.get("frames")
    data_info = stripped_payload.get("data")
    if not isinstance(frames_info, dict) or not isinstance(data_info, dict):
        raise TypeError("Project state labels payload must contain frames/data mappings")

    frame_index = np.asarray(frames_info.get("frame_index", []), dtype=np.int32)
    video_index = np.asarray(frames_info.get("video_index", []), dtype=np.int32)
    num_instances = np.asarray(frames_info.get("num_instances", []), dtype=np.int32)
    keypoints = np.asarray(data_info.get("keypoints", []), dtype=np.float32)
    flags = np.asarray(data_info.get("flags", []), dtype=np.uint8)
    max_instances = keypoints.shape[1] if keypoints.ndim >= 2 else 1
    keypoint_count = keypoints.shape[2] if keypoints.ndim >= 3 else 0
    track_ids = _label_track_ids_array(
        data_info,
        row_count=keypoints.shape[0] if keypoints.ndim >= 1 else 0,
        max_instances=max_instances,
    )

    row_count = min(
        frame_index.shape[0],
        video_index.shape[0],
        num_instances.shape[0],
        keypoints.shape[0] if keypoints.ndim >= 1 else 0,
    )
    kept_rows: list[tuple[int, list[int]]] = []
    for row in range(row_count):
        key = (int(video_index[row]), int(frame_index[row]))
        predicted_signatures = prediction_map.get(key)
        inst_count = min(int(num_instances[row]), max_instances)
        if not predicted_signatures:
            kept_rows.append((row, list(range(inst_count))))
            continue

        matched_indices: set[int] = set()
        for predicted_points, predicted_track_id in predicted_signatures:
            matched_idx: int | None = None
            for inst_idx in range(inst_count):
                if inst_idx in matched_indices:
                    continue
                if int(track_ids[row, inst_idx]) != predicted_track_id:
                    continue
                if np.allclose(keypoints[row, inst_idx], predicted_points, equal_nan=True):
                    matched_idx = inst_idx
                    break
            if matched_idx is None:
                raise ValueError(
                    "Project state labels/predictions are inconsistent for "
                    f"video_index={key[0]}, frame_index={key[1]}"
                )
            matched_indices.add(matched_idx)

        kept_indices = [
            inst_idx for inst_idx in range(inst_count) if inst_idx not in matched_indices
        ]
        if kept_indices:
            kept_rows.append((row, kept_indices))

    kept_row_count = len(kept_rows)
    kept_max_instances = max(
        (len(indices) for _, indices in kept_rows),
        default=max(max_instances, 1),
    )
    kept_video_index = np.zeros((kept_row_count,), dtype=np.int32)
    kept_frame_index = np.zeros((kept_row_count,), dtype=np.int32)
    kept_num_instances = np.zeros((kept_row_count,), dtype=np.int32)
    kept_keypoints = np.full(
        (kept_row_count, kept_max_instances, keypoint_count, 3),
        np.nan,
        dtype=np.float32,
    )
    kept_flags = np.zeros((kept_row_count, kept_max_instances, keypoint_count), dtype=np.uint8)
    kept_track_ids = np.full((kept_row_count, kept_max_instances), -1, dtype=np.int32)

    for out_row, (src_row, kept_indices) in enumerate(kept_rows):
        kept_video_index[out_row] = int(video_index[src_row])
        kept_frame_index[out_row] = int(frame_index[src_row])
        kept_num_instances[out_row] = len(kept_indices)
        for out_inst, src_inst in enumerate(kept_indices):
            kept_keypoints[out_row, out_inst] = keypoints[src_row, src_inst]
            kept_flags[out_row, out_inst] = flags[src_row, src_inst]
            kept_track_ids[out_row, out_inst] = track_ids[src_row, src_inst]

    frames_info["video_index"] = kept_video_index.tolist()
    frames_info["frame_index"] = kept_frame_index.tolist()
    frames_info["num_instances"] = kept_num_instances.tolist()
    data_info["keypoints"] = kept_keypoints.tolist()
    data_info["flags"] = kept_flags.tolist()
    data_info["track_ids"] = kept_track_ids.tolist()
    return stripped_payload


