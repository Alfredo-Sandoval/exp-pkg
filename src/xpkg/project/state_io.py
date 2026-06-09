"""Project-native JSON state serialization and cache-digest routines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from xpkg._core.hashing import sha256_file
from xpkg._core.json_utils import load_json_dict, parse_json_dict, write_json
from xpkg._core.logging_utils import get_logger
from xpkg._core.path_registry import ensure_dir, resolve_path
from xpkg.io.labels.json_format import (
    XPKG_LABELS_JSON_FORMAT,
    XPKG_LABELS_JSON_VERSION,
    labels_to_json_payload,
    read_labels_json_payload,
)
from xpkg.io.predictions import (
    PredictionAppendItem,
    PredictionLabelsView,
    coerce_predictions_from_labels,
)
from xpkg.project.layout import (
    CURRENT_STATE_FILENAME,
    project_media_root,
    project_state_root,
)

if TYPE_CHECKING:
    from xpkg.model import Labels


PROJECT_COMMIT_ID_KEY = "xpkg_commit_id"
PROJECT_CACHE_DIGEST_KEY = "sha256"

_logger = get_logger(__name__)


def current_project_state_path(path: str | Path) -> Path:
    """Return the canonical project state path."""
    return project_state_root(path) / CURRENT_STATE_FILENAME


def project_state_cache_digest_path(path: str | Path) -> Path:
    """Return the private digest sidecar path for a project state cache."""

    target = resolve_path(path)
    return target.with_name(f"{target.name}.sha256")


def write_project_state_cache_digest(path: str | Path, *, commit_id: str) -> Path:
    """Write the digest sidecar used to trust a current state cache quickly."""

    target = resolve_path(path)
    digest_path = project_state_cache_digest_path(target)
    write_json(
        digest_path,
        {
            PROJECT_COMMIT_ID_KEY: str(commit_id),
            PROJECT_CACHE_DIGEST_KEY: sha256_file(target),
        },
        indent=2,
        sort_keys=False,
        ensure_ascii=True,
    )
    return digest_path


def project_state_cache_digest_matches(path: str | Path, *, commit_id: str) -> bool:
    """Return whether the state cache matches its private digest sidecar."""

    target = resolve_path(path)
    digest_path = project_state_cache_digest_path(target)
    if not digest_path.is_file():
        return False
    try:
        payload = load_json_dict(digest_path)
    except FileNotFoundError:
        return False
    except (OSError, ValueError, TypeError) as exc:
        _logger.warning("Unreadable state cache digest %s: %s", digest_path, exc)
        return False
    if str(payload.get(PROJECT_COMMIT_ID_KEY, "")).strip() != str(commit_id):
        return False
    expected = str(payload.get(PROJECT_CACHE_DIGEST_KEY, "")).strip()
    return bool(expected) and expected == sha256_file(target)


def _materialize_json_value(value: Any) -> Any:
    materialize = getattr(value, "materialize", None)
    if callable(materialize):
        value = materialize()

    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, bytes | bytearray | np.bytes_):
        return bytes(value).decode("utf-8")
    if isinstance(value, Mapping):
        return {str(key): _materialize_json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_materialize_json_value(item) for item in value]
    return value


def _relative_project_path(
    path: str | Path,
    project_root: Path,
    *,
    resolved_project_root: Path | None = None,
) -> str:
    resolved = resolve_path(path)
    root = (
        resolved_project_root
        if resolved_project_root is not None
        else project_root.resolve()
    )
    return resolved.relative_to(root).as_posix()


def _relativize_video_payload_paths(videos_payload: dict[str, Any], project_root: Path) -> None:
    resolved_project_root = project_root.resolve()
    filenames = list(videos_payload.get("filenames") or [])
    image_sequences = list(videos_payload.get("image_filenames") or [])

    relative_filenames: list[str] = []
    for raw_filename in filenames:
        filename = str(raw_filename).strip()
        if not filename:
            relative_filenames.append("")
            continue
        relative_filenames.append(
            _relative_project_path(
                filename,
                project_root,
                resolved_project_root=resolved_project_root,
            )
        )

    relative_sequences: list[list[str]] = []
    for sequence_entry in image_sequences:
        if not isinstance(sequence_entry, Sequence) or isinstance(
            sequence_entry, str | bytes | bytearray
        ):
            relative_sequences.append([])
            continue
        relative_sequences.append(
            [
                _relative_project_path(
                    frame_path,
                    project_root,
                    resolved_project_root=resolved_project_root,
                )
                for frame_path in sequence_entry
            ]
        )

    videos_payload["filenames"] = relative_filenames
    videos_payload["resolved_paths"] = list(relative_filenames)
    videos_payload["image_filenames"] = relative_sequences


def _relative_media_path_from_name(raw_path: str, *, project_root: Path) -> str:
    target_name = Path(raw_path).name
    if not target_name:
        return ""
    media_root = project_media_root(project_root).resolve()
    matches = sorted(candidate.resolve() for candidate in media_root.rglob(target_name))
    if len(matches) != 1:
        return ""
    return matches[0].relative_to(project_root.resolve()).as_posix()


def _rewrite_session_state_paths(session_state: dict[str, Any], *, project_root: Path) -> None:
    active_video_path = session_state.get("active_video_path")
    if not isinstance(active_video_path, str):
        return
    session_state["active_video_path"] = _relative_media_path_from_name(
        active_video_path,
        project_root=project_root,
    )


def rewrite_project_metadata_paths(
    metadata: dict[str, Any] | None,
    *,
    project_root: Path,
) -> dict[str, Any]:
    normalized = _materialize_json_value(metadata or {})
    if not isinstance(normalized, dict):
        raise TypeError("Project metadata must normalize to a mapping")

    session_state = normalized.get("session_json")
    if isinstance(session_state, str):
        session_state = parse_json_dict(session_state)
        normalized["session_json"] = session_state
    if isinstance(session_state, dict):
        _rewrite_session_state_paths(session_state, project_root=project_root)

    training_state = normalized.get("training_state_json")
    if isinstance(training_state, str):
        training_state = parse_json_dict(training_state)
        normalized["training_state_json"] = training_state

    return normalized


def _empty_predictions_payload() -> dict[str, Any]:
    return {
        "frames": {
            "video_index": [],
            "frame_index": [],
            "num_instances": [],
        },
        "data": {
            "keypoints": [],
            "keypoint_score": [],
            "instance_score": [],
            "track_id": [],
            "deleted": [],
            "heatmaps": None,
        },
        "attrs": {"committed_length": 0},
        "metadata": {
            "num_frames": 0,
            "max_instances": 0,
            "num_keypoints": 0,
            "heatmap_height": 0,
            "heatmap_width": 0,
        },
    }


def normalize_predictions_payload(predictions: dict[str, Any] | None) -> dict[str, Any]:
    if predictions is None:
        return _empty_predictions_payload()
    normalized = _materialize_json_value(predictions)
    if not isinstance(normalized, dict):
        raise TypeError("Predictions payload must normalize to a mapping")
    return normalized


def _normalized_state_metadata(
    metadata: dict[str, Any] | None,
    *,
    project_root: Path,
    commit_id: str | None,
) -> dict[str, Any]:
    normalized_metadata = rewrite_project_metadata_paths(
        metadata,
        project_root=project_root,
    )
    if commit_id is None:
        normalized_metadata.pop(PROJECT_COMMIT_ID_KEY, None)
    else:
        normalized_metadata[PROJECT_COMMIT_ID_KEY] = str(commit_id)
    return normalized_metadata


def build_project_state_payload(
    *,
    labels: Labels,
    project_root: Path,
    metadata: dict[str, Any] | None = None,
    predictions: dict[str, Any] | None = None,
    commit_id: str | None = None,
) -> dict[str, Any]:
    normalized_metadata = _normalized_state_metadata(
        metadata,
        project_root=project_root,
        commit_id=commit_id,
    )
    document = labels_to_json_payload(labels, metadata=normalized_metadata)
    payload = document["payload"]
    _relativize_video_payload_paths(payload["videos"], project_root)
    session_state = payload.get("session")
    if isinstance(session_state, dict):
        _rewrite_session_state_paths(session_state, project_root=project_root)
    payload["predictions"] = normalize_predictions_payload(predictions)
    return payload


def _state_payload_document(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "format": XPKG_LABELS_JSON_FORMAT,
        "version": XPKG_LABELS_JSON_VERSION,
        "payload": payload,
    }


def write_project_state_payload(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    commit_id: str | None = None,
) -> Path:
    target = resolve_path(path)
    ensure_dir(target.parent)
    normalized_payload = _materialize_json_value(payload)
    if not isinstance(normalized_payload, dict):
        raise TypeError("Project state payload must normalize to a mapping")

    metadata = normalized_payload.get("metadata")
    normalized_metadata = dict(metadata) if isinstance(metadata, dict) else {}
    if commit_id is None:
        normalized_metadata.pop(PROJECT_COMMIT_ID_KEY, None)
    else:
        normalized_metadata[PROJECT_COMMIT_ID_KEY] = str(commit_id)
    normalized_payload["metadata"] = normalized_metadata
    write_json(
        target,
        _state_payload_document(normalized_payload),
        indent=None,
        sort_keys=False,
        ensure_ascii=True,
        compact=True,
    )
    digest_path = project_state_cache_digest_path(target)
    if commit_id is None:
        digest_path.unlink(missing_ok=True)
    else:
        write_project_state_cache_digest(target, commit_id=str(commit_id))
    return target


def _prediction_heatmap_shape(
    prediction_items: Sequence[PredictionAppendItem],
) -> tuple[int, int] | None:
    expected_shape: tuple[int, int] | None = None
    for item in prediction_items:
        if item.heatmaps is None:
            continue
        heatmaps = np.asarray(item.heatmaps)
        if heatmaps.ndim != 3:
            raise ValueError("Prediction heatmaps must be a (K, H, W) array")
        candidate_shape = (int(heatmaps.shape[1]), int(heatmaps.shape[2]))
        if expected_shape is None:
            expected_shape = candidate_shape
            continue
        if candidate_shape != expected_shape:
            raise ValueError("Prediction heatmaps must share a consistent spatial shape")
    return expected_shape


def _prediction_payload_from_items(
    prediction_items: Sequence[PredictionAppendItem],
    *,
    keypoint_count: int,
) -> dict[str, Any]:
    if not prediction_items:
        payload = _empty_predictions_payload()
        payload["metadata"]["num_keypoints"] = int(keypoint_count)
        return payload

    row_count = len(prediction_items)
    max_instances = max(len(item.instances or []) for item in prediction_items)
    heatmap_shape = _prediction_heatmap_shape(prediction_items)

    video_index = np.zeros((row_count,), dtype=np.int32)
    frame_index = np.zeros((row_count,), dtype=np.int32)
    num_instances = np.zeros((row_count,), dtype=np.int32)
    keypoints = np.full((row_count, max_instances, keypoint_count, 3), np.nan, dtype=np.float32)
    keypoint_score = np.zeros((row_count, max_instances, keypoint_count), dtype=np.float32)
    instance_score = np.zeros((row_count, max_instances), dtype=np.float32)
    track_id = np.full((row_count, max_instances), -1, dtype=np.int32)
    deleted = np.zeros((row_count, max_instances), dtype=np.uint8)
    heatmaps: np.ndarray | None = None
    if heatmap_shape is not None:
        heatmaps = np.zeros(
            (row_count, keypoint_count, heatmap_shape[0], heatmap_shape[1]),
            dtype=np.float16,
        )

    for row_idx, item in enumerate(prediction_items):
        video_index[row_idx] = int(item.video_index)
        frame_index[row_idx] = int(item.frame_index)
        num_instances[row_idx] = int(len(item.instances or []))
        if heatmaps is not None and item.heatmaps is not None:
            heatmaps[row_idx] = np.asarray(item.heatmaps, dtype=np.float16)

        for inst_idx, instance in enumerate(item.instances or []):
            points = instance.point_records(copy=False)
            keypoints[row_idx, inst_idx, :, 0] = np.asarray(points["x"], dtype=np.float32)
            keypoints[row_idx, inst_idx, :, 1] = np.asarray(points["y"], dtype=np.float32)
            if "score" in points.dtype.names:
                scores = np.asarray(points["score"], dtype=np.float32)
            else:
                scores = np.asarray(points["visible"], dtype=np.float32)
            keypoints[row_idx, inst_idx, :, 2] = scores
            keypoint_score[row_idx, inst_idx] = scores
            instance_score[row_idx, inst_idx] = float(getattr(instance, "score", 0.0))
            if instance.track is not None:
                track_id[row_idx, inst_idx] = int(instance.track.id)

    return normalize_predictions_payload(
        {
            "frames": {
                "video_index": video_index,
                "frame_index": frame_index,
                "num_instances": num_instances,
            },
            "data": {
                "keypoints": keypoints,
                "keypoint_score": keypoint_score,
                "instance_score": instance_score,
                "track_id": track_id,
                "deleted": deleted,
                "heatmaps": heatmaps,
            },
            "attrs": {"committed_length": row_count},
            "metadata": {
                "num_frames": row_count,
                "max_instances": max_instances,
                "num_keypoints": keypoint_count,
                "heatmap_height": heatmap_shape[0] if heatmap_shape is not None else 0,
                "heatmap_width": heatmap_shape[1] if heatmap_shape is not None else 0,
            },
        }
    )


def predictions_payload_from_labels(labels: PredictionLabelsView) -> dict[str, Any]:
    prediction_items = coerce_predictions_from_labels(labels)
    skeletons = getattr(labels, "skeletons", ())
    skeleton = getattr(labels, "skeleton", None)
    keypoint_count = len(skeleton.keypoints) if skeletons and skeleton is not None else 0
    return _prediction_payload_from_items(prediction_items, keypoint_count=keypoint_count)


def write_project_state(
    path: str | Path,
    *,
    labels: Labels,
    project_root: Path,
    metadata: dict[str, Any] | None = None,
    predictions: dict[str, Any] | None = None,
    commit_id: str | None = None,
) -> Path:
    payload = build_project_state_payload(
        labels=labels,
        project_root=project_root,
        metadata=metadata,
        predictions=predictions,
        commit_id=commit_id,
    )
    return write_project_state_payload(path, payload, commit_id=commit_id)


def read_project_state(path: str | Path) -> dict[str, Any]:
    return read_labels_json_payload(path)


def read_project_state_payload(path: str | Path) -> dict[str, Any]:
    """Alias for the canonical project state payload reader."""
    return read_project_state(path)


def state_commit_id(payload: Mapping[str, Any]) -> str | None:
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    raw_commit_id = metadata.get(PROJECT_COMMIT_ID_KEY)
    if not isinstance(raw_commit_id, str):
        return None
    commit_id = raw_commit_id.strip()
    return commit_id or None


__all__ = [
    "build_project_state_payload",
    "current_project_state_path",
    "normalize_predictions_payload",
    "predictions_payload_from_labels",
    "read_project_state",
    "read_project_state_payload",
    "rewrite_project_metadata_paths",
    "state_commit_id",
    "PROJECT_COMMIT_ID_KEY",
    "project_state_cache_digest_matches",
    "project_state_cache_digest_path",
    "write_project_state_cache_digest",
    "write_project_state",
    "write_project_state_payload",
]
