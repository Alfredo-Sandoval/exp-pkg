"""Project save/import boundary backed by the private ``.xpkg`` store."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from xpkg._core.hashing import sha256_file
from xpkg._core.json_utils import write_json
from xpkg._core.path_registry import ensure_dir, resolve_path, slugify_path_component
from xpkg.adapters.vicon import (
    read_vicon_json_payload,
    vicon_recording_from_json_payload,
    vicon_recording_to_json_payload,
)
from xpkg.project.artifact import validate_project
from xpkg.project.layout import (
    CURRENT_STATE_FILENAME,
    EXPORTS_DIRNAME,
    MEDIA_DIRNAME,
    PROJECT_DESCRIPTOR_FILENAME,
    STORE_DIRNAME,
    STORE_STATE_DIRNAME,
    ProjectDescriptor,
    _candidate_project_root,
    _now_utc_iso,
    load_project_descriptor,
    project_current_state_path,
    project_exports_root,
    project_media_root,
    project_store_root,
    resolve_project_root,
    write_project_descriptor,
)
from xpkg.project.state import (
    project_state_commit_id_from_document,
    project_state_kind,
    project_state_payload_from_document,
    read_project_state_document,
)
from xpkg.project.state_io import (
    PROJECT_COMMIT_ID_KEY,
    normalize_predictions_payload,
    predictions_payload_from_labels,
    project_state_cache_digest_matches,
    read_project_state,
    rewrite_project_metadata_paths,
    write_project_state,
    write_project_state_cache_digest,
    write_project_state_payload,
)

if TYPE_CHECKING:
    from xpkg.model import Labels, ViconRecording


def load_project_payload(path: str | Path) -> dict[str, Any]:
    """Return the current committed project payload on the public project surface."""
    root = resolve_project_root(path)
    if root is None:
        raise FileNotFoundError(f"Not an xpkg project: {path}")
    state = _current_project_state_payload(root)
    if state is None:
        return {"metadata": {}}
    payload, source_kind = state
    metadata = _state_metadata_from_state_payload(payload) or {}
    if source_kind == "state_labels":
        public_payload = _public_payload_from_state_labels(payload, metadata=metadata)
        rebase_project_payload_videos(public_payload, root)
        return public_payload
    if source_kind == "state_vicon":
        return {
            "recording": deepcopy(payload),
            "metadata": metadata,
        }
    raise ValueError(f"Unsupported project state source: {source_kind!r}")


def current_project_state_path(path: str | Path) -> Path:
    """Return the project state cache path under `.xpkg/state/current.json`."""

    return project_current_state_path(path)


def current_project_commit_id(path: str | Path) -> str | None:
    return _project_store(path).current_commit_id()


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


@dataclass(slots=True)
class ProjectStore:
    """Private project store boundary for `.xpkg/`."""

    project_root: Path

    @property
    def store_root(self) -> Path:
        return project_store_root(self.project_root)

    @property
    def staging_root(self) -> Path:
        return self.store_root / "project"

    def has_durable_store(self) -> bool:
        return (self.store_root / "superblock.a.json").exists() or (
            self.store_root / "superblock.b.json"
        ).exists()

    def has_current_state(self) -> bool:
        if not self.has_durable_store():
            return False
        return self.open().has_current_root("state")

    def current_commit_id(self) -> str | None:
        if not self.has_durable_store():
            return None
        return self.open().load_current_commit().commit_id

    def open(self):
        from xpkg.project.durable_store import ProjectDurableStore

        return ProjectDurableStore.open(self.store_root)

    def current_state_path(self) -> Path:
        if not self.has_durable_store():
            raise FileNotFoundError(f"Project has no durable state root: {self.store_root}")
        return self.open().current_root_path("state")

    def commit_state(
        self,
        state_path: str | Path,
        *,
        reason: str,
        created_by: dict[str, Any] | None = None,
    ) -> Path:
        candidate = resolve_path(state_path)
        if not candidate.is_file():
            raise FileNotFoundError(f"Staged project state not found: {candidate}")

        if self.has_durable_store():
            store = self.open()
            store.commit_new_roots({"state": candidate}, reason=reason, created_by=created_by)
            return store.current_root_path("state")

        from xpkg.project.durable_store import ProjectDurableStore

        store = ProjectDurableStore.create_from_roots(
            store_root=self.store_root,
            initial_roots={"state": candidate},
            created_by=created_by,
            reason=reason,
        )
        return store.current_root_path("state")


def _project_store(path: str | Path) -> ProjectStore:
    root = resolve_project_root(path) or _candidate_project_root(path)
    return ProjectStore(project_root=root)


def init_project(
    project: str | Path,
    *,
    title: str | None = None,
    project_id: str | None = None,
    force: bool = False,
) -> ProjectDescriptor:
    root = _candidate_project_root(project)
    if root.exists() and not root.is_dir():
        raise NotADirectoryError(f"Project root is not a directory: {root}")
    if root.exists():
        entries = list(root.iterdir())
        if entries and not force:
            raise FileExistsError(f"Project directory is not empty: {root}")
    else:
        ensure_dir(root)

    ensure_dir(project_store_root(root))
    ensure_dir(project_media_root(root))
    ensure_dir(project_exports_root(root))

    descriptor = ProjectDescriptor.new(
        title=(title or root.name or "exp-pkg Project").strip(),
        project_id=project_id,
    )
    write_project_descriptor(root, descriptor)
    return descriptor


def _clone_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    return dict(metadata or {})


def _stage_project_parent(project_root: Path) -> Path:
    return ensure_dir(_project_store(project_root).staging_root)


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


def _normalized_project_metadata(
    metadata: dict[str, Any] | None,
    *,
    project_root: Path,
    commit_id: str | None,
) -> dict[str, Any]:
    normalized = rewrite_project_metadata_paths(
        metadata,
        project_root=project_root,
    )
    if commit_id is None:
        normalized.pop(PROJECT_COMMIT_ID_KEY, None)
    else:
        normalized[PROJECT_COMMIT_ID_KEY] = str(commit_id)
    return normalized


def _project_state_cache_matches_committed_head(
    project_root: Path,
    state_path: Path,
) -> bool:
    from xpkg.project.durable_store import ProjectDurableStore

    store = ProjectDurableStore.open(project_store_root(project_root))
    commit = store.load_current_commit()
    if not commit.has_root("state"):
        return False
    if project_state_cache_digest_matches(state_path, commit_id=commit.commit_id):
        return True
    root_entry = commit.root_entry("state")
    committed_state_path = store.paths.object_path(root_entry.object_id, ext=root_entry.ext)
    if not committed_state_path.exists():
        return False
    if f"obj_{sha256_file(state_path)}" == root_entry.object_id:
        write_project_state_cache_digest(state_path, commit_id=commit.commit_id)
        return True

    cache_document = read_project_state_document(state_path)
    committed_document = read_project_state_document(committed_state_path)
    if _project_state_documents_match_cache(cache_document, committed_document):
        write_project_state_cache_digest(state_path, commit_id=commit.commit_id)
        return True
    return False


def _metadata_matches_without_commit_id(
    cache_metadata: Mapping[str, Any],
    committed_metadata: Mapping[str, Any],
) -> bool:
    cache_keys = set(cache_metadata)
    committed_keys = set(committed_metadata)
    cache_keys.discard(PROJECT_COMMIT_ID_KEY)
    committed_keys.discard(PROJECT_COMMIT_ID_KEY)
    if cache_keys != committed_keys:
        return False
    return all(cache_metadata[key] == committed_metadata[key] for key in cache_keys)


def _project_payload_matches_cache(
    cache_payload: Mapping[str, Any],
    committed_payload: Mapping[str, Any],
) -> bool:
    cache_keys = set(cache_payload)
    committed_keys = set(committed_payload)
    if cache_keys != committed_keys:
        return False

    for key in cache_keys:
        if key == "metadata":
            continue
        if cache_payload[key] != committed_payload[key]:
            return False

    cache_metadata = cache_payload.get("metadata")
    committed_metadata = committed_payload.get("metadata")
    if isinstance(cache_metadata, Mapping) and isinstance(committed_metadata, Mapping):
        return _metadata_matches_without_commit_id(cache_metadata, committed_metadata)
    return cache_metadata == committed_metadata


def _project_state_documents_match_cache(
    cache_document: Mapping[str, Any],
    committed_document: Mapping[str, Any],
) -> bool:
    cache_payload = project_state_payload_from_document(cache_document)
    committed_payload = project_state_payload_from_document(committed_document)

    if cache_payload is not cache_document or committed_payload is not committed_document:
        cache_keys = set(cache_document)
        committed_keys = set(committed_document)
        if cache_keys != committed_keys:
            return False
        for key in cache_keys:
            if key == "payload":
                continue
            if cache_document[key] != committed_document[key]:
                return False
        return _project_payload_matches_cache(cache_payload, committed_payload)

    return _project_payload_matches_cache(cache_document, committed_document)


def ensure_current_project_state_cache(project_root: Path) -> Path | None:
    """Materialize the current project state cache from the committed head when needed."""

    state_path = project_current_state_path(project_root)
    if state_path.exists():
        current_head = current_project_commit_id(project_root)
        if current_head is None:
            return state_path
        state_document = read_project_state_document(state_path)
        state_head = project_state_commit_id_from_document(state_document)
        if (
            state_head == current_head
            and _project_state_cache_matches_committed_head(
                project_root,
                state_path,
            )
        ):
            return state_path

    state = _current_project_state_payload(project_root)
    if state is None:
        return None
    return rebuild_project_state_cache(project_root)


def _current_project_state_payload(
    project_root: Path,
) -> tuple[dict[str, Any], str] | None:
    store = _project_store(project_root)
    if store.has_durable_store():
        mounted = store.open()
        if mounted.has_current_root("state"):
            state_path = mounted.current_root_path("state")
            state_kind = project_state_kind(state_path)
            if state_kind == "labels":
                return read_project_state(state_path), "state_labels"
            return read_vicon_json_payload(state_path), "state_vicon"

    return None


def _project_state_components(
    state_payload: dict[str, Any],
    *,
    source_kind: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if source_kind in {"state_labels", "state_vicon"}:
        metadata = _state_metadata_from_state_payload(state_payload)
    else:
        raise ValueError(f"Unsupported project state source: {source_kind!r}")
    predictions = (
        _predictions_payload_from_state_payload(state_payload)
        if source_kind == "state_labels"
        else None
    )
    return metadata, predictions


def _write_project_state(
    project_root: Path,
    *,
    labels: Labels,
    metadata: dict[str, Any] | None = None,
    predictions: dict[str, Any] | None = None,
    commit_id: str | None = None,
) -> Path:
    _manage_labels_media(labels, project_root)
    return write_project_state(
        project_current_state_path(project_root),
        labels=labels,
        project_root=project_root,
        metadata=metadata,
        predictions=predictions,
        commit_id=commit_id,
    )


def _write_vicon_project_state(
    project_root: Path,
    *,
    recording: ViconRecording,
    metadata: dict[str, Any] | None = None,
    commit_id: str | None = None,
) -> Path:
    document = vicon_recording_to_json_payload(
        recording,
        metadata=_normalized_project_metadata(
            metadata,
            project_root=project_root,
            commit_id=commit_id,
        ),
        source_root=project_root,
    )
    target = project_current_state_path(project_root)
    ensure_dir(target.parent)
    write_json(
        target,
        document,
        indent=None,
        sort_keys=False,
        ensure_ascii=True,
        compact=True,
    )
    if commit_id is not None:
        write_project_state_cache_digest(target, commit_id=str(commit_id))
    return target


def _commit_labels_to_project(
    project_root: Path,
    *,
    labels: Labels,
    metadata: dict[str, Any] | None = None,
    predictions: dict[str, Any] | None = None,
    reason: str,
) -> Path:
    _manage_labels_media(labels, project_root)
    normalized_metadata = rewrite_project_metadata_paths(
        metadata,
        project_root=project_root,
    )
    normalized_predictions = (
        predictions_payload_from_labels(labels)
        if predictions is None
        else normalize_predictions_payload(predictions)
    )

    stage_parent = _stage_project_parent(project_root)
    with tempfile.TemporaryDirectory(
        prefix=".project_commit_",
        dir=str(stage_parent),
    ) as tmp_dir:
        staged_state = Path(tmp_dir) / CURRENT_STATE_FILENAME
        write_project_state(
            staged_state,
            labels=labels,
            project_root=project_root,
            metadata=normalized_metadata,
            predictions=normalized_predictions,
        )
        store = _project_store(project_root)
        store.commit_state(staged_state, reason=reason)
        commit_id = store.current_commit_id()

    return _write_project_state(
        project_root,
        labels=labels,
        metadata=normalized_metadata,
        predictions=normalized_predictions,
        commit_id=commit_id,
    )


def _commit_state_metadata_to_project(
    project_root: Path,
    *,
    state_payload: dict[str, Any],
    metadata: dict[str, Any] | None,
    reason: str,
) -> Path:
    normalized_metadata = rewrite_project_metadata_paths(
        metadata,
        project_root=project_root,
    )
    existing_metadata = state_payload.get("metadata")
    if "preferences" not in normalized_metadata:
        preferences = (
            existing_metadata.get("preferences")
            if isinstance(existing_metadata, dict)
            else None
        )
        normalized_metadata["preferences"] = dict(preferences or {})
    staged_payload = deepcopy(state_payload)
    staged_payload["metadata"] = _normalized_project_metadata(
        normalized_metadata,
        project_root=project_root,
        commit_id=None,
    )

    stage_parent = _stage_project_parent(project_root)
    with tempfile.TemporaryDirectory(
        prefix=".project_commit_",
        dir=str(stage_parent),
    ) as tmp_dir:
        staged_state = Path(tmp_dir) / CURRENT_STATE_FILENAME
        write_project_state_payload(staged_state, staged_payload)
        store = _project_store(project_root)
        store.commit_state(staged_state, reason=reason)
        commit_id = store.current_commit_id()

    current_payload = deepcopy(state_payload)
    current_payload["metadata"] = normalized_metadata
    return write_project_state_payload(
        project_current_state_path(project_root),
        current_payload,
        commit_id=commit_id,
    )


def _commit_vicon_to_project(
    project_root: Path,
    *,
    recording: ViconRecording,
    metadata: dict[str, Any] | None = None,
    reason: str,
) -> Path:
    normalized_metadata = rewrite_project_metadata_paths(
        metadata,
        project_root=project_root,
    )

    stage_parent = _stage_project_parent(project_root)
    with tempfile.TemporaryDirectory(
        prefix=".project_commit_",
        dir=str(stage_parent),
    ) as tmp_dir:
        staged_state = Path(tmp_dir) / CURRENT_STATE_FILENAME
        document = vicon_recording_to_json_payload(
            recording,
            metadata=_normalized_project_metadata(
                normalized_metadata,
                project_root=project_root,
                commit_id=None,
            ),
            source_root=project_root,
        )
        write_json(
            staged_state,
            document,
            indent=None,
            sort_keys=False,
            ensure_ascii=True,
            compact=True,
        )
        store = _project_store(project_root)
        store.commit_state(staged_state, reason=reason)
        commit_id = store.current_commit_id()

    return _write_vicon_project_state(
        project_root,
        recording=recording,
        metadata=normalized_metadata,
        commit_id=commit_id,
    )


_POSE_PREDICTION_TOOLS: dict[str, tuple[str, str]] = {
    "dlc_csv_import": ("DeepLabCut", "csv"),
    "dlc_h5_import": ("DeepLabCut", "h5"),
    "dlc_project_import": ("DeepLabCut", "project"),
    "lightning_pose_csv_import": ("Lightning Pose", "csv"),
    "mediapipe_pose_landmarks_json_import": ("MediaPipe", "json"),
    "mmpose_topdown_json_import": ("MMPose", "json"),
    "sleap_h5_import": ("SLEAP", "h5"),
    "sleap_pkg_import": ("SLEAP", "pkg.slp"),
}


def _merge_metadata_dict(base: dict[str, Any], extra: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = dict(base)
    if extra is None:
        return merged
    for key, value in extra.items():
        key_text = str(key)
        if (
            isinstance(value, Mapping)
            and isinstance(merged.get(key_text), dict)
        ):
            merged[key_text] = _merge_metadata_dict(merged[key_text], value)
            continue
        merged[key_text] = value
    return merged


def _config_snapshot_payload(path: str | Path) -> dict[str, str]:
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Prediction provenance config snapshot not found: {resolved}")
    return {
        "path": resolved.as_posix(),
        "sha256": sha256_file(resolved),
    }


def _source_inputs_from_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    inputs: dict[str, Any] = {}
    for key, value in metadata.items():
        if key in {"source", "project_name"}:
            continue
        if key.startswith("source"):
            inputs[key] = deepcopy(value)
    return inputs


def _normalized_prediction_provenance(
    metadata: Mapping[str, Any],
    extra: Mapping[str, Any] | None,
) -> dict[str, Any]:
    source_label = str(metadata.get("source") or "unknown_pose_import")
    tool_name, source_format = _POSE_PREDICTION_TOOLS.get(
        source_label,
        ("unknown", "unknown"),
    )
    provenance: dict[str, Any] = {
        "schema_version": 1,
        "importer": source_label,
        "tool": {"name": tool_name},
        "source_format": source_format,
        "inputs": _source_inputs_from_metadata(metadata),
    }
    if extra is None:
        return provenance

    extra_payload = dict(extra)
    tool_payload = dict(provenance["tool"])
    model_payload: dict[str, Any] = {}
    metadata_payload: dict[str, Any] = {}

    for flat_key, target_key in (
        ("tool_name", "name"),
        ("tool_version", "version"),
        ("framework_version", "framework_version"),
    ):
        value = extra_payload.pop(flat_key, None)
        if value is not None:
            tool_payload[target_key] = value
    for flat_key, target_key in (
        ("model_name", "name"),
        ("model_version", "version"),
        ("training_set", "training_set"),
        ("training_set_ref", "training_set"),
    ):
        value = extra_payload.pop(flat_key, None)
        if value is not None:
            model_payload[target_key] = value

    config_path = extra_payload.pop("config_snapshot_path", None)
    if config_path is None:
        config_path = extra_payload.pop("config_path", None)
    if config_path is not None:
        extra_payload["config_snapshot"] = _merge_metadata_dict(
            _config_snapshot_payload(config_path),
            extra_payload.get("config_snapshot")
            if isinstance(extra_payload.get("config_snapshot"), Mapping)
            else None,
        )

    nested_tool = extra_payload.pop("tool", None)
    if isinstance(nested_tool, Mapping):
        tool_payload = _merge_metadata_dict(tool_payload, nested_tool)
    nested_model = extra_payload.pop("model", None)
    if isinstance(nested_model, Mapping):
        model_payload = _merge_metadata_dict(model_payload, nested_model)
    nested_metadata = extra_payload.pop("metadata", None)
    if isinstance(nested_metadata, Mapping):
        metadata_payload = _merge_metadata_dict(metadata_payload, nested_metadata)

    provenance["tool"] = tool_payload
    if model_payload:
        provenance["model"] = model_payload
    if metadata_payload:
        provenance["metadata"] = metadata_payload
    return _merge_metadata_dict(provenance, extra_payload)


def _attach_prediction_provenance(
    labels: Labels,
    metadata: dict[str, Any],
    extra: Mapping[str, Any] | None,
) -> dict[str, Any]:
    provenance = _normalized_prediction_provenance(metadata, extra)
    labels.provenance = dict(labels.provenance)
    labels.provenance["pose_prediction"] = provenance
    metadata["prediction_provenance"] = provenance
    return provenance


def _import_project_from_conversion(
    project: str | Path,
    *,
    force: bool,
    reason: str,
    convert: Callable[[Path], Any],
    prediction_provenance: Mapping[str, Any] | None = None,
) -> Path:
    root = _ensure_project_for_import(
        project,
        force=force,
    )
    stage_parent = _stage_project_parent(root)
    with tempfile.TemporaryDirectory(
        prefix=".project_import_",
        dir=str(stage_parent),
    ) as tmp_dir:
        result = convert(Path(tmp_dir))
        result.metadata = dict(result.metadata)
        _attach_prediction_provenance(
            result.labels,
            result.metadata,
            prediction_provenance,
        )
        state_path = _commit_labels_to_project(
            root,
            labels=result.labels,
            metadata=result.metadata,
            reason=reason,
        )
    _touch_descriptor(root)
    return state_path


def _unify_matching_skeletons(base_labels: Labels, new_labels: Labels) -> None:
    mapping: dict[int, Any] = {}
    for skeleton in new_labels.skeletons:
        target = next(
            (existing for existing in base_labels.skeletons if existing.matches(skeleton)),
            None,
        )
        if target is not None:
            mapping[id(skeleton)] = target

    if not mapping:
        return

    for labeled_frame in new_labels.labeled_frames:
        for instance in labeled_frame.instances:
            target = mapping.get(id(instance.skeleton))
            if target is None or instance.skeleton is target:
                continue
            instance.skeleton = target
            instance.realign_points()

    deduped_skeletons: list[Any] = []
    seen_ids: set[int] = set()
    for skeleton in new_labels.skeletons:
        target = mapping.get(id(skeleton), skeleton)
        target_id = id(target)
        if target_id in seen_ids:
            continue
        seen_ids.add(target_id)
        deduped_skeletons.append(target)
    new_labels.skeletons = deduped_skeletons


def _merge_labels_for_import(
    merged_labels: Labels | None,
    new_labels: Labels,
) -> Labels:
    if merged_labels is None:
        return new_labels

    _unify_matching_skeletons(merged_labels, new_labels)
    merged_labels.extend_from(new_labels, unify=False)
    return merged_labels


def rebuild_project_state_cache(project_root: Path) -> Path:
    state = _current_project_state_payload(project_root)
    if state is None:
        raise FileNotFoundError(f"Project has no committed state: {project_root}")

    state_payload, source_kind = state
    commit_id = _project_store(project_root).current_commit_id()
    if source_kind == "state_labels":
        return write_project_state_payload(
            project_current_state_path(project_root),
            state_payload,
            commit_id=commit_id,
        )
    if source_kind == "state_vicon":
        recording = vicon_recording_from_json_payload(state_payload, source_root=project_root)
        return _write_vicon_project_state(
            project_root,
            recording=recording,
            metadata=_state_metadata_from_state_payload(state_payload),
            commit_id=commit_id,
        )

    raise ValueError(f"Unsupported project state source: {source_kind!r}")


def load_project_vicon_recording(project: str | Path) -> ViconRecording:
    """Load the current Vicon recording from project-managed state."""

    root = resolve_project_root(project)
    if root is None:
        raise FileNotFoundError(f"Not an xpkg project: {project}")

    state_path = ensure_current_project_state_cache(root)
    if state_path is None:
        raise FileNotFoundError(f"Project has no committed state: {root}")
    if state_path.suffix.lower() != ".json":
        raise ValueError(
            "Project current state is not a Vicon JSON state; "
            "only project-native state documents can be loaded as Vicon recordings."
        )
    if project_state_kind(state_path) != "vicon":
        raise ValueError(
            "Project current state is not a Vicon recording. "
            "Use Labels.load_file(...) or ProjectService.load_labels()."
        )
    return vicon_recording_from_json_payload(
        read_project_state_document(state_path),
        source_root=root,
    )


def load_project_metadata(project: str | Path) -> dict[str, Any]:
    """Return the current project metadata payload from the managed head."""

    root = resolve_project_root(project)
    if root is None:
        raise FileNotFoundError(f"Not an xpkg project: {project}")

    current_state = _current_project_state_payload(root)
    if current_state is None:
        return {}

    state_payload, source_kind = current_state
    metadata, _predictions = _project_state_components(
        state_payload,
        source_kind=source_kind,
    )
    return _clone_metadata(metadata)


def save_project_metadata(
    project: str | Path,
    metadata: Mapping[str, Any] | None,
    *,
    reason: str = "project.save.metadata",
) -> Path:
    """Commit updated metadata onto the current project head."""

    root = resolve_project_root(project)
    if root is None:
        raise FileNotFoundError(f"Not an xpkg project: {project}")

    descriptor = load_project_descriptor(root)
    descriptor.validate()
    ensure_dir(project_store_root(root))
    ensure_dir(project_media_root(root))
    ensure_dir(project_exports_root(root))

    current_state = _current_project_state_payload(root)
    if current_state is None:
        raise FileNotFoundError(f"Project has no committed state: {root}")

    normalized_metadata = rewrite_project_metadata_paths(
        None if metadata is None else dict(metadata),
        project_root=root,
    )
    state_payload, source_kind = current_state
    if source_kind == "state_labels":
        state_path = _commit_state_metadata_to_project(
            root,
            state_payload=state_payload,
            metadata=normalized_metadata,
            reason=reason,
        )
        _touch_descriptor(root)
        return state_path

    if source_kind == "state_vicon":
        state_path = _commit_vicon_to_project(
            root,
            recording=load_project_vicon_recording(root),
            metadata=normalized_metadata,
            reason=reason,
        )
        _touch_descriptor(root)
        return state_path

    raise ValueError(f"Unsupported project state source: {source_kind!r}")

def _is_within_resolved(path: Path, resolved_parent: Path) -> bool:
    try:
        path.relative_to(resolved_parent)
        return True
    except ValueError:
        return False


def _dedupe_file_target(target: Path) -> Path:
    if not target.exists():
        return target
    stem = target.stem
    suffix = target.suffix
    parent = target.parent
    counter = 1
    while True:
        candidate = parent / f"{stem}-{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def _dedupe_dir_target(target: Path) -> Path:
    if not target.exists():
        return target
    parent = target.parent
    name = target.name or "media"
    counter = 1
    while True:
        candidate = parent / f"{name}-{counter}"
        if not candidate.exists():
            return candidate
        counter += 1


def _copy_vicon_sidecar_into_bundle(sidecar_path: Path | None, bundle_root: Path) -> Path | None:
    if sidecar_path is None:
        return None
    resolved_sidecar = resolve_path(sidecar_path)
    if not resolved_sidecar.is_file():
        raise FileNotFoundError(f"Vicon sidecar not found: {resolved_sidecar}")
    target = bundle_root / resolved_sidecar.name
    shutil.copy2(resolved_sidecar, target)
    return target.resolve()


def _copy_vicon_import_bundle(recording: ViconRecording, project_root: Path) -> ViconRecording:
    imports_root = ensure_dir(project_store_root(project_root) / "imports" / "vicon")
    bundle_root = ensure_dir(
        _dedupe_dir_target(imports_root / slugify_path_component(recording.path))
    )

    source_path = resolve_path(recording.path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Vicon recording not found: {source_path}")

    managed_recording_path = bundle_root / source_path.name
    shutil.copy2(source_path, managed_recording_path)
    managed_xcp_path = _copy_vicon_sidecar_into_bundle(recording.xcp_path, bundle_root)
    managed_vsk_path = _copy_vicon_sidecar_into_bundle(recording.vsk_path, bundle_root)

    return replace(
        recording,
        path=managed_recording_path.resolve(),
        xcp_path=managed_xcp_path,
        vsk_path=managed_vsk_path,
    )


def _copy_file_into_media(source: Path, media_root: Path, copied: dict[Path, Path]) -> Path:
    resolved_source = source.resolve()
    cached = copied.get(resolved_source)
    if cached is not None:
        return cached
    resolved_media_root = media_root.resolve()
    if _is_within_resolved(resolved_source, resolved_media_root):
        copied[resolved_source] = resolved_source
        return resolved_source
    target = _dedupe_file_target(media_root / resolved_source.name)
    ensure_dir(target.parent)
    shutil.copy2(resolved_source, target)
    resolved_target = target.resolve()
    copied[resolved_source] = resolved_target
    return resolved_target


def _copy_sequence_into_media(
    frames: list[Path],
    media_root: Path,
    copied: dict[Path, Path],
) -> tuple[Path, list[Path]]:
    resolved_frames = [frame.resolve() for frame in frames]
    resolved_media_root = media_root.resolve()
    if resolved_frames and all(
        _is_within_resolved(frame, resolved_media_root) for frame in resolved_frames
    ):
        sequence_root = resolved_frames[0].parent
        return sequence_root, resolved_frames

    source_root = resolved_frames[0].parent
    dir_name = source_root.name or resolved_frames[0].stem or "sequence"
    target_dir = _dedupe_dir_target(media_root / dir_name)
    ensure_dir(target_dir)
    copied_frames: list[Path] = []
    for frame in resolved_frames:
        cached = copied.get(frame)
        if cached is not None:
            copied_frames.append(cached)
            continue
        target = target_dir / frame.name
        shutil.copy2(frame, target)
        resolved_target = target.resolve()
        copied[frame] = resolved_target
        copied_frames.append(resolved_target)
    return target_dir.resolve(), copied_frames


def _manage_labels_media(labels: Labels, project_root: Path) -> None:
    media_root = ensure_dir(project_media_root(project_root))
    copied: dict[Path, Path] = {}

    for video in labels.videos:
        raw_image_filenames = getattr(video, "image_filenames", None) or []
        image_filenames = [Path(str(path)) for path in raw_image_filenames if str(path).strip()]
        if image_filenames:
            sequence_root, copied_frames = _copy_sequence_into_media(
                image_filenames,
                media_root,
                copied,
            )
            video._image_filenames = [path.as_posix() for path in copied_frames]
            video.filename = sequence_root.as_posix()
            continue

        filename = getattr(video, "filename", None)
        if not filename:
            continue
        copied_file = _copy_file_into_media(Path(str(filename)), media_root, copied)
        video.filename = copied_file.as_posix()


def rebase_project_payload_videos(payload: dict[str, Any], project_root: Path) -> None:
    project_root = project_root.resolve()

    def _rebase_videos_info(videos_info: dict[str, Any]) -> None:
        raw_filenames = list(videos_info.get("filenames") or [])
        raw_sequences = list(videos_info.get("image_filenames") or [])
        total = max(
            len(raw_filenames),
            len(raw_sequences),
            len(videos_info.get("resolved_paths") or []),
        )
        rebased_resolved_paths: list[str] = []
        rebased_exists: list[bool] = []
        rebased_sequences: list[list[str]] = []

        for idx in range(total):
            raw_name = str(raw_filenames[idx]).strip() if idx < len(raw_filenames) else ""
            if raw_name:
                filename_path = Path(raw_name)
                resolved_path = (
                    filename_path.resolve()
                    if filename_path.is_absolute()
                    else (project_root / filename_path).resolve()
                )
                rebased_resolved_paths.append(resolved_path.as_posix())
                rebased_exists.append(resolved_path.exists())
            else:
                rebased_resolved_paths.append("")
                rebased_exists.append(False)

            sequence_entry = raw_sequences[idx] if idx < len(raw_sequences) else []
            rebased_frames: list[str] = []
            if isinstance(sequence_entry, list):
                for frame in sequence_entry:
                    frame_path = Path(str(frame))
                    resolved_frame = (
                        frame_path.resolve()
                        if frame_path.is_absolute()
                        else (project_root / frame_path).resolve()
                    )
                    rebased_frames.append(resolved_frame.as_posix())
            rebased_sequences.append(rebased_frames)

        videos_info["filenames"] = rebased_resolved_paths
        videos_info["resolved_paths"] = rebased_resolved_paths
        videos_info["resolved_exists"] = rebased_exists
        videos_info["image_filenames"] = rebased_sequences

    labels_payload = payload.get("labels")
    if isinstance(labels_payload, dict):
        labels_videos = labels_payload.get("videos")
        if isinstance(labels_videos, dict):
            _rebase_videos_info(labels_videos)
    else:
        labels_videos = payload.get("videos")
        if isinstance(labels_videos, dict):
            _rebase_videos_info(labels_videos)

    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        metadata_videos = metadata.get("videos")
        if isinstance(metadata_videos, dict):
            _rebase_videos_info(metadata_videos)


def _ensure_project_for_import(
    project: str | Path,
    *,
    title: str | None = None,
    force: bool = False,
) -> Path:
    root = resolve_project_root(project)
    if root is None:
        init_project(
            project,
            title=title,
            force=force,
        )
        return _candidate_project_root(project)
    validate_project(root)
    return root


def _touch_descriptor(root: Path) -> None:
    descriptor = load_project_descriptor(root)
    descriptor.updated_at = _now_utc_iso()
    write_project_descriptor(root, descriptor)


def _import_pose_project(
    project: str | Path,
    *,
    force: bool,
    reason: str,
    convert: Callable[[Path], Any],
    prediction_provenance: Mapping[str, Any] | None,
    provenance: Any,
    default_tool: str,
    source_path: str | Path,
) -> Path:
    state_path = _import_project_from_conversion(
        project,
        force=force,
        reason=reason,
        convert=convert,
        prediction_provenance=prediction_provenance,
    )
    if provenance is not None:
        root = resolve_project_root(project)
        if root is None:
            root = state_path.parents[2]
        _persist_pose_provenance(
            root,
            provenance,
            default_tool=default_tool,
            source_path=source_path,
        )
    return state_path


def _persist_pose_provenance(
    root: Path,
    provenance: Any,
    *,
    default_tool: str,
    source_path: str | Path,
) -> None:
    if provenance is None:
        return
    from xpkg.model import PoseModelProvenance
    from xpkg.project.metadata import save_project_pose_provenance

    if isinstance(provenance, PoseModelProvenance):
        record = provenance
    elif isinstance(provenance, Mapping):
        payload = dict(provenance)
        if not str(payload.get("tool", "")).strip():
            payload["tool"] = default_tool
        record = PoseModelProvenance.from_dict(payload)
    else:
        raise TypeError(
            f"provenance must be PoseModelProvenance or mapping, got {provenance!r}."
        )

    fields: dict[str, Any] = {}
    if record.imported_from is None:
        fields["imported_from"] = resolve_path(source_path).as_posix()
    if record.imported_at is None:
        fields["imported_at"] = _now_utc_iso()
    if fields:
        record = PoseModelProvenance.from_dict({**record.to_dict(), **fields})
    save_project_pose_provenance(root, record)


def _import_vicon_project_recording(
    recording_path: str | Path,
    project: str | Path,
    *,
    force: bool,
    reason: str,
    progress_callback: Any | None,
    reader: Callable[[str | Path], ViconRecording],
    source_name: str,
) -> Path:
    root = _ensure_project_for_import(
        project,
        force=force,
    )
    if progress_callback is not None:
        progress_callback(f"Reading {source_name} recording")
    recording = reader(recording_path)
    if progress_callback is not None:
        progress_callback("Copying Vicon recording bundle into project store")
    managed_recording = _copy_vicon_import_bundle(recording, root)
    metadata = {
        "source": source_name,
        "source_recording": resolve_path(recording_path).as_posix(),
    }
    if recording.xcp_path is not None:
        metadata["source_xcp"] = resolve_path(recording.xcp_path).as_posix()
    if recording.vsk_path is not None:
        metadata["source_vsk"] = resolve_path(recording.vsk_path).as_posix()
    state_path = _commit_vicon_to_project(
        root,
        recording=managed_recording,
        metadata=metadata,
        reason=reason,
    )
    _touch_descriptor(root)
    return state_path


def import_vicon_csv_project(
    csv_path: str | Path,
    project: str | Path,
    *,
    force: bool = False,
    progress_callback: Any | None = None,
) -> Path:
    """Import a Vicon CSV recording into a project."""
    from xpkg.io.readers import read_vicon_csv

    return _import_vicon_project_recording(
        csv_path,
        project,
        force=force,
        reason="project.import.vicon_csv",
        progress_callback=progress_callback,
        reader=read_vicon_csv,
        source_name="vicon_csv_import",
    )


def import_vicon_c3d_project(
    c3d_path: str | Path,
    project: str | Path,
    *,
    force: bool = False,
    progress_callback: Any | None = None,
) -> Path:
    """Import a Vicon C3D recording into a project."""
    from xpkg.io.readers import read_vicon_c3d

    return _import_vicon_project_recording(
        c3d_path,
        project,
        force=force,
        reason="project.import.vicon_c3d",
        progress_callback=progress_callback,
        reader=read_vicon_c3d,
        source_name="vicon_c3d_import",
    )


def import_vicon_project(
    recording_path: str | Path,
    project: str | Path,
    *,
    force: bool = False,
    progress_callback: Any | None = None,
) -> Path:
    """Import a Vicon CSV or C3D recording into a project."""
    from xpkg.io.readers import read_vicon_recording

    return _import_vicon_project_recording(
        recording_path,
        project,
        force=force,
        reason="project.import.vicon",
        progress_callback=progress_callback,
        reader=read_vicon_recording,
        source_name="vicon_import",
    )


def import_dlc_csv_project(
    csv_path: str | Path,
    video_path: str | Path,
    project: str | Path,
    *,
    skeleton_name: str = "imported",
    likelihood_threshold: float = 0.0,
    prediction_provenance: Mapping[str, Any] | None = None,
    provenance: Any = None,
    force: bool = False,
    progress_callback: Any | None = None,
) -> Path:
    """Import a DeepLabCut CSV plus video into a project."""
    from xpkg.io.converters.dlc_import import convert_dlc_csv

    return _import_pose_project(
        project,
        force=force,
        reason="project.import.dlc_csv",
        prediction_provenance=prediction_provenance,
        provenance=provenance,
        default_tool="deeplabcut",
        source_path=csv_path,
        convert=lambda _tmp_dir: convert_dlc_csv(
            csv_path,
            video_path,
            skeleton_name=skeleton_name,
            likelihood_threshold=likelihood_threshold,
            progress_callback=progress_callback,
        ),
    )


def import_lightning_pose_csv_project(
    csv_path: str | Path,
    video_path: str | Path,
    project: str | Path,
    *,
    skeleton_name: str = "imported",
    likelihood_threshold: float = 0.0,
    prediction_provenance: Mapping[str, Any] | None = None,
    provenance: Any = None,
    force: bool = False,
    progress_callback: Any | None = None,
) -> Path:
    """Import a Lightning Pose prediction CSV plus video into a project."""
    from xpkg.io.converters.dlc_import import convert_lightning_pose_csv

    return _import_pose_project(
        project,
        force=force,
        reason="project.import.lightning_pose_csv",
        prediction_provenance=prediction_provenance,
        provenance=provenance,
        default_tool="lightning-pose",
        source_path=csv_path,
        convert=lambda _tmp_dir: convert_lightning_pose_csv(
            csv_path,
            video_path,
            skeleton_name=skeleton_name,
            likelihood_threshold=likelihood_threshold,
            progress_callback=progress_callback,
        ),
    )


def import_dlc_h5_project(
    h5_path: str | Path,
    video_path: str | Path,
    project: str | Path,
    *,
    skeleton_name: str = "imported",
    likelihood_threshold: float = 0.0,
    prediction_provenance: Mapping[str, Any] | None = None,
    provenance: Any = None,
    force: bool = False,
    progress_callback: Any | None = None,
) -> Path:
    """Import a DeepLabCut H5 export plus video into a project."""
    from xpkg.io.converters.dlc_import import convert_dlc_h5

    return _import_pose_project(
        project,
        force=force,
        reason="project.import.dlc_h5",
        prediction_provenance=prediction_provenance,
        provenance=provenance,
        default_tool="deeplabcut",
        source_path=h5_path,
        convert=lambda _tmp_dir: convert_dlc_h5(
            h5_path,
            video_path,
            skeleton_name=skeleton_name,
            likelihood_threshold=likelihood_threshold,
            progress_callback=progress_callback,
        ),
    )


def import_dlc_project_directory(
    project_dir: str | Path,
    project: str | Path,
    *,
    skeleton_name: str | None = None,
    likelihood_threshold: float = 0.0,
    prediction_provenance: Mapping[str, Any] | None = None,
    provenance: Any = None,
    force: bool = False,
    progress_callback: Any | None = None,
) -> Path:
    """Import a supported DeepLabCut project into one project."""
    from xpkg.io.converters.converter_helpers import ConversionResult
    from xpkg.io.converters.dlc_import import (
        _discover_dlc_project_items,
        _stored_project_path,
        convert_dlc_csv,
        convert_dlc_h5,
    )

    resolved_project_dir = resolve_path(project_dir)
    resolved_skeleton_name = skeleton_name or resolved_project_dir.name or "dlc"

    def _convert_project(_tmp_dir: Path) -> ConversionResult:
        project_items, skipped_items = _discover_dlc_project_items(
            resolved_project_dir,
            progress_callback=progress_callback,
        )
        if not project_items:
            raise ValueError(f"No supported DLC project items found in {resolved_project_dir}")

        merged_labels = None
        videos: list[Path] = []
        source_items: list[dict[str, str]] = []
        for project_item in project_items:
            if project_item.source_type == "h5":
                result = convert_dlc_h5(
                    project_item.data_path,
                    project_item.video_path,
                    skeleton_name=resolved_skeleton_name,
                    likelihood_threshold=likelihood_threshold,
                    progress_callback=progress_callback,
                )
            else:
                result = convert_dlc_csv(
                    project_item.data_path,
                    project_item.video_path,
                    skeleton_name=resolved_skeleton_name,
                    likelihood_threshold=likelihood_threshold,
                    progress_callback=progress_callback,
                )

            merged_labels = _merge_labels_for_import(merged_labels, result.labels)
            videos.extend(result.videos)
            source_items.append(
                {
                    "name": project_item.name,
                    "source": f"dlc_{project_item.source_type}_import",
                    "source_data": _stored_project_path(
                        project_item.data_path,
                        project_root=resolved_project_dir,
                    ),
                    "source_video": _stored_project_path(
                        project_item.video_path,
                        project_root=resolved_project_dir,
                    ),
                }
            )

        assert merged_labels is not None
        merged_labels.validate()
        metadata = {
            "project_name": resolved_project_dir.name,
            "source": "dlc_project_import",
            "source_project": resolved_project_dir.as_posix(),
            "source_items": source_items,
            "skipped_items": [
                {"name": skipped_item.name, "reason": skipped_item.reason}
                for skipped_item in skipped_items
            ],
        }
        return ConversionResult(
            source_dir=resolved_project_dir,
            project_root=resolved_project_dir,
            videos=videos,
            labels=merged_labels,
            metadata=metadata,
        )

    return _import_pose_project(
        project,
        force=force,
        reason="project.import.dlc_project",
        prediction_provenance=prediction_provenance,
        provenance=provenance,
        default_tool="deeplabcut",
        source_path=resolved_project_dir,
        convert=_convert_project,
    )


def import_sleap_package_project(
    slp: str | Path,
    project: str | Path,
    *,
    fps: int = 30,
    encode_videos: bool | None = None,
    prediction_provenance: Mapping[str, Any] | None = None,
    provenance: Any = None,
    force: bool = False,
    progress_callback: Any | None = None,
) -> Path:
    """Import a SLEAP package into a project."""
    from xpkg.io.converters.sleap_import import convert_sleap_package

    return _import_pose_project(
        project,
        force=force,
        reason="project.import.sleap",
        prediction_provenance=prediction_provenance,
        provenance=provenance,
        default_tool="sleap",
        source_path=slp,
        convert=lambda tmp_dir: convert_sleap_package(
            slp,
            tmp_dir,
            fps=int(fps),
            encode_videos=encode_videos,
            progress_callback=progress_callback,
        ),
    )


def import_sleap_h5_project(
    h5_path: str | Path,
    video_path: str | Path,
    project: str | Path,
    *,
    skeleton_name: str = "imported",
    likelihood_threshold: float = 0.0,
    prediction_provenance: Mapping[str, Any] | None = None,
    provenance: Any = None,
    force: bool = False,
    progress_callback: Any | None = None,
) -> Path:
    """Import a SLEAP analysis H5 export plus video into a project."""
    from xpkg.io.converters.sleap_import import convert_sleap_h5

    return _import_pose_project(
        project,
        force=force,
        reason="project.import.sleap_h5",
        prediction_provenance=prediction_provenance,
        provenance=provenance,
        default_tool="sleap",
        source_path=h5_path,
        convert=lambda _tmp_dir: convert_sleap_h5(
            h5_path,
            video_path,
            skeleton_name=skeleton_name,
            likelihood_threshold=likelihood_threshold,
            progress_callback=progress_callback,
        ),
    )


def import_mmpose_topdown_json_project(
    json_path: str | Path,
    video_path: str | Path,
    project: str | Path,
    *,
    skeleton_name: str = "imported",
    instance_index: int = 0,
    likelihood_threshold: float = 0.0,
    prediction_provenance: Mapping[str, Any] | None = None,
    provenance: Any = None,
    force: bool = False,
    progress_callback: Any | None = None,
) -> Path:
    """Import an MMPose top-down JSON export plus video into a project."""
    from xpkg.io.converters.mmpose_import import convert_mmpose_topdown_json

    return _import_pose_project(
        project,
        force=force,
        reason="project.import.mmpose_topdown_json",
        prediction_provenance=prediction_provenance,
        provenance=provenance,
        default_tool="mmpose",
        source_path=json_path,
        convert=lambda _tmp_dir: convert_mmpose_topdown_json(
            json_path,
            video_path,
            skeleton_name=skeleton_name,
            instance_index=int(instance_index),
            likelihood_threshold=likelihood_threshold,
            progress_callback=progress_callback,
        ),
    )


def import_mediapipe_pose_landmarks_json_project(
    json_path: str | Path,
    video_path: str | Path,
    project: str | Path,
    *,
    skeleton_name: str = "mediapipe_pose",
    likelihood_threshold: float = 0.0,
    prediction_provenance: Mapping[str, Any] | None = None,
    provenance: Any = None,
    force: bool = False,
    progress_callback: Any | None = None,
) -> Path:
    """Import MediaPipe pose-landmarks JSON plus video into a project."""
    from xpkg.io.converters.mediapipe_import import convert_mediapipe_pose_landmarks_json

    return _import_pose_project(
        project,
        force=force,
        reason="project.import.mediapipe_pose_landmarks_json",
        prediction_provenance=prediction_provenance,
        provenance=provenance,
        default_tool="mediapipe",
        source_path=json_path,
        convert=lambda _tmp_dir: convert_mediapipe_pose_landmarks_json(
            json_path,
            video_path,
            skeleton_name=skeleton_name,
            likelihood_threshold=likelihood_threshold,
            progress_callback=progress_callback,
        ),
    )


def save_project_labels(
    project: str | Path,
    labels: Labels,
    *,
    metadata: dict[str, Any] | None = None,
    journal: bool = True,
    regenerate_predictions: bool = False,
) -> Path:
    """Commit a label save into the project's private store."""
    root = resolve_project_root(project)
    if root is None:
        raise FileNotFoundError(f"Not an xpkg project: {project}")

    descriptor = load_project_descriptor(root)
    descriptor.validate()
    ensure_dir(project_store_root(root))
    ensure_dir(project_media_root(root))
    ensure_dir(project_exports_root(root))

    state_path = current_project_state_path(root)
    current_state = _current_project_state_payload(root)
    has_state_cache = state_path.exists()
    has_committed_state = current_state is not None
    if metadata is not None and (has_state_cache or has_committed_state):
        raise ValueError(
            "Project saves with existing history do not accept metadata overrides. "
            "Update project metadata through a dedicated metadata API."
        )

    if not has_state_cache and not has_committed_state:
        initial_metadata = rewrite_project_metadata_paths(
            metadata,
            project_root=root,
        )
        state_path = _commit_labels_to_project(
            root,
            labels=labels,
            metadata=initial_metadata,
            predictions=predictions_payload_from_labels(labels),
            reason="project.save.init",
        )
        _touch_descriptor(root)
        labels.path = root
        return state_path

    del journal
    state_metadata: dict[str, Any] | None = None
    predictions: dict[str, Any] | None = None
    if current_state is not None:
        state_payload, source_kind = current_state
        state_metadata, predictions = _project_state_components(
            state_payload,
            source_kind=source_kind,
        )
    elif has_state_cache:
        state_payload = read_project_state(state_path)
        state_metadata = _state_metadata_from_state_payload(state_payload)
        predictions = _predictions_payload_from_state_payload(state_payload)

    candidate_predictions = predictions_payload_from_labels(labels)
    if regenerate_predictions:
        predictions = candidate_predictions
    elif (
        _predictions_committed_length(predictions) <= 0
        and _predictions_committed_length(candidate_predictions) > 0
    ):
        predictions = candidate_predictions

    state_path = _commit_labels_to_project(
        root,
        labels=labels,
        metadata=state_metadata,
        predictions=predictions,
        reason="project.save",
    )
    _touch_descriptor(root)
    labels.path = root
    return state_path


__all__ = [
    "CURRENT_STATE_FILENAME",
    "EXPORTS_DIRNAME",
    "import_vicon_c3d_project",
    "import_vicon_csv_project",
    "import_vicon_project",
    "import_dlc_csv_project",
    "import_dlc_h5_project",
    "import_dlc_project_directory",
    "import_lightning_pose_csv_project",
    "import_mediapipe_pose_landmarks_json_project",
    "import_mmpose_topdown_json_project",
    "import_sleap_h5_project",
    "import_sleap_package_project",
    "MEDIA_DIRNAME",
    "PROJECT_DESCRIPTOR_FILENAME",
    "ProjectDescriptor",
    "STORE_DIRNAME",
    "STORE_STATE_DIRNAME",
    "current_project_state_path",
    "init_project",
    "load_project_metadata",
    "load_project_payload",
    "load_project_vicon_recording",
    "load_project_descriptor",
    "resolve_project_root",
    "rebase_project_payload_videos",
    "save_project_metadata",
    "save_project_labels",
    "validate_project",
    "project_exports_root",
    "project_media_root",
    "project_store_root",
    "write_project_descriptor",
]
