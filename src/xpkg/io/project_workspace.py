"""Workspace save/import and store helpers for xpkg v1."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from xpkg.codecs.vicon import (
    read_vicon_json_payload,
    vicon_recording_from_json_payload,
    vicon_recording_to_json_payload,
)
from xpkg.core.json_utils import write_json
from xpkg.core.path_registry import ensure_dir, resolve_path, slugify_path_component
from xpkg.io.archive_format import read_archive
from xpkg.io.archive_store.hashing import sha256_file
from xpkg.io.project_artifact import validate_workspace
from xpkg.io.project_layout import (
    CANONICAL_ARCHIVE_SUFFIX,
    CURRENT_ARCHIVE_FILENAME,
    CURRENT_SNAPSHOT_FILENAME,
    EXPORTS_DIRNAME,
    MEDIA_DIRNAME,
    PROJECT_DESCRIPTOR_FILENAME,
    STORE_DIRNAME,
    STORE_STATE_DIRNAME,
    SUPPORTED_CURRENT_ARCHIVE_FILENAMES,
    PackMode,
    ProjectDescriptor,
    _candidate_workspace_root,
    _now_utc_iso,
    load_project_descriptor,
    resolve_workspace_root,
    workspace_current_snapshot_path,
    workspace_exports_root,
    workspace_media_root,
    workspace_store_root,
    write_project_descriptor,
)
from xpkg.io.workspace_snapshot_backend import (
    WORKSPACE_COMMIT_ID_KEY,
    normalize_predictions_payload,
    predictions_payload_from_labels,
    read_workspace_snapshot,
    rewrite_workspace_metadata_paths,
    workspace_snapshot_cache_digest_matches,
    write_workspace_snapshot,
    write_workspace_snapshot_cache_digest,
    write_workspace_snapshot_payload,
)
from xpkg.io.workspace_state import (
    read_workspace_state_document,
    workspace_state_commit_id_from_document,
    workspace_state_kind,
    workspace_state_payload_from_document,
)

if TYPE_CHECKING:
    from xpkg.model import Labels, ViconRecording


class LegacyWorkspaceMigrationRequiredError(FileNotFoundError):
    """Raised when a pre-cutover workspace must be migrated explicitly."""


def export_project_archive(
    path: str | Path,
    *,
    out: str | Path | None = None,
) -> Path:
    """Materialize a compatibility `.xpkg` archive from the committed workspace head."""

    return _workspace_store(path).export_archive(out=out)


def load_workspace_payload(path: str | Path) -> dict[str, Any]:
    """Return the current committed workspace payload on the public bundle surface."""
    root = resolve_workspace_root(path)
    if root is None:
        raise FileNotFoundError(f"Not an xpkg workspace: {path}")
    state = _current_workspace_state_payload(root)
    if state is None:
        return {"metadata": {}}
    payload, source_kind = state
    if source_kind == "archive":
        rebase_workspace_payload_videos(payload, root)
        return payload

    metadata = _snapshot_metadata_from_state_payload(payload) or {}
    if source_kind == "snapshot_labels":
        public_payload = _public_payload_from_snapshot_labels(payload, metadata=metadata)
        rebase_workspace_payload_videos(public_payload, root)
        return public_payload
    if source_kind == "snapshot_vicon":
        return {
            "recording": deepcopy(payload),
            "metadata": metadata,
        }
    raise ValueError(f"Unsupported workspace state source: {source_kind!r}")


def current_project_snapshot_path(path: str | Path) -> Path:
    """Return the workspace snapshot cache path under `.xpkg/state/current.json`."""

    return workspace_current_snapshot_path(path)


def current_project_commit_id(path: str | Path) -> str | None:
    return _workspace_store(path).current_commit_id()


def _public_payload_from_snapshot_labels(
    snapshot_payload: dict[str, Any],
    *,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    labels_payload = _public_labels_payload_from_snapshot(snapshot_payload, metadata=metadata)
    keypoint_count = int(labels_payload["metadata"]["num_keypoints"])
    return {
        "labels": labels_payload,
        "predictions": _public_predictions_payload_from_snapshot(
            snapshot_payload,
            keypoint_count=keypoint_count,
        ),
        "metrics": {
            "schema_version": 0,
            "tables": {},
            "metadata": {},
        },
        "suggestions": _public_suggestions_payload_from_snapshot(snapshot_payload),
        "runs": _empty_runs_payload(),
        "metadata": dict(metadata),
        "provenance": deepcopy(snapshot_payload.get("provenance") or {"events": []}),
        "session": _public_session_payload_from_snapshot(snapshot_payload, metadata=metadata),
        "segmentation": _public_segmentation_payload_from_snapshot(snapshot_payload),
    }


def _public_labels_payload_from_snapshot(
    snapshot_payload: dict[str, Any],
    *,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    labels_snapshot = _strip_prediction_instances_from_snapshot_payload(snapshot_payload)
    frames_info = labels_snapshot.get("frames")
    data_info = labels_snapshot.get("data")
    if not isinstance(frames_info, dict) or not isinstance(data_info, dict):
        raise TypeError("Workspace snapshot labels payload must contain frames/data mappings")

    skeleton_info = deepcopy(labels_snapshot.get("skeleton") or {})
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
        "videos": _public_videos_payload_from_snapshot(labels_snapshot),
        "tracks": deepcopy(labels_snapshot.get("tracks") or {}),
        "provenance": deepcopy(labels_snapshot.get("provenance") or {"events": []}),
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


def _public_videos_payload_from_snapshot(snapshot_payload: dict[str, Any]) -> dict[str, Any]:
    videos_info = deepcopy(snapshot_payload.get("videos") or {})
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


def _public_predictions_payload_from_snapshot(
    snapshot_payload: dict[str, Any],
    *,
    keypoint_count: int,
) -> dict[str, Any]:
    predictions = normalize_predictions_payload(
        _predictions_payload_from_state_payload(snapshot_payload)
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


def _public_suggestions_payload_from_snapshot(snapshot_payload: dict[str, Any]) -> dict[str, Any]:
    suggestions = snapshot_payload.get("suggestions")
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


def _public_session_payload_from_snapshot(
    snapshot_payload: dict[str, Any],
    *,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    session_json = metadata.get("session_json")
    if isinstance(session_json, dict):
        return deepcopy(session_json)
    session_payload = snapshot_payload.get("session")
    return deepcopy(session_payload) if isinstance(session_payload, dict) else {}


def _public_segmentation_payload_from_snapshot(snapshot_payload: dict[str, Any]) -> dict[str, Any]:
    segmentation = snapshot_payload.get("segmentation")
    if isinstance(segmentation, dict):
        return deepcopy(segmentation)
    return {"masks": [], "rois": [], "schema_version": ""}


def current_project_state_path(path: str | Path) -> Path:
    """Return the currently addressable workspace state path.

    Normal workspace-first flows use `.xpkg/state/current.json`. Archive roots
    are returned only for explicit compatibility-backed durable heads.
    """

    store = _workspace_store(path)
    snapshot_path = current_project_snapshot_path(path)
    if snapshot_path.exists() or store.has_current_snapshot():
        return snapshot_path
    if store.has_current_archive():
        return store.current_archive_root_path()
    return snapshot_path


@dataclass(slots=True)
class WorkspaceStore:
    """Private workspace store boundary for `.xpkg/`."""

    workspace_root: Path

    @property
    def store_root(self) -> Path:
        return workspace_store_root(self.workspace_root)

    @property
    def legacy_state_root(self) -> Path:
        return self.store_root / STORE_STATE_DIRNAME

    @property
    def legacy_current_archive_path(self) -> Path:
        return self.legacy_state_root / CURRENT_ARCHIVE_FILENAME

    @property
    def legacy_current_archive_paths(self) -> tuple[Path, ...]:
        return tuple(
            self.legacy_state_root / filename for filename in SUPPORTED_CURRENT_ARCHIVE_FILENAMES
        )

    @property
    def staging_root(self) -> Path:
        return self.store_root / "workspace"

    def has_durable_store(self) -> bool:
        return (self.store_root / "superblock.a.json").exists() or (
            self.store_root / "superblock.b.json"
        ).exists()

    def has_legacy_archive(self) -> bool:
        return self._find_legacy_archive_path() is not None

    def has_current_archive(self) -> bool:
        if not self.has_durable_store():
            return False
        return self.open().has_current_root("archive")

    def has_current_snapshot(self) -> bool:
        if not self.has_durable_store():
            return False
        return self.open().has_current_root("snapshot")

    def current_commit_id(self) -> str | None:
        if not self.has_durable_store():
            return None
        return self.open().load_current_commit().commit_id

    def open(self):
        from xpkg.io.archive_store import ArchiveStore

        return ArchiveStore.open(self.store_root)

    def _cleanup_legacy_state(self) -> None:
        for legacy_archive in self.legacy_current_archive_paths:
            if legacy_archive.exists():
                legacy_archive.unlink()

        state_root = self.legacy_state_root
        if state_root.is_dir():
            try:
                state_root.rmdir()
            except OSError:
                return

    def _find_legacy_archive_path(self) -> Path | None:
        for candidate in self.legacy_current_archive_paths:
            if candidate.is_file():
                return candidate
        return None

    def _legacy_migration_required_error(self) -> LegacyWorkspaceMigrationRequiredError:
        legacy_archive = self._find_legacy_archive_path() or self.legacy_current_archive_path
        return LegacyWorkspaceMigrationRequiredError(
            "Workspace still uses legacy archive-backed state at "
            f"{legacy_archive}. Run migrate_legacy_archive(...) before using "
            "workspace load/save/archive helpers."
        )

    def current_archive_path(self) -> Path:
        return self.export_archive()

    def current_archive_root_path(self) -> Path:
        if self.has_durable_store():
            store = self.open()
            if store.has_current_root("archive"):
                return store.current_archive_path()
        if self.has_legacy_archive():
            raise self._legacy_migration_required_error()
        raise FileNotFoundError(f"Workspace has no committed archive root: {self.store_root}")

    def export_archive(self, *, out: str | Path | None = None) -> Path:
        target_archive_path = (
            self.legacy_current_archive_path if out is None else _archive_export_path(out)
        )
        if self.has_durable_store():
            store = self.open()
            if store.has_current_root("archive"):
                current_archive = store.current_archive_path()
                if out is None:
                    return current_archive
                return _copy_archive_export(current_archive, target_archive_path)
            if store.has_current_root("snapshot"):
                return _export_archive_from_snapshot(
                    self.workspace_root,
                    store.current_root_path("snapshot"),
                    archive_path=target_archive_path,
                )
        if self.has_legacy_archive():
            raise self._legacy_migration_required_error()
        raise FileNotFoundError(f"Workspace has no committed state to export: {self.store_root}")

    def current_snapshot_path(self) -> Path:
        if not self.has_durable_store():
            raise FileNotFoundError(f"Workspace has no durable snapshot root: {self.store_root}")
        return self.open().current_root_path("snapshot")

    def commit_archive(
        self,
        archive_path: str | Path,
        *,
        reason: str,
        created_by: dict[str, Any] | None = None,
    ) -> Path:
        candidate = resolve_path(archive_path)
        if not candidate.is_file():
            raise FileNotFoundError(f"Staged archive not found: {candidate}")

        if self.has_durable_store():
            store = self.open()
            store.commit_new_archive(candidate, reason=reason, created_by=created_by)
            self._cleanup_legacy_state()
            return store.current_archive_path()

        if self.has_legacy_archive():
            raise self._legacy_migration_required_error()

        from xpkg.io.archive_store import ArchiveStore

        store = ArchiveStore.create_from_archive(
            store_root=self.store_root,
            initial_archive=candidate,
            created_by=created_by,
            reason=reason,
        )
        return store.current_archive_path()

    def commit_snapshot(
        self,
        snapshot_path: str | Path,
        *,
        reason: str,
        created_by: dict[str, Any] | None = None,
    ) -> Path:
        candidate = resolve_path(snapshot_path)
        if not candidate.is_file():
            raise FileNotFoundError(f"Staged workspace snapshot not found: {candidate}")

        if self.has_durable_store():
            store = self.open()
            store.commit_new_roots({"snapshot": candidate}, reason=reason, created_by=created_by)
            self._cleanup_legacy_state()
            return store.current_root_path("snapshot")

        if self.has_legacy_archive():
            raise self._legacy_migration_required_error()

        from xpkg.io.archive_store import ArchiveStore

        store = ArchiveStore.create_from_roots(
            store_root=self.store_root,
            initial_roots={"snapshot": candidate},
            created_by=created_by,
            reason=reason,
        )
        return store.current_root_path("snapshot")


def _workspace_store(path: str | Path) -> WorkspaceStore:
    root = resolve_workspace_root(path) or _candidate_workspace_root(path)
    return WorkspaceStore(workspace_root=root)


def init_project(
    workspace: str | Path,
    *,
    title: str | None = None,
    project_id: str | None = None,
    default_pack_mode: PackMode = "portable",
    force: bool = False,
) -> ProjectDescriptor:
    root = _candidate_workspace_root(workspace)
    if root.exists() and not root.is_dir():
        raise NotADirectoryError(f"Project root is not a directory: {root}")
    if root.exists():
        entries = list(root.iterdir())
        if entries and not force:
            raise FileExistsError(f"Project directory is not empty: {root}")
    else:
        ensure_dir(root)

    ensure_dir(workspace_store_root(root))
    ensure_dir(workspace_media_root(root))
    ensure_dir(workspace_exports_root(root))

    descriptor = ProjectDescriptor.new(
        title=(title or root.name or "exp-pkg Project").strip(),
        project_id=project_id,
        default_pack_mode=default_pack_mode,
    )
    write_project_descriptor(root, descriptor)
    return descriptor


def _clone_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    return dict(metadata or {})


def _stage_archive_parent(workspace_root: Path) -> Path:
    return ensure_dir(_workspace_store(workspace_root).staging_root)


def _archive_export_path(path: str | Path) -> Path:
    target = resolve_path(path)
    if target.suffix.lower() != CANONICAL_ARCHIVE_SUFFIX:
        raise ValueError(f"Archive exports must use {CANONICAL_ARCHIVE_SUFFIX}: {target}")
    return target


def _copy_archive_export(source_archive: Path, target_archive: Path) -> Path:
    if source_archive.resolve() == target_archive.resolve():
        return source_archive
    ensure_dir(target_archive.parent)
    shutil.copy2(source_archive, target_archive)
    return target_archive


def _snapshot_metadata(archive_payload: dict[str, Any]) -> dict[str, Any] | None:
    metadata = archive_payload.get("metadata")
    if not isinstance(metadata, dict):
        return None
    normalized = dict(metadata)
    normalized.pop("manifest", None)
    normalized.pop(WORKSPACE_COMMIT_ID_KEY, None)
    return normalized


def _snapshot_metadata_from_state_payload(
    state_payload: dict[str, Any],
) -> dict[str, Any] | None:
    metadata = state_payload.get("metadata")
    if not isinstance(metadata, dict):
        return None
    normalized = dict(metadata)
    normalized.pop(WORKSPACE_COMMIT_ID_KEY, None)
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


def _strip_prediction_instances_from_snapshot_payload(
    snapshot_payload: dict[str, Any],
) -> dict[str, Any]:
    stripped_payload = deepcopy(snapshot_payload)
    predictions_payload = _predictions_payload_from_state_payload(snapshot_payload)
    prediction_map = _prediction_instance_signatures(predictions_payload)
    if not prediction_map:
        return stripped_payload

    frames_info = stripped_payload.get("frames")
    data_info = stripped_payload.get("data")
    if not isinstance(frames_info, dict) or not isinstance(data_info, dict):
        raise TypeError("Workspace snapshot labels payload must contain frames/data mappings")

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
                    "Workspace snapshot labels/predictions are inconsistent for "
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


def _normalized_workspace_metadata(
    metadata: dict[str, Any] | None,
    *,
    workspace_root: Path,
    commit_id: str | None,
) -> dict[str, Any]:
    normalized = rewrite_workspace_metadata_paths(
        metadata,
        workspace_root=workspace_root,
    )
    if commit_id is None:
        normalized.pop(WORKSPACE_COMMIT_ID_KEY, None)
    else:
        normalized[WORKSPACE_COMMIT_ID_KEY] = str(commit_id)
    return normalized


def _workspace_snapshot_cache_matches_committed_head(
    workspace_root: Path,
    snapshot_path: Path,
) -> bool:
    from xpkg.io.archive_store import ArchiveStore

    store = ArchiveStore.open(workspace_store_root(workspace_root))
    commit = store.load_current_commit()
    if not commit.has_root("snapshot"):
        return False
    if workspace_snapshot_cache_digest_matches(snapshot_path, commit_id=commit.commit_id):
        return True
    root_entry = commit.root_entry("snapshot")
    committed_snapshot_path = store.paths.object_path(root_entry.object_id, ext=root_entry.ext)
    if not committed_snapshot_path.exists():
        return False
    if f"obj_{sha256_file(snapshot_path)}" == root_entry.object_id:
        write_workspace_snapshot_cache_digest(snapshot_path, commit_id=commit.commit_id)
        return True

    cache_document = read_workspace_state_document(snapshot_path)
    committed_document = read_workspace_state_document(committed_snapshot_path)
    if _workspace_state_documents_match_cache(cache_document, committed_document):
        write_workspace_snapshot_cache_digest(snapshot_path, commit_id=commit.commit_id)
        return True
    return False


def _metadata_matches_without_commit_id(
    cache_metadata: Mapping[str, Any],
    committed_metadata: Mapping[str, Any],
) -> bool:
    cache_keys = set(cache_metadata)
    committed_keys = set(committed_metadata)
    cache_keys.discard(WORKSPACE_COMMIT_ID_KEY)
    committed_keys.discard(WORKSPACE_COMMIT_ID_KEY)
    if cache_keys != committed_keys:
        return False
    return all(cache_metadata[key] == committed_metadata[key] for key in cache_keys)


def _workspace_payload_matches_cache(
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


def _workspace_state_documents_match_cache(
    cache_document: Mapping[str, Any],
    committed_document: Mapping[str, Any],
) -> bool:
    cache_payload = workspace_state_payload_from_document(cache_document)
    committed_payload = workspace_state_payload_from_document(committed_document)

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
        return _workspace_payload_matches_cache(cache_payload, committed_payload)

    return _workspace_payload_matches_cache(cache_document, committed_document)


def ensure_current_workspace_snapshot_cache(workspace_root: Path) -> Path | None:
    """Materialize the current workspace snapshot cache from the committed head when needed."""

    snapshot_path = workspace_current_snapshot_path(workspace_root)
    if snapshot_path.exists():
        current_head = current_project_commit_id(workspace_root)
        if current_head is None:
            return snapshot_path
        snapshot_document = read_workspace_state_document(snapshot_path)
        snapshot_head = workspace_state_commit_id_from_document(snapshot_document)
        if (
            snapshot_head == current_head
            and _workspace_snapshot_cache_matches_committed_head(
                workspace_root,
                snapshot_path,
            )
        ):
            return snapshot_path

    state = _current_workspace_state_payload(workspace_root)
    if state is None:
        return None
    return rebuild_workspace_snapshot_cache(workspace_root)


def _current_workspace_state_payload(
    workspace_root: Path,
) -> tuple[dict[str, Any], str] | None:
    store = _workspace_store(workspace_root)
    if store.has_durable_store():
        mounted = store.open()
        if mounted.has_current_root("snapshot"):
            snapshot_path = mounted.current_root_path("snapshot")
            snapshot_kind = workspace_state_kind(snapshot_path)
            if snapshot_kind == "labels":
                return read_workspace_snapshot(snapshot_path), "snapshot_labels"
            return read_vicon_json_payload(snapshot_path), "snapshot_vicon"
        if mounted.has_current_root("archive"):
            return read_archive(mounted.current_archive_path(), lazy=False), "archive"

    if store.has_legacy_archive():
        raise store._legacy_migration_required_error()

    return None


def _workspace_state_components(
    state_payload: dict[str, Any],
    *,
    source_kind: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if source_kind in {"snapshot_labels", "snapshot_vicon"}:
        metadata = _snapshot_metadata_from_state_payload(state_payload)
    elif source_kind == "archive":
        metadata = _snapshot_metadata(state_payload)
    else:
        raise ValueError(f"Unsupported workspace state source: {source_kind!r}")
    predictions = (
        _predictions_payload_from_state_payload(state_payload)
        if source_kind in {"snapshot_labels", "archive"}
        else None
    )
    return metadata, predictions


def _write_workspace_state(
    workspace_root: Path,
    *,
    labels: Labels,
    metadata: dict[str, Any] | None = None,
    predictions: dict[str, Any] | None = None,
    commit_id: str | None = None,
) -> Path:
    _manage_labels_media(labels, workspace_root)
    return write_workspace_snapshot(
        workspace_current_snapshot_path(workspace_root),
        labels=labels,
        workspace_root=workspace_root,
        metadata=metadata,
        predictions=predictions,
        commit_id=commit_id,
    )


def _write_vicon_workspace_state(
    workspace_root: Path,
    *,
    recording: ViconRecording,
    metadata: dict[str, Any] | None = None,
    commit_id: str | None = None,
) -> Path:
    document = vicon_recording_to_json_payload(
        recording,
        metadata=_normalized_workspace_metadata(
            metadata,
            workspace_root=workspace_root,
            commit_id=commit_id,
        ),
        source_root=workspace_root,
    )
    target = workspace_current_snapshot_path(workspace_root)
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
        write_workspace_snapshot_cache_digest(target, commit_id=str(commit_id))
    return target


def _prediction_items_from_payload(predictions: dict[str, Any] | None) -> list[Any]:
    from xpkg.io.archive_format.predictions_datasets import (
        PredictionAppendItem,
        SerializerPredictedInstance,
    )

    normalized = normalize_predictions_payload(predictions)
    attrs = normalized.get("attrs")
    frames = normalized.get("frames")
    data = normalized.get("data")
    if not isinstance(attrs, dict) or not isinstance(frames, dict) or not isinstance(data, dict):
        return []

    committed_length = int(attrs.get("committed_length", 0) or 0)
    if committed_length <= 0:
        return []

    video_index = list(frames.get("video_index") or [])
    frame_index = list(frames.get("frame_index") or [])
    num_instances = list(frames.get("num_instances") or [])
    keypoints = list(data.get("keypoints") or [])
    keypoint_score = list(data.get("keypoint_score") or [])
    instance_score = list(data.get("instance_score") or [])
    track_id = list(data.get("track_id") or [])
    deleted = list(data.get("deleted") or [])
    heatmaps = data.get("heatmaps")

    row_count = min(committed_length, len(video_index), len(frame_index), len(num_instances))
    items: list[Any] = []

    for row_idx in range(row_count):
        instance_count = int(num_instances[row_idx] or 0)
        row_keypoints = keypoints[row_idx] if row_idx < len(keypoints) else []
        row_keypoint_scores = keypoint_score[row_idx] if row_idx < len(keypoint_score) else []
        row_instance_scores = instance_score[row_idx] if row_idx < len(instance_score) else []
        row_track_ids = track_id[row_idx] if row_idx < len(track_id) else []
        row_deleted = deleted[row_idx] if row_idx < len(deleted) else []
        row_heatmaps = None
        if isinstance(heatmaps, list) and row_idx < len(heatmaps):
            row_heatmaps = heatmaps[row_idx]

        instances: list[Any] = []
        for inst_idx in range(instance_count):
            point_rows = row_keypoints[inst_idx] if inst_idx < len(row_keypoints) else []
            point_score_rows = (
                row_keypoint_scores[inst_idx] if inst_idx < len(row_keypoint_scores) else []
            )
            serialized_keypoints: list[tuple[float, float, float]] = []
            serialized_scores: list[float] = []

            for point_idx, point in enumerate(point_rows):
                if not isinstance(point, list | tuple) or len(point) < 3:
                    continue
                x = float(point[0])
                y = float(point[1])
                score = float(point[2])
                serialized_keypoints.append((x, y, score))
                if point_idx < len(point_score_rows):
                    serialized_scores.append(float(point_score_rows[point_idx]))
                else:
                    serialized_scores.append(score)

            score_value = None
            if inst_idx < len(row_instance_scores):
                score_value = float(row_instance_scores[inst_idx])

            track_value = None
            if inst_idx < len(row_track_ids):
                track_value = int(row_track_ids[inst_idx])

            deleted_value = False
            if inst_idx < len(row_deleted):
                deleted_value = bool(row_deleted[inst_idx])

            instances.append(
                SerializerPredictedInstance(
                    keypoints=serialized_keypoints,
                    keypoint_scores=serialized_scores,
                    score=score_value,
                    track_id=track_value,
                    deleted=deleted_value,
                )
            )

        items.append(
            PredictionAppendItem(
                video_index=int(video_index[row_idx]),
                frame_index=int(frame_index[row_idx]),
                instances=instances,
                heatmaps=row_heatmaps,
            )
        )

    return items


def _commit_labels_to_workspace(
    workspace_root: Path,
    *,
    labels: Labels,
    metadata: dict[str, Any] | None = None,
    predictions: dict[str, Any] | None = None,
    reason: str,
) -> Path:
    _manage_labels_media(labels, workspace_root)
    normalized_metadata = rewrite_workspace_metadata_paths(
        metadata,
        workspace_root=workspace_root,
    )
    normalized_predictions = (
        predictions_payload_from_labels(labels)
        if predictions is None
        else normalize_predictions_payload(predictions)
    )

    stage_parent = _stage_archive_parent(workspace_root)
    with tempfile.TemporaryDirectory(
        prefix=".workspace_commit_",
        dir=str(stage_parent),
    ) as tmp_dir:
        staged_snapshot = Path(tmp_dir) / CURRENT_SNAPSHOT_FILENAME
        write_workspace_snapshot(
            staged_snapshot,
            labels=labels,
            workspace_root=workspace_root,
            metadata=normalized_metadata,
            predictions=normalized_predictions,
        )
        store = _workspace_store(workspace_root)
        store.commit_snapshot(staged_snapshot, reason=reason)
        commit_id = store.current_commit_id()

    return _write_workspace_state(
        workspace_root,
        labels=labels,
        metadata=normalized_metadata,
        predictions=normalized_predictions,
        commit_id=commit_id,
    )


def _commit_snapshot_metadata_to_workspace(
    workspace_root: Path,
    *,
    state_payload: dict[str, Any],
    metadata: dict[str, Any] | None,
    reason: str,
) -> Path:
    normalized_metadata = rewrite_workspace_metadata_paths(
        metadata,
        workspace_root=workspace_root,
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
    staged_payload["metadata"] = _normalized_workspace_metadata(
        normalized_metadata,
        workspace_root=workspace_root,
        commit_id=None,
    )

    stage_parent = _stage_archive_parent(workspace_root)
    with tempfile.TemporaryDirectory(
        prefix=".workspace_commit_",
        dir=str(stage_parent),
    ) as tmp_dir:
        staged_snapshot = Path(tmp_dir) / CURRENT_SNAPSHOT_FILENAME
        write_workspace_snapshot_payload(staged_snapshot, staged_payload)
        store = _workspace_store(workspace_root)
        store.commit_snapshot(staged_snapshot, reason=reason)
        commit_id = store.current_commit_id()

    current_payload = deepcopy(state_payload)
    current_payload["metadata"] = normalized_metadata
    return write_workspace_snapshot_payload(
        workspace_current_snapshot_path(workspace_root),
        current_payload,
        commit_id=commit_id,
    )


def _commit_vicon_to_workspace(
    workspace_root: Path,
    *,
    recording: ViconRecording,
    metadata: dict[str, Any] | None = None,
    reason: str,
) -> Path:
    normalized_metadata = rewrite_workspace_metadata_paths(
        metadata,
        workspace_root=workspace_root,
    )

    stage_parent = _stage_archive_parent(workspace_root)
    with tempfile.TemporaryDirectory(
        prefix=".workspace_commit_",
        dir=str(stage_parent),
    ) as tmp_dir:
        staged_snapshot = Path(tmp_dir) / CURRENT_SNAPSHOT_FILENAME
        document = vicon_recording_to_json_payload(
            recording,
            metadata=_normalized_workspace_metadata(
                normalized_metadata,
                workspace_root=workspace_root,
                commit_id=None,
            ),
            source_root=workspace_root,
        )
        write_json(
            staged_snapshot,
            document,
            indent=None,
            sort_keys=False,
            ensure_ascii=True,
            compact=True,
        )
        store = _workspace_store(workspace_root)
        store.commit_snapshot(staged_snapshot, reason=reason)
        commit_id = store.current_commit_id()

    return _write_vicon_workspace_state(
        workspace_root,
        recording=recording,
        metadata=normalized_metadata,
        commit_id=commit_id,
    )


def _load_staged_archive_labels(
    staged_archive: Path,
) -> tuple[Labels, dict[str, Any] | None, dict[str, Any] | None]:
    from xpkg.model import Labels

    labels = Labels.load_file(staged_archive.as_posix())
    payload = read_archive(staged_archive, lazy=False)
    return (
        labels,
        _snapshot_metadata(payload),
        _predictions_payload_from_state_payload(payload),
    )


def _import_workspace_from_staged_archive(
    workspace: str | Path,
    *,
    default_pack_mode: PackMode,
    force: bool,
    reason: str,
    build_staged_archive: Callable[[Path], Path],
) -> Path:
    root = _ensure_workspace_for_import(
        workspace,
        default_pack_mode=default_pack_mode,
        force=force,
    )
    stage_parent = _stage_archive_parent(root)
    with tempfile.TemporaryDirectory(
        prefix=".workspace_import_",
        dir=str(stage_parent),
    ) as tmp_dir:
        staged_archive = build_staged_archive(Path(tmp_dir))
        labels, metadata, predictions = _load_staged_archive_labels(staged_archive)
        snapshot_path = _commit_labels_to_workspace(
            root,
            labels=labels,
            metadata=metadata,
            predictions=predictions,
            reason=reason,
        )
    _touch_descriptor(root)
    return snapshot_path


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


def rebuild_workspace_snapshot_cache(workspace_root: Path) -> Path:
    state = _current_workspace_state_payload(workspace_root)
    if state is None:
        raise FileNotFoundError(f"Workspace has no committed state: {workspace_root}")

    state_payload, source_kind = state
    commit_id = _workspace_store(workspace_root).current_commit_id()
    if source_kind == "snapshot_labels":
        return write_workspace_snapshot_payload(
            workspace_current_snapshot_path(workspace_root),
            state_payload,
            commit_id=commit_id,
        )
    if source_kind == "snapshot_vicon":
        recording = vicon_recording_from_json_payload(state_payload, source_root=workspace_root)
        return _write_vicon_workspace_state(
            workspace_root,
            recording=recording,
            metadata=_snapshot_metadata_from_state_payload(state_payload),
            commit_id=commit_id,
        )

    from xpkg.model import Labels

    archive_path = _workspace_store(workspace_root).current_archive_root_path()
    labels = Labels.load_file(archive_path.as_posix())
    metadata, predictions = _workspace_state_components(
        state_payload,
        source_kind=source_kind,
    )
    return _write_workspace_state(
        workspace_root,
        labels=labels,
        metadata=metadata,
        predictions=predictions,
        commit_id=commit_id,
    )


def load_workspace_vicon_recording(workspace: str | Path) -> ViconRecording:
    """Load the current Vicon recording from a workspace-managed state snapshot."""

    root = resolve_workspace_root(workspace)
    if root is None:
        raise FileNotFoundError(f"Not an xpkg workspace: {workspace}")

    state_path = ensure_current_workspace_snapshot_cache(root)
    if state_path is None:
        raise FileNotFoundError(f"Workspace has no committed state: {root}")
    if state_path.suffix.lower() != ".json":
        raise ValueError(
            "Workspace current state is not a Vicon JSON snapshot; "
            "only workspace-native snapshots can be loaded as Vicon recordings."
        )
    if workspace_state_kind(state_path) != "vicon":
        raise ValueError(
            "Workspace current state is not a Vicon recording. "
            "Use Labels.load_file(...) or WorkspaceService.load_labels()."
        )
    return vicon_recording_from_json_payload(
        read_workspace_state_document(state_path),
        source_root=root,
    )


def load_workspace_metadata(workspace: str | Path) -> dict[str, Any]:
    """Return the current workspace metadata payload from the managed head."""

    root = resolve_workspace_root(workspace)
    if root is None:
        raise FileNotFoundError(f"Not an xpkg workspace: {workspace}")

    current_state = _current_workspace_state_payload(root)
    if current_state is None:
        return {}

    state_payload, source_kind = current_state
    metadata, _predictions = _workspace_state_components(
        state_payload,
        source_kind=source_kind,
    )
    return _clone_metadata(metadata)


def save_workspace_metadata(
    workspace: str | Path,
    metadata: Mapping[str, Any] | None,
    *,
    reason: str = "workspace.save.metadata",
) -> Path:
    """Commit updated metadata onto the current workspace head."""

    root = resolve_workspace_root(workspace)
    if root is None:
        raise FileNotFoundError(f"Not an xpkg workspace: {workspace}")

    descriptor = load_project_descriptor(root)
    descriptor.validate()
    ensure_dir(workspace_store_root(root))
    ensure_dir(workspace_media_root(root))
    ensure_dir(workspace_exports_root(root))

    current_state = _current_workspace_state_payload(root)
    if current_state is None:
        raise FileNotFoundError(f"Workspace has no committed state: {root}")

    normalized_metadata = rewrite_workspace_metadata_paths(
        None if metadata is None else dict(metadata),
        workspace_root=root,
    )
    state_payload, source_kind = current_state
    if source_kind == "snapshot_labels":
        state_path = _commit_snapshot_metadata_to_workspace(
            root,
            state_payload=state_payload,
            metadata=normalized_metadata,
            reason=reason,
        )
        _touch_descriptor(root)
        return state_path

    if source_kind == "snapshot_vicon":
        state_path = _commit_vicon_to_workspace(
            root,
            recording=load_workspace_vicon_recording(root),
            metadata=normalized_metadata,
            reason=reason,
        )
        _touch_descriptor(root)
        return state_path

    from xpkg.model import Labels

    labels = Labels.load_file(root.as_posix())
    _state_metadata, predictions = _workspace_state_components(
        state_payload,
        source_kind=source_kind,
    )
    state_path = _commit_labels_to_workspace(
        root,
        labels=labels,
        metadata=normalized_metadata,
        predictions=predictions,
        reason=reason,
    )
    _touch_descriptor(root)
    labels.path = root
    return state_path


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


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


def _copy_vicon_import_bundle(recording: ViconRecording, workspace_root: Path) -> ViconRecording:
    imports_root = ensure_dir(workspace_store_root(workspace_root) / "imports" / "vicon")
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


def _manage_labels_media(labels: Labels, workspace_root: Path) -> None:
    media_root = ensure_dir(workspace_media_root(workspace_root))
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


def rebase_workspace_payload_videos(payload: dict[str, Any], workspace_root: Path) -> None:
    workspace_root = workspace_root.resolve()

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
                    else (workspace_root / filename_path).resolve()
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
                        else (workspace_root / frame_path).resolve()
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


def _export_archive_from_snapshot(
    workspace_root: Path,
    committed_snapshot_path: Path,
    *,
    archive_path: Path,
) -> Path:
    from xpkg.io.archive_format import write_archive
    from xpkg.io.labels.json_format import labels_from_json_payload

    snapshot_payload = read_workspace_snapshot(committed_snapshot_path)
    labels_payload = _strip_prediction_instances_from_snapshot_payload(snapshot_payload)
    hydrated_payload = deepcopy(labels_payload)
    rebase_workspace_payload_videos(hydrated_payload, workspace_root)
    labels = labels_from_json_payload(hydrated_payload)

    target_archive = _archive_export_path(archive_path)
    ensure_dir(target_archive.parent)
    write_archive(
        target_archive,
        labels,
        metadata=_snapshot_metadata_from_state_payload(snapshot_payload),
        predictions=_prediction_items_from_payload(
            _predictions_payload_from_state_payload(snapshot_payload)
        ),
    )
    return target_archive


def _absolutize_label_media(labels: Labels, *, source_root: Path) -> None:
    for video in labels.videos:
        filename = getattr(video, "filename", None)
        if filename:
            path = Path(str(filename))
            if path.is_absolute():
                video.filename = resolve_path(path).as_posix()
            else:
                video.filename = (source_root / path).resolve().as_posix()

        image_filenames = list(getattr(video, "image_filenames", []) or [])
        if not image_filenames:
            continue
        normalized: list[str] = []
        for frame in image_filenames:
            frame_path = Path(str(frame))
            if frame_path.is_absolute():
                normalized.append(resolve_path(frame_path).as_posix())
            else:
                normalized.append((source_root / frame_path).resolve().as_posix())
        video._image_filenames = normalized
        if not getattr(video, "filename", None) and normalized:
            video.filename = normalized[0]


def _apply_resolved_video_paths_from_payload(labels: Labels, payload: dict[str, Any]) -> None:
    labels_payload = payload.get("labels")
    if not isinstance(labels_payload, dict):
        return
    videos_payload = labels_payload.get("videos")
    if not isinstance(videos_payload, dict):
        return

    resolved_paths = list(videos_payload.get("resolved_paths") or [])
    for idx, video in enumerate(labels.videos):
        if idx >= len(resolved_paths):
            continue
        raw_path = str(resolved_paths[idx]).strip()
        if not raw_path:
            continue
        resolved = resolve_path(raw_path)
        video.filename = resolved.as_posix()


def migrate_legacy_archive(
    legacy_archive: str | Path,
    workspace: str | Path,
    *,
    title: str | None = None,
    default_pack_mode: PackMode = "portable",
    force: bool = False,
) -> Path:
    """Migrate an explicit legacy `.xpkg` archive into a workspace."""
    legacy_path = resolve_path(legacy_archive)
    if not legacy_path.is_file():
        raise FileNotFoundError(f"Legacy archive not found: {legacy_path}")

    root = resolve_workspace_root(workspace)
    if root is None:
        init_project(
            workspace,
            title=title,
            default_pack_mode=default_pack_mode,
            force=force,
        )
        root = _candidate_workspace_root(workspace)
    else:
        validate_workspace(root)

    from xpkg.model import Labels

    labels = Labels.load_file(str(legacy_path))
    _absolutize_label_media(labels, source_root=legacy_path.parent)
    payload = read_archive(legacy_path, lazy=False)
    _apply_resolved_video_paths_from_payload(labels, payload)
    _manage_labels_media(labels, root)
    metadata = rewrite_workspace_metadata_paths(
        _snapshot_metadata(payload),
        workspace_root=root,
        legacy_root=legacy_path.parent,
    )
    predictions = _predictions_payload_from_state_payload(payload)
    snapshot_path = _commit_labels_to_workspace(
        root,
        labels=labels,
        metadata=metadata,
        predictions=predictions,
        reason="workspace.migrate",
    )

    descriptor = load_project_descriptor(root)
    descriptor.updated_at = _now_utc_iso()
    if title is not None and title.strip():
        descriptor.title = title.strip()
    write_project_descriptor(root, descriptor)
    return snapshot_path


def _ensure_workspace_for_import(
    workspace: str | Path,
    *,
    title: str | None = None,
    default_pack_mode: PackMode = "portable",
    force: bool = False,
) -> Path:
    root = resolve_workspace_root(workspace)
    if root is None:
        init_project(
            workspace,
            title=title,
            default_pack_mode=default_pack_mode,
            force=force,
        )
        return _candidate_workspace_root(workspace)
    validate_workspace(root)
    return root


def _touch_descriptor(root: Path) -> None:
    descriptor = load_project_descriptor(root)
    descriptor.updated_at = _now_utc_iso()
    write_project_descriptor(root, descriptor)


def _build_dlc_csv_import_archive(
    tmp_dir: Path,
    *,
    csv_path: str | Path,
    video_path: str | Path,
    skeleton_name: str,
    likelihood_threshold: float,
    progress_callback: Any | None,
    convert_dlc_csv: Callable[..., Any],
) -> Path:
    staged_archive = tmp_dir / f"dlc_csv{CANONICAL_ARCHIVE_SUFFIX}"
    convert_dlc_csv(
        csv_path,
        video_path,
        staged_archive,
        skeleton_name=skeleton_name,
        likelihood_threshold=likelihood_threshold,
        progress_callback=progress_callback,
    )
    return staged_archive


def _build_lightning_pose_csv_import_archive(
    tmp_dir: Path,
    *,
    csv_path: str | Path,
    video_path: str | Path,
    skeleton_name: str,
    likelihood_threshold: float,
    progress_callback: Any | None,
    convert_lightning_pose_csv: Callable[..., Any],
) -> Path:
    staged_archive = tmp_dir / f"lightning_pose_csv{CANONICAL_ARCHIVE_SUFFIX}"
    convert_lightning_pose_csv(
        csv_path,
        video_path,
        staged_archive,
        skeleton_name=skeleton_name,
        likelihood_threshold=likelihood_threshold,
        progress_callback=progress_callback,
    )
    return staged_archive


def _build_dlc_h5_import_archive(
    tmp_dir: Path,
    *,
    h5_path: str | Path,
    video_path: str | Path,
    skeleton_name: str,
    likelihood_threshold: float,
    progress_callback: Any | None,
    convert_dlc_h5: Callable[..., Any],
) -> Path:
    staged_archive = tmp_dir / f"dlc_h5{CANONICAL_ARCHIVE_SUFFIX}"
    convert_dlc_h5(
        h5_path,
        video_path,
        staged_archive,
        skeleton_name=skeleton_name,
        likelihood_threshold=likelihood_threshold,
        progress_callback=progress_callback,
    )
    return staged_archive


def _build_sleap_h5_import_archive(
    tmp_dir: Path,
    *,
    h5_path: str | Path,
    video_path: str | Path,
    skeleton_name: str,
    likelihood_threshold: float,
    progress_callback: Any | None,
    convert_sleap_h5: Callable[..., Any],
) -> Path:
    staged_archive = tmp_dir / f"sleap_h5{CANONICAL_ARCHIVE_SUFFIX}"
    convert_sleap_h5(
        h5_path,
        video_path,
        staged_archive,
        skeleton_name=skeleton_name,
        likelihood_threshold=likelihood_threshold,
        progress_callback=progress_callback,
    )
    return staged_archive


def _build_mediapipe_pose_landmarks_json_import_archive(
    tmp_dir: Path,
    *,
    json_path: str | Path,
    video_path: str | Path,
    skeleton_name: str,
    likelihood_threshold: float,
    progress_callback: Any | None,
    convert_mediapipe_pose_landmarks_json: Callable[..., Any],
) -> Path:
    staged_archive = tmp_dir / f"mediapipe_pose_landmarks{CANONICAL_ARCHIVE_SUFFIX}"
    convert_mediapipe_pose_landmarks_json(
        json_path,
        video_path,
        staged_archive,
        skeleton_name=skeleton_name,
        likelihood_threshold=likelihood_threshold,
        progress_callback=progress_callback,
    )
    return staged_archive


def _build_mmpose_topdown_json_import_archive(
    tmp_dir: Path,
    *,
    json_path: str | Path,
    video_path: str | Path,
    skeleton_name: str,
    instance_index: int,
    likelihood_threshold: float,
    progress_callback: Any | None,
    convert_mmpose_topdown_json: Callable[..., Any],
) -> Path:
    staged_archive = tmp_dir / f"mmpose_topdown_json{CANONICAL_ARCHIVE_SUFFIX}"
    convert_mmpose_topdown_json(
        json_path,
        video_path,
        staged_archive,
        skeleton_name=skeleton_name,
        instance_index=int(instance_index),
        likelihood_threshold=likelihood_threshold,
        progress_callback=progress_callback,
    )
    return staged_archive


def _import_vicon_workspace_recording(
    recording_path: str | Path,
    workspace: str | Path,
    *,
    default_pack_mode: PackMode,
    force: bool,
    reason: str,
    progress_callback: Any | None,
    reader: Callable[[str | Path], ViconRecording],
    source_name: str,
) -> Path:
    root = _ensure_workspace_for_import(
        workspace,
        default_pack_mode=default_pack_mode,
        force=force,
    )
    if progress_callback is not None:
        progress_callback(f"Reading {source_name} recording")
    recording = reader(recording_path)
    if progress_callback is not None:
        progress_callback("Copying Vicon recording bundle into workspace store")
    managed_recording = _copy_vicon_import_bundle(recording, root)
    metadata = {
        "source": source_name,
        "source_recording": resolve_path(recording_path).as_posix(),
    }
    if recording.xcp_path is not None:
        metadata["source_xcp"] = resolve_path(recording.xcp_path).as_posix()
    if recording.vsk_path is not None:
        metadata["source_vsk"] = resolve_path(recording.vsk_path).as_posix()
    snapshot_path = _commit_vicon_to_workspace(
        root,
        recording=managed_recording,
        metadata=metadata,
        reason=reason,
    )
    _touch_descriptor(root)
    return snapshot_path


def import_vicon_csv_workspace(
    csv_path: str | Path,
    workspace: str | Path,
    *,
    default_pack_mode: PackMode = "portable",
    force: bool = False,
    progress_callback: Any | None = None,
) -> Path:
    """Import a Vicon CSV recording into a workspace."""
    from xpkg.io.readers import read_vicon_csv

    return _import_vicon_workspace_recording(
        csv_path,
        workspace,
        default_pack_mode=default_pack_mode,
        force=force,
        reason="workspace.import.vicon_csv",
        progress_callback=progress_callback,
        reader=read_vicon_csv,
        source_name="vicon_csv_import",
    )


def import_vicon_c3d_workspace(
    c3d_path: str | Path,
    workspace: str | Path,
    *,
    default_pack_mode: PackMode = "portable",
    force: bool = False,
    progress_callback: Any | None = None,
) -> Path:
    """Import a Vicon C3D recording into a workspace."""
    from xpkg.io.readers import read_vicon_c3d

    return _import_vicon_workspace_recording(
        c3d_path,
        workspace,
        default_pack_mode=default_pack_mode,
        force=force,
        reason="workspace.import.vicon_c3d",
        progress_callback=progress_callback,
        reader=read_vicon_c3d,
        source_name="vicon_c3d_import",
    )


def import_vicon_workspace(
    recording_path: str | Path,
    workspace: str | Path,
    *,
    default_pack_mode: PackMode = "portable",
    force: bool = False,
    progress_callback: Any | None = None,
) -> Path:
    """Import a Vicon CSV or C3D recording into a workspace."""
    from xpkg.io.readers import read_vicon_recording

    return _import_vicon_workspace_recording(
        recording_path,
        workspace,
        default_pack_mode=default_pack_mode,
        force=force,
        reason="workspace.import.vicon",
        progress_callback=progress_callback,
        reader=read_vicon_recording,
        source_name="vicon_import",
    )


def import_dlc_csv_workspace(
    csv_path: str | Path,
    video_path: str | Path,
    workspace: str | Path,
    *,
    skeleton_name: str = "imported",
    likelihood_threshold: float = 0.0,
    default_pack_mode: PackMode = "portable",
    force: bool = False,
    progress_callback: Any | None = None,
) -> Path:
    """Import a DeepLabCut CSV plus video into a workspace."""
    from xpkg.io.converters.dlc_import import convert_dlc_csv

    return _import_workspace_from_staged_archive(
        workspace,
        default_pack_mode=default_pack_mode,
        force=force,
        reason="workspace.import.dlc_csv",
        build_staged_archive=lambda tmp_dir: _build_dlc_csv_import_archive(
            tmp_dir,
            csv_path=csv_path,
            video_path=video_path,
            skeleton_name=skeleton_name,
            likelihood_threshold=likelihood_threshold,
            progress_callback=progress_callback,
            convert_dlc_csv=convert_dlc_csv,
        ),
    )


def import_lightning_pose_csv_workspace(
    csv_path: str | Path,
    video_path: str | Path,
    workspace: str | Path,
    *,
    skeleton_name: str = "imported",
    likelihood_threshold: float = 0.0,
    default_pack_mode: PackMode = "portable",
    force: bool = False,
    progress_callback: Any | None = None,
) -> Path:
    """Import a Lightning Pose prediction CSV plus video into a workspace."""
    from xpkg.io.converters.dlc_import import convert_lightning_pose_csv

    return _import_workspace_from_staged_archive(
        workspace,
        default_pack_mode=default_pack_mode,
        force=force,
        reason="workspace.import.lightning_pose_csv",
        build_staged_archive=lambda tmp_dir: _build_lightning_pose_csv_import_archive(
            tmp_dir,
            csv_path=csv_path,
            video_path=video_path,
            skeleton_name=skeleton_name,
            likelihood_threshold=likelihood_threshold,
            progress_callback=progress_callback,
            convert_lightning_pose_csv=convert_lightning_pose_csv,
        ),
    )


def import_dlc_h5_workspace(
    h5_path: str | Path,
    video_path: str | Path,
    workspace: str | Path,
    *,
    skeleton_name: str = "imported",
    likelihood_threshold: float = 0.0,
    default_pack_mode: PackMode = "portable",
    force: bool = False,
    progress_callback: Any | None = None,
) -> Path:
    """Import a DeepLabCut H5 export plus video into a workspace."""
    from xpkg.io.converters.dlc_import import convert_dlc_h5

    return _import_workspace_from_staged_archive(
        workspace,
        default_pack_mode=default_pack_mode,
        force=force,
        reason="workspace.import.dlc_h5",
        build_staged_archive=lambda tmp_dir: _build_dlc_h5_import_archive(
            tmp_dir,
            h5_path=h5_path,
            video_path=video_path,
            skeleton_name=skeleton_name,
            likelihood_threshold=likelihood_threshold,
            progress_callback=progress_callback,
            convert_dlc_h5=convert_dlc_h5,
        ),
    )


def import_dlc_project_workspace(
    project_dir: str | Path,
    workspace: str | Path,
    *,
    skeleton_name: str | None = None,
    likelihood_threshold: float = 0.0,
    default_pack_mode: PackMode = "portable",
    force: bool = False,
    progress_callback: Any | None = None,
) -> Path:
    """Import a supported DeepLabCut project into one workspace."""
    from xpkg.io.archive_format import write_archive
    from xpkg.io.converters.dlc_import import (
        _discover_dlc_project_items,
        _stored_project_path,
        convert_dlc_csv,
        convert_dlc_h5,
    )

    resolved_project_dir = resolve_path(project_dir)
    resolved_skeleton_name = skeleton_name or resolved_project_dir.name or "dlc"

    def _build_archive(tmp_dir: Path) -> Path:
        project_items, skipped_items = _discover_dlc_project_items(
            resolved_project_dir,
            progress_callback=progress_callback,
        )
        if not project_items:
            raise ValueError(f"No supported DLC project items found in {resolved_project_dir}")

        staged_items_root = ensure_dir(tmp_dir / "items")
        merged_labels = None
        source_items: list[dict[str, str]] = []
        for project_item in project_items:
            staged_archive = staged_items_root / f"{project_item.name}{CANONICAL_ARCHIVE_SUFFIX}"
            if project_item.source_type == "h5":
                convert_dlc_h5(
                    project_item.data_path,
                    project_item.video_path,
                    staged_archive,
                    skeleton_name=resolved_skeleton_name,
                    likelihood_threshold=likelihood_threshold,
                    progress_callback=progress_callback,
                )
            else:
                convert_dlc_csv(
                    project_item.data_path,
                    project_item.video_path,
                    staged_archive,
                    skeleton_name=resolved_skeleton_name,
                    likelihood_threshold=likelihood_threshold,
                    progress_callback=progress_callback,
                )

            labels, _metadata, _predictions = _load_staged_archive_labels(staged_archive)
            merged_labels = _merge_labels_for_import(merged_labels, labels)
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
            "source_items_json": source_items,
            "skipped_items_json": [
                {"name": skipped_item.name, "reason": skipped_item.reason}
                for skipped_item in skipped_items
            ],
        }
        staged_archive = (
            tmp_dir / f"{resolved_project_dir.name or 'dlc_project'}{CANONICAL_ARCHIVE_SUFFIX}"
        )
        write_archive(staged_archive, merged_labels, metadata=metadata)
        return staged_archive

    return _import_workspace_from_staged_archive(
        workspace,
        default_pack_mode=default_pack_mode,
        force=force,
        reason="workspace.import.dlc_project",
        build_staged_archive=_build_archive,
    )


def import_sleap_package_workspace(
    slp: str | Path,
    workspace: str | Path,
    *,
    fps: int = 30,
    encode_videos: bool | None = None,
    default_pack_mode: PackMode = "portable",
    force: bool = False,
    progress_callback: Any | None = None,
) -> Path:
    """Import a SLEAP package into a workspace."""
    from xpkg.io.converters.sleap_import import convert_sleap_package

    return _import_workspace_from_staged_archive(
        workspace,
        default_pack_mode=default_pack_mode,
        force=force,
        reason="workspace.import.sleap",
        build_staged_archive=lambda tmp_dir: convert_sleap_package(
            slp,
            tmp_dir,
            fps=int(fps),
            encode_videos=encode_videos,
            progress_callback=progress_callback,
        ).archive_path,
    )


def import_sleap_h5_workspace(
    h5_path: str | Path,
    video_path: str | Path,
    workspace: str | Path,
    *,
    skeleton_name: str = "imported",
    likelihood_threshold: float = 0.0,
    default_pack_mode: PackMode = "portable",
    force: bool = False,
    progress_callback: Any | None = None,
) -> Path:
    """Import a SLEAP analysis H5 export plus video into a workspace."""
    from xpkg.io.converters.sleap_import import convert_sleap_h5

    return _import_workspace_from_staged_archive(
        workspace,
        default_pack_mode=default_pack_mode,
        force=force,
        reason="workspace.import.sleap_h5",
        build_staged_archive=lambda tmp_dir: _build_sleap_h5_import_archive(
            tmp_dir,
            h5_path=h5_path,
            video_path=video_path,
            skeleton_name=skeleton_name,
            likelihood_threshold=likelihood_threshold,
            progress_callback=progress_callback,
            convert_sleap_h5=convert_sleap_h5,
        ),
    )


def import_mmpose_topdown_json_workspace(
    json_path: str | Path,
    video_path: str | Path,
    workspace: str | Path,
    *,
    skeleton_name: str = "imported",
    instance_index: int = 0,
    likelihood_threshold: float = 0.0,
    default_pack_mode: PackMode = "portable",
    force: bool = False,
    progress_callback: Any | None = None,
) -> Path:
    """Import an MMPose top-down JSON export plus video into a workspace."""
    from xpkg.io.converters.mmpose_import import convert_mmpose_topdown_json

    return _import_workspace_from_staged_archive(
        workspace,
        default_pack_mode=default_pack_mode,
        force=force,
        reason="workspace.import.mmpose_topdown_json",
        build_staged_archive=lambda tmp_dir: _build_mmpose_topdown_json_import_archive(
            tmp_dir,
            json_path=json_path,
            video_path=video_path,
            skeleton_name=skeleton_name,
            instance_index=int(instance_index),
            likelihood_threshold=likelihood_threshold,
            progress_callback=progress_callback,
            convert_mmpose_topdown_json=convert_mmpose_topdown_json,
        ),
    )


def import_mediapipe_pose_landmarks_json_workspace(
    json_path: str | Path,
    video_path: str | Path,
    workspace: str | Path,
    *,
    skeleton_name: str = "mediapipe_pose",
    likelihood_threshold: float = 0.0,
    default_pack_mode: PackMode = "portable",
    force: bool = False,
    progress_callback: Any | None = None,
) -> Path:
    """Import MediaPipe pose-landmarks JSON plus video into a workspace."""
    from xpkg.io.converters.mediapipe_import import convert_mediapipe_pose_landmarks_json

    return _import_workspace_from_staged_archive(
        workspace,
        default_pack_mode=default_pack_mode,
        force=force,
        reason="workspace.import.mediapipe_pose_landmarks_json",
        build_staged_archive=lambda tmp_dir: _build_mediapipe_pose_landmarks_json_import_archive(
            tmp_dir,
            json_path=json_path,
            video_path=video_path,
            skeleton_name=skeleton_name,
            likelihood_threshold=likelihood_threshold,
            progress_callback=progress_callback,
            convert_mediapipe_pose_landmarks_json=convert_mediapipe_pose_landmarks_json,
        ),
    )


def save_workspace_labels(
    workspace: str | Path,
    labels: Labels,
    *,
    metadata: dict[str, Any] | None = None,
    journal: bool = True,
    regenerate_predictions: bool = False,
) -> Path:
    """Commit a label save into the workspace's private store."""
    root = resolve_workspace_root(workspace)
    if root is None:
        raise FileNotFoundError(f"Not an xpkg workspace: {workspace}")

    descriptor = load_project_descriptor(root)
    descriptor.validate()
    ensure_dir(workspace_store_root(root))
    ensure_dir(workspace_media_root(root))
    ensure_dir(workspace_exports_root(root))

    snapshot_path = current_project_snapshot_path(root)
    current_state = _current_workspace_state_payload(root)
    has_snapshot_cache = snapshot_path.exists()
    has_committed_state = current_state is not None
    if metadata is not None and (has_snapshot_cache or has_committed_state):
        raise ValueError(
            "Workspace saves with existing history do not accept metadata overrides. "
            "Update workspace metadata through a dedicated metadata API."
        )

    if not has_snapshot_cache and not has_committed_state:
        initial_metadata = rewrite_workspace_metadata_paths(
            metadata,
            workspace_root=root,
        )
        state_path = _commit_labels_to_workspace(
            root,
            labels=labels,
            metadata=initial_metadata,
            predictions=predictions_payload_from_labels(labels),
            reason="workspace.save.init",
        )
        _touch_descriptor(root)
        labels.path = root
        return state_path

    del journal
    state_metadata: dict[str, Any] | None = None
    predictions: dict[str, Any] | None = None
    if current_state is not None:
        state_payload, source_kind = current_state
        state_metadata, predictions = _workspace_state_components(
            state_payload,
            source_kind=source_kind,
        )
    elif has_snapshot_cache:
        snapshot_payload = read_workspace_snapshot(snapshot_path)
        state_metadata = _snapshot_metadata_from_state_payload(snapshot_payload)
        predictions = _predictions_payload_from_state_payload(snapshot_payload)

    candidate_predictions = predictions_payload_from_labels(labels)
    if regenerate_predictions:
        predictions = candidate_predictions
    elif (
        _predictions_committed_length(predictions) <= 0
        and _predictions_committed_length(candidate_predictions) > 0
    ):
        predictions = candidate_predictions

    state_path = _commit_labels_to_workspace(
        root,
        labels=labels,
        metadata=state_metadata,
        predictions=predictions,
        reason="workspace.save",
    )
    _touch_descriptor(root)
    labels.path = root
    return state_path


__all__ = [
    "CANONICAL_ARCHIVE_SUFFIX",
    "CURRENT_ARCHIVE_FILENAME",
    "CURRENT_SNAPSHOT_FILENAME",
    "EXPORTS_DIRNAME",
    "export_project_archive",
    "import_vicon_c3d_workspace",
    "import_vicon_csv_workspace",
    "import_vicon_workspace",
    "import_dlc_csv_workspace",
    "import_dlc_h5_workspace",
    "import_dlc_project_workspace",
    "import_lightning_pose_csv_workspace",
    "import_mediapipe_pose_landmarks_json_workspace",
    "import_mmpose_topdown_json_workspace",
    "import_sleap_h5_workspace",
    "import_sleap_package_workspace",
    "MEDIA_DIRNAME",
    "PROJECT_DESCRIPTOR_FILENAME",
    "ProjectDescriptor",
    "STORE_DIRNAME",
    "STORE_STATE_DIRNAME",
    "current_project_snapshot_path",
    "current_project_state_path",
    "init_project",
    "load_workspace_metadata",
    "load_workspace_payload",
    "load_workspace_vicon_recording",
    "load_project_descriptor",
    "migrate_legacy_archive",
    "resolve_workspace_root",
    "rebase_workspace_payload_videos",
    "save_workspace_metadata",
    "save_workspace_labels",
    "validate_workspace",
    "workspace_exports_root",
    "workspace_media_root",
    "workspace_store_root",
    "write_project_descriptor",
]
