"""Workspace save/import and store helpers for xpkg v1."""

from __future__ import annotations

import shutil
import tempfile
import warnings
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from xpkg.core.path_registry import ensure_dir, resolve_path
from xpkg.io.archive_format import read_archive
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
    write_workspace_snapshot,
    write_workspace_snapshot_payload,
)

if TYPE_CHECKING:
    from xpkg.model import Labels


class LegacyWorkspaceMigrationRequiredError(FileNotFoundError):
    """Raised when a pre-cutover workspace must be migrated explicitly."""


def current_project_archive_path(path: str | Path) -> Path:
    """Compatibility alias for explicit `.xpkg` archive export."""

    warnings.warn(
        "current_project_archive_path(...) is compatibility-only; "
        "use export_project_archive(...) for explicit archive export.",
        DeprecationWarning,
        stacklevel=2,
    )
    return export_project_archive(path)


def export_project_archive(
    path: str | Path,
    *,
    out: str | Path | None = None,
) -> Path:
    """Materialize a compatibility `.xpkg` archive from the committed workspace head."""

    return _workspace_store(path).export_archive(out=out)


def current_project_snapshot_path(path: str | Path) -> Path:
    """Return the workspace snapshot cache path under `.xpkg/state/current.json`."""

    return workspace_current_snapshot_path(path)


def current_project_commit_id(path: str | Path) -> str | None:
    return _workspace_store(path).current_commit_id()


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
                return _export_workspace_archive_from_snapshot(
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


def _current_workspace_state_payload(
    workspace_root: Path,
) -> tuple[dict[str, Any], str] | None:
    store = _workspace_store(workspace_root)
    if store.has_durable_store():
        mounted = store.open()
        if mounted.has_current_root("snapshot"):
            return read_workspace_snapshot(mounted.current_root_path("snapshot")), "snapshot"
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
    if source_kind == "snapshot":
        metadata = _snapshot_metadata_from_state_payload(state_payload)
    elif source_kind == "archive":
        metadata = _snapshot_metadata(state_payload)
    else:
        raise ValueError(f"Unsupported workspace state source: {source_kind!r}")
    return metadata, _predictions_payload_from_state_payload(state_payload)


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
    if source_kind == "snapshot":
        return write_workspace_snapshot_payload(
            workspace_current_snapshot_path(workspace_root),
            state_payload,
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


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
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


def _copy_file_into_media(source: Path, media_root: Path, copied: dict[Path, Path]) -> Path:
    resolved_source = source.resolve()
    cached = copied.get(resolved_source)
    if cached is not None:
        return cached
    if _is_within(resolved_source, media_root):
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
    if resolved_frames and all(_is_within(frame, media_root) for frame in resolved_frames):
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


def _export_workspace_archive_from_snapshot(
    workspace_root: Path,
    committed_snapshot_path: Path,
    *,
    archive_path: Path,
) -> Path:
    from xpkg.io.archive_format import write_archive
    from xpkg.io.labels.json_format import labels_from_json_payload

    snapshot_payload = read_workspace_snapshot(committed_snapshot_path)
    hydrated_payload = deepcopy(snapshot_payload)
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


def import_legacy_archive(
    legacy_archive: str | Path,
    workspace: str | Path,
    *,
    title: str | None = None,
    default_pack_mode: PackMode = "portable",
    force: bool = False,
) -> Path:
    return migrate_legacy_archive(
        legacy_archive,
        workspace,
        title=title,
        default_pack_mode=default_pack_mode,
        force=force,
    )


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


def _build_openpose_json_import_archive(
    tmp_dir: Path,
    *,
    json_dir: str | Path,
    video_path: str | Path,
    skeleton_name: str,
    likelihood_threshold: float,
    progress_callback: Any | None,
    convert_openpose_json: Callable[..., Any],
) -> Path:
    staged_archive = tmp_dir / f"openpose_json{CANONICAL_ARCHIVE_SUFFIX}"
    convert_openpose_json(
        json_dir,
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


def _build_detectron2_coco_import_archive(
    tmp_dir: Path,
    *,
    predictions_path: str | Path,
    dataset_json_path: str | Path,
    image_root: str | Path,
    category_id: int | None,
    skeleton_name: str | None,
    likelihood_threshold: float,
    progress_callback: Any | None,
    convert_detectron2_coco: Callable[..., Any],
) -> Path:
    staged_archive = tmp_dir / f"detectron2_coco{CANONICAL_ARCHIVE_SUFFIX}"
    convert_detectron2_coco(
        predictions_path,
        dataset_json_path,
        image_root,
        staged_archive,
        category_id=category_id,
        skeleton_name=skeleton_name,
        likelihood_threshold=likelihood_threshold,
        progress_callback=progress_callback,
    )
    return staged_archive


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


def import_openpose_json_workspace(
    json_dir: str | Path,
    video_path: str | Path,
    workspace: str | Path,
    *,
    skeleton_name: str = "imported",
    likelihood_threshold: float = 0.0,
    default_pack_mode: PackMode = "portable",
    force: bool = False,
    progress_callback: Any | None = None,
) -> Path:
    from xpkg.io.converters.openpose_import import convert_openpose_json

    return _import_workspace_from_staged_archive(
        workspace,
        default_pack_mode=default_pack_mode,
        force=force,
        reason="workspace.import.openpose_json",
        build_staged_archive=lambda tmp_dir: _build_openpose_json_import_archive(
            tmp_dir,
            json_dir=json_dir,
            video_path=video_path,
            skeleton_name=skeleton_name,
            likelihood_threshold=likelihood_threshold,
            progress_callback=progress_callback,
            convert_openpose_json=convert_openpose_json,
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


def import_detectron2_coco_workspace(
    predictions_path: str | Path,
    dataset_json_path: str | Path,
    image_root: str | Path,
    workspace: str | Path,
    *,
    category_id: int | None = None,
    skeleton_name: str | None = None,
    likelihood_threshold: float = 0.0,
    default_pack_mode: PackMode = "portable",
    force: bool = False,
    progress_callback: Any | None = None,
) -> Path:
    from xpkg.io.converters.detectron2_import import convert_detectron2_coco

    return _import_workspace_from_staged_archive(
        workspace,
        default_pack_mode=default_pack_mode,
        force=force,
        reason="workspace.import.detectron2_coco",
        build_staged_archive=lambda tmp_dir: _build_detectron2_coco_import_archive(
            tmp_dir,
            predictions_path=predictions_path,
            dataset_json_path=dataset_json_path,
            image_root=image_root,
            category_id=category_id,
            skeleton_name=skeleton_name,
            likelihood_threshold=likelihood_threshold,
            progress_callback=progress_callback,
            convert_detectron2_coco=convert_detectron2_coco,
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

    if regenerate_predictions:
        predictions = predictions_payload_from_labels(labels)
    elif predictions is None:
        candidate_predictions = predictions_payload_from_labels(labels)
        if int(candidate_predictions["attrs"]["committed_length"]) > 0:
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
    "import_dlc_csv_workspace",
    "import_dlc_h5_workspace",
    "import_dlc_project_workspace",
    "import_detectron2_coco_workspace",
    "import_mediapipe_pose_landmarks_json_workspace",
    "import_mmpose_topdown_json_workspace",
    "import_openpose_json_workspace",
    "import_sleap_h5_workspace",
    "import_sleap_package_workspace",
    "MEDIA_DIRNAME",
    "PROJECT_DESCRIPTOR_FILENAME",
    "ProjectDescriptor",
    "STORE_DIRNAME",
    "STORE_STATE_DIRNAME",
    "current_project_archive_path",
    "current_project_snapshot_path",
    "current_project_state_path",
    "import_legacy_archive",
    "init_project",
    "load_project_descriptor",
    "resolve_workspace_root",
    "rebase_workspace_payload_videos",
    "save_workspace_labels",
    "validate_workspace",
    "workspace_exports_root",
    "workspace_media_root",
    "workspace_store_root",
    "write_project_descriptor",
]
