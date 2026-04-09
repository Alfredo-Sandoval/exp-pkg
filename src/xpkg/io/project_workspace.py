"""Workspace save/import and store helpers for xpkg v1."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from xpkg.core.path_registry import ensure_dir, resolve_path
from xpkg.io.project_artifact import validate_workspace
from xpkg.io.project_layout import (
    CANONICAL_BUNDLE_SUFFIX,
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
from xpkg.io.archive_format import read_archive
from xpkg.io.workspace_snapshot_backend import (
    WORKSPACE_COMMIT_ID_KEY,
    normalize_predictions_payload,
    predictions_payload_from_labels,
    read_workspace_snapshot,
    rewrite_workspace_metadata_paths,
    write_workspace_snapshot,
)

if TYPE_CHECKING:
    from xpkg.model import Labels


def current_project_archive_path(path: str | Path) -> Path:
    return _workspace_store(path).current_archive_path()


def current_project_snapshot_path(path: str | Path) -> Path:
    return workspace_current_snapshot_path(path)


def current_project_commit_id(path: str | Path) -> str | None:
    return _workspace_store(path).current_commit_id()


def current_project_state_path(path: str | Path) -> Path:
    snapshot_path = current_project_snapshot_path(path)
    if snapshot_path.exists():
        return snapshot_path
    return current_project_archive_path(path)


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
        if self.has_durable_store():
            return self.open().current_archive_path().exists()
        return self.has_legacy_archive()

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

    def ensure_store(self):
        from xpkg.io.archive_store import ArchiveStore

        if self.has_durable_store():
            return self.open()

        legacy_archive = self._find_legacy_archive_path()
        if legacy_archive is None:
            raise FileNotFoundError(
                f"Workspace has no committed archive in {self.store_root}: "
                f"{self.legacy_current_archive_path}"
            )

        store = ArchiveStore.create_from_archive(
            store_root=self.store_root,
            initial_archive=legacy_archive,
            reason="workspace-adopt-legacy",
        )
        self._cleanup_legacy_state()
        return store

    def current_archive_path(self) -> Path:
        if self.has_durable_store():
            return self.open().current_archive_path()
        if self.has_legacy_archive():
            return self.ensure_store().current_archive_path()
        return self.legacy_current_archive_path

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
            legacy_archive = self._find_legacy_archive_path()
            assert legacy_archive is not None
            store = self.ensure_store()
            if candidate != legacy_archive:
                store.commit_new_archive(candidate, reason=reason, created_by=created_by)
            return store.current_archive_path()

        from xpkg.io.archive_store import ArchiveStore

        store = ArchiveStore.create_from_archive(
            store_root=self.store_root,
            initial_archive=candidate,
            created_by=created_by,
            reason=reason,
        )
        return store.current_archive_path()


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


def _snapshot_metadata(bundle_payload: dict[str, Any]) -> dict[str, Any] | None:
    metadata = bundle_payload.get("metadata")
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
    from xpkg.io.archive_format import write_archive

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
        staged_archive = Path(tmp_dir) / f"workspace{CANONICAL_BUNDLE_SUFFIX}"
        write_archive(
            staged_archive,
            labels,
            metadata=normalized_metadata,
            predictions=_prediction_items_from_payload(normalized_predictions),
        )
        store = _workspace_store(workspace_root)
        store.commit_archive(staged_archive, reason=reason)
        commit_id = store.current_commit_id()

    return _write_workspace_state(
        workspace_root,
        labels=labels,
        metadata=normalized_metadata,
        predictions=normalized_predictions,
        commit_id=commit_id,
    )


def rebuild_workspace_snapshot_cache(workspace_root: Path) -> Path:
    from xpkg.model import Labels

    store = _workspace_store(workspace_root)
    archive_path = store.current_archive_path()
    bundle_payload = read_archive(archive_path, lazy=False)
    labels = Labels.load_file(archive_path.as_posix())
    return _write_workspace_state(
        workspace_root,
        labels=labels,
        metadata=_snapshot_metadata(bundle_payload),
        predictions=_predictions_payload_from_state_payload(bundle_payload),
        commit_id=store.current_commit_id(),
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
        image_filenames = [
            Path(str(path))
            for path in raw_image_filenames
            if str(path).strip()
        ]
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

    root = _ensure_workspace_for_import(
        workspace,
        default_pack_mode=default_pack_mode,
        force=force,
    )
    from xpkg.model import Labels

    stage_parent = _stage_archive_parent(root)
    with tempfile.TemporaryDirectory(
        prefix=".workspace_import_",
        dir=str(stage_parent),
    ) as tmp_dir:
        staged_archive = Path(tmp_dir) / f"dlc_csv{CANONICAL_BUNDLE_SUFFIX}"
        convert_dlc_csv(
            csv_path,
            video_path,
            staged_archive,
            skeleton_name=skeleton_name,
            likelihood_threshold=likelihood_threshold,
            progress_callback=progress_callback,
        )
        labels = Labels.load_file(staged_archive.as_posix())
        payload = read_archive(staged_archive, lazy=False)
        snapshot_path = _commit_labels_to_workspace(
            root,
            labels=labels,
            metadata=_snapshot_metadata(payload),
            predictions=_predictions_payload_from_state_payload(payload),
            reason="workspace.import.dlc_csv",
        )
    _touch_descriptor(root)
    return snapshot_path


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

    root = _ensure_workspace_for_import(
        workspace,
        default_pack_mode=default_pack_mode,
        force=force,
    )
    from xpkg.model import Labels

    stage_parent = _stage_archive_parent(root)
    with tempfile.TemporaryDirectory(
        prefix=".workspace_import_",
        dir=str(stage_parent),
    ) as tmp_dir:
        staged_archive = Path(tmp_dir) / f"dlc_h5{CANONICAL_BUNDLE_SUFFIX}"
        convert_dlc_h5(
            h5_path,
            video_path,
            staged_archive,
            skeleton_name=skeleton_name,
            likelihood_threshold=likelihood_threshold,
            progress_callback=progress_callback,
        )
        labels = Labels.load_file(staged_archive.as_posix())
        payload = read_archive(staged_archive, lazy=False)
        snapshot_path = _commit_labels_to_workspace(
            root,
            labels=labels,
            metadata=_snapshot_metadata(payload),
            predictions=_predictions_payload_from_state_payload(payload),
            reason="workspace.import.dlc_h5",
        )
    _touch_descriptor(root)
    return snapshot_path


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

    root = _ensure_workspace_for_import(
        workspace,
        default_pack_mode=default_pack_mode,
        force=force,
    )
    from xpkg.model import Labels

    stage_parent = _stage_archive_parent(root)
    with tempfile.TemporaryDirectory(
        prefix=".workspace_import_",
        dir=str(stage_parent),
    ) as tmp_dir:
        import_root = Path(tmp_dir)
        result = convert_sleap_package(
            slp,
            import_root,
            fps=int(fps),
            encode_videos=encode_videos,
            progress_callback=progress_callback,
        )
        labels = Labels.load_file(result.bundle_path.as_posix())
        payload = read_archive(result.bundle_path, lazy=False)
        snapshot_path = _commit_labels_to_workspace(
            root,
            labels=labels,
            metadata=_snapshot_metadata(payload),
            predictions=_predictions_payload_from_state_payload(payload),
            reason="workspace.import.sleap",
        )
    _touch_descriptor(root)
    return snapshot_path


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

    store = _workspace_store(root)
    snapshot_path = current_project_snapshot_path(root)
    has_snapshot = snapshot_path.exists()
    has_archive = store.has_current_archive()
    if metadata is not None and (has_snapshot or has_archive):
        raise ValueError(
            "Workspace saves with existing history do not accept metadata overrides. "
            "Update workspace metadata through a dedicated metadata API."
        )

    if not has_snapshot and not has_archive:
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
    if has_snapshot:
        state_payload = read_workspace_snapshot(snapshot_path)
        state_metadata = _snapshot_metadata_from_state_payload(state_payload)
        predictions = _predictions_payload_from_state_payload(state_payload)
    elif has_archive:
        bundle_payload = read_archive(store.current_archive_path(), lazy=False)
        state_metadata = _snapshot_metadata(bundle_payload)
        predictions = _predictions_payload_from_state_payload(bundle_payload)

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
    "CANONICAL_BUNDLE_SUFFIX",
    "CURRENT_ARCHIVE_FILENAME",
    "CURRENT_SNAPSHOT_FILENAME",
    "EXPORTS_DIRNAME",
    "import_dlc_csv_workspace",
    "import_dlc_h5_workspace",
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
