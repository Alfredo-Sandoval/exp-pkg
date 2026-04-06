"""Workspace save/import and store helpers for Posetta v1."""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

from posetta.core.path_registry import ensure_dir, resolve_path
from posetta.io.project_artifact import validate_workspace
from posetta.io.project_layout import (
    CANONICAL_BUNDLE_SUFFIX,
    CURRENT_ARCHIVE_FILENAME,
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
    workspace_exports_root,
    workspace_media_root,
    workspace_store_root,
    write_project_descriptor,
)
from posetta.io.siesta_format import read_siesta, write_siesta
from posetta.io.siesta_format.shared import _serialize_json

if TYPE_CHECKING:
    from posetta.model import Labels


def current_project_archive_path(path: str | Path) -> Path:
    return _workspace_store(path).current_archive_path()


@dataclass(slots=True)
class WorkspaceStore:
    """Private workspace store boundary for `.posetta/`."""

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

    def open(self):
        from posetta.io.siesta_store import SiestaStore

        return SiestaStore.open(self.store_root)

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
        from posetta.io.siesta_store import SiestaStore

        if self.has_durable_store():
            return self.open()

        legacy_archive = self._find_legacy_archive_path()
        if legacy_archive is None:
            raise FileNotFoundError(
                f"Workspace has no committed archive in {self.store_root}: "
                f"{self.legacy_current_archive_path}"
            )

        store = SiestaStore.create_from_archive(
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

        from posetta.io.siesta_store import SiestaStore

        store = SiestaStore.create_from_archive(
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
        title=(title or root.name or "Posetta Project").strip(),
        project_id=project_id,
        default_pack_mode=default_pack_mode,
    )
    write_project_descriptor(root, descriptor)
    return descriptor


def _clone_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    return dict(metadata or {})


def _copy_metrics_group(src_path: Path, dst_path: Path) -> None:
    with h5py.File(str(src_path), mode="r") as src, h5py.File(str(dst_path), mode="a") as dst:
        if "metrics" not in src:
            return
        if "metrics" in dst:
            del dst["metrics"]
        src.copy("metrics", dst)


def _drop_manifest_attr(archive_path: Path) -> None:
    with h5py.File(str(archive_path), mode="r+") as handle:
        metadata_group = handle.get("project_metadata")
        if not isinstance(metadata_group, h5py.Group):
            return
        attrs = metadata_group.attrs
        if "manifest_json" in attrs:
            del attrs["manifest_json"]


def _stage_archive_parent(workspace_root: Path) -> Path:
    return ensure_dir(_workspace_store(workspace_root).staging_root)


def _commit_labels_to_workspace(
    workspace_root: Path,
    *,
    labels: Labels,
    metadata: dict[str, Any] | None = None,
    manifest: dict[str, Any] | None = None,
    metrics_source: Path | None = None,
    reason: str,
) -> Path:
    _manage_labels_media(labels, workspace_root)
    stage_parent = _stage_archive_parent(workspace_root)

    with tempfile.TemporaryDirectory(prefix=".workspace_stage_", dir=str(stage_parent)) as tmp_dir:
        staged_archive = Path(tmp_dir) / f"staged{CANONICAL_BUNDLE_SUFFIX}"
        write_siesta(
            staged_archive,
            labels,
            metadata=metadata,
            manifest=manifest,
        )
        _rewrite_internal_archive_video_paths(
            staged_archive,
            labels=labels,
            workspace_root=workspace_root,
        )
        if metrics_source is not None:
            _copy_metrics_group(metrics_source, staged_archive)
        return _workspace_store(workspace_root).commit_archive(
            staged_archive,
            reason=reason,
        )


def _relative_workspace_path(path: Path, workspace_root: Path) -> str:
    resolved = path.resolve()
    return resolved.relative_to(workspace_root.resolve()).as_posix()


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


def _rewrite_internal_archive_video_paths(
    archive_path: Path,
    *,
    labels: Labels,
    workspace_root: Path,
) -> None:
    workspace_root = workspace_root.resolve()
    str_dtype = h5py.string_dtype("utf-8")
    filenames = [
        _relative_workspace_path(Path(str(video.filename)), workspace_root)
        if getattr(video, "filename", None)
        else ""
        for video in labels.videos
    ]
    image_filenames_json = []
    for video in labels.videos:
        raw_frames = [
            str(path)
            for path in (getattr(video, "image_filenames", None) or [])
            if str(path).strip()
        ]
        if not raw_frames:
            image_filenames_json.append("")
            continue
        relative_frames = [
            _relative_workspace_path(Path(frame_path), workspace_root) for frame_path in raw_frames
        ]
        image_filenames_json.append(json.dumps(relative_frames))

    with h5py.File(str(archive_path), mode="r+") as handle:
        videos_group = handle.get("videos")
        if not isinstance(videos_group, h5py.Group):
            raise FileNotFoundError(f"Archive is missing the /videos group: {archive_path}")
        for dataset_name in ("filenames", "image_filenames_json"):
            if dataset_name in videos_group:
                del videos_group[dataset_name]
        videos_group.create_dataset(
            "filenames",
            data=np.array(filenames, dtype=object),
            dtype=str_dtype,
        )
        videos_group.create_dataset(
            "image_filenames_json",
            data=np.array(image_filenames_json, dtype=object),
            dtype=str_dtype,
        )
    _drop_manifest_attr(archive_path)


def _load_project_metadata_json_attr(
    metadata_group: h5py.Group,
    attr_key: str,
) -> dict[str, Any] | None:
    raw_value = metadata_group.attrs.get(attr_key)
    if raw_value is None:
        return None
    if isinstance(raw_value, bytes | bytearray | np.bytes_):
        raw_text = raw_value.decode("utf-8")
    else:
        raw_text = str(raw_value)
    payload = json.loads(raw_text)
    if not isinstance(payload, dict):
        raise TypeError(f"project_metadata.{attr_key} must decode to a JSON object")
    return dict(payload)


def _rebase_legacy_workspace_path(
    raw_path: str,
    *,
    legacy_root: Path,
    workspace_root: Path,
) -> str:
    candidate = Path(raw_path)
    if not raw_path.strip() or not candidate.is_absolute():
        return raw_path
    try:
        relative = candidate.resolve().relative_to(legacy_root.resolve())
    except ValueError:
        return raw_path
    rebased = workspace_root.resolve() / relative
    if not rebased.exists():
        return ""
    return rebased.as_posix()


def _rewrite_training_state_entry_paths(
    entry: dict[str, Any],
    *,
    legacy_root: Path,
    workspace_root: Path,
) -> None:
    for field in ("output_dir", "source_bundle"):
        raw_value = entry.get(field)
        if isinstance(raw_value, str):
            entry[field] = _rebase_legacy_workspace_path(
                raw_value,
                legacy_root=legacy_root,
                workspace_root=workspace_root,
            )


def _rewrite_training_state_attr_paths(
    training_state: dict[str, Any],
    *,
    legacy_root: Path,
    workspace_root: Path,
) -> None:
    latest = training_state.get("latest")
    if latest is not None:
        if not isinstance(latest, dict):
            raise TypeError("project_metadata.training_state_json.latest must be a mapping")
        _rewrite_training_state_entry_paths(
            latest,
            legacy_root=legacy_root,
            workspace_root=workspace_root,
        )
    runs = training_state.get("runs")
    if runs is None:
        return
    if not isinstance(runs, list):
        raise TypeError("project_metadata.training_state_json.runs must be a JSON array")
    for entry in runs:
        if not isinstance(entry, dict):
            raise TypeError("project_metadata.training_state_json.runs[] must be mappings")
        _rewrite_training_state_entry_paths(
            entry,
            legacy_root=legacy_root,
            workspace_root=workspace_root,
        )


def _match_workspace_media_relative_path(raw_path: str, *, workspace_root: Path) -> str:
    target_name = Path(raw_path).name
    if not target_name:
        return ""
    media_root = workspace_media_root(workspace_root).resolve()
    matches = sorted(
        candidate.resolve()
        for candidate in media_root.rglob(target_name)
        if candidate.is_file() or candidate.is_dir()
    )
    if len(matches) != 1:
        return ""
    return matches[0].relative_to(workspace_root.resolve()).as_posix()


def _rewrite_session_attr_paths(session_state: dict[str, Any], *, workspace_root: Path) -> None:
    active_video_path = session_state.get("active_video_path")
    if active_video_path is None:
        return
    if not isinstance(active_video_path, str):
        raise TypeError("project_metadata.session_json.active_video_path must be a string")
    session_state["active_video_path"] = _match_workspace_media_relative_path(
        active_video_path,
        workspace_root=workspace_root,
    )


def _rewrite_staged_archive_project_metadata_paths(
    archive_path: Path,
    *,
    legacy_root: Path,
    workspace_root: Path,
) -> None:
    with h5py.File(archive_path, "a") as handle:
        metadata_group = handle.get("project_metadata")
        if not isinstance(metadata_group, h5py.Group):
            return
        training_state = _load_project_metadata_json_attr(
            metadata_group,
            "training_state_json",
        )
        if training_state is not None:
            _rewrite_training_state_attr_paths(
                training_state,
                legacy_root=legacy_root,
                workspace_root=workspace_root,
            )
            metadata_group.attrs["training_state_json"] = _serialize_json(training_state)
        session_state = _load_project_metadata_json_attr(metadata_group, "session_json")
        if session_state is not None:
            _rewrite_session_attr_paths(session_state, workspace_root=workspace_root)
            metadata_group.attrs["session_json"] = _serialize_json(session_state)


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

    from posetta.io.siesta_format import update_labels_siesta
    from posetta.model import Labels

    labels = Labels.load_file(str(legacy_path))
    _absolutize_label_media(labels, source_root=legacy_path.parent)
    payload = read_siesta(legacy_path, lazy=False)
    _apply_resolved_video_paths_from_payload(labels, payload)
    _manage_labels_media(labels, root)

    stage_parent = _stage_archive_parent(root)
    with tempfile.TemporaryDirectory(
        prefix=".workspace_migrate_",
        dir=str(stage_parent),
    ) as tmp_dir:
        staged_archive = Path(tmp_dir) / f"migrate{CANONICAL_BUNDLE_SUFFIX}"
        shutil.copy2(legacy_path, staged_archive)
        update_labels_siesta(
            staged_archive,
            labels,
            journal=False,
            regenerate_predictions=False,
        )
        _rewrite_internal_archive_video_paths(
            staged_archive,
            labels=labels,
            workspace_root=root,
        )
        _rewrite_staged_archive_project_metadata_paths(
            staged_archive,
            legacy_root=legacy_path.parent,
            workspace_root=root,
        )
        archive_path = _workspace_store(root).commit_archive(
            staged_archive,
            reason="migrate.legacy",
        )

    descriptor = load_project_descriptor(root)
    descriptor.updated_at = _now_utc_iso()
    if title is not None and title.strip():
        descriptor.title = title.strip()
    write_project_descriptor(root, descriptor)
    return archive_path


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
    from posetta.io.converters.dlc_import import convert_dlc_csv

    root = _ensure_workspace_for_import(
        workspace,
        default_pack_mode=default_pack_mode,
        force=force,
    )
    from posetta.model import Labels

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
        payload = read_siesta(staged_archive, lazy=False)
        metadata = _clone_metadata(payload.get("metadata") if isinstance(payload, dict) else None)
        metadata.pop("manifest", None)
        archive_path = _commit_labels_to_workspace(
            root,
            labels=labels,
            metadata=metadata,
            reason="import.dlc_csv",
        )
    _touch_descriptor(root)
    return archive_path


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
    from posetta.io.converters.dlc_import import convert_dlc_h5

    root = _ensure_workspace_for_import(
        workspace,
        default_pack_mode=default_pack_mode,
        force=force,
    )
    from posetta.model import Labels

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
        payload = read_siesta(staged_archive, lazy=False)
        metadata = _clone_metadata(payload.get("metadata") if isinstance(payload, dict) else None)
        metadata.pop("manifest", None)
        archive_path = _commit_labels_to_workspace(
            root,
            labels=labels,
            metadata=metadata,
            reason="import.dlc_h5",
        )
    _touch_descriptor(root)
    return archive_path


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
    from posetta.io.converters.sleap_import import convert_sleap_package

    root = _ensure_workspace_for_import(
        workspace,
        default_pack_mode=default_pack_mode,
        force=force,
    )
    from posetta.model import Labels

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
        payload = read_siesta(result.bundle_path, lazy=False)
        metadata = _clone_metadata(payload.get("metadata") if isinstance(payload, dict) else None)
        metadata.pop("manifest", None)
        archive_path = _commit_labels_to_workspace(
            root,
            labels=labels,
            metadata=metadata,
            reason="import.sleap",
        )
    _touch_descriptor(root)
    return archive_path


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
        raise FileNotFoundError(f"Not a Posetta workspace: {workspace}")

    descriptor = load_project_descriptor(root)
    descriptor.validate()
    ensure_dir(workspace_store_root(root))
    ensure_dir(workspace_media_root(root))
    ensure_dir(workspace_exports_root(root))

    store = _workspace_store(root)
    if metadata is not None and store.has_current_archive():
        raise ValueError(
            "Workspace saves with existing history do not accept metadata overrides. "
            "Update workspace metadata through a dedicated metadata API."
        )

    if not store.has_current_archive():
        archive_path = _commit_labels_to_workspace(
            root,
            labels=labels,
            metadata=metadata,
            reason="workspace.save.init",
        )
        _touch_descriptor(root)
        labels.path = root
        return archive_path

    current_archive = store.current_archive_path()
    stage_parent = _stage_archive_parent(root)
    from posetta.io.siesta_format import update_labels_siesta

    with tempfile.TemporaryDirectory(
        prefix=".workspace_save_",
        dir=str(stage_parent),
    ) as tmp_dir:
        staged_archive = Path(tmp_dir) / f"save{CANONICAL_BUNDLE_SUFFIX}"
        shutil.copy2(current_archive, staged_archive)
        _manage_labels_media(labels, root)
        update_labels_siesta(
            staged_archive,
            labels,
            journal=journal,
            regenerate_predictions=regenerate_predictions,
        )
        _rewrite_internal_archive_video_paths(
            staged_archive,
            labels=labels,
            workspace_root=root,
        )
        archive_path = store.commit_archive(
            staged_archive,
            reason="workspace.save",
        )

    _touch_descriptor(root)
    labels.path = root
    return archive_path


__all__ = [
    "CANONICAL_BUNDLE_SUFFIX",
    "CURRENT_ARCHIVE_FILENAME",
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
