"""Generated shallow project inventory summary.

The summary index is the cheap project-row contract behind ``PROJECT.json``.
It records enough inventory to route and display projects without loading
labels, predictions, dense masks, motion arrays, or media payloads.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from xpkg.adapters.vicon import XPKG_VICON_JSON_FORMAT
from xpkg.io.labels.json_format import XPKG_LABELS_JSON_FORMAT
from xpkg.project.durable_store import ProjectDurableStore, ProjectDurableStoreError
from xpkg.project.layout import (
    _now_utc_iso,
    load_project_descriptor,
    project_artifacts_root,
    project_current_state_path,
    project_store_root,
    project_summary_path,
    resolve_project_root,
)

from .._core.json_utils import load_json_dict, write_json
from .._core.path_registry import ensure_dir

if TYPE_CHECKING:
    from xpkg.model import Labels, ViconRecording


PROJECT_SUMMARY_SCHEMA_VERSION = 1
_STATE_PREFIX_BYTES = 8192
_ARTIFACT_INDEX_FILENAME = "index.json"
_METADATA_SLOT_FILES = {
    "acquisition": "acquisition.json",
    "dataset-share": "dataset_share.json",
    "datasheet": "datasheet.json",
    "model-card": "model_card.json",
    "pose-provenance": "pose_provenance.json",
}

ProjectSummaryStateKind = Literal["empty", "labels", "vicon", "present", "unreadable"]
JsonScalar = str | int | float | bool | None
MediaSummaryItem = dict[str, JsonScalar]


@dataclass(frozen=True, slots=True)
class ProjectSummaryIndex:
    """Shallow generated inventory for one xpkg project."""

    schema_version: int
    generated_at: str
    project_id: str
    title: str
    descriptor_updated_at: str
    state_kind: ProjectSummaryStateKind
    has_current_state: bool
    state_bytes: int | None
    commit_id: str | None
    state_summary: dict[str, JsonScalar] = field(default_factory=dict)
    media: tuple[MediaSummaryItem, ...] = ()
    metadata_slots: tuple[str, ...] = ()
    artifact_count: int = 0
    artifact_types: dict[str, int] = field(default_factory=dict)
    modalities: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProjectSummaryIndex:
        schema_version = int(data.get("schema_version", 0))
        if schema_version != PROJECT_SUMMARY_SCHEMA_VERSION:
            raise ValueError(f"Unsupported project summary schema: {schema_version!r}")
        project = _mapping(data.get("project"), name="project_summary.project")
        state = _mapping(data.get("state"), name="project_summary.state")
        media = _mapping(data.get("media") or {}, name="project_summary.media")
        artifacts = _mapping(data.get("artifacts"), name="project_summary.artifacts")
        metadata = _mapping(data.get("metadata"), name="project_summary.metadata")
        return cls(
            schema_version=schema_version,
            generated_at=str(data.get("generated_at", "")),
            project_id=str(project.get("project_id", "")),
            title=str(project.get("title", "")),
            descriptor_updated_at=str(project.get("descriptor_updated_at", "")),
            state_kind=_state_kind(str(state.get("kind", "present"))),
            has_current_state=bool(state.get("has_current_state", False)),
            state_bytes=_optional_int(state.get("bytes")),
            commit_id=_optional_str(state.get("commit_id")),
            state_summary=_scalar_dict(state.get("summary") or {}),
            media=_media_items(media.get("items") or ()),
            metadata_slots=_string_tuple(metadata.get("slots") or ()),
            artifact_count=int(artifacts.get("count", 0) or 0),
            artifact_types=_int_dict(artifacts.get("types") or {}),
            modalities=_string_tuple(data.get("modalities") or ()),
            warnings=_string_tuple(data.get("warnings") or ()),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "generated_at": self.generated_at,
            "project": {
                "project_id": self.project_id,
                "title": self.title,
                "descriptor_updated_at": self.descriptor_updated_at,
            },
            "state": {
                "kind": self.state_kind,
                "has_current_state": self.has_current_state,
                "bytes": self.state_bytes,
                "commit_id": self.commit_id,
                "summary": dict(self.state_summary),
            },
            "media": {"items": [dict(item) for item in self.media]},
            "metadata": {"slots": list(self.metadata_slots)},
            "artifacts": {
                "count": int(self.artifact_count),
                "types": dict(self.artifact_types),
            },
            "modalities": list(self.modalities),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class _StateSource:
    kind: ProjectSummaryStateKind
    has_current_state: bool
    bytes: int | None
    commit_id: str | None
    warnings: tuple[str, ...]


def labels_state_summary(
    labels: Labels,
    predictions: Mapping[str, Any] | None = None,
) -> dict[str, JsonScalar]:
    """Return shallow counts from in-memory labels during save/import."""

    keypoint_count = 0
    if labels.skeletons:
        keypoint_count = len(labels.skeletons[0].keypoints)
    return {
        "video_count": len(labels.videos),
        "skeleton_count": len(labels.skeletons),
        "keypoint_count": keypoint_count,
        "track_count": len(labels.tracks),
        "suggestion_count": len(labels.suggestions),
        "label_frame_count": len(labels.user_labeled_frames),
        "prediction_frame_count": _prediction_frame_count(predictions),
    }


def labels_media_summary(
    labels: Labels,
    predictions: Mapping[str, Any] | None = None,
    *,
    project_root: str | Path | None = None,
) -> tuple[MediaSummaryItem, ...]:
    """Return cheap media inventory from in-memory labels during save/import."""

    root = Path(project_root).resolve() if project_root is not None else None
    video_lookup: dict[object, int] = {video: idx for idx, video in enumerate(labels.videos)}
    label_stats = _frame_stats_by_video(labels.labeled_frames, video_lookup)
    prediction_stats = _prediction_stats_by_video(predictions)
    items: list[MediaSummaryItem] = []
    for index, video in enumerate(labels.videos):
        image_filenames = tuple(str(path) for path in (video.image_filenames or ()))
        raw_path = str(video.filename or "")
        if image_filenames and not raw_path:
            raw_path = str(Path(image_filenames[0]).parent)
        fallback_label = Path(raw_path).name if raw_path else f"video-{index}"
        label_entry = label_stats.get(index, {})
        prediction_entry = prediction_stats.get(index, {})
        fps = _safe_float(getattr(video, "fps", 0.0))
        frame_count = _safe_int(getattr(video, "frames", 0))
        items.append(
            {
                "index": index,
                "kind": "image_sequence" if image_filenames else "video_file",
                "path": _summary_media_path(raw_path, root),
                "backend": str(video.backend or ""),
                "video_id": str(video.id or f"video_{index}"),
                "label": str(video.label or fallback_label),
                "frame_count": frame_count,
                "fps": fps,
                "duration_s": _duration_seconds(frame_count, fps),
                "timebase": "frame_index" if fps > 0.0 else None,
                "height": _safe_int(getattr(video, "height", 0)),
                "width": _safe_int(getattr(video, "width", 0)),
                "channels": _safe_int(getattr(video, "channels", 0)),
                "image_count": len(image_filenames),
                "label_frame_count": int(label_entry.get("count", 0) or 0),
                "max_label_frame_index": label_entry.get("max_frame_index"),
                "prediction_frame_count": int(prediction_entry.get("count", 0) or 0),
                "max_prediction_frame_index": prediction_entry.get("max_frame_index"),
            }
        )
    return tuple(items)


def vicon_state_summary(recording: ViconRecording) -> dict[str, JsonScalar]:
    """Return shallow counts from an in-memory Vicon recording during import."""

    return {
        "source_type": recording.source_type,
        "fps": int(recording.fps),
        "frame_count": int(recording.positions.shape[0]),
        "marker_count": len(recording.marker_names),
        "event_count": len(recording.events),
        "camera_count": len(recording.cameras),
        "has_analog": recording.analog is not None,
        "has_force_platform": recording.force_platform is not None,
    }


def load_project_summary(project: str | Path) -> ProjectSummaryIndex:
    """Load the generated project summary index."""

    return ProjectSummaryIndex.from_dict(load_json_dict(project_summary_path(project)))


def snapshot_project_summary(
    project: str | Path,
    *,
    state_summary: Mapping[str, JsonScalar] | None = None,
    media_summary: Sequence[Mapping[str, JsonScalar]] | None = None,
) -> ProjectSummaryIndex:
    """Build the generated shallow summary index without writing it."""

    project_root = resolve_project_root(project)
    if project_root is None:
        raise FileNotFoundError(f"Not an xpkg project: {project}")
    descriptor = load_project_descriptor(project_root)
    state = _state_source(project_root)
    existing, existing_warnings = _existing_summary(project_root)
    state_details = _state_details(state, existing, state_summary)
    media_details = _media_details(state, existing, media_summary)
    artifact_count, artifact_types, artifact_warnings = _artifact_inventory(project_root)
    metadata_slots = _metadata_slots(project_root)
    warnings = (*state.warnings, *existing_warnings, *artifact_warnings)
    summary = ProjectSummaryIndex(
        schema_version=PROJECT_SUMMARY_SCHEMA_VERSION,
        generated_at=_now_utc_iso(),
        project_id=descriptor.project_id,
        title=descriptor.title,
        descriptor_updated_at=descriptor.updated_at,
        state_kind=state.kind,
        has_current_state=state.has_current_state,
        state_bytes=state.bytes,
        commit_id=state.commit_id,
        state_summary=state_details,
        media=media_details,
        metadata_slots=metadata_slots,
        artifact_count=artifact_count,
        artifact_types=artifact_types,
        modalities=_modalities(state.kind, state_details, metadata_slots, artifact_count),
        warnings=warnings,
    )
    return summary


def refresh_project_summary(
    project: str | Path,
    *,
    state_summary: Mapping[str, JsonScalar] | None = None,
    media_summary: Sequence[Mapping[str, JsonScalar]] | None = None,
) -> ProjectSummaryIndex:
    """Refresh and write the generated shallow summary index."""

    project_root = resolve_project_root(project)
    if project_root is None:
        raise FileNotFoundError(f"Not an xpkg project: {project}")
    existing, _existing_warnings = _existing_summary(project_root)
    summary = snapshot_project_summary(
        project_root,
        state_summary=state_summary,
        media_summary=media_summary,
    )
    if existing is not None and _equivalent_summary(existing, summary):
        return existing
    target = project_summary_path(project_root)
    ensure_dir(target.parent)
    write_json(target, summary.to_dict(), indent=2, sort_keys=False, ensure_ascii=True)
    return summary


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return cast("Mapping[str, Any]", value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(cast("str | bytes | bytearray | int | float | bool", value))


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _state_kind(value: str) -> ProjectSummaryStateKind:
    if value in {"empty", "labels", "vicon", "present", "unreadable"}:
        return cast("ProjectSummaryStateKind", value)
    raise ValueError(f"Unsupported project summary state kind: {value!r}")


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise TypeError("Project summary value must be a list of strings")
    return tuple(str(item) for item in value)


def _scalar_dict(value: object) -> dict[str, JsonScalar]:
    payload = _mapping(value, name="project_summary.scalar_dict")
    out: dict[str, JsonScalar] = {}
    for key, item in payload.items():
        if item is not None and not isinstance(item, str | int | float | bool):
            continue
        out[str(key)] = item
    return out


def _media_items(value: object) -> tuple[MediaSummaryItem, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise TypeError("Project summary media items must be a list")
    return tuple(_scalar_dict(item) for item in value)


def _int_dict(value: object) -> dict[str, int]:
    payload = _mapping(value, name="project_summary.int_dict")
    return {str(key): int(item) for key, item in payload.items()}


def _prediction_frame_count(predictions: Mapping[str, Any] | None) -> int:
    if predictions is None:
        return 0
    attrs = predictions.get("attrs")
    if not isinstance(attrs, Mapping):
        return 0
    return int(attrs.get("committed_length", 0) or 0)


def _equivalent_summary(
    existing: ProjectSummaryIndex,
    candidate: ProjectSummaryIndex,
) -> bool:
    existing_payload = existing.to_dict()
    candidate_payload = candidate.to_dict()
    existing_payload["generated_at"] = ""
    candidate_payload["generated_at"] = ""
    return existing_payload == candidate_payload


def _existing_summary(project_root: Path) -> tuple[ProjectSummaryIndex | None, tuple[str, ...]]:
    path = project_summary_path(project_root)
    if not path.is_file():
        return None, ()
    try:
        return load_project_summary(project_root), ()
    except (OSError, TypeError, ValueError) as exc:
        return None, (f"Could not read previous project summary: {exc}",)


def _state_details(
    state: _StateSource,
    existing: ProjectSummaryIndex | None,
    state_summary: Mapping[str, JsonScalar] | None,
) -> dict[str, JsonScalar]:
    if state_summary is not None:
        return _scalar_dict(dict(state_summary))
    if existing is None:
        return {}
    if existing.state_kind == state.kind:
        return dict(existing.state_summary)
    return {}


def _media_details(
    state: _StateSource,
    existing: ProjectSummaryIndex | None,
    media_summary: Sequence[Mapping[str, JsonScalar]] | None,
) -> tuple[MediaSummaryItem, ...]:
    if media_summary is not None:
        return _media_items(tuple(dict(item) for item in media_summary))
    if existing is None:
        return ()
    if existing.state_kind == state.kind:
        return existing.media
    return ()


def _state_source(project_root: Path) -> _StateSource:
    commit_id, state_path, warnings = _durable_state(project_root)
    if state_path is None:
        cache_path = project_current_state_path(project_root)
        state_path = cache_path if cache_path.exists() else None
    if state_path is None:
        return _StateSource("empty", False, None, commit_id, tuple(warnings))
    try:
        state_bytes = int(state_path.stat().st_size)
        return _StateSource(
            _state_kind_from_prefix(state_path),
            True,
            state_bytes,
            commit_id,
            tuple(warnings),
        )
    except OSError as exc:
        warnings.append(f"Could not read current project state summary: {exc}")
        return _StateSource("unreadable", True, None, commit_id, tuple(warnings))


def _durable_state(project_root: Path) -> tuple[str | None, Path | None, list[str]]:
    store_root = project_store_root(project_root)
    warnings: list[str] = []
    if not (store_root / "superblock.a.json").exists() and not (
        store_root / "superblock.b.json"
    ).exists():
        return None, None, warnings
    try:
        store = ProjectDurableStore.open(store_root)
        commit = store.load_current_commit()
        if not commit.has_root("state"):
            return commit.commit_id, None, warnings
        return commit.commit_id, store.current_root_path("state"), warnings
    except (OSError, ProjectDurableStoreError) as exc:
        warnings.append(f"Could not read durable project head: {exc}")
        return None, None, warnings


def _state_kind_from_prefix(state_path: Path) -> ProjectSummaryStateKind:
    with state_path.open("rb") as handle:
        prefix = handle.read(_STATE_PREFIX_BYTES).decode("utf-8", errors="replace")
    if XPKG_LABELS_JSON_FORMAT in prefix:
        return "labels"
    if XPKG_VICON_JSON_FORMAT in prefix:
        return "vicon"
    return "present"


def _metadata_slots(project_root: Path) -> tuple[str, ...]:
    metadata_root = project_store_root(project_root) / "metadata"
    return tuple(
        slot
        for slot, filename in _METADATA_SLOT_FILES.items()
        if (metadata_root / filename).is_file()
    )


def _artifact_inventory(project_root: Path) -> tuple[int, dict[str, int], tuple[str, ...]]:
    index_path = project_artifacts_root(project_root) / _ARTIFACT_INDEX_FILENAME
    if not index_path.is_file():
        return 0, {}, ()
    try:
        payload = load_json_dict(index_path)
        raw_entries = payload.get("artifacts", []) or []
        entries = _artifact_entries(raw_entries)
    except (OSError, TypeError, ValueError) as exc:
        return 0, {}, (f"Could not read artifact index summary: {exc}",)
    artifact_types: dict[str, int] = {}
    for entry in entries:
        artifact_type = str(entry.get("artifact_type", "")).strip()
        if artifact_type:
            artifact_types[artifact_type] = artifact_types.get(artifact_type, 0) + 1
    return len(entries), artifact_types, ()


def _artifact_entries(raw_entries: object) -> list[Mapping[str, Any]]:
    if not isinstance(raw_entries, Sequence) or isinstance(
        raw_entries, str | bytes | bytearray
    ):
        raise TypeError("Artifact index artifacts must be a list")
    entries: list[Mapping[str, Any]] = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, Mapping):
            raise TypeError("Artifact index entries must be objects")
        entries.append(cast("Mapping[str, Any]", raw_entry))
    return entries


def _modalities(
    state_kind: ProjectSummaryStateKind,
    state_summary: Mapping[str, JsonScalar],
    metadata_slots: Sequence[str],
    artifact_count: int,
) -> tuple[str, ...]:
    modalities: list[str] = []
    if state_kind == "labels":
        modalities.append("labels")
        if int(state_summary.get("prediction_frame_count") or 0) > 0:
            modalities.append("pose_predictions")
    elif state_kind == "vicon":
        modalities.append("motion")
    if metadata_slots:
        modalities.append("project_metadata")
    if artifact_count > 0:
        modalities.append("artifacts")
    return tuple(modalities)


def _safe_int(value: object) -> int:
    try:
        return int(cast("str | bytes | bytearray | int | float | bool", value))
    except (TypeError, ValueError):
        return 0


def _safe_float(value: object) -> float:
    try:
        return float(cast("str | bytes | bytearray | int | float | bool", value))
    except (TypeError, ValueError):
        return 0.0


def _duration_seconds(frame_count: int, fps: float) -> float | None:
    if frame_count <= 0 or fps <= 0.0:
        return None
    return float(frame_count / fps)


def _summary_media_path(raw_path: str, project_root: Path | None) -> str:
    path_text = str(raw_path).strip()
    if not path_text:
        return ""
    if project_root is None:
        return path_text
    path = Path(path_text)
    resolved = path.resolve() if path.is_absolute() else (project_root / path).resolve()
    try:
        return resolved.relative_to(project_root).as_posix()
    except ValueError:
        return path_text


def _frame_stats_by_video(
    labeled_frames: Sequence[Any],
    video_lookup: Mapping[object, int],
) -> dict[int, dict[str, JsonScalar]]:
    stats: dict[int, dict[str, JsonScalar]] = {}
    for frame in labeled_frames:
        video_index = video_lookup.get(getattr(frame, "video", None))
        if video_index is None or not getattr(frame, "user_instances", ()):
            continue
        frame_index = _safe_int(getattr(frame, "frame_idx", 0))
        entry = stats.setdefault(video_index, {"count": 0, "max_frame_index": None})
        entry["count"] = int(entry.get("count", 0) or 0) + 1
        current_max = entry.get("max_frame_index")
        if current_max is None or frame_index > int(current_max):
            entry["max_frame_index"] = frame_index
    return stats


def _prediction_stats_by_video(
    predictions: Mapping[str, Any] | None,
) -> dict[int, dict[str, JsonScalar]]:
    if predictions is None:
        return {}
    frames = predictions.get("frames")
    if not isinstance(frames, Mapping):
        return {}
    video_indices = _json_sequence(frames.get("video_index"))
    frame_indices = _json_sequence(frames.get("frame_index"))
    stats: dict[int, dict[str, JsonScalar]] = {}
    for raw_video, raw_frame in zip(video_indices, frame_indices, strict=False):
        video_index = _safe_int(raw_video)
        frame_index = _safe_int(raw_frame)
        entry = stats.setdefault(video_index, {"count": 0, "max_frame_index": None})
        entry["count"] = int(entry.get("count", 0) or 0) + 1
        current_max = entry.get("max_frame_index")
        if current_max is None or frame_index > int(current_max):
            entry["max_frame_index"] = frame_index
    return stats


def _json_sequence(value: object) -> list[object]:
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        return []
    return list(value)


__all__ = [
    "PROJECT_SUMMARY_SCHEMA_VERSION",
    "ProjectSummaryIndex",
    "ProjectSummaryStateKind",
    "labels_state_summary",
    "labels_media_summary",
    "load_project_summary",
    "refresh_project_summary",
    "snapshot_project_summary",
    "vicon_state_summary",
]
