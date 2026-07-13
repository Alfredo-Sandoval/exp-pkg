"""Generated shallow project inventory summary.

The summary index is the cheap project-row contract behind ``PROJECT.json``.
It records enough inventory to route and display projects without loading
labels, predictions, dense masks, or media payloads.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from xpkg._core.json_utils import load_json_dict, write_json
from xpkg._core.path_registry import ensure_dir
from xpkg._core.time import now_utc_iso
from xpkg.io.experiment_json import EXPERIMENT_FORMAT
from xpkg.io.labels.model import Labels
from xpkg.model.experiment import Experiment
from xpkg.model.signals import PhotometryRecording
from xpkg.project.durable_store import ProjectDurableStore, ProjectDurableStoreError
from xpkg.project.layout import (
    load_project_descriptor,
    project_artifacts_root,
    project_current_state_path,
    project_store_root,
    project_summary_path,
    resolve_project_root,
)

if TYPE_CHECKING:
    from xpkg.model import RecordingSession


PROJECT_SUMMARY_SCHEMA_VERSION = 2
_STATE_PREFIX_BYTES = 8192
_ARTIFACT_INDEX_FILENAME = "index.json"
_METADATA_SLOT_FILES = {
    "datasheet": "datasheet.json",
    "model-card": "model_card.json",
}

ProjectSummaryStateKind = Literal["empty", "experiment", "present", "unreadable"]
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
    session_ids: tuple[str, ...] = ()
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
            session_ids=_string_tuple(data.get("sessions") or ()),
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
            "sessions": list(self.session_ids),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class _StateSource:
    kind: ProjectSummaryStateKind
    has_current_state: bool
    bytes: int | None
    commit_id: str | None
    warnings: tuple[str, ...]


def _recording_state_summary(session: RecordingSession) -> dict[str, JsonScalar]:
    """Return shallow counts from one in-memory recording session."""
    series = [
        recording.series if isinstance(recording, PhotometryRecording) else recording
        for recording in (link.recording for link in session.signals)
    ]
    time_range = session.time_range
    pose_labels = [link.data for link in session.poses if isinstance(link.data, Labels)]
    trajectories = [
        link.data for link in session.poses if not isinstance(link.data, Labels)
    ]
    return {
        "signal_count": len(session.signals),
        "channel_count": sum(item.n_channels for item in series),
        "sample_count": sum(item.n_samples for item in series),
        "video_count": len(session.videos),
        "event_count": len(session.events),
        "pose_count": len(session.poses),
        "label_frame_count": sum(len(labels.user_labeled_frames) for labels in pose_labels),
        "prediction_frame_count": sum(
            sum(bool(frame.predicted_instances) for frame in labels.labeled_frames)
            for labels in pose_labels
        ),
        "trajectory_frame_count": sum(item.n_frames for item in trajectories),
        "behavior_count": len(session.behaviors),
        "behavior_interval_count": sum(
            len(link.labels.intervals) for link in session.behaviors
        ),
        "calibration_count": len(session.calibrations),
        "alignment_count": len(session.alignments),
        "start_s": None if time_range is None else time_range.start_s,
        "end_s": None if time_range is None else time_range.end_s,
    }


def experiment_state_summary(experiment: Experiment) -> dict[str, JsonScalar]:
    """Return aggregate shallow counts for an experiment and all its sessions."""
    session_summaries = [_recording_state_summary(session) for session in experiment.sessions]
    count_keys = (
        "signal_count",
        "channel_count",
        "sample_count",
        "video_count",
        "event_count",
        "pose_count",
        "label_frame_count",
        "prediction_frame_count",
        "trajectory_frame_count",
        "behavior_count",
        "behavior_interval_count",
        "calibration_count",
        "alignment_count",
    )
    starts = [item["start_s"] for item in session_summaries if item["start_s"] is not None]
    ends = [item["end_s"] for item in session_summaries if item["end_s"] is not None]
    summary: dict[str, JsonScalar] = {
        "experiment_id": experiment.experiment_id,
        "subject_count": len(experiment.subjects),
        "protocol_count": len(experiment.protocols),
        "condition_count": len(experiment.conditions),
        "acquisition_session_count": sum(
            session.acquisition is not None for session in experiment.sessions
        ),
        "has_dataset_share": experiment.dataset_share is not None,
        "session_count": len(experiment.sessions),
        "start_s": min(starts) if starts else None,
        "end_s": max(ends) if ends else None,
    }
    summary.update(
        {key: sum(int(item[key] or 0) for item in session_summaries) for key in count_keys}
    )
    return summary


def _recording_media_summary(
    session: RecordingSession,
    *,
    project_root: str | Path,
) -> tuple[MediaSummaryItem, ...]:
    """Return project-relative video inventory from a recording session."""
    root = Path(project_root).resolve()
    items: list[MediaSummaryItem] = []
    for index, video in enumerate(session.videos):
        resolved = (root / video.path).resolve()
        frame_count = 0 if video.frame_count is None else video.frame_count
        frame_rate_hz = 0.0 if video.frame_rate_hz is None else video.frame_rate_hz
        items.append(
            {
                "index": index,
                "kind": "image_sequence" if resolved.is_dir() else "video_file",
                "path": video.path.as_posix(),
                "role": video.role,
                "video_id": str(video.metadata.get("video_id", "")),
                "label": str(video.metadata.get("label", video.path.name)),
                "backend": str(video.metadata.get("backend", "")),
                "frame_count": frame_count,
                "fps": frame_rate_hz,
                "duration_s": _duration_seconds(frame_count, frame_rate_hz),
                "timebase": video.timebase.name,
                "height": int(video.metadata.get("height", 0)),
                "width": int(video.metadata.get("width", 0)),
                "channels": int(video.metadata.get("channels", 0)),
                "image_count": int(video.metadata.get("image_count", 0)),
                "label_frame_count": int(video.metadata.get("label_frame_count", 0)),
                "max_label_frame_index": video.metadata.get("max_label_frame_index"),
                "prediction_frame_count": int(
                    video.metadata.get("prediction_frame_count", 0)
                ),
                "max_prediction_frame_index": video.metadata.get(
                    "max_prediction_frame_index"
                ),
                "exists": resolved.exists(),
                "size_bytes": resolved.stat().st_size if resolved.is_file() else None,
            }
        )
    return tuple(items)


def experiment_media_summary(
    experiment: Experiment,
    *,
    project_root: str | Path,
) -> tuple[MediaSummaryItem, ...]:
    """Return a project-relative video inventory across all sessions."""
    items: list[MediaSummaryItem] = []
    for session in experiment.sessions:
        for item in _recording_media_summary(session, project_root=project_root):
            item["index"] = len(items)
            item["session_id"] = session.session_id
            items.append(item)
    return tuple(items)


def load_project_summary(project: str | Path) -> ProjectSummaryIndex:
    """Load the generated project summary index."""

    return ProjectSummaryIndex.from_dict(load_json_dict(project_summary_path(project)))


def snapshot_project_summary(
    project: str | Path,
    *,
    state_summary: Mapping[str, JsonScalar] | None = None,
    media_summary: Sequence[Mapping[str, JsonScalar]] | None = None,
    session_ids: Sequence[str] | None = None,
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
        generated_at=now_utc_iso(drop_microseconds=True),
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
        session_ids=_session_ids(state, existing, session_ids),
        warnings=warnings,
    )
    return summary


def refresh_project_summary(
    project: str | Path,
    *,
    state_summary: Mapping[str, JsonScalar] | None = None,
    media_summary: Sequence[Mapping[str, JsonScalar]] | None = None,
    session_ids: Sequence[str] | None = None,
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
        session_ids=session_ids,
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
    if value in {"empty", "experiment", "present", "unreadable"}:
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


def _session_ids(
    state: _StateSource,
    existing: ProjectSummaryIndex | None,
    session_ids: Sequence[str] | None,
) -> tuple[str, ...]:
    if session_ids is not None:
        return tuple(str(session_id) for session_id in session_ids)
    if existing is not None and existing.state_kind == state.kind:
        return existing.session_ids
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
    if (
        not (store_root / "superblock.a.json").exists()
        and not (store_root / "superblock.b.json").exists()
    ):
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
    if EXPERIMENT_FORMAT in prefix:
        return "experiment"
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
    if not isinstance(raw_entries, Sequence) or isinstance(raw_entries, str | bytes | bytearray):
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
    if state_kind == "experiment":
        if int(state_summary.get("acquisition_session_count") or 0) > 0:
            modalities.append("acquisition")
        if int(state_summary.get("signal_count") or 0) > 0:
            modalities.append("signals")
        if int(state_summary.get("event_count") or 0) > 0:
            modalities.append("events")
        if int(state_summary.get("video_count") or 0) > 0:
            modalities.append("videos")
        if int(state_summary.get("pose_count") or 0) > 0:
            modalities.append("pose")
        if int(state_summary.get("behavior_count") or 0) > 0:
            modalities.append("behavior")
        if int(state_summary.get("calibration_count") or 0) > 0:
            modalities.append("calibration")
        if int(state_summary.get("alignment_count") or 0) > 0:
            modalities.append("synchronization")
        if bool(state_summary.get("has_dataset_share")):
            modalities.append("dataset_share")
    if metadata_slots:
        modalities.append("project_metadata")
    if artifact_count > 0:
        modalities.append("artifacts")
    return tuple(modalities)


def _duration_seconds(frame_count: int, fps: float) -> float | None:
    if frame_count <= 0 or fps <= 0.0:
        return None
    return float(frame_count / fps)




__all__ = [
    "PROJECT_SUMMARY_SCHEMA_VERSION",
    "ProjectSummaryIndex",
    "ProjectSummaryStateKind",
    "experiment_media_summary",
    "experiment_state_summary",
    "load_project_summary",
    "refresh_project_summary",
    "snapshot_project_summary",
]
