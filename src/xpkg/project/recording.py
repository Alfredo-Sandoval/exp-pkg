"""Governed project actions for experiment and recording-session state."""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from xpkg._core.hashing import sha256_file
from xpkg._core.path_registry import ensure_dir
from xpkg.io.experiment_json import read_experiment_json, write_experiment_json
from xpkg.io.readers import (
    read_behavior_events_csv,
    read_behavior_events_json,
    read_boris_csv,
    read_bsoid_csv,
    read_events_csv,
    read_keypoint_moseq_syllables_csv,
    read_photometry_csv,
    read_simba_csv,
    read_synchronization_csv,
)
from xpkg.model.behavior import BehaviorLabels
from xpkg.model.events import EventTable
from xpkg.model.experiment import Experiment
from xpkg.model.experiment_actions import (
    add_experiment_session,
    replace_experiment_dataset_share,
    replace_experiment_metadata,
    replace_experiment_session,
)
from xpkg.model.metadata import (
    AcquisitionMetadata,
    DatasetShareMetadata,
    PoseModelProvenance,
    SourceProvenance,
)
from xpkg.model.session import (
    AlignmentModel,
    RecordingSession,
    SessionBehavior,
    SessionEventStream,
    SessionPose,
    SessionSignal,
    SessionVideo,
    SynchronizationMethod,
    TimebaseAlignment,
)
from xpkg.model.session_actions import (
    add_session_behavior,
    add_session_event_stream,
    add_session_pose,
    add_session_signal,
    add_timebase_alignment,
    replace_session_acquisition,
    replace_session_behavior,
    replace_session_event_stream,
    replace_session_pose,
    replace_session_signal,
    replace_timebase_alignment,
)
from xpkg.model.time import Timebase
from xpkg.pose.trajectory import PoseTrajectory
from xpkg.project.layout import (
    CURRENT_STATE_FILENAME,
    load_project_descriptor,
    project_current_state_path,
    project_exports_root,
    project_media_root,
    project_store_root,
    resolve_project_root,
)
from xpkg.project.state import PROJECT_COMMIT_ID_KEY, project_state_kind
from xpkg.project.store._helpers import (
    _ensure_project_for_import,
    _project_store,
    _stage_project_parent,
    _touch_descriptor,
)
from xpkg.project.store.media import _copy_file_into_media, _manage_labels_media


def save_project_experiment(
    project: str | Path,
    experiment: Experiment,
    *,
    reason: str = "project.save.experiment",
) -> Path:
    """Commit one experiment as the project's canonical state."""
    root = resolve_project_root(project)
    if root is None:
        raise FileNotFoundError(f"Not an xpkg project: {project}")
    if not isinstance(experiment, Experiment):
        raise TypeError(f"experiment must be an Experiment, got {experiment!r}.")
    descriptor = load_project_descriptor(root)
    if experiment.experiment_id != descriptor.project_id:
        raise ValueError(
            f"Experiment id {experiment.experiment_id!r} does not match project id "
            f"{descriptor.project_id!r}."
        )
    for session in experiment.sessions:
        for pose in session.poses:
            from xpkg.model.labels import Labels

            if isinstance(pose.data, Labels):
                _manage_labels_media(pose.data, root)
        _require_portable_session_paths(session, root)
        _require_portable_source_paths(session, root)
    ensure_dir(project_store_root(root))
    ensure_dir(project_media_root(root))
    ensure_dir(project_exports_root(root))

    stage_parent = _stage_project_parent(root)
    with tempfile.TemporaryDirectory(prefix=".experiment_commit_", dir=stage_parent) as tmp_dir:
        staged_state = Path(tmp_dir) / CURRENT_STATE_FILENAME
        write_experiment_json(staged_state, experiment, project_root=root)
        store = _project_store(root)
        store.commit_state(staged_state, reason=reason)
        commit_id = store.current_commit_id()
    current_state = project_current_state_path(root)
    ensure_dir(current_state.parent)
    state_path = write_experiment_json(
        current_state,
        experiment,
        document_metadata={PROJECT_COMMIT_ID_KEY: commit_id},
        project_root=root,
    )

    from xpkg.project.summary import experiment_media_summary, experiment_state_summary

    _touch_descriptor(
        root,
        state_summary=experiment_state_summary(experiment),
        media_summary=experiment_media_summary(experiment, project_root=root),
        session_ids=experiment.session_ids,
    )
    return state_path


def load_project_experiment(project: str | Path) -> Experiment:
    """Load the canonical experiment ontology object for a project."""
    root = resolve_project_root(project)
    if root is None:
        raise FileNotFoundError(f"Not an xpkg project: {project}")
    store = _project_store(root)
    if not store.has_current_state():
        raise FileNotFoundError(f"Project has no committed state: {root}")
    from xpkg.project.store.cache import ensure_current_project_state_cache

    ensure_current_project_state_cache(root)
    state_path = store.current_state_path()
    state_kind = project_state_kind(state_path)
    if state_kind != "experiment":
        raise ValueError(f"Unsupported project state kind: {state_kind!r}.")
    experiment = read_experiment_json(state_path, project_root=root)
    descriptor = load_project_descriptor(root)
    if experiment.experiment_id != descriptor.project_id:
        raise ValueError(
            f"Stored experiment id {experiment.experiment_id!r} does not match project id "
            f"{descriptor.project_id!r}."
        )
    return experiment


def save_project_session(
    project: str | Path,
    session: RecordingSession,
    *,
    reason: str = "project.save.session",
) -> Path:
    """Add or replace one recording session in the project's experiment."""
    if not isinstance(session, RecordingSession):
        raise TypeError(f"session must be a RecordingSession, got {session!r}.")
    experiment = _load_or_create_experiment(project)
    if session.session_id in experiment.session_ids:
        experiment = replace_experiment_session(experiment, session)
    else:
        experiment = add_experiment_session(experiment, session)
    return save_project_experiment(project, experiment, reason=reason)


def load_project_session(project: str | Path, *, session_id: str | None = None) -> RecordingSession:
    """Load a named session, or the sole session when selection is unambiguous."""
    experiment = load_project_experiment(project)
    if session_id is not None:
        return experiment.session(session_id)
    if len(experiment.sessions) == 1:
        return experiment.sessions[0]
    if not experiment.sessions:
        raise FileNotFoundError(
            f"Experiment {experiment.experiment_id!r} has no recording sessions."
        )
    raise ValueError(
        f"Experiment {experiment.experiment_id!r} has {len(experiment.sessions)} sessions; "
        "session_id is required."
    )


def save_project_acquisition(
    project: str | Path,
    acquisition: AcquisitionMetadata,
    *,
    session_id: str | None = None,
) -> Path:
    """Commit acquisition context to one recording session."""
    if not isinstance(acquisition, AcquisitionMetadata):
        raise TypeError("acquisition must be AcquisitionMetadata.")
    experiment = _load_or_create_experiment(project)
    session = _select_or_create_session(experiment, requested_session_id=session_id)
    session = replace_session_acquisition(session, acquisition)
    return save_project_session(project, session, reason="project.save.acquisition")


def load_project_acquisition(
    project: str | Path, *, session_id: str | None = None
) -> AcquisitionMetadata | None:
    """Load acquisition context from one unambiguously selected session."""
    root = resolve_project_root(project)
    if root is None:
        raise FileNotFoundError(f"Not an xpkg project: {project}")
    if not _project_store(root).has_current_state():
        return None
    return load_project_session(project, session_id=session_id).acquisition


def save_project_dataset_share(
    project: str | Path,
    dataset_share: DatasetShareMetadata,
) -> Path:
    """Commit dataset-sharing metadata to the experiment aggregate."""
    if not isinstance(dataset_share, DatasetShareMetadata):
        raise TypeError("dataset_share must be DatasetShareMetadata.")
    experiment = _load_or_create_experiment(project)
    return save_project_experiment(
        project,
        replace_experiment_dataset_share(experiment, dataset_share),
        reason="project.save.dataset_share",
    )


def load_project_dataset_share(project: str | Path) -> DatasetShareMetadata | None:
    """Load the experiment's dataset-sharing metadata."""
    root = resolve_project_root(project)
    if root is None:
        raise FileNotFoundError(f"Not an xpkg project: {project}")
    if not _project_store(root).has_current_state():
        return None
    return load_project_experiment(project).dataset_share


def import_photometry_csv_project(
    path: str | Path,
    *,
    project: str | Path,
    session_id: str | None = None,
    signal_name: str = "photometry",
    force: bool = False,
) -> Path:
    """Import one photometry CSV through the canonical project action layer."""
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Photometry CSV not found: {source}")
    root = _ensure_project_for_import(project, title=session_id or source.stem, force=force)
    managed_source = _copy_import_source(source, root, media_kind="signals")
    recording = read_photometry_csv(managed_source, name=signal_name)
    recording = replace(
        recording,
        series=replace(
            recording.series,
            provenance=_without_source(recording.series.provenance),
        ),
        metadata=_without_source(recording.metadata),
    )
    provenance = _portable_source(managed_source, root, source_type="photometry_csv")
    experiment = _load_or_create_experiment(root)
    session = _select_or_create_session(
        experiment,
        requested_session_id=session_id,
        fallback_session_id=source.stem,
    )
    link = SessionSignal(
        name=signal_name,
        recording=recording,
        provenance=provenance,
    )
    if signal_name in session.signal_names:
        if not force:
            raise FileExistsError(
                f"Recording session already has signal {signal_name!r}. "
                "Pass force=True to replace it."
            )
        session = replace_session_signal(session, link)
    else:
        session = add_session_signal(session, link)
    return save_project_session(root, session, reason="project.import.photometry_csv")


def import_events_csv_project(
    path: str | Path,
    *,
    project: str | Path,
    event_stream_name: str = "events",
    session_id: str | None = None,
    force: bool = False,
) -> Path:
    """Import one event CSV as a named session event stream."""
    source = _require_source_file(path, role="Event CSV")
    root = _ensure_project_for_import(project, title=session_id or source.stem, force=force)
    experiment = _load_or_create_experiment(root)
    session = _select_or_create_session(
        experiment,
        requested_session_id=session_id,
        fallback_session_id=source.stem,
    )
    existing = {stream.name for stream in session.event_streams}
    if event_stream_name in existing and not force:
        raise FileExistsError(
            f"Recording session {session.session_id!r} already has event stream "
            f"{event_stream_name!r}. Pass force=True to replace it."
        )
    events = read_events_csv(source)
    if len(events) == 0:
        raise ValueError(f"Event CSV contains no events: {source}")
    managed_source = _copy_import_source(source, root, media_kind="events")
    events = _without_source_metadata(events)
    stream = SessionEventStream(
        name=event_stream_name,
        events=events,
        provenance=_portable_source(managed_source, root, source_type="events_csv"),
    )
    if event_stream_name in existing:
        session = replace_session_event_stream(session, stream)
    else:
        session = add_session_event_stream(session, stream)
    return save_project_session(root, session, reason="project.import.events_csv")


def import_behavior_project(
    format: str,
    path: str | Path,
    *,
    project: str | Path,
    behavior_name: str = "behavior",
    session_id: str | None = None,
    video_roles: tuple[str, ...] = (),
    pose_names: tuple[str, ...] = (),
    force: bool = False,
) -> Path:
    """Import behavior labels through one canonical format dispatch boundary."""
    source = _require_source_file(path, role="Behavior source")
    root = _ensure_project_for_import(project, title=session_id or source.stem, force=force)
    experiment = _load_or_create_experiment(root)
    session = _select_or_create_session(
        experiment,
        requested_session_id=session_id,
        fallback_session_id=source.stem,
    )
    existing = {link.name for link in session.behaviors}
    if behavior_name in existing and not force:
        raise FileExistsError(
            f"Recording session {session.session_id!r} already has behavior "
            f"{behavior_name!r}. Pass force=True to replace it."
        )
    behavior_videos, behavior_poses = _resolve_behavior_links(
        session, video_roles=video_roles, pose_names=pose_names
    )
    labels = _read_behavior_source(format, source)
    managed_source = _copy_import_source(source, root, media_kind="behavior")
    labels = _portable_behavior_labels(
        labels,
        managed_source,
        root,
        videos=behavior_videos,
    )
    link = SessionBehavior(
        name=behavior_name,
        labels=labels,
        videos=behavior_videos,
        poses=behavior_poses,
        provenance=_portable_source(managed_source, root, source_type=labels.source_type),
    )
    if behavior_name in existing:
        session = replace_session_behavior(session, link)
    else:
        session = add_session_behavior(session, link)
    return save_project_session(root, session, reason=f"project.import.behavior.{format}")


def import_synchronization_csv_project(
    path: str | Path,
    *,
    project: str | Path,
    source_timebase: Timebase,
    target_timebase: Timebase,
    model: AlignmentModel,
    method: SynchronizationMethod,
    alignment_name: str | None = None,
    session_id: str | None = None,
    force: bool = False,
) -> Path:
    """Import paired clock observations as one typed timebase alignment."""
    source = _require_source_file(path, role="Synchronization CSV")
    root = _ensure_project_for_import(project, title=session_id or source.stem, force=force)
    experiment = _load_or_create_experiment(root)
    session = _select_or_create_session(
        experiment,
        requested_session_id=session_id,
        fallback_session_id=source.stem,
    )
    name = alignment_name or f"{source_timebase.name}-to-{target_timebase.name}"
    existing = {item.name for item in session.alignments}
    if name in existing and not force:
        raise FileExistsError(
            f"Recording session {session.session_id!r} already has alignment {name!r}. "
            "Pass force=True to replace it."
        )
    alignment = read_synchronization_csv(
        source,
        source_timebase=source_timebase,
        target_timebase=target_timebase,
        model=model,
        method=method,
        name=name,
    )
    managed_source = _copy_import_source(source, root, media_kind="synchronization")
    alignment = _portable_timebase_alignment(alignment, managed_source, root)
    if name in existing:
        session = replace_timebase_alignment(session, alignment)
    else:
        session = add_timebase_alignment(session, alignment)
    return save_project_session(root, session, reason="project.import.synchronization_csv")


def save_project_event_stream(
    project: str | Path,
    stream: SessionEventStream,
    *,
    session_id: str | None = None,
    replace_existing: bool = True,
) -> Path:
    """Add or replace one named event stream on a recording session."""
    if not isinstance(stream, SessionEventStream):
        raise TypeError("stream must be a SessionEventStream.")
    experiment = _load_or_create_experiment(project)
    session = _select_or_create_session(experiment, requested_session_id=session_id)
    existing = {item.name for item in session.event_streams}
    if stream.name in existing:
        if not replace_existing:
            raise FileExistsError(f"Recording session already has event stream {stream.name!r}.")
        session = replace_session_event_stream(session, stream)
    else:
        session = add_session_event_stream(session, stream)
    return save_project_session(project, session, reason="project.save.event_stream")


def load_project_event_stream(
    project: str | Path,
    *,
    event_stream_name: str,
    session_id: str | None = None,
) -> EventTable:
    """Load one named event table from a recording session."""
    session = load_project_session(project, session_id=session_id)
    try:
        return session.event_stream(event_stream_name)
    except KeyError as exc:
        raise FileNotFoundError(
            f"Recording session {session.session_id!r} has no event stream {event_stream_name!r}."
        ) from exc


def save_project_behavior(
    project: str | Path,
    behavior: SessionBehavior,
    *,
    session_id: str | None = None,
    replace_existing: bool = True,
) -> Path:
    """Add or replace one typed behavior-label link on a recording session."""
    if not isinstance(behavior, SessionBehavior):
        raise TypeError("behavior must be a SessionBehavior.")
    experiment = _load_or_create_experiment(project)
    session = _select_or_create_session(experiment, requested_session_id=session_id)
    existing = {item.name for item in session.behaviors}
    if behavior.name in existing:
        if not replace_existing:
            raise FileExistsError(f"Recording session already has behavior {behavior.name!r}.")
        session = replace_session_behavior(session, behavior)
    else:
        session = add_session_behavior(session, behavior)
    return save_project_session(project, session, reason="project.save.behavior")


def load_project_behavior(
    project: str | Path,
    *,
    behavior_name: str = "behavior",
    session_id: str | None = None,
) -> BehaviorLabels:
    """Load one named behavior-label link from a recording session."""
    session = load_project_session(project, session_id=session_id)
    try:
        return session.behavior(behavior_name)
    except KeyError as exc:
        raise FileNotFoundError(
            f"Recording session {session.session_id!r} has no behavior {behavior_name!r}."
        ) from exc


def save_project_alignment(
    project: str | Path,
    alignment: TimebaseAlignment,
    *,
    session_id: str | None = None,
    replace_existing: bool = True,
) -> Path:
    """Add or replace one typed timebase alignment on a recording session."""
    if not isinstance(alignment, TimebaseAlignment):
        raise TypeError(f"alignment must be a TimebaseAlignment, got {alignment!r}.")
    experiment = _load_or_create_experiment(project)
    session = _select_or_create_session(experiment, requested_session_id=session_id)
    existing = {item.name for item in session.alignments}
    if alignment.name in existing:
        if not replace_existing:
            raise FileExistsError(f"Recording session already has alignment {alignment.name!r}.")
        session = replace_timebase_alignment(session, alignment)
    else:
        session = add_timebase_alignment(session, alignment)
    return save_project_session(project, session, reason="project.save.alignment")


def load_project_alignment(
    project: str | Path,
    *,
    alignment_name: str,
    session_id: str | None = None,
) -> TimebaseAlignment:
    """Load one named timebase alignment from a recording session."""
    session = load_project_session(project, session_id=session_id)
    try:
        return session.alignment(alignment_name)
    except KeyError as exc:
        raise FileNotFoundError(
            f"Recording session {session.session_id!r} has no alignment {alignment_name!r}."
        ) from exc


def save_project_labels(
    project: str | Path,
    labels,
    *,
    metadata: Mapping[str, object] | None = None,
    pose_metadata: Mapping[str, object] | None = None,
    provenance: PoseModelProvenance | None = None,
    session_id: str | None = None,
    pose_name: str = "pose",
    reason: str = "project.save.pose",
    replace_existing: bool = True,
) -> Path:
    """Commit canonical pose labels as a typed link on the project session."""
    from xpkg.model.labels import Labels

    if not isinstance(labels, Labels):
        raise TypeError(f"labels must be Labels, got {labels!r}.")
    if provenance is not None and not isinstance(provenance, PoseModelProvenance):
        raise TypeError(f"provenance must be PoseModelProvenance or None, got {provenance!r}.")
    root = resolve_project_root(project)
    if root is None:
        raise FileNotFoundError(f"Not an xpkg project: {project}")
    _manage_labels_media(labels, root)
    videos = _pose_videos(labels, root)
    pose = SessionPose(
        name=pose_name,
        data=labels,
        videos=videos,
        provenance=provenance,
        metadata=dict(pose_metadata or {}),
    )
    experiment = _load_or_create_experiment(root)
    if metadata is not None:
        experiment = replace_experiment_metadata(experiment, metadata)
    session = _select_or_create_session(experiment, requested_session_id=session_id)
    existing = {link.name for link in session.poses}
    if pose_name in existing:
        if not replace_existing:
            raise FileExistsError(
                f"Recording session already has pose {pose_name!r}. Pass force=True to replace it."
            )
        session = replace_session_pose(session, pose)
    else:
        session = add_session_pose(session, pose)
    if session.session_id in experiment.session_ids:
        experiment = replace_experiment_session(experiment, session)
    else:
        experiment = add_experiment_session(experiment, session)
    state_path = save_project_experiment(root, experiment, reason=reason)
    labels.path = root
    return state_path


def load_project_labels(
    project: str | Path,
    *,
    pose_name: str = "pose",
    session_id: str | None = None,
):
    """Load one typed pose link from the canonical project session."""
    from xpkg.model.labels import Labels

    root = resolve_project_root(project)
    if root is None:
        raise FileNotFoundError(f"Not an xpkg project: {project}")
    store = _project_store(root)
    if not store.has_current_state():
        labels = Labels()
        labels.path = root
        return labels
    session = load_project_session(root, session_id=session_id)
    try:
        pose = session.pose(pose_name)
    except KeyError as exc:
        raise FileNotFoundError(
            f"Recording session {session.session_id!r} has no pose {pose_name!r}."
        ) from exc
    if not isinstance(pose, Labels):
        raise TypeError(
            f"Recording session {session.session_id!r} pose {pose_name!r} is a "
            "PoseTrajectory, not Labels."
        )
    pose.path = root
    return pose


def _require_source_file(path: str | Path, *, role: str) -> Path:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{role} not found: {source}")
    return source


def _copy_import_source(source: Path, project_root: Path, *, media_kind: str) -> Path:
    managed_root = ensure_dir(project_media_root(project_root) / media_kind)
    existing = managed_root / source.name
    if existing.is_file() and sha256_file(existing) == sha256_file(source):
        return existing.resolve()
    return _copy_file_into_media(source, managed_root, {})


def _read_behavior_source(format: str, source: Path) -> BehaviorLabels:
    if format == "behavior-csv":
        return read_behavior_events_csv(source)
    if format == "behavior-json":
        return read_behavior_events_json(source)
    if format == "boris-csv":
        return read_boris_csv(source)
    if format == "bsoid-csv":
        return read_bsoid_csv(source)
    if format == "simba-csv":
        return read_simba_csv(source)
    if format == "keypoint-moseq-csv":
        return read_keypoint_moseq_syllables_csv(source)
    raise ValueError(f"Unknown behavior format: {format!r}")


def _portable_source(
    managed_source: Path, project_root: Path, *, source_type: str
) -> SourceProvenance:
    relative = managed_source.resolve().relative_to(project_root.resolve()).as_posix()
    return SourceProvenance(
        source_type=source_type,
        source_path=relative,
        size_bytes=managed_source.stat().st_size,
        sha256=sha256_file(managed_source),
    )


def _portable_behavior_labels(
    labels: BehaviorLabels,
    managed_source: Path,
    project_root: Path,
    *,
    videos: tuple[SessionVideo, ...],
) -> BehaviorLabels:
    metadata = dict(labels.metadata)
    metadata.pop("source", None)
    if labels.media_path is not None:
        metadata["reported_media_path"] = labels.media_path
    media_path = None
    if len(videos) == 1:
        media_path = videos[0].path.as_posix()
    return replace(labels, media_path=media_path, metadata=metadata)


def _resolve_behavior_links(
    session: RecordingSession,
    *,
    video_roles: tuple[str, ...],
    pose_names: tuple[str, ...],
) -> tuple[tuple[SessionVideo, ...], tuple[SessionPose, ...]]:
    videos = tuple(session.video(role) for role in video_roles)
    pose_lookup = {pose.name: pose for pose in session.poses}
    missing = sorted(set(pose_names) - pose_lookup.keys())
    if missing:
        raise KeyError(f"Recording session has no poses: {', '.join(missing)}.")
    return videos, tuple(pose_lookup[name] for name in pose_names)


def _portable_timebase_alignment(
    alignment: TimebaseAlignment,
    managed_source: Path,
    project_root: Path,
) -> TimebaseAlignment:
    return replace(
        alignment,
        metadata=_without_source(alignment.metadata),
        provenance=_portable_source(
            managed_source,
            project_root,
            source_type="synchronization_csv",
        ),
    )


def _without_source_metadata(events: EventTable) -> EventTable:
    return replace(
        events,
        events=tuple(
            replace(event, metadata=_without_source(event.metadata)) for event in events
        ),
        metadata=_without_source(events.metadata),
    )


def _without_source(metadata: Mapping[str, object]) -> dict[str, object]:
    values = dict(metadata)
    values.pop("source", None)
    return values


def _load_or_create_experiment(project: str | Path) -> Experiment:
    root = resolve_project_root(project)
    if root is None:
        raise FileNotFoundError(f"Not an xpkg project: {project}")
    store = _project_store(root)
    if store.has_current_state():
        return load_project_experiment(root)
    descriptor = load_project_descriptor(root)
    return Experiment(experiment_id=descriptor.project_id, title=descriptor.title)


def _select_or_create_session(
    experiment: Experiment,
    *,
    requested_session_id: str | None = None,
    fallback_session_id: str | None = None,
) -> RecordingSession:
    if requested_session_id is not None:
        if requested_session_id in experiment.session_ids:
            return experiment.session(requested_session_id)
        return RecordingSession(session_id=requested_session_id, title=requested_session_id)
    if len(experiment.sessions) == 1:
        return experiment.sessions[0]
    if len(experiment.sessions) > 1:
        raise ValueError(
            f"Experiment {experiment.experiment_id!r} has {len(experiment.sessions)} sessions; "
            "session_id is required."
        )
    return RecordingSession(
        session_id=fallback_session_id or experiment.experiment_id,
        title=fallback_session_id or experiment.title,
    )


def _pose_videos(labels, project_root: Path) -> tuple[SessionVideo, ...]:
    root = project_root.resolve()
    videos: list[SessionVideo] = []
    for index, video in enumerate(labels.videos):
        raw_path = str(video.filename or "").strip()
        if not raw_path:
            raise ValueError(f"Pose video {index} has no filename.")
        path = Path(raw_path).resolve().relative_to(root)
        fps = float(video.fps) if float(video.fps) > 0.0 else None
        frame_count = int(video.frames) if int(video.frames) >= 0 else None
        matching_frames = [frame for frame in labels.labeled_frames if frame.video is video]
        user_frames = [frame for frame in matching_frames if frame.user_instances]
        prediction_frames = [frame for frame in matching_frames if frame.predicted_instances]
        videos.append(
            SessionVideo(
                role=f"pose-video-{index}",
                path=path,
                timebase=Timebase(name="frame_index"),
                frame_rate_hz=fps,
                frame_count=frame_count,
                metadata={
                    "video_id": str(video.id or f"video_{index}"),
                    "label": str(video.label or Path(raw_path).name),
                    "backend": str(video.backend or ""),
                    "sha256": str(video.sha256 or ""),
                    "height": int(video.height),
                    "width": int(video.width),
                    "channels": int(video.channels),
                    "image_count": len(video.image_filenames or ()),
                    "label_frame_count": len(user_frames),
                    "max_label_frame_index": max(
                        (int(frame.frame_idx) for frame in user_frames), default=None
                    ),
                    "prediction_frame_count": len(prediction_frames),
                    "max_prediction_frame_index": max(
                        (int(frame.frame_idx) for frame in prediction_frames), default=None
                    ),
                },
            )
        )
    return tuple(videos)


def _require_portable_session_paths(session: RecordingSession, project_root: Path) -> None:
    root = project_root.resolve()
    for video in session.videos:
        path = video.path
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(
                f"Session video paths must be project-relative, got {path.as_posix()!r}."
            )
        resolved = (root / path).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Session video path escapes the project: {path}") from exc
        if not resolved.exists():
            raise FileNotFoundError(f"Session video source not found in project: {resolved}")


def _require_portable_source_paths(session: RecordingSession, project_root: Path) -> None:
    for path in _session_source_paths(session):
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(
                f"Session source paths must be project-relative, got {path.as_posix()!r}."
            )
        resolved = (project_root.resolve() / path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Session source not found in project: {resolved}")


def _session_source_paths(session: RecordingSession) -> tuple[Path, ...]:
    paths = list(_signal_source_paths(session))
    for stream in session.event_streams:
        source_path = _provenance_path(stream.provenance)
        if source_path is not None:
            paths.append(source_path)
    for link in session.behaviors:
        source_path = _provenance_path(link.provenance)
        if source_path is not None:
            paths.append(source_path)
    for alignment in session.alignments:
        source_path = _provenance_path(alignment.provenance)
        if source_path is not None:
            paths.append(source_path)
    for link in session.poses:
        if isinstance(link.data, PoseTrajectory) and link.data.source_path is not None:
            paths.append(link.data.source_path)
    for link in session.calibrations:
        source = link.calibration.source
        if source is not None and source.imported_from is not None:
            paths.append(Path(source.imported_from))
    return tuple(dict.fromkeys(paths))


def _signal_source_paths(session: RecordingSession) -> tuple[Path, ...]:
    paths: list[Path] = []
    for link in session.signals:
        source_path = _provenance_path(link.provenance)
        if source_path is not None:
            paths.append(source_path)
    return tuple(dict.fromkeys(paths))


def _provenance_path(provenance: SourceProvenance | None) -> Path | None:
    if provenance is None or provenance.source_path is None:
        return None
    return Path(provenance.source_path)


__all__ = [
    "import_behavior_project",
    "import_events_csv_project",
    "import_photometry_csv_project",
    "import_synchronization_csv_project",
    "load_project_acquisition",
    "load_project_alignment",
    "load_project_behavior",
    "load_project_dataset_share",
    "load_project_event_stream",
    "load_project_experiment",
    "load_project_labels",
    "load_project_session",
    "save_project_acquisition",
    "save_project_alignment",
    "save_project_behavior",
    "save_project_dataset_share",
    "save_project_event_stream",
    "save_project_labels",
    "save_project_experiment",
    "save_project_session",
]
