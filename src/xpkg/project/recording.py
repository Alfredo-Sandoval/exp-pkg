"""Governed project actions for experiment and recording-session state."""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from xpkg._core.hashing import sha256_file
from xpkg._core.path_registry import ensure_dir
from xpkg.io.experiment_json import read_experiment_json, write_experiment_json
from xpkg.io.readers import read_photometry_csv
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
)
from xpkg.model.session import (
    RecordingSession,
    SessionPose,
    SessionSignal,
    SessionVideo,
)
from xpkg.model.session_actions import (
    add_session_pose,
    add_session_signal,
    replace_session_acquisition,
    replace_session_pose,
    replace_session_signal,
)
from xpkg.model.signals import PhotometryRecording
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
            from xpkg.io.labels.model import Labels

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


def load_project_session(
    project: str | Path, *, session_id: str | None = None
) -> RecordingSession:
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
    acquisition: AcquisitionMetadata | Mapping[str, object],
    *,
    session_id: str | None = None,
) -> Path:
    """Commit acquisition context to one recording session."""
    value = (
        acquisition
        if isinstance(acquisition, AcquisitionMetadata)
        else AcquisitionMetadata.from_dict(acquisition)
    )
    experiment = _load_or_create_experiment(project)
    session = _select_or_create_session(experiment, requested_session_id=session_id)
    session = replace_session_acquisition(session, value)
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
    dataset_share: DatasetShareMetadata | Mapping[str, object],
) -> Path:
    """Commit dataset-sharing metadata to the experiment aggregate."""
    value = (
        dataset_share
        if isinstance(dataset_share, DatasetShareMetadata)
        else DatasetShareMetadata.from_dict(dataset_share)
    )
    experiment = _load_or_create_experiment(project)
    return save_project_experiment(
        project,
        replace_experiment_dataset_share(experiment, value),
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
    managed_root = ensure_dir(project_media_root(root) / "signals")
    managed_source = _copy_file_into_media(source, managed_root, {})
    recording = read_photometry_csv(managed_source, name=signal_name)
    recording = _portable_photometry_recording(recording, managed_source, root)
    experiment = _load_or_create_experiment(root)
    session = _select_or_create_session(
        experiment,
        requested_session_id=session_id,
        fallback_session_id=source.stem,
    )
    link = SessionSignal(name=signal_name, recording=recording)
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
    from xpkg.io.labels.model import Labels
    if not isinstance(labels, Labels):
        raise TypeError(f"labels must be Labels, got {labels!r}.")
    if provenance is not None and not isinstance(provenance, PoseModelProvenance):
        raise TypeError(
            "provenance must be PoseModelProvenance or None, "
            f"got {provenance!r}."
        )
    root = resolve_project_root(project)
    if root is None:
        raise FileNotFoundError(f"Not an xpkg project: {project}")
    _manage_labels_media(labels, root)
    videos = _pose_videos(labels, root)
    pose = SessionPose(
        name=pose_name,
        data=labels,
        video_roles=tuple(video.role for video in videos),
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
        session = replace_session_pose(session, pose, videos=videos)
    else:
        session = add_session_pose(session, pose, videos=videos)
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
    from xpkg.io.labels.model import Labels

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


def _portable_photometry_recording(
    recording: PhotometryRecording,
    managed_source: Path,
    project_root: Path,
) -> PhotometryRecording:
    relative_source = managed_source.resolve().relative_to(project_root.resolve()).as_posix()
    source = {
        "type": "photometry_csv",
        "path": relative_source,
        "size_bytes": managed_source.stat().st_size,
        "sha256": sha256_file(managed_source),
    }
    provenance = dict(recording.series.provenance)
    provenance["source"] = source
    metadata = dict(recording.metadata)
    metadata["source"] = source
    return replace(
        recording,
        series=replace(recording.series, provenance=provenance),
        metadata=metadata,
    )


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
        recording = link.recording
        series = recording.series if isinstance(recording, PhotometryRecording) else recording
        source_path = _source_path(series.provenance.get("source"))
        if source_path is not None:
            paths.append(source_path)
        if isinstance(recording, PhotometryRecording):
            source_path = _source_path(recording.metadata.get("source"))
            if source_path is not None:
                paths.append(source_path)
    return tuple(dict.fromkeys(paths))


def _source_path(source: object) -> Path | None:
    if source is None:
        return None
    if not isinstance(source, Mapping):
        raise TypeError("Signal provenance source must be an object when present.")
    raw_path = source.get("path")
    if raw_path is None:
        return None
    if not isinstance(raw_path, str) or not raw_path:
        raise TypeError("Signal provenance source.path must be a non-empty string.")
    return Path(raw_path)


__all__ = [
    "import_photometry_csv_project",
    "load_project_acquisition",
    "load_project_dataset_share",
    "load_project_experiment",
    "load_project_labels",
    "load_project_session",
    "save_project_acquisition",
    "save_project_dataset_share",
    "save_project_labels",
    "save_project_experiment",
    "save_project_session",
]
