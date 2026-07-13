"""Governed state transitions for recording-session ontology objects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from xpkg.model.metadata import AcquisitionMetadata
from xpkg.model.session import (
    AlignmentModel,
    RecordingSession,
    SessionBehavior,
    SessionCalibration,
    SessionEventStream,
    SessionPose,
    SessionSignal,
    SessionVideo,
    SynchronizationMethod,
    TimebaseAlignment,
    TimebaseCorrespondence,
)
from xpkg.model.time import Timebase


class InvalidSessionTransitionError(ValueError):
    """Raised when a recording-session action violates its preconditions."""


def fit_timebase_alignment(
    *,
    name: str,
    source: Timebase,
    target: Timebase,
    model: AlignmentModel,
    method: SynchronizationMethod,
    evidence: tuple[TimebaseCorrespondence, ...],
    metadata: Mapping[str, Any] | None = None,
) -> TimebaseAlignment:
    """Fit one offset or affine transform from paired time observations."""
    pairs = tuple(evidence)
    if not pairs:
        raise InvalidSessionTransitionError(
            "Fitting a timebase alignment requires correspondence evidence."
        )
    if method is SynchronizationMethod.MANUAL:
        raise InvalidSessionTransitionError(
            "Manual timebase alignments must declare coefficients explicitly."
        )
    if model is AlignmentModel.OFFSET:
        scale = 1.0
        offset_s = sum(pair.target_time_s - pair.source_time_s for pair in pairs) / len(pairs)
    elif model is AlignmentModel.AFFINE:
        scale, offset_s = _fit_affine_coefficients(pairs)
    else:
        raise TypeError(f"model must be an AlignmentModel, got {model!r}.")
    return TimebaseAlignment(
        name=name,
        source=source,
        target=target,
        model=model,
        method=method,
        scale=scale,
        offset_s=offset_s,
        evidence=pairs,
        metadata=dict(metadata or {}),
    )


def _fit_affine_coefficients(
    evidence: tuple[TimebaseCorrespondence, ...],
) -> tuple[float, float]:
    source_mean = sum(item.source_time_s for item in evidence) / len(evidence)
    target_mean = sum(item.target_time_s for item in evidence) / len(evidence)
    denominator = sum((item.source_time_s - source_mean) ** 2 for item in evidence)
    if denominator == 0.0:
        raise InvalidSessionTransitionError(
            "Affine alignment requires at least two distinct source times."
        )
    scale = (
        sum(
            (item.source_time_s - source_mean) * (item.target_time_s - target_mean)
            for item in evidence
        )
        / denominator
    )
    if scale <= 0.0:
        raise InvalidSessionTransitionError(
            "Affine alignment evidence produced a non-positive scale."
        )
    return float(scale), float(target_mean - (scale * source_mean))


def add_session_signal(session: RecordingSession, signal: SessionSignal) -> RecordingSession:
    """Return a session with one new uniquely named signal link."""
    if any(existing.name == signal.name for existing in session.signals):
        raise InvalidSessionTransitionError(
            f"Recording session {session.session_id!r} already has signal {signal.name!r}."
        )
    return replace(
        session,
        signals=(*session.signals, signal),
        timebases=_merged_timebases(session, signal.recording.timeline.timebase),
    )


def replace_session_signal(session: RecordingSession, signal: SessionSignal) -> RecordingSession:
    """Replace an existing signal link while preserving link order."""
    if not any(existing.name == signal.name for existing in session.signals):
        raise InvalidSessionTransitionError(
            f"Recording session {session.session_id!r} has no signal {signal.name!r} to replace."
        )
    signals = tuple(
        signal if existing.name == signal.name else existing for existing in session.signals
    )
    return replace(
        session,
        signals=signals,
        timebases=_merged_timebases(session, signal.recording.timeline.timebase),
    )


def add_session_video(session: RecordingSession, video: SessionVideo) -> RecordingSession:
    """Return a session with one new uniquely named video link."""
    if any(existing.role == video.role for existing in session.videos):
        raise InvalidSessionTransitionError(
            f"Recording session {session.session_id!r} already has video role {video.role!r}."
        )
    return replace(
        session,
        videos=(*session.videos, video),
        timebases=_merged_timebases(session, video.timebase),
    )


def replace_session_video(session: RecordingSession, video: SessionVideo) -> RecordingSession:
    """Replace an existing video link while preserving link order."""
    if not any(existing.role == video.role for existing in session.videos):
        raise InvalidSessionTransitionError(
            f"Recording session {session.session_id!r} has no video role {video.role!r}."
        )
    current = next(existing for existing in session.videos if existing.role == video.role)
    videos = tuple(video if existing is current else existing for existing in session.videos)
    poses = tuple(
        replace(pose, videos=tuple(video if item is current else item for item in pose.videos))
        for pose in session.poses
    )
    behaviors = tuple(
        replace(
            behavior,
            videos=tuple(video if item is current else item for item in behavior.videos),
        )
        for behavior in session.behaviors
    )
    return replace(
        session,
        videos=videos,
        poses=poses,
        behaviors=behaviors,
        timebases=_merged_timebases(session, video.timebase),
    )


def add_session_pose(
    session: RecordingSession,
    pose: SessionPose,
) -> RecordingSession:
    """Add one pose link and its referenced videos atomically."""
    if any(existing.name == pose.name for existing in session.poses):
        raise InvalidSessionTransitionError(
            f"Recording session {session.session_id!r} already has pose {pose.name!r}."
        )
    merged_videos = _merge_video_refs(session, pose.videos)
    pose = replace(pose, videos=_canonical_video_refs(merged_videos, pose.videos))
    return replace(
        session,
        videos=merged_videos,
        poses=(*session.poses, pose),
        timebases=_merged_timebases(session, *(video.timebase for video in pose.videos)),
    )


def replace_session_pose(
    session: RecordingSession,
    pose: SessionPose,
) -> RecordingSession:
    """Replace one pose link and update behavior references atomically."""
    current = next((existing for existing in session.poses if existing.name == pose.name), None)
    if current is None:
        raise InvalidSessionTransitionError(
            f"Recording session {session.session_id!r} has no pose {pose.name!r}."
        )
    replaceable_roles = {video.role for video in current.videos}
    working = session
    for incoming in pose.videos:
        existing = next(
            (video for video in working.videos if video.role == incoming.role), None
        )
        if existing is not None and existing != incoming:
            if incoming.role not in replaceable_roles:
                raise InvalidSessionTransitionError(
                    f"Video role {incoming.role!r} is not linked to pose {pose.name!r}."
                )
            working = replace_session_video(working, incoming)
    current = next(existing for existing in working.poses if existing.name == pose.name)
    merged_videos = _merge_video_refs(working, pose.videos)
    pose = replace(pose, videos=_canonical_video_refs(merged_videos, pose.videos))
    poses = tuple(pose if existing.name == pose.name else existing for existing in working.poses)
    behaviors = tuple(
        replace(
            behavior,
            poses=tuple(pose if item is current else item for item in behavior.poses),
        )
        for behavior in working.behaviors
    )
    return replace(
        working,
        videos=merged_videos,
        poses=poses,
        behaviors=behaviors,
        timebases=_merged_timebases(session, *(video.timebase for video in pose.videos)),
    )


def add_session_behavior(session: RecordingSession, behavior: SessionBehavior) -> RecordingSession:
    """Add one uniquely named behavior-label link."""
    return replace(
        session,
        behaviors=(*session.behaviors, _require_new_link(session, behavior, "behavior")),
        timebases=_merged_timebases(session, behavior.labels.timebase),
    )


def replace_session_behavior(
    session: RecordingSession, behavior: SessionBehavior
) -> RecordingSession:
    """Replace one behavior-label link while preserving link order."""
    _require_existing_link(session, behavior, "behavior")
    return replace(
        session,
        behaviors=tuple(
            behavior if existing.name == behavior.name else existing
            for existing in session.behaviors
        ),
        timebases=_merged_timebases(session, behavior.labels.timebase),
    )


def add_session_calibration(
    session: RecordingSession, calibration: SessionCalibration
) -> RecordingSession:
    """Add one uniquely named calibration link."""
    return replace(
        session,
        calibrations=(
            *session.calibrations,
            _require_new_link(session, calibration, "calibration"),
        ),
    )


def replace_session_calibration(
    session: RecordingSession, calibration: SessionCalibration
) -> RecordingSession:
    """Replace one calibration link while preserving link order."""
    current = next(
        (item for item in session.calibrations if item.name == calibration.name), None
    )
    if current is None:
        raise InvalidSessionTransitionError(
            f"Recording session {session.session_id!r} has no calibration "
            f"{calibration.name!r}."
        )
    return replace(
        session,
        calibrations=tuple(
            calibration if existing.name == calibration.name else existing
            for existing in session.calibrations
        ),
        poses=tuple(
            replace(pose, calibration=calibration) if pose.calibration is current else pose
            for pose in session.poses
        ),
    )


def add_timebase_alignment(
    session: RecordingSession, alignment: TimebaseAlignment
) -> RecordingSession:
    """Add one uniquely named synchronization relationship."""
    return replace(
        session,
        alignments=(*session.alignments, _require_new_link(session, alignment, "alignment")),
        timebases=_merged_timebases(session, alignment.source, alignment.target),
    )


def replace_timebase_alignment(
    session: RecordingSession, alignment: TimebaseAlignment
) -> RecordingSession:
    """Replace one synchronization relationship while preserving link order."""
    _require_existing_link(session, alignment, "alignment")
    return replace(
        session,
        alignments=tuple(
            alignment if existing.name == alignment.name else existing
            for existing in session.alignments
        ),
        timebases=_merged_timebases(session, alignment.source, alignment.target),
    )


def add_session_event_stream(
    session: RecordingSession, stream: SessionEventStream
) -> RecordingSession:
    """Add one uniquely named event stream."""
    return replace(
        session,
        event_streams=(
            *session.event_streams,
            _require_new_link(session, stream, "event stream"),
        ),
        timebases=_merged_timebases(session, stream.events.timebase),
    )


def replace_session_event_stream(
    session: RecordingSession, stream: SessionEventStream
) -> RecordingSession:
    """Replace one event stream while preserving stream order."""
    _require_existing_link(session, stream, "event stream")
    return replace(
        session,
        event_streams=tuple(
            stream if existing.name == stream.name else existing
            for existing in session.event_streams
        ),
        timebases=_merged_timebases(session, stream.events.timebase),
    )


def add_session_timebase(session: RecordingSession, timebase: Timebase) -> RecordingSession:
    """Register one new named timebase without changing modality links."""
    if not isinstance(timebase, Timebase):
        raise TypeError("timebase must be a Timebase.")
    if any(current.name == timebase.name for current in session.timebases):
        raise InvalidSessionTransitionError(
            f"Recording session {session.session_id!r} already has timebase {timebase.name!r}."
        )
    return replace(session, timebases=(*session.timebases, timebase))


def replace_session_metadata(
    session: RecordingSession, metadata: Mapping[str, Any] | None
) -> RecordingSession:
    """Replace session metadata through the governed action layer."""
    return replace(session, metadata=dict(metadata or {}))


def replace_session_acquisition(
    session: RecordingSession,
    acquisition: AcquisitionMetadata | None,
) -> RecordingSession:
    """Replace the acquisition context owned by one recording session."""
    if acquisition is not None and not isinstance(acquisition, AcquisitionMetadata):
        raise TypeError("acquisition must be AcquisitionMetadata or None.")
    return replace(session, acquisition=acquisition)


def _merge_video_refs(
    session: RecordingSession, videos: tuple[SessionVideo, ...]
) -> tuple[SessionVideo, ...]:
    existing = {video.role: video for video in session.videos}
    for video in videos:
        current = existing.get(video.role)
        if current is not None and current != video:
            raise InvalidSessionTransitionError(
                f"Video role {video.role!r} conflicts with its session object."
            )
        existing[video.role] = video
    return tuple(existing.values())


def _canonical_video_refs(
    session_videos: tuple[SessionVideo, ...], referenced: tuple[SessionVideo, ...]
) -> tuple[SessionVideo, ...]:
    lookup = {video.role: video for video in session_videos}
    return tuple(lookup[video.role] for video in referenced)


def _link_collection(session: RecordingSession, kind: str):
    if kind == "behavior":
        return session.behaviors
    if kind == "calibration":
        return session.calibrations
    if kind == "alignment":
        return session.alignments
    if kind == "event stream":
        return session.event_streams
    raise ValueError(f"Unsupported session link kind: {kind!r}.")


def _require_new_link(session: RecordingSession, link, kind: str):
    if any(existing.name == link.name for existing in _link_collection(session, kind)):
        raise InvalidSessionTransitionError(
            f"Recording session {session.session_id!r} already has {kind} {link.name!r}."
        )
    return link


def _require_existing_link(session: RecordingSession, link, kind: str) -> None:
    if not any(existing.name == link.name for existing in _link_collection(session, kind)):
        raise InvalidSessionTransitionError(
            f"Recording session {session.session_id!r} has no {kind} {link.name!r}."
        )


def _merged_timebases(session: RecordingSession, *incoming: Timebase) -> tuple[Timebase, ...]:
    merged = {timebase.name: timebase for timebase in session.timebases}
    for timebase in incoming:
        existing = merged.get(timebase.name)
        if existing is not None and existing != timebase:
            raise InvalidSessionTransitionError(
                f"Timebase {timebase.name!r} conflicts with its registered definition."
            )
        merged[timebase.name] = timebase
    return tuple(merged.values())


__all__ = [
    "InvalidSessionTransitionError",
    "add_session_behavior",
    "add_session_calibration",
    "add_session_pose",
    "add_session_event_stream",
    "add_session_signal",
    "add_session_timebase",
    "add_session_video",
    "add_timebase_alignment",
    "fit_timebase_alignment",
    "replace_session_behavior",
    "replace_session_calibration",
    "replace_session_acquisition",
    "replace_session_event_stream",
    "replace_session_metadata",
    "replace_session_pose",
    "replace_session_signal",
    "replace_session_video",
    "replace_timebase_alignment",
]
