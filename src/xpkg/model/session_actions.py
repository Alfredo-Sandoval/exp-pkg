"""Governed state transitions for recording-session ontology objects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from xpkg.model.events import EventTable
from xpkg.model.metadata import AcquisitionMetadata
from xpkg.model.session import (
    RecordingSession,
    SessionBehavior,
    SessionCalibration,
    SessionPose,
    SessionSignal,
    SessionVideo,
    TimebaseAlignment,
)


class InvalidSessionTransitionError(ValueError):
    """Raised when a recording-session action violates its preconditions."""


def add_session_signal(session: RecordingSession, signal: SessionSignal) -> RecordingSession:
    """Return a session with one new uniquely named signal link."""
    if any(existing.name == signal.name for existing in session.signals):
        raise InvalidSessionTransitionError(
            f"Recording session {session.session_id!r} already has signal {signal.name!r}."
        )
    return replace(session, signals=(*session.signals, signal))


def replace_session_signal(session: RecordingSession, signal: SessionSignal) -> RecordingSession:
    """Replace an existing signal link while preserving link order."""
    if not any(existing.name == signal.name for existing in session.signals):
        raise InvalidSessionTransitionError(
            f"Recording session {session.session_id!r} has no signal {signal.name!r} to replace."
        )
    signals = tuple(
        signal if existing.name == signal.name else existing for existing in session.signals
    )
    return replace(session, signals=signals)


def add_session_video(session: RecordingSession, video: SessionVideo) -> RecordingSession:
    """Return a session with one new uniquely named video link."""
    if any(existing.role == video.role for existing in session.videos):
        raise InvalidSessionTransitionError(
            f"Recording session {session.session_id!r} already has video role {video.role!r}."
        )
    return replace(session, videos=(*session.videos, video))


def replace_session_video(session: RecordingSession, video: SessionVideo) -> RecordingSession:
    """Replace an existing video link while preserving link order."""
    if not any(existing.role == video.role for existing in session.videos):
        raise InvalidSessionTransitionError(
            f"Recording session {session.session_id!r} has no video role {video.role!r}."
        )
    videos = tuple(
        video if existing.role == video.role else existing for existing in session.videos
    )
    return replace(session, videos=videos)


def add_session_pose(
    session: RecordingSession,
    pose: SessionPose,
    *,
    videos: tuple[SessionVideo, ...] = (),
) -> RecordingSession:
    """Add one pose link and its previously unlinked videos atomically."""
    if any(existing.name == pose.name for existing in session.poses):
        raise InvalidSessionTransitionError(
            f"Recording session {session.session_id!r} already has pose {pose.name!r}."
        )
    merged_videos = _add_videos(session, videos)
    return replace(session, videos=merged_videos, poses=(*session.poses, pose))


def replace_session_pose(
    session: RecordingSession,
    pose: SessionPose,
    *,
    videos: tuple[SessionVideo, ...] = (),
) -> RecordingSession:
    """Replace one pose link and its same-role videos atomically."""
    current = next((existing for existing in session.poses if existing.name == pose.name), None)
    if current is None:
        raise InvalidSessionTransitionError(
            f"Recording session {session.session_id!r} has no pose {pose.name!r}."
        )
    retained_videos = tuple(
        video for video in session.videos if video.role not in current.video_roles
    )
    merged_videos = _replace_or_add_videos(retained_videos, videos)
    poses = tuple(pose if existing.name == pose.name else existing for existing in session.poses)
    return replace(session, videos=merged_videos, poses=poses)


def add_session_behavior(
    session: RecordingSession, behavior: SessionBehavior
) -> RecordingSession:
    """Add one uniquely named behavior-label link."""
    return replace(
        session,
        behaviors=(*session.behaviors, _require_new_link(session, behavior, "behavior")),
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
    _require_existing_link(session, calibration, "calibration")
    return replace(
        session,
        calibrations=tuple(
            calibration if existing.name == calibration.name else existing
            for existing in session.calibrations
        ),
    )


def add_timebase_alignment(
    session: RecordingSession, alignment: TimebaseAlignment
) -> RecordingSession:
    """Add one uniquely named synchronization relationship."""
    return replace(
        session,
        alignments=(*session.alignments, _require_new_link(session, alignment, "alignment")),
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
    )


def replace_session_events(session: RecordingSession, events: EventTable) -> RecordingSession:
    """Return a session whose event link points to ``events``."""
    if not isinstance(events, EventTable):
        raise TypeError(f"events must be an EventTable, got {events!r}.")
    return replace(session, events=events)


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


def _add_videos(
    session: RecordingSession, videos: tuple[SessionVideo, ...]
) -> tuple[SessionVideo, ...]:
    existing = {video.role for video in session.videos}
    incoming = {video.role for video in videos}
    overlap = sorted(existing & incoming)
    if overlap:
        raise InvalidSessionTransitionError(
            f"Recording session {session.session_id!r} already has video roles: "
            f"{', '.join(overlap)}."
        )
    return (*session.videos, *videos)


def _replace_or_add_videos(
    current: tuple[SessionVideo, ...], videos: tuple[SessionVideo, ...]
) -> tuple[SessionVideo, ...]:
    replacements = {video.role: video for video in videos}
    merged = tuple(replacements.pop(video.role, video) for video in current)
    return (*merged, *replacements.values())


def _link_collection(session: RecordingSession, kind: str):
    if kind == "behavior":
        return session.behaviors
    if kind == "calibration":
        return session.calibrations
    if kind == "alignment":
        return session.alignments
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


__all__ = [
    "InvalidSessionTransitionError",
    "add_session_behavior",
    "add_session_calibration",
    "add_session_pose",
    "add_session_signal",
    "add_session_video",
    "add_timebase_alignment",
    "replace_session_behavior",
    "replace_session_calibration",
    "replace_session_acquisition",
    "replace_session_events",
    "replace_session_metadata",
    "replace_session_pose",
    "replace_session_signal",
    "replace_session_video",
    "replace_timebase_alignment",
]
