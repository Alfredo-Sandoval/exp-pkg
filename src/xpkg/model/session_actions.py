"""Governed state transitions for recording-session ontology objects."""

from __future__ import annotations

from dataclasses import replace

from xpkg.model.events import EventTable
from xpkg.model.session import RecordingSession, SessionSignal, SessionVideo


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


def replace_session_events(session: RecordingSession, events: EventTable) -> RecordingSession:
    """Return a session whose event link points to ``events``."""
    if not isinstance(events, EventTable):
        raise TypeError(f"events must be an EventTable, got {events!r}.")
    return replace(session, events=events)


__all__ = [
    "InvalidSessionTransitionError",
    "add_session_signal",
    "add_session_video",
    "replace_session_events",
    "replace_session_signal",
]
