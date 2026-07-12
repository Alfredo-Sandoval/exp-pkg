from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest

from xpkg.model import (
    Event,
    EventTable,
    PhotometryChannel,
    PhotometryRecording,
    RecordingSession,
    SessionSignal,
    SessionVideo,
    SignalChannel,
    SyncEvent,
    Timebase,
    Timeline,
    TimeRange,
    TimeSeries,
    add_session_signal,
    add_session_video,
    replace_session_events,
)
from xpkg.model.session_actions import InvalidSessionTransitionError


def test_timeline_from_sample_rate_and_nearest_index() -> None:
    timeline = Timeline.from_sample_rate(n_samples=5, sample_rate_hz=10.0, start_s=1.0)

    np.testing.assert_allclose(timeline.timestamps_s, [1.0, 1.1, 1.2, 1.3, 1.4])
    assert timeline.n_samples == 5
    assert timeline.estimated_sample_rate_hz == pytest.approx(10.0)
    assert timeline.nearest_index(1.24) == 2
    assert timeline.nearest_index(0.5) == 0
    assert timeline.nearest_index(2.0) == 4


def test_timeline_rejects_non_monotonic_timestamps() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        Timeline(timestamps_s=np.array([0.0, 0.2, 0.1]))


@pytest.mark.parametrize(
    ("kwargs", "exc_type", "message"),
    [
        ({"name": 1}, TypeError, "timebase name must be a string"),
        ({"name": ""}, ValueError, "timebase name must be a non-empty string"),
        (
            {"name": " session"},
            ValueError,
            "timebase name must not contain surrounding whitespace",
        ),
        ({"unit": 1}, TypeError, "timebase unit must be a string"),
        (
            {"unit": " s"},
            ValueError,
            "timebase unit must not contain surrounding whitespace",
        ),
    ],
)
def test_timebase_rejects_unclean_text(
    kwargs: dict[str, object],
    exc_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exc_type, match=message):
        Timebase(**kwargs)


def test_event_table_queries_and_round_trips() -> None:
    events = EventTable.from_events(
        [
            Event(kind="cue", start_s=1.5, duration_s=0.25, label="tone"),
            Event(kind="trial", start_s=1.0, duration_s=2.0, label="A"),
        ]
    )
    events = EventTable(events=events.events, metadata={"source": "unit-test"})

    assert [event.kind for event in events] == ["trial", "cue"]
    assert events.query(kind="trial") == (events.events[0],)
    assert events.query(time_s=1.6) == events.events
    assert events.query(overlaps=TimeRange(1.4, 1.55)) == events.events
    assert events.append(Event(kind="cue", start_s=3.0)).metadata == events.metadata

    hydrated = EventTable.from_dict(events.to_dict())
    assert hydrated.to_dict() == events.to_dict()


def test_event_table_from_dict_rejects_unclean_timebase_text() -> None:
    payload = {
        "timebase": {"name": " session", "unit": "s", "offset_s": 0.0},
        "events": [{"kind": "cue", "start_s": 0.0}],
    }

    with pytest.raises(ValueError, match="timebase name must not contain surrounding whitespace"):
        EventTable.from_dict(payload)


@pytest.mark.parametrize(
    ("kwargs", "exc_type", "message"),
    [
        ({"kind": 1, "start_s": 0.0}, TypeError, "event kind must be a string"),
        ({"kind": "", "start_s": 0.0}, ValueError, "event kind must be a non-empty string"),
        (
            {"kind": " cue", "start_s": 0.0},
            ValueError,
            "event kind must not contain surrounding whitespace",
        ),
        ({"kind": "cue", "start_s": 0.0, "label": 1}, TypeError, "event label must be a string"),
        (
            {"kind": "cue", "start_s": 0.0, "label": ""},
            ValueError,
            "event label must be a non-empty string",
        ),
        (
            {"kind": "cue", "start_s": 0.0, "label": " tone"},
            ValueError,
            "event label must not contain surrounding whitespace",
        ),
    ],
)
def test_event_rejects_unclean_text(
    kwargs: dict[str, object],
    exc_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exc_type, match=message):
        Event(**kwargs)


def test_event_from_dict_rejects_coerced_text() -> None:
    with pytest.raises(TypeError, match="event kind must be a string"):
        Event.from_dict({"kind": 1, "start_s": 0.0})


@pytest.mark.parametrize(
    ("query_kwargs", "exc_type", "message"),
    [
        ({"kind": 1}, TypeError, "event query kind must be a string"),
        (
            {"kind": " cue"},
            ValueError,
            "event query kind must not contain surrounding whitespace",
        ),
        ({"label": ""}, ValueError, "event query label must be a non-empty string"),
        (
            {"label": " tone"},
            ValueError,
            "event query label must not contain surrounding whitespace",
        ),
    ],
)
def test_event_table_query_rejects_unclean_filters(
    query_kwargs: dict[str, object],
    exc_type: type[Exception],
    message: str,
) -> None:
    events = EventTable.from_events([Event(kind="cue", start_s=0.0, label="tone")])

    with pytest.raises(exc_type, match=message):
        events.query(**query_kwargs)


def test_sync_event_payload_preserves_source() -> None:
    event = SyncEvent(kind="sync", start_s=1.0, source="ttl")

    assert event.to_dict()["source"] == "ttl"


@pytest.mark.parametrize(
    ("source", "exc_type", "message"),
    [
        (1, TypeError, "sync event source must be a string"),
        ("", ValueError, "sync event source must be a non-empty string"),
        (" ttl", ValueError, "sync event source must not contain surrounding whitespace"),
    ],
)
def test_sync_event_rejects_unclean_source(
    source: object,
    exc_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exc_type, match=message):
        SyncEvent(kind="sync", start_s=1.0, source=source)


def test_time_series_and_photometry_recording_validate_channels() -> None:
    series = TimeSeries.from_samples(
        [[1.0, 0.5], [1.2, 0.45], [1.3, 0.4]],
        sample_rate_hz=20.0,
        channel_names=["gcamp", "isosbestic"],
        units=["dff", "dff"],
        name="fiber",
    )
    recording = PhotometryRecording(
        series=series,
        signal_channel="gcamp",
        reference_channel="isosbestic",
    )

    assert series.n_samples == 3
    assert series.n_channels == 2
    assert series.channel_index("gcamp") == 0
    assert recording.timeline is series.timeline
    assert recording.channel_names == ("gcamp", "isosbestic")

    with pytest.raises(ValueError, match="channel names must be unique"):
        TimeSeries(
            values=np.ones((3, 2)),
            timeline=series.timeline,
            channels=(SignalChannel("dup"), SignalChannel("dup")),
        )

    with pytest.raises(ValueError, match="channel_names length"):
        TimeSeries.from_samples(
            np.ones((3, 2)),
            sample_rate_hz=10.0,
            channel_names=["gcamp"],
        )


@pytest.mark.parametrize(
    ("kwargs", "exc_type", "message"),
    [
        ({"name": 1}, TypeError, "signal channel name must be a string"),
        ({"name": ""}, ValueError, "signal channel name must be a non-empty string"),
        (
            {"name": " gcamp"},
            ValueError,
            "signal channel name must not contain surrounding whitespace",
        ),
        (
            {"name": "gcamp", "unit": 1},
            TypeError,
            "signal channel unit must be a string",
        ),
        (
            {"name": "gcamp", "unit": " dff"},
            ValueError,
            "signal channel unit must not contain surrounding whitespace",
        ),
        (
            {"name": "gcamp", "description": " primary"},
            ValueError,
            "signal channel description must not contain surrounding whitespace",
        ),
    ],
)
def test_signal_channel_rejects_unclean_text(
    kwargs: dict[str, object],
    exc_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exc_type, match=message):
        SignalChannel(**kwargs)


def test_photometry_channel_rejects_unclean_excitation() -> None:
    with pytest.raises(
        ValueError,
        match="photometry channel excitation must not contain surrounding whitespace",
    ):
        PhotometryChannel(name="gcamp", excitation=" 470")


@pytest.mark.parametrize(
    ("operation", "exc_type", "message"),
    [
        (
            lambda series: series.channel_index(" gcamp"),
            ValueError,
            "channel name must not contain surrounding whitespace",
        ),
        (
            lambda series: TimeSeries(
                values=np.ones((3, 1)),
                timeline=series.timeline,
                channels=(SignalChannel("gcamp"),),
                name=" signals",
            ),
            ValueError,
            "time series name must not contain surrounding whitespace",
        ),
        (
            lambda series: TimeSeries.from_samples(
                np.ones((3, 1)),
                sample_rate_hz=10.0,
                channel_names=[" gcamp"],
            ),
            ValueError,
            "signal channel name must not contain surrounding whitespace",
        ),
        (
            lambda series: PhotometryRecording(series=series, signal_channel=" gcamp"),
            ValueError,
            "signal_channel must not contain surrounding whitespace",
        ),
        (
            lambda series: PhotometryRecording(
                series=series,
                signal_channel="gcamp",
                reference_channel=" isosbestic",
            ),
            ValueError,
            "reference_channel must not contain surrounding whitespace",
        ),
    ],
)
def test_signal_models_reject_unclean_text(
    operation,
    exc_type: type[Exception],
    message: str,
) -> None:
    series = TimeSeries.from_samples(
        np.ones((3, 2)),
        sample_rate_hz=10.0,
        channel_names=["gcamp", "isosbestic"],
    )

    with pytest.raises(exc_type, match=message):
        operation(series)


def test_recording_session_collects_signal_and_event_time_range() -> None:
    series = TimeSeries.from_samples(
        [1.0, 1.1, 1.2],
        sample_rate_hz=10.0,
        channel_names=["gcamp"],
    )
    recording = PhotometryRecording(series=series, signal_channel="gcamp")
    events = EventTable.from_events([Event(kind="trial", start_s=0.0, duration_s=1.0)])

    session = add_session_signal(
        RecordingSession(session_id="session-001"),
        SessionSignal("fiber", recording),
    )
    session = replace_session_events(session, events)

    assert session.modality_names == ("signals", "events")
    assert session.signal("fiber") is recording
    assert session.time_range == TimeRange(0.0, 1.0)


@pytest.mark.parametrize(
    ("kwargs", "exc_type", "message"),
    [
        ({"session_id": 1}, TypeError, "session_id must be a string"),
        ({"session_id": ""}, ValueError, "session_id must be a non-empty string"),
        (
            {"session_id": " session-001"},
            ValueError,
            "session_id must not contain surrounding whitespace",
        ),
        (
            {"session_id": "session-001", "title": ""},
            ValueError,
            "session title must be a non-empty string",
        ),
        (
            {"session_id": "session-001", "title": " Session"},
            ValueError,
            "session title must not contain surrounding whitespace",
        ),
    ],
)
def test_recording_session_rejects_unclean_text(
    kwargs: dict[str, object],
    exc_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exc_type, match=message):
        RecordingSession(**kwargs)


def test_recording_session_rejects_unclean_signal_keys() -> None:
    series = TimeSeries.from_samples(
        [1.0, 1.1, 1.2],
        sample_rate_hz=10.0,
        channel_names=["gcamp"],
    )
    recording = PhotometryRecording(series=series, signal_channel="gcamp")

    with pytest.raises(
        ValueError,
        match="session signal name must not contain surrounding whitespace",
    ):
        SessionSignal(" fiber", recording)


def test_add_session_signal_rejects_duplicate_name() -> None:
    series = TimeSeries.from_samples(
        [1.0, 1.1, 1.2],
        sample_rate_hz=10.0,
        channel_names=["gcamp"],
    )
    recording = PhotometryRecording(series=series, signal_channel="gcamp")

    session = RecordingSession(
        session_id="session-001",
        signals=(SessionSignal("fiber", recording),),
    )

    with pytest.raises(InvalidSessionTransitionError, match="already has signal 'fiber'"):
        add_session_signal(session, SessionSignal("fiber", recording))


def test_add_session_video_creates_typed_role_link() -> None:
    session = add_session_video(
        RecordingSession(session_id="session-001"),
        SessionVideo(role="behavior", path=Path("recordings/session.mp4")),
    )

    assert session.modality_names == ("videos",)
    assert session.video("behavior").path == Path("recordings/session.mp4")


def test_recording_session_rejects_crossed_link_types() -> None:
    video = SessionVideo(role="behavior", path=Path("recordings/session.mp4"))

    with pytest.raises(TypeError, match="session signal entries must be SessionSignal"):
        RecordingSession(
            session_id="session-001",
            signals=cast(tuple[SessionSignal, ...], (video,)),
        )
