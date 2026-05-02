from __future__ import annotations

import numpy as np
import pytest

from xpkg.model import (
    Event,
    EventTable,
    PhotometryRecording,
    RecordingSession,
    SignalChannel,
    SyncEvent,
    Timeline,
    TimeRange,
    TimeSeries,
)


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


def test_event_table_queries_and_round_trips() -> None:
    events = EventTable.from_events(
        [
            Event(kind="cue", start_s=1.5, duration_s=0.25, label="tone"),
            Event(kind="trial", start_s=1.0, duration_s=2.0, label="A"),
        ]
    )

    assert [event.kind for event in events] == ["trial", "cue"]
    assert events.query(kind="trial") == (events.events[0],)
    assert events.query(time_s=1.6) == events.events
    assert events.query(overlaps=TimeRange(1.4, 1.55)) == events.events

    hydrated = EventTable.from_dict(events.to_dict())
    assert hydrated.to_dict() == events.to_dict()


def test_sync_event_payload_preserves_source() -> None:
    event = SyncEvent(kind="sync", start_s=1.0, source="ttl")

    assert event.to_dict()["source"] == "ttl"


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


def test_recording_session_collects_signal_and_event_time_range() -> None:
    series = TimeSeries.from_samples(
        [1.0, 1.1, 1.2],
        sample_rate_hz=10.0,
        channel_names=["gcamp"],
    )
    recording = PhotometryRecording(series=series, signal_channel="gcamp")
    events = EventTable.from_events([Event(kind="trial", start_s=0.0, duration_s=1.0)])

    session = RecordingSession(session_id="session-001").with_signal(
        "fiber",
        recording,
    ).with_events(events)

    assert session.modality_names == ("signals", "events")
    assert session.signals["fiber"] is recording
    assert session.time_range == TimeRange(0.0, 1.0)
