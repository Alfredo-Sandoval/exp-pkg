from __future__ import annotations

import numpy as np
import pytest

from xpkg.io.readers import read_events_csv, read_photometry_csv
from xpkg.model import EventTable, PhotometryRecording


def test_read_photometry_csv_uses_explicit_time_and_channels(tmp_path) -> None:
    path = tmp_path / "photometry.csv"
    path.write_text(
        "\n".join(
            [
                "time,gcamp,isosbestic,ttl",
                "0.0,1.0,0.5,0",
                "0.1,1.1,0.4,1",
                "0.2,1.2,0.3,0",
            ]
        ),
        encoding="utf-8",
    )

    recording = read_photometry_csv(
        path,
        time_column="time",
        signal_columns=["gcamp", "isosbestic"],
        signal_channel="gcamp",
        reference_channel="isosbestic",
        units={"gcamp": "a.u.", "isosbestic": "a.u."},
    )

    assert isinstance(recording, PhotometryRecording)
    assert recording.signal_channel == "gcamp"
    assert recording.reference_channel == "isosbestic"
    assert recording.channel_names == ("gcamp", "isosbestic")
    assert recording.series.sample_rate_hz == pytest.approx(10.0)
    np.testing.assert_allclose(recording.timeline.timestamps_s, [0.0, 0.1, 0.2])
    np.testing.assert_allclose(recording.series.values[:, 0], [1.0, 1.1, 1.2])


def test_read_photometry_csv_can_build_regular_timeline_from_sample_rate(tmp_path) -> None:
    path = tmp_path / "samples.csv"
    path.write_text("gcamp,isosbestic\n1.0,0.5\n1.1,0.4\n", encoding="utf-8")

    recording = read_photometry_csv(
        path,
        signal_columns=["gcamp", "isosbestic"],
        sample_rate_hz=20.0,
        start_s=2.0,
    )

    np.testing.assert_allclose(recording.timeline.timestamps_s, [2.0, 2.05])
    assert recording.series.sample_rate_hz == pytest.approx(20.0)


def test_read_photometry_csv_rejects_missing_timebase(tmp_path) -> None:
    path = tmp_path / "single.csv"
    path.write_text("gcamp\n1.0\n1.1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="time column or sample_rate_hz"):
        read_photometry_csv(path)


def test_read_events_csv_maps_labels_kinds_and_millisecond_times(tmp_path) -> None:
    path = tmp_path / "events.csv"
    path.write_text(
        "\n".join(
            [
                "timestamp,type,label,duration",
                "1000,cue,tone,100",
                "1500,trial,A,500",
            ]
        ),
        encoding="utf-8",
    )

    events = read_events_csv(path, time_unit="ms")

    assert isinstance(events, EventTable)
    assert [event.kind for event in events] == ["cue", "trial"]
    assert [event.label for event in events] == ["tone", "A"]
    np.testing.assert_allclose([event.start_s for event in events], [1.0, 1.5])
    np.testing.assert_allclose([event.duration_s for event in events], [0.1, 0.5])
    assert events.query(kind="trial")[0].label == "A"


def test_read_events_csv_uses_default_kind_without_label_columns(tmp_path) -> None:
    path = tmp_path / "timestamps.csv"
    path.write_text("timestamps\n0.2\n0.1\n", encoding="utf-8")

    events = read_events_csv(path, default_kind="ttl")

    assert [event.kind for event in events] == ["ttl", "ttl"]
    np.testing.assert_allclose([event.start_s for event in events], [0.1, 0.2])


@pytest.mark.parametrize("column", ["event", "events"])
def test_read_events_csv_accepts_event_named_time_columns(
    tmp_path,
    column: str,
) -> None:
    path = tmp_path / f"{column}.csv"
    path.write_text(f"{column},label\n0.2,cue\n0.1,reward\n", encoding="utf-8")

    events = read_events_csv(path)

    assert [event.label for event in events] == ["reward", "cue"]
    np.testing.assert_allclose([event.start_s for event in events], [0.1, 0.2])


def test_csv_readers_enforce_file_size_limit(tmp_path) -> None:
    path = tmp_path / "photometry.csv"
    path.write_text("time,gcamp\n0,1\n1,2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exceeds max load size"):
        read_photometry_csv(path, max_mb=0.000001)
