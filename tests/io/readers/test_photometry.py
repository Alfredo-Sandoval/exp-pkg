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
    assert recording.metadata["sampling_rate_hz"] == pytest.approx(10.0)
    assert recording.metadata["sampling_rate_source"] == "time.timestamps_uniform"
    np.testing.assert_allclose(recording.timeline.timestamps_s, [0.0, 0.1, 0.2])
    np.testing.assert_allclose(recording.series.values[:, 0], [1.0, 1.1, 1.2])


@pytest.mark.parametrize(
    ("kwargs", "exc_type", "message"),
    [
        (
            {"time_column": " time"},
            ValueError,
            "time_column must not contain surrounding whitespace",
        ),
        (
            {"time_column": 0},
            TypeError,
            "time_column must be a string",
        ),
        (
            {"signal_columns": "gcamp"},
            TypeError,
            "signal_columns must be a sequence of strings, not a string",
        ),
        (
            {"signal_columns": [" gcamp"]},
            ValueError,
            "signal_columns\\[0\\] must not contain surrounding whitespace",
        ),
        (
            {"signal_columns": []},
            ValueError,
            "signal_columns must be a non-empty sequence of strings",
        ),
    ],
)
def test_read_photometry_csv_rejects_unclean_explicit_column_selectors(
    tmp_path,
    kwargs: dict[str, object],
    exc_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exc_type, match=message):
        read_photometry_csv(tmp_path / "missing.csv", **kwargs)  # type: ignore[arg-type]


def test_read_photometry_csv_reports_resolved_layout_metadata(tmp_path) -> None:
    path = tmp_path / "photometry.csv"
    path.write_text(
        "timestamp,gcamp,isosbestic\n0,1.0,0.5\n1,1.1,0.4\n",
        encoding="utf-8",
    )

    recording = read_photometry_csv(path)

    assert recording.metadata["time_column"] == "timestamp"
    assert recording.metadata["signal_columns"] == ["gcamp", "isosbestic"]
    assert recording.metadata["size_bytes"] == path.stat().st_size
    assert recording.metadata["sampling_rate_hz"] == pytest.approx(1.0)
    assert recording.metadata["sampling_rate_source"] == "timestamp.timestamps_uniform"
    assert recording.channel_names == ("gcamp", "isosbestic")


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
    assert recording.metadata["sampling_rate_hz"] == pytest.approx(20.0)
    assert recording.metadata["sampling_rate_source"] == "sample_rate_hz.argument"


def test_read_photometry_csv_can_use_sample_rate_for_single_timestamp(tmp_path) -> None:
    path = tmp_path / "single_sample.csv"
    path.write_text("time,gcamp\n2.0,1.0\n", encoding="utf-8")

    recording = read_photometry_csv(path, sample_rate_hz=20.0)

    np.testing.assert_allclose(recording.timeline.timestamps_s, [2.0])
    assert recording.metadata["sampling_rate_hz"] == pytest.approx(20.0)
    assert recording.metadata["sampling_rate_source"] == "sample_rate_hz.argument"


def test_read_photometry_csv_rejects_single_timestamp_without_sample_rate(
    tmp_path,
) -> None:
    path = tmp_path / "single_sample.csv"
    path.write_text("time,gcamp\n2.0,1.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="sampling_rate_hz"):
        read_photometry_csv(path)


def test_read_photometry_csv_rejects_irregular_timebase(tmp_path) -> None:
    path = tmp_path / "irregular.csv"
    path.write_text(
        "time,gcamp\n0.0,1.0\n0.1,1.1\n0.25,1.2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="uniformly sampled"):
        read_photometry_csv(path)


def test_read_photometry_csv_rejects_mismatched_sample_rate_hint(tmp_path) -> None:
    path = tmp_path / "photometry.csv"
    path.write_text(
        "time,gcamp\n0.0,1.0\n0.1,1.1\n0.2,1.2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sample_rate_hz"):
        read_photometry_csv(path, sample_rate_hz=5.0)


def test_read_photometry_csv_rejects_missing_timebase(tmp_path) -> None:
    path = tmp_path / "single.csv"
    path.write_text("gcamp\n1.0\n1.1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="time column or sample_rate_hz"):
        read_photometry_csv(path)


def test_read_photometry_csv_can_require_named_time_column(tmp_path) -> None:
    path = tmp_path / "implicit_time.csv"
    path.write_text("sample,gcamp\n0,1.0\n1,1.1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must include a time column"):
        read_photometry_csv(path, allow_implicit_time_column=False)


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
    assert events.metadata == {
        "source_type": "events_csv",
        "source": {"type": "events_csv", "path": str(path)},
        "time_column": "timestamp",
        "kind_column": "type",
        "label_column": "label",
        "duration_column": "duration",
        "columns": ["timestamp", "type", "label", "duration"],
        "rows": 2,
        "event_records": [
            {
                "source_row": 0,
                "time_s": 1.0,
                "duration_s": 0.1,
                "kind": "cue",
                "label": "tone",
            },
            {
                "source_row": 1,
                "time_s": 1.5,
                "duration_s": 0.5,
                "kind": "trial",
                "label": "A",
            },
        ],
        "time_unit": "ms",
        "default_kind": "event",
    }


def test_read_events_csv_preserves_empty_table_metadata(tmp_path) -> None:
    path = tmp_path / "empty_events.csv"
    path.write_text("timestamps,label,duration\n", encoding="utf-8")

    events = read_events_csv(path, default_kind="ttl")

    assert list(events) == []
    assert events.metadata == {
        "source_type": "events_csv",
        "source": {"type": "events_csv", "path": str(path)},
        "time_column": "timestamps",
        "kind_column": None,
        "label_column": "label",
        "duration_column": "duration",
        "columns": ["timestamps", "label", "duration"],
        "rows": 0,
        "event_records": [],
        "time_unit": "s",
        "default_kind": "ttl",
    }


def test_read_events_csv_records_normalized_event_rows(tmp_path) -> None:
    path = tmp_path / "state_events.csv"
    path.write_text(
        "timestamps,state,duration\n0.2,2,0.01\n0.1,1,0.02\n",
        encoding="utf-8",
    )

    events = read_events_csv(path, default_kind="ttl")

    assert [event.metadata["source_row"] for event in events] == [1, 0]
    assert events.metadata["event_records"] == [
        {
            "source_row": 1,
            "time_s": 0.1,
            "duration_s": 0.02,
            "kind": "ttl",
            "label": "state=1",
        },
        {
            "source_row": 0,
            "time_s": 0.2,
            "duration_s": 0.01,
            "kind": "ttl",
            "label": "state=2",
        },
    ]


def test_read_events_csv_rejects_empty_file_without_time_column(tmp_path) -> None:
    path = tmp_path / "empty_without_time.csv"
    path.write_text("label\n", encoding="utf-8")

    with pytest.raises(ValueError, match="timestamp column"):
        read_events_csv(path)


@pytest.mark.parametrize(
    ("kwargs", "exc_type", "message"),
    [
        (
            {"time_column": " timestamp"},
            ValueError,
            "time_column must not contain surrounding whitespace",
        ),
        (
            {"kind_column": 0},
            TypeError,
            "kind_column must be a string",
        ),
        (
            {"label_column": ""},
            ValueError,
            "label_column must be a non-empty string",
        ),
        (
            {"duration_column": " duration"},
            ValueError,
            "duration_column must not contain surrounding whitespace",
        ),
    ],
)
def test_read_events_csv_rejects_unclean_explicit_column_selectors(
    tmp_path,
    kwargs: dict[str, object],
    exc_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exc_type, match=message):
        read_events_csv(tmp_path / "missing.csv", **kwargs)  # type: ignore[arg-type]


def test_read_events_csv_uses_default_kind_without_label_columns(tmp_path) -> None:
    path = tmp_path / "timestamps.csv"
    path.write_text("timestamps\n0.2\n0.1\n", encoding="utf-8")

    events = read_events_csv(path, default_kind="ttl")

    assert [event.kind for event in events] == ["ttl", "ttl"]
    np.testing.assert_allclose([event.start_s for event in events], [0.1, 0.2])


def test_read_events_csv_accepts_state_as_numeric_label_column(tmp_path) -> None:
    path = tmp_path / "state_events.csv"
    path.write_text("timestamps,state\n0.2,2\n0.1,1\n0.3,1\n", encoding="utf-8")

    events = read_events_csv(path)

    assert [event.kind for event in events] == ["event", "event", "event"]
    assert [event.label for event in events] == ["state=1", "state=2", "state=1"]
    np.testing.assert_allclose([event.start_s for event in events], [0.1, 0.2, 0.3])


def test_read_events_csv_rejects_padded_label(tmp_path) -> None:
    path = tmp_path / "events.csv"
    path.write_text("timestamps,label\n0.1, cue\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"label column 'label' at row 0"):
        read_events_csv(path)


def test_read_events_csv_rejects_padded_kind(tmp_path) -> None:
    path = tmp_path / "events.csv"
    path.write_text("timestamps,type,label\n0.1, cue,tone\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"kind column 'type' at row 0"):
        read_events_csv(path)


def test_read_events_csv_rejects_missing_label_cell(tmp_path) -> None:
    path = tmp_path / "events.csv"
    path.write_text("timestamps,label\n0.1,\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"label column 'label' at row 0"):
        read_events_csv(path)


def test_read_events_csv_rejects_padded_default_kind(tmp_path) -> None:
    path = tmp_path / "events.csv"
    path.write_text("timestamps\n0.1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="default_kind"):
        read_events_csv(path, default_kind=" ttl")


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
