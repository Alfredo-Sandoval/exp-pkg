from __future__ import annotations

from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from xpkg.io.readers import (
    find_first_doric_photometry_file,
    find_first_neurophotometrics_csv,
    find_first_nwb_photometry_file,
    find_first_teleopto_h5,
    find_photometry_session_entries,
    find_tdt_block_directories,
    is_doric_photometry_file,
    is_neurophotometrics_csv,
    is_nwb_photometry_file,
    is_rwd_ofrs_session,
    is_tdt_block,
    is_teleopto_h5,
    neurophotometrics_channel_selection_from_label,
    neurophotometrics_source_column_from_label,
    parse_teleopto_h5_arrays,
    read_doric_photometry,
    read_neurophotometrics_csv,
    read_nwb_photometry,
    read_pmat_events_csv,
    read_pmat_photometry_csv,
    read_rwd_ofrs_session,
    read_tdt_photometry_block,
    read_teleopto_h5,
    resolve_tdt_block_path,
)
from xpkg.model import EventTable, PhotometryRecording, RecordingSession


def test_read_pmat_csv_and_events(tmp_path) -> None:
    photometry_path = tmp_path / "pmat.csv"
    photometry_path.write_text(
        "Time,Signal,Control\n0.0,1.0,0.5\n0.1,1.1,0.4\n",
        encoding="utf-8",
    )
    event_path = tmp_path / "events.csv"
    event_path.write_text(
        "Name,Onset,Offset\ncue,0.25,0.4\n",
        encoding="utf-8",
    )

    photometry = read_pmat_photometry_csv(photometry_path)
    events = read_pmat_events_csv(event_path)

    assert isinstance(photometry, PhotometryRecording)
    assert photometry.signal_channel == "Signal"
    assert photometry.reference_channel == "Control"
    assert photometry.metadata["time_column"] == "Time"
    assert photometry.metadata["signal_column"] == "Signal"
    assert photometry.metadata["reference_column"] == "Control"
    assert photometry.metadata["signal_columns"] == ["Signal", "Control"]
    assert photometry.metadata["columns"] == ["Time", "Signal", "Control"]
    assert photometry.metadata["rows"] == 2
    assert photometry.metadata["time_unit"] == "s"
    assert photometry.metadata["sampling_rate_hz"] == pytest.approx(10.0)
    assert photometry.metadata["sampling_rate_source"] == "Time.timestamps_uniform"
    np.testing.assert_allclose(photometry.timeline.timestamps_s, [0.0, 0.1])
    assert isinstance(events, EventTable)
    assert events.events[0].label == "cue"
    assert events.events[0].duration_s == pytest.approx(0.15)
    assert events.metadata["source_type"] == "pmat_events_csv"
    assert events.metadata["label_column"] == "Name"
    assert events.metadata["onset_column"] == "Onset"
    assert events.metadata["offset_column"] == "Offset"
    assert events.metadata["columns"] == ["Name", "Onset", "Offset"]
    assert events.metadata["rows"] == 1
    assert events.metadata["time_unit"] == "s"


def test_read_pmat_events_rejects_nonfinite_offset(tmp_path) -> None:
    event_path = tmp_path / "events.csv"
    event_path.write_text(
        "Name,Onset,Offset\ncue,0.25,nan\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"Offset.*non-finite"):
        read_pmat_events_csv(event_path)


def test_read_pmat_events_rejects_offset_before_onset(tmp_path) -> None:
    event_path = tmp_path / "events.csv"
    event_path.write_text(
        "Name,Onset,Offset\ncue,0.40,0.25\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"offset column 'Offset'.*onset column"):
        read_pmat_events_csv(event_path)


def test_read_pmat_events_rejects_padded_label(tmp_path) -> None:
    event_path = tmp_path / "events.csv"
    event_path.write_text(
        "Name,Onset,Offset\n cue,0.25,0.4\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"label column 'Name' at row 0"):
        read_pmat_events_csv(event_path)


def test_read_pmat_events_rejects_numeric_label(tmp_path) -> None:
    event_path = tmp_path / "events.csv"
    event_path.write_text(
        "Name,Onset,Offset\n1,0.25,0.4\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"label column 'Name' at row 0"):
        read_pmat_events_csv(event_path)


def test_read_pmat_csv_rejects_file_exceeding_max_mb(tmp_path) -> None:
    photometry_path = tmp_path / "pmat.csv"
    photometry_path.write_text(
        "Time,Signal,Control\n" + "\n".join(f"{i},{i}.0,{i}.5" for i in range(3000)),
        encoding="utf-8",
    )
    event_path = tmp_path / "events.csv"
    event_path.write_text(
        "Name,Onset,Offset\n" + "\n".join(f"cue,{i}.0,{i}.1" for i in range(3000)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exceeds max load size"):
        read_pmat_photometry_csv(photometry_path, max_mb=0.0001)
    with pytest.raises(ValueError, match="exceeds max load size"):
        read_pmat_events_csv(event_path, max_mb=0.0001)


def test_is_neurophotometrics_csv_detects_led_state_contract(tmp_path) -> None:
    flags_path = tmp_path / "flags.csv"
    flags_path.write_text(
        "FrameCounter,Timestamp,Flags,Region0G\n0,0.0,1,0.2\n",
        encoding="utf-8",
    )
    led_state_path = tmp_path / "ledstate.csv"
    led_state_path.write_text(
        "Timestamp,LedState,Region0G\n0.0,2,0.2\n",
        encoding="utf-8",
    )
    plain_path = tmp_path / "plain.csv"
    plain_path.write_text("time,signal\n0.0,1.0\n", encoding="utf-8")

    assert is_neurophotometrics_csv(flags_path) is True
    assert is_neurophotometrics_csv(led_state_path) is True
    assert is_neurophotometrics_csv(plain_path) is False
    assert is_neurophotometrics_csv(tmp_path / "missing.csv") is False


def test_find_first_neurophotometrics_csv_uses_detector_contract(tmp_path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    plain_path = tmp_path / "plain.csv"
    plain_path.write_text("time,signal\n0.0,1.0\n", encoding="utf-8")
    match_path = nested / "npm.csv"
    match_path.write_text(
        "Timestamp,LedState,Region0G\n0.0,2,0.2\n",
        encoding="utf-8",
    )

    assert find_first_neurophotometrics_csv(tmp_path) == match_path
    assert find_first_neurophotometrics_csv(plain_path) is None


def test_read_neurophotometrics_csv_demuxes_led_state_channels(tmp_path) -> None:
    path = tmp_path / "npm.csv"
    path.write_text(
        "\n".join(
            [
                "FrameCounter,Timestamp,Flags,Region0G,Region0R",
                "0,0.0,1,0.2,0.1",
                "1,0.1,2,0.3,0.2",
                "2,0.2,1,0.25,0.15",
                "3,0.3,2,0.35,0.25",
            ]
        ),
        encoding="utf-8",
    )

    session = read_neurophotometrics_csv(path, reference_column="Region0R")
    photometry = session.signals["photometry"]

    assert isinstance(session, RecordingSession)
    assert isinstance(photometry, PhotometryRecording)
    assert photometry.channel_names == ("Region0G_470nm", "Region0R_415nm")
    assert photometry.signal_channel == "Region0G_470nm"
    assert photometry.reference_channel == "Region0R_415nm"
    assert photometry.metadata["led_demux"]["applied"] is True
    assert photometry.metadata["led_demux"]["codes_present"] == [1, 2]
    assert photometry.metadata["led_demux"]["code_to_nm"] == {
        "1": 415,
        "2": 470,
        "4": 560,
    }
    assert photometry.metadata["sampling_rate_hz"] == pytest.approx(5.0)
    assert (
        photometry.metadata["sampling_rate_source"] == "Timestamp.demux_signal.timestamps_uniform"
    )
    assert session.metadata["sampling_rate_hz"] == pytest.approx(5.0)
    assert session.metadata["sampling_rate_source"] == "Timestamp.demux_signal.timestamps_uniform"
    np.testing.assert_allclose(photometry.timeline.timestamps_s, [0.1, 0.3])
    np.testing.assert_allclose(photometry.series.values, [[0.3, 0.1], [0.35, 0.15]])
    assert "flags" in session.signals
    assert session.metadata["frame_counter"] == [0, 1, 2, 3]


def test_neurophotometrics_source_column_from_label_strips_demux_suffix() -> None:
    assert neurophotometrics_source_column_from_label("Region1G_470nm") == "Region1G"
    assert neurophotometrics_source_column_from_label("Region1G_led_state_7") == "Region1G"
    assert neurophotometrics_source_column_from_label("Region1G") == "Region1G"
    assert neurophotometrics_source_column_from_label(None) is None
    assert neurophotometrics_source_column_from_label("") is None


def test_neurophotometrics_channel_selection_from_label_parses_led_identity() -> None:
    assert neurophotometrics_channel_selection_from_label("Region1G_560nm") == (
        "Region1G",
        560,
        None,
    )
    assert neurophotometrics_channel_selection_from_label("Region1G_led_state_7") == (
        "Region1G",
        None,
        7,
    )
    assert neurophotometrics_channel_selection_from_label("Region1G") == (
        "Region1G",
        None,
        None,
    )
    assert neurophotometrics_channel_selection_from_label(None) == (None, None, None)


@pytest.mark.parametrize(
    ("label", "exc_type", "message"),
    [
        (
            " Region1G_470nm",
            ValueError,
            "Neurophotometrics label must be a non-empty string without",
        ),
        (
            "Region1G_led_state_7 ",
            ValueError,
            "Neurophotometrics label must be a non-empty string without",
        ),
        (
            470,
            TypeError,
            "Neurophotometrics label must be a string or None",
        ),
        (
            "_470nm",
            ValueError,
            "source column before the demux suffix",
        ),
    ],
)
def test_neurophotometrics_channel_selection_from_label_rejects_unclean_label(
    label: object,
    exc_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exc_type, match=message):
        neurophotometrics_channel_selection_from_label(label)  # type: ignore[arg-type]


def test_read_neurophotometrics_csv_accepts_custom_led_map(tmp_path) -> None:
    path = tmp_path / "custom_npm.csv"
    path.write_text(
        "\n".join(
            [
                "Timestamp,LedState,Region0G",
                "0.0,7,100.0",
                "0.1,1,50.0",
                "0.2,7,101.0",
                "0.3,1,51.0",
            ]
        ),
        encoding="utf-8",
    )

    photometry = read_neurophotometrics_csv(
        path,
        led_code_to_nm={7: 470, 1: 415},
    ).signals["photometry"]

    assert photometry.signal_channel == "Region0G_470nm"
    assert photometry.reference_channel == "Region0G_415nm"
    assert photometry.metadata["led_demux"]["signal_code"] == 7


def test_read_neurophotometrics_csv_selects_requested_signal_wavelength(
    tmp_path,
) -> None:
    path = tmp_path / "npm_560.csv"
    path.write_text(
        "\n".join(
            [
                "Timestamp,LedState,Region0G",
                "0.0,2,100.0",
                "0.1,1,50.0",
                "0.2,4,700.0",
                "0.3,2,101.0",
                "0.4,1,51.0",
                "0.5,4,701.0",
            ]
        ),
        encoding="utf-8",
    )

    photometry = read_neurophotometrics_csv(
        path,
        signal_nm=560,
        reference_nm=415,
    ).signals["photometry"]

    assert photometry.signal_channel == "Region0G_560nm"
    assert photometry.reference_channel == "Region0G_415nm"
    assert photometry.metadata["led_demux"]["signal_code"] == 4
    np.testing.assert_allclose(photometry.series.values[:, 0], [700.0, 701.0])
    np.testing.assert_allclose(photometry.series.values[:, 1], [50.0, 51.0])


def test_read_neurophotometrics_csv_selects_requested_led_code(
    tmp_path,
) -> None:
    path = tmp_path / "npm_led_code.csv"
    path.write_text(
        "\n".join(
            [
                "Timestamp,LedState,Region0G",
                "0.0,2,100.0",
                "0.1,1,50.0",
                "0.2,4,700.0",
                "0.3,2,101.0",
                "0.4,1,51.0",
                "0.5,4,701.0",
            ]
        ),
        encoding="utf-8",
    )

    photometry = read_neurophotometrics_csv(
        path,
        signal_led_code=4,
        reference_led_code=1,
    ).signals["photometry"]

    assert photometry.signal_channel == "Region0G_560nm"
    assert photometry.reference_channel == "Region0G_415nm"
    assert photometry.metadata["led_demux"]["signal_selection"] == "led_state_code"
    assert photometry.metadata["led_demux"]["reference_selection"] == "led_state_code"
    np.testing.assert_allclose(photometry.series.values[:, 0], [700.0, 701.0])
    np.testing.assert_allclose(photometry.series.values[:, 1], [50.0, 51.0])


def test_read_neurophotometrics_csv_rejects_missing_requested_led_code(
    tmp_path,
) -> None:
    path = tmp_path / "npm.csv"
    path.write_text(
        "\n".join(
            [
                "Timestamp,LedState,Region0G",
                "0.0,2,100.0",
                "0.1,1,50.0",
                "0.2,2,101.0",
                "0.3,1,51.0",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requested signal LedState code 4"):
        read_neurophotometrics_csv(path, signal_led_code=4)


def test_read_neurophotometrics_csv_rejects_same_signal_and_reference_wavelength(
    tmp_path,
) -> None:
    path = tmp_path / "npm.csv"
    path.write_text(
        "\n".join(
            [
                "Timestamp,LedState,Region0G",
                "0.0,2,100.0",
                "0.1,1,50.0",
                "0.2,2,101.0",
                "0.3,1,51.0",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="same LedState code"):
        read_neurophotometrics_csv(path, signal_led_code=2, reference_nm=470)


def test_read_neurophotometrics_csv_preserves_raw_channels_without_state(
    tmp_path,
) -> None:
    path = tmp_path / "raw_npm.csv"
    path.write_text(
        "\n".join(
            [
                "Timestamp,Region0G,Region0R",
                "0.0,0.2,0.1",
                "0.1,0.3,0.2",
            ]
        ),
        encoding="utf-8",
    )

    photometry = read_neurophotometrics_csv(
        path,
        reference_column="Region0R",
    ).signals["photometry"]

    assert photometry.channel_names == ("Region0G", "Region0R")
    assert photometry.reference_channel == "Region0R"
    assert photometry.metadata["sampling_rate_hz"] == pytest.approx(10.0)
    assert photometry.metadata["sampling_rate_source"] == "Timestamp.timestamps_uniform"
    assert photometry.metadata["led_demux"] == {"applied": False}


def test_read_neurophotometrics_csv_rejects_irregular_demux_timebase(tmp_path) -> None:
    path = tmp_path / "irregular_npm.csv"
    path.write_text(
        "\n".join(
            [
                "Timestamp,LedState,Region0G",
                "0.0,1,50.0",
                "0.1,2,100.0",
                "0.25,1,51.0",
                "0.4,2,101.0",
                "0.55,1,52.0",
                "0.8,2,102.0",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="uniformly sampled"):
        read_neurophotometrics_csv(path, led_code_to_nm={2: 470, 1: 415})


def test_read_neurophotometrics_csv_rejects_irregular_raw_timebase(tmp_path) -> None:
    path = tmp_path / "irregular_raw_npm.csv"
    path.write_text(
        "\n".join(
            [
                "Timestamp,Region0G",
                "0.0,0.2",
                "0.1,0.3",
                "0.25,0.4",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="uniformly sampled"):
        read_neurophotometrics_csv(path)


def test_is_rwd_ofrs_session_detects_fluorescence_bundle(tmp_path) -> None:
    session_dir = tmp_path / "rwd-session"
    session_dir.mkdir()
    (session_dir / "Fluorescence.csv").write_text(
        "\n".join(
            [
                '{"Fps":30.0;"Channels":[{"Name":"CH1"}]}',
                "TimeStamp,Events,CH1-410,CH1-470,",
                "0.000,,1.0,2.0,",
            ]
        ),
        encoding="utf-8",
    )
    no_metadata_dir = tmp_path / "rwd-session-no-metadata"
    no_metadata_dir.mkdir()
    (no_metadata_dir / "Fluorescence.csv").write_text(
        "Timestamp,Events,CH1-560,\n0.000,,3.0,\n",
        encoding="utf-8",
    )
    plain_dir = tmp_path / "plain"
    plain_dir.mkdir()
    (plain_dir / "Fluorescence.csv").write_text(
        "time,signal\n0.0,1.0\n",
        encoding="utf-8",
    )

    assert is_rwd_ofrs_session(session_dir) is True
    assert is_rwd_ofrs_session(no_metadata_dir) is True
    assert is_rwd_ofrs_session(plain_dir) is False
    assert is_rwd_ofrs_session(tmp_path / "missing") is False


def test_read_rwd_ofrs_session_parses_multicolor_bundle_and_events(tmp_path) -> None:
    session_dir = tmp_path / "rwd-session"
    session_dir.mkdir()
    (session_dir / "Fluorescence.csv").write_text(
        "\n".join(
            [
                '{"Fps":30.0;"Channels":[{"Name":"CH1"}]}',
                "TimeStamp,Events,CH1-410,CH1-470,CH1-560,",
                "0.000,,1.0,2.0,3.0,",
                "33.333,,1.1,2.1,3.1,",
                "66.666,,1.2,2.2,3.2,",
            ]
        ),
        encoding="utf-8",
    )
    (session_dir / "Events.csv").write_text(
        "TimeStamp,Name,State\n33.333,brush,0\n66.666,brush,1\n",
        encoding="utf-8",
    )

    session = read_rwd_ofrs_session(session_dir)
    photometry = session.signals["photometry"]

    assert isinstance(photometry, PhotometryRecording)
    assert photometry.channel_names == ("CH1-410", "CH1-470", "CH1-560")
    assert photometry.signal_channel == "CH1-470"
    assert photometry.reference_channel == "CH1-410"
    assert photometry.metadata["time_scale_inference"] == "declared_fps_milliseconds"
    assert photometry.metadata["declared_fps_hz"] == pytest.approx(30.0)
    assert photometry.metadata["median_raw_time_delta"] == pytest.approx(33.333)
    assert session.metadata["events_csv"] == {
        "present": True,
        "path": str(session_dir / "Events.csv"),
        "row_count": 2,
        "time_column": "TimeStamp",
        "unique_labels": ["brush"],
        "time_monotonic": True,
    }
    assert [event.label for event in session.events] == ["brush", "brush_offset"]
    np.testing.assert_allclose([event.start_s for event in session.events], [0.033333, 0.066666])


def test_read_rwd_ofrs_session_records_source_event_order_metadata(tmp_path) -> None:
    session_dir = tmp_path / "rwd-session"
    session_dir.mkdir()
    (session_dir / "Fluorescence.csv").write_text(
        "\n".join(
            [
                '{"Fps":30.0;"Channels":[{"Name":"CH1"}]}',
                "TimeStamp,Events,CH1-410,CH1-470,",
                "0.000,,1.0,2.0,",
                "33.333,,1.1,2.1,",
                "66.666,,1.2,2.2,",
            ]
        ),
        encoding="utf-8",
    )
    (session_dir / "Events.csv").write_text(
        "TimeStamp,Name,State\n66.666,brush,1\n33.333,brush,0\n",
        encoding="utf-8",
    )

    session = read_rwd_ofrs_session(session_dir)

    assert session.metadata["events_csv"]["time_monotonic"] is False
    assert [event.label for event in session.events] == ["brush", "brush_offset"]
    np.testing.assert_allclose(
        [event.start_s for event in session.events],
        [0.033333, 0.066666],
    )


@pytest.mark.parametrize("state", ["2", "0.5"])
def test_read_rwd_ofrs_session_rejects_invalid_event_state(
    tmp_path,
    state: str,
) -> None:
    session_dir = tmp_path / "rwd-session"
    session_dir.mkdir()
    (session_dir / "Fluorescence.csv").write_text(
        "\n".join(
            [
                '{"Fps":30.0;"Channels":[{"Name":"CH1"}]}',
                "TimeStamp,Events,CH1-410,CH1-470,",
                "0.000,,1.0,2.0,",
                "33.333,,1.1,2.1,",
            ]
        ),
        encoding="utf-8",
    )
    (session_dir / "Events.csv").write_text(
        f"TimeStamp,Name,State\n33.333,brush,{state}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="State values must be 0"):
        read_rwd_ofrs_session(session_dir)


def test_read_rwd_ofrs_session_rejects_empty_event_name(tmp_path) -> None:
    session_dir = tmp_path / "rwd-session"
    session_dir.mkdir()
    (session_dir / "Fluorescence.csv").write_text(
        "\n".join(
            [
                '{"Fps":30.0;"Channels":[{"Name":"CH1"}]}',
                "TimeStamp,Events,CH1-410,CH1-470,",
                "0.000,,1.0,2.0,",
                "33.333,,1.1,2.1,",
            ]
        ),
        encoding="utf-8",
    )
    (session_dir / "Events.csv").write_text(
        "TimeStamp,Name,State\n33.333,,0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"Name column at row 0"):
        read_rwd_ofrs_session(session_dir)


@pytest.mark.parametrize("name", [" brush", "1"])
def test_read_rwd_ofrs_session_rejects_malformed_event_name(
    tmp_path,
    name: str,
) -> None:
    session_dir = tmp_path / "rwd-session"
    session_dir.mkdir()
    (session_dir / "Fluorescence.csv").write_text(
        "\n".join(
            [
                '{"Fps":30.0;"Channels":[{"Name":"CH1"}]}',
                "TimeStamp,Events,CH1-410,CH1-470,",
                "0.000,,1.0,2.0,",
                "33.333,,1.1,2.1,",
            ]
        ),
        encoding="utf-8",
    )
    (session_dir / "Events.csv").write_text(
        f"TimeStamp,Name,State\n33.333,{name},0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"Name column at row 0"):
        read_rwd_ofrs_session(session_dir)


def _write_rwd_fluorescence(
    session_dir, *, metadata_line: str | None, timestamps: list[float]
) -> None:
    session_dir.mkdir()
    rows = ["TimeStamp,Events,CH1-410,CH1-470,"]
    for index, stamp in enumerate(timestamps):
        rows.append(f"{stamp:.6f},,{1.0 + 0.1 * index},{2.0 + 0.1 * index},")
    lines = ([metadata_line] if metadata_line is not None else []) + rows
    (session_dir / "Fluorescence.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_find_photometry_session_entries_uses_xpkg_format_detectors(tmp_path) -> None:
    generic_h5 = tmp_path / "generic.h5"
    with h5py.File(generic_h5, "w") as handle:
        handle.create_dataset("signal", data=np.asarray([1.0, 2.0]))

    hidden_dir = tmp_path / ".hidden"
    hidden_dir.mkdir()
    hidden_h5 = hidden_dir / "hidden.h5"
    with h5py.File(hidden_h5, "w") as handle:
        handle.create_dataset("signal", data=np.asarray([3.0, 4.0]))

    ofrs_dir = tmp_path / "ofrs"
    _write_rwd_fluorescence(
        ofrs_dir,
        metadata_line='{"Fps":30.0;"Channels":[{"Name":"CH1"}]}',
        timestamps=[0.0, 33.333],
    )

    tdt_dir = tmp_path / "tdt"
    tdt_dir.mkdir()
    (tdt_dir / "block.tsq").write_bytes(b"")
    (tdt_dir / "block.tev").write_bytes(b"")

    ppd_path = tmp_path / "session.ppd"
    ppd_header = b'{"sampling_rate": 130.0}'
    ppd_path.write_bytes(len(ppd_header).to_bytes(2, "little") + ppd_header)

    pyphotometry_csv = tmp_path / "pyphotometry.csv"
    pyphotometry_csv.write_text("Analog1, Analog2\n1.0,0.5\n", encoding="utf-8")

    entries = find_photometry_session_entries(tmp_path)

    assert entries == sorted(
        [
            generic_h5.resolve(),
            ofrs_dir.resolve(),
            ppd_path.resolve(),
            pyphotometry_csv.resolve(),
            tdt_dir.resolve(),
        ],
        key=str,
    )
    assert hidden_h5.resolve() not in entries

    with_hidden = find_photometry_session_entries(
        tmp_path,
        include_hidden_dirs=True,
    )
    assert hidden_h5.resolve() in with_hidden


def test_find_photometry_session_entries_can_require_known_formats(tmp_path) -> None:
    generic_h5 = tmp_path / "generic.h5"
    with h5py.File(generic_h5, "w") as handle:
        handle.create_dataset("signal", data=np.asarray([1.0, 2.0]))

    assert (
        find_photometry_session_entries(
            tmp_path,
            include_generic_hdf5=False,
        )
        == []
    )


def test_read_rwd_ofrs_slow_seconds_anchors_to_declared_fps(tmp_path) -> None:
    # 0.5 Hz acquisition in seconds (2 s/sample) must stay in seconds, not be
    # divided by 1000 because the spacing is >= 1.
    _write_rwd_fluorescence(
        tmp_path / "slow",
        metadata_line='{"Fps":0.5;"Channels":[{"Name":"CH1"}]}',
        timestamps=[0.0, 2.0, 4.0, 6.0],
    )
    session = read_rwd_ofrs_session(tmp_path / "slow")
    photometry = session.signals["photometry"]
    assert photometry.metadata["sampling_rate_hz"] == pytest.approx(0.5)
    assert photometry.metadata["time_scale"] == pytest.approx(1.0)
    assert photometry.metadata["time_scale_inference"] == "declared_fps_seconds"
    assert photometry.metadata["declared_fps_hz"] == pytest.approx(0.5)
    assert photometry.metadata["median_raw_time_delta"] == pytest.approx(2.0)
    np.testing.assert_allclose(photometry.timeline.timestamps_s, [0.0, 2.0, 4.0, 6.0])


def test_read_rwd_ofrs_subsecond_without_fps_infers_milliseconds(tmp_path) -> None:
    _write_rwd_fluorescence(tmp_path / "sub", metadata_line=None, timestamps=[0.0, 0.4, 0.8, 1.2])
    session = read_rwd_ofrs_session(tmp_path / "sub")
    photometry = session.signals["photometry"]
    assert photometry.metadata["time_scale"] == pytest.approx(0.001)
    assert photometry.metadata["time_scale_inference"] == "subsecond_spacing_milliseconds"
    assert photometry.metadata["declared_fps_hz"] is None
    assert photometry.metadata["median_raw_time_delta"] == pytest.approx(0.4)
    assert photometry.metadata["sampling_rate_hz"] == pytest.approx(2500.0, rel=1e-6)


def test_read_rwd_ofrs_ambiguous_spacing_without_fps_raises(tmp_path) -> None:
    _write_rwd_fluorescence(tmp_path / "amb", metadata_line=None, timestamps=[0.0, 2.0, 4.0, 6.0])
    with pytest.raises(ValueError, match="ambiguous"):
        read_rwd_ofrs_session(tmp_path / "amb")


def test_is_doric_photometry_file_detects_doric_hdf5_contract(tmp_path) -> None:
    path = tmp_path / "recording.doric"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("Signal470", data=np.asarray([1.0, 1.1, 1.2]))
        handle.create_dataset("Time", data=np.asarray([0.0, 0.1, 0.2]))

    wrong_suffix = tmp_path / "recording.h5"
    with h5py.File(wrong_suffix, "w") as handle:
        handle.create_dataset("Signal470", data=np.asarray([1.0, 1.1, 1.2]))

    not_hdf5 = tmp_path / "not-hdf5.doric"
    not_hdf5.write_text("not hdf5", encoding="utf-8")

    only_matrix = tmp_path / "matrix-only.doric"
    with h5py.File(only_matrix, "w") as handle:
        handle.create_dataset("Signal470", data=np.asarray([[1.0, 1.1], [1.2, 1.3]]))

    assert is_doric_photometry_file(path) is True
    assert is_doric_photometry_file(wrong_suffix) is False
    assert is_doric_photometry_file(not_hdf5) is False
    assert is_doric_photometry_file(only_matrix) is False
    assert is_doric_photometry_file(tmp_path / "missing.doric") is False


def test_find_first_doric_photometry_file_uses_detector_contract(tmp_path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    not_doric = tmp_path / "plain.h5"
    with h5py.File(not_doric, "w") as handle:
        handle.create_dataset("signal", data=np.arange(3, dtype=float))
    match_path = nested / "recording.doric"
    with h5py.File(match_path, "w") as handle:
        handle.create_dataset("signal", data=np.arange(3, dtype=float))

    assert find_first_doric_photometry_file(tmp_path) == match_path
    assert find_first_doric_photometry_file(not_doric) is None


def test_read_doric_photometry_uses_hdf5_datasets(tmp_path) -> None:
    path = tmp_path / "recording.doric"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("Data")
        group.create_dataset("Signal", data=np.asarray([1.0, 1.1, 1.2]))
        group.create_dataset("Control", data=np.asarray([0.5, 0.4, 0.3]))
        handle.create_dataset("Time", data=np.asarray([0.0, 0.1, 0.2]))

    session = read_doric_photometry(path)
    photometry = session.signals["photometry"]

    assert isinstance(photometry, PhotometryRecording)
    assert photometry.signal_channel == "Data/Signal"
    assert photometry.reference_channel == "Data/Control"
    np.testing.assert_allclose(photometry.timeline.timestamps_s, [0.0, 0.1, 0.2])
    assert photometry.metadata["sampling_rate_hz"] == pytest.approx(10.0)
    assert photometry.metadata["sampling_rate_source"] == "Time.timestamps_uniform"
    assert session.metadata["sampling_rate_hz"] == pytest.approx(10.0)
    assert session.metadata["sampling_rate_source"] == "Time.timestamps_uniform"
    assert photometry.metadata["channel_inference"] == "wavelength_tokens"


def test_read_doric_photometry_records_sampling_rate_attribute_source(tmp_path) -> None:
    path = tmp_path / "rate_attr.doric"
    with h5py.File(path, "w") as handle:
        signal = handle.create_dataset("Signal470", data=np.asarray([1.0, 1.1, 1.2]))
        signal.attrs["SamplingRate"] = 20.0

    session = read_doric_photometry(path)
    photometry = session.signals["photometry"]

    assert photometry.metadata["sampling_rate_hz"] == pytest.approx(20.0)
    assert photometry.metadata["sampling_rate_source"] == "Signal470.attrs.SamplingRate"
    assert session.metadata["sampling_rate_hz"] == pytest.approx(20.0)
    assert session.metadata["sampling_rate_source"] == "Signal470.attrs.SamplingRate"


def test_read_doric_photometry_rejects_duplicate_signal_reference_path(
    tmp_path,
) -> None:
    path = tmp_path / "duplicate.doric"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("Signal470", data=np.asarray([1.0, 1.1, 1.2]))
        handle.create_dataset("Time", data=np.asarray([0.0, 0.1, 0.2]))

    with pytest.raises(ValueError, match="reference dataset must differ"):
        read_doric_photometry(
            path,
            signal_path="Signal470",
            reference_path="Signal470",
        )


def test_read_doric_photometry_rejects_time_dataset_as_signal(
    tmp_path,
) -> None:
    path = tmp_path / "time_signal.doric"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("Signal470", data=np.asarray([1.0, 1.1, 1.2]))
        handle.create_dataset("Time", data=np.asarray([0.0, 0.1, 0.2]))

    with pytest.raises(ValueError, match="signal dataset must not be the time"):
        read_doric_photometry(
            path,
            signal_path="Time",
            time_path="Time",
        )


def test_read_doric_photometry_rejects_irregular_time_dataset(tmp_path) -> None:
    path = tmp_path / "irregular.doric"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("Signal470", data=np.asarray([1.0, 1.1, 1.2]))
        handle.create_dataset("Time", data=np.asarray([0.0, 0.1, 0.25]))

    with pytest.raises(ValueError, match="uniformly sampled"):
        read_doric_photometry(path)


def test_read_doric_photometry_selects_560_signal_over_isosbestic(tmp_path) -> None:
    path = tmp_path / "rg.doric"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("Data")
        group.create_dataset("Sig560", data=np.asarray([1.0, 1.1, 1.2]))
        group.create_dataset("Ref405", data=np.asarray([0.5, 0.4, 0.3]))
        handle.create_dataset("Time", data=np.asarray([0.0, 0.1, 0.2]))

    photometry = read_doric_photometry(path).signals["photometry"]
    assert photometry.signal_channel == "Data/Sig560"
    assert photometry.reference_channel == "Data/Ref405"
    assert photometry.metadata["channel_inference"] == "wavelength_tokens"


def test_read_doric_photometry_flags_storage_order_inference(tmp_path) -> None:
    path = tmp_path / "unnamed.doric"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("ChannelA", data=np.asarray([1.0, 1.1, 1.2]))
        handle.create_dataset("ChannelB", data=np.asarray([0.5, 0.4, 0.3]))
        handle.create_dataset("Time", data=np.asarray([0.0, 0.1, 0.2]))

    photometry = read_doric_photometry(path).signals["photometry"]
    assert photometry.metadata["channel_inference"] == "storage_order"


def _write_nwb_series(
    parent: h5py.Group,
    name: str,
    data: np.ndarray,
    *,
    rate: float | None = 100.0,
    start: float = 0.0,
    timestamps: np.ndarray | None = None,
) -> h5py.Group:
    group = parent.create_group(name)
    group.create_dataset("data", data=data)
    if timestamps is not None:
        group.create_dataset("timestamps", data=timestamps)
    elif rate is not None:
        starting_time = group.create_dataset("starting_time", data=start)
        starting_time.attrs["rate"] = rate
    return group


def test_read_nwb_photometry_prefers_dff_and_extracts_events(tmp_path) -> None:
    path = tmp_path / "community.nwb"
    signal = np.tile(np.linspace(0.0, 1.0, 2000)[:, None], (1, 2))
    control = np.tile(np.full(2000, 0.5)[:, None], (1, 2))
    reward = np.zeros(20_000, dtype=np.float64)
    for onset_s in (2.0, 5.0, 9.0, 14.0):
        index = int(onset_s * 1000.0)
        reward[index : index + 50] = 1.0

    with h5py.File(path, "w") as handle:
        handle.create_dataset("identifier", data=np.bytes_("session-1"))
        acquisition = handle.create_group("acquisition")
        _write_nwb_series(acquisition, "Fluorescence", signal * 100.0)
        _write_nwb_series(acquisition, "FiberPhotometryResponseSeriesIsosbestic", control)
        _write_nwb_series(acquisition, "Reward", reward, rate=1000.0)
        annotations = acquisition.create_group("events")
        annotations.attrs["neurodata_type"] = "AnnotationSeries"
        annotations.create_dataset("data", data=np.asarray([b"lick", b"cue"]))
        annotations.create_dataset("timestamps", data=np.asarray([0.75, 6.5]))
        ttl = acquisition.create_group("TtlsTable")
        ttl.create_dataset("timestamp", data=np.linspace(0.0, 20.0, 4000))
        ttl.create_dataset("ttl_type", data=np.zeros(4000, dtype=np.int64))

        ophys = handle.create_group("processing/ophys")
        _write_nwb_series(ophys, "DfOverFResponseSeries", signal)

        peaks = handle.create_group("analysis/PeakFluorescenceEvents")
        peaks.create_dataset("timestamp", data=np.array([1.5, 7.2, 13.0]))
        subject = handle.create_group("general/subject")
        subject.create_dataset("genotype", data=np.bytes_("Anxa1-iCre"))

    session = read_nwb_photometry(path)
    photometry = session.signals["photometry"]

    assert isinstance(session, RecordingSession)
    assert isinstance(photometry, PhotometryRecording)
    assert session.session_id == "session-1"
    assert photometry.signal_channel == "DfOverFResponseSeries"
    assert photometry.reference_channel == "FiberPhotometryResponseSeriesIsosbestic"
    assert photometry.metadata["signal_is_dff"] is True
    assert photometry.metadata["sampling_rate_hz"] == pytest.approx(100.0)
    assert photometry.metadata["sampling_rate_source"] == "starting_time.rate"
    assert photometry.channel_names == (
        "DfOverFResponseSeries",
        "DfOverFResponseSeries_fiber1",
        "FiberPhotometryResponseSeriesIsosbestic",
        "FiberPhotometryResponseSeriesIsosbestic_fiber1",
    )
    np.testing.assert_allclose(photometry.timeline.timestamps_s[:3], [0.0, 0.01, 0.02])
    assert [event.label for event in session.events].count("Reward") == 4
    np.testing.assert_allclose(
        [event.start_s for event in session.events if event.label == "Reward"],
        [2.0, 5.0, 9.0, 14.0],
    )
    np.testing.assert_allclose(
        [event.start_s for event in session.events if event.label == "PeakFluorescenceEvents"],
        [1.5, 7.2, 13.0],
    )
    np.testing.assert_allclose(
        [event.start_s for event in session.events if event.label == "lick"],
        [0.75],
    )
    np.testing.assert_allclose(
        [event.start_s for event in session.events if event.label == "cue"],
        [6.5],
    )
    assert "TtlsTable" not in {event.label for event in session.events}
    assert session.metadata["subject"]["genotype"] == "Anxa1-iCre"
    assert session.metadata["sampling_rate_hz"] == pytest.approx(100.0)
    assert session.metadata["sampling_rate_source"] == "starting_time.rate"


def test_is_nwb_photometry_file_detects_photometry_series(tmp_path) -> None:
    path = tmp_path / "photometry.nwb"
    with h5py.File(path, "w") as handle:
        acquisition = handle.create_group("acquisition")
        _write_nwb_series(
            acquisition,
            "FiberPhotometryResponseSeries",
            np.asarray([1.0, 1.1, 1.2]),
        )

    no_signal = tmp_path / "behavior-only.nwb"
    with h5py.File(no_signal, "w") as handle:
        acquisition = handle.create_group("acquisition")
        _write_nwb_series(acquisition, "RunningSpeed", np.asarray([1.0, 1.1, 1.2]))

    wrong_suffix = tmp_path / "photometry.h5"
    with h5py.File(wrong_suffix, "w") as handle:
        acquisition = handle.create_group("acquisition")
        _write_nwb_series(
            acquisition,
            "FiberPhotometryResponseSeries",
            np.asarray([1.0, 1.1, 1.2]),
        )

    not_hdf5 = tmp_path / "not-hdf5.nwb"
    not_hdf5.write_text("not hdf5", encoding="utf-8")

    assert is_nwb_photometry_file(path) is True
    assert is_nwb_photometry_file(no_signal) is False
    assert is_nwb_photometry_file(wrong_suffix) is False
    assert is_nwb_photometry_file(not_hdf5) is False
    assert is_nwb_photometry_file(tmp_path / "missing.nwb") is False


def test_find_first_nwb_photometry_file_uses_detector_contract(tmp_path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    not_nwb = tmp_path / "plain.h5"
    with h5py.File(not_nwb, "w") as handle:
        handle.create_dataset("signal", data=np.arange(3, dtype=float))
    match_path = nested / "recording.nwb"
    with h5py.File(match_path, "w") as handle:
        acquisition = handle.create_group("acquisition")
        _write_nwb_series(
            acquisition,
            "FiberPhotometryResponseSeries",
            np.arange(3, dtype=float),
        )

    assert find_first_nwb_photometry_file(tmp_path) == match_path
    assert find_first_nwb_photometry_file(not_nwb) is None


def test_read_nwb_photometry_records_timestamp_sampling_rate_source(tmp_path) -> None:
    path = tmp_path / "timestamped.nwb"
    timestamps = np.arange(6, dtype=np.float64) * 0.25
    with h5py.File(path, "w") as handle:
        acquisition = handle.create_group("acquisition")
        _write_nwb_series(
            acquisition,
            "FiberPhotometryResponseSeries",
            np.linspace(0.0, 1.0, timestamps.size),
            rate=None,
            timestamps=timestamps,
        )

    session = read_nwb_photometry(path)
    photometry = session.signals["photometry"]

    assert photometry.metadata["sampling_rate_hz"] == pytest.approx(4.0)
    assert photometry.metadata["sampling_rate_source"] == "timestamps_uniform"
    assert session.metadata["sampling_rate_hz"] == pytest.approx(4.0)
    assert session.metadata["sampling_rate_source"] == "timestamps_uniform"


def test_read_nwb_photometry_rejects_missing_timebase(tmp_path) -> None:
    path = tmp_path / "missing-timebase.nwb"
    with h5py.File(path, "w") as handle:
        acquisition = handle.create_group("acquisition")
        _write_nwb_series(
            acquisition,
            "FiberPhotometryResponseSeries",
            np.asarray([1.0, 1.1, 1.2]),
            rate=None,
        )

    with pytest.raises(ValueError, match="missing timestamps or starting_time"):
        read_nwb_photometry(path)


def test_read_nwb_photometry_rejects_misaligned_control(tmp_path) -> None:
    path = tmp_path / "misaligned-control.nwb"
    with h5py.File(path, "w") as handle:
        acquisition = handle.create_group("acquisition")
        _write_nwb_series(acquisition, "FiberPhotometryResponseSeries", np.ones(10))
        _write_nwb_series(acquisition, "FiberPhotometryResponseSeriesIsosbestic", np.ones(9))

    with pytest.raises(ValueError, match="control series timeline must match"):
        read_nwb_photometry(path)


def test_read_nwb_photometry_rejects_multicolumn_event_channel(tmp_path) -> None:
    path = tmp_path / "multicolumn-event.nwb"
    with h5py.File(path, "w") as handle:
        acquisition = handle.create_group("acquisition")
        _write_nwb_series(
            acquisition,
            "FiberPhotometryResponseSeries",
            np.asarray([1.0, 1.1, 1.2]),
        )
        _write_nwb_series(
            acquisition,
            "Reward",
            np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        )

    with pytest.raises(ValueError, match=r"event channel .* one-dimensional"):
        read_nwb_photometry(path)


@pytest.mark.parametrize("label", [b"", b" cue"])
def test_read_nwb_photometry_rejects_malformed_annotation_label(
    tmp_path,
    label: bytes,
) -> None:
    path = tmp_path / "malformed-annotation.nwb"
    with h5py.File(path, "w") as handle:
        acquisition = handle.create_group("acquisition")
        _write_nwb_series(
            acquisition,
            "FiberPhotometryResponseSeries",
            np.asarray([1.0, 1.1, 1.2]),
        )
        annotations = acquisition.create_group("events")
        annotations.attrs["neurodata_type"] = "AnnotationSeries"
        annotations.create_dataset("data", data=np.asarray([b"cue", label]))
        annotations.create_dataset("timestamps", data=np.asarray([0.1, 0.2]))

    with pytest.raises(ValueError, match=r"label at row 1"):
        read_nwb_photometry(path)


def test_read_nwb_photometry_rejects_nonfinite_samples_by_default(tmp_path) -> None:
    path = tmp_path / "strict-gaps.nwb"
    signal = np.linspace(0.0, 1.0, 2000)
    signal[100] = np.nan
    with h5py.File(path, "w") as handle:
        ophys = handle.create_group("processing/ophys")
        _write_nwb_series(ophys, "DfOverFResponseSeries", signal)

    with pytest.raises(ValueError, match="non-finite samples"):
        read_nwb_photometry(path)


def test_read_nwb_photometry_repairs_sparse_nonfinite_gaps_when_opted_in(tmp_path) -> None:
    # Archive dF/F traces can carry a handful of dropped-frame NaN/inf samples.
    # The opt-in repair policy interpolates internal gaps this sparse.
    path = tmp_path / "gaps.nwb"
    clean = np.linspace(0.0, 1.0, 2000)
    signal = clean.copy()
    signal[100] = np.nan
    signal[101] = np.inf
    with h5py.File(path, "w") as handle:
        ophys = handle.create_group("processing/ophys")
        _write_nwb_series(ophys, "DfOverFResponseSeries", signal)

    session = read_nwb_photometry(path, nonfinite_policy="interpolate_sparse")
    photometry = session.signals["photometry"]
    values = photometry.series.values[:, 0]

    assert np.isfinite(values).all()
    np.testing.assert_allclose(values, clean, atol=1e-12)
    assert (
        photometry.metadata["nonfinite_repairs"]["processing/ophys/DfOverFResponseSeries"][
            "nonfinite_samples"
        ]
        == 2
    )


def test_read_nwb_photometry_rejects_heavily_nonfinite_series(tmp_path) -> None:
    path = tmp_path / "corrupt.nwb"
    signal = np.linspace(0.0, 1.0, 1000)
    signal[:50] = np.nan  # 5% non-finite, beyond the repair threshold
    with h5py.File(path, "w") as handle:
        ophys = handle.create_group("processing/ophys")
        _write_nwb_series(ophys, "DfOverFResponseSeries", signal)

    with pytest.raises(ValueError, match="non-finite"):
        read_nwb_photometry(path, nonfinite_policy="interpolate_sparse")


def test_read_nwb_photometry_rejects_edge_nonfinite_repair(tmp_path) -> None:
    path = tmp_path / "edge-gaps.nwb"
    signal = np.linspace(0.0, 1.0, 1000)
    signal[0] = np.nan
    with h5py.File(path, "w") as handle:
        ophys = handle.create_group("processing/ophys")
        _write_nwb_series(ophys, "DfOverFResponseSeries", signal)

    with pytest.raises(ValueError, match="edge samples"):
        read_nwb_photometry(path, nonfinite_policy="interpolate_sparse")


def test_read_teleopto_h5_extracts_channels_and_ttl(tmp_path) -> None:
    path = tmp_path / "teleopto.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("d1", data=np.asarray([1.0, 1.1, 1.2, 1.3]))
        handle.create_dataset("d2", data=np.asarray([0.0, 0.0, 5.0, 0.0]))
        handle.create_dataset("num", data=np.asarray([0.0, 10.0]))
        handle.create_dataset("st1", data=np.asarray([0.4]))
        handle.create_dataset("str", data=np.asarray([b"Signal", b"", b"TTL"]))
        handle.create_dataset("ct1", data=np.asarray([0.15]))

    assert is_teleopto_h5(path) is True

    session = read_teleopto_h5(path)
    photometry = session.signals["photometry"]

    assert isinstance(photometry, PhotometryRecording)
    assert photometry.channel_names == ("Signal", "TTL")
    assert photometry.signal_channel == "Signal"
    assert photometry.reference_channel is None
    assert photometry.metadata["secondary_channel"] == "TTL"
    assert photometry.metadata["sampling_rate_hz"] == pytest.approx(10.0)
    assert photometry.metadata["sampling_rate_source"] == "num[1]"
    assert photometry.metadata["event_label_scheme"] == "teleopto_native"
    assert session.metadata["sampling_rate_hz"] == pytest.approx(10.0)
    assert session.metadata["sampling_rate_source"] == "num[1]"
    assert session.metadata["event_label_scheme"] == "teleopto_native"
    by_label = {event.label: event.start_s for event in session.events}
    assert {"ct1", "TTL_ttl", "press_on_times", "press_off_times"} <= set(by_label)
    assert by_label["ct1"] == pytest.approx(0.15)
    assert by_label["TTL_ttl"] == pytest.approx(0.2)


def test_is_teleopto_h5_rejects_missing_contract_and_non_hdf5(tmp_path) -> None:
    missing_key_path = tmp_path / "missing-key.h5"
    with h5py.File(missing_key_path, "w") as handle:
        handle.create_dataset("d1", data=np.asarray([1.0]))
        handle.create_dataset("num", data=np.asarray([0.0, 10.0]))
        handle.create_dataset("st1", data=np.asarray([0.1]))

    text_path = tmp_path / "not-hdf5.h5"
    text_path.write_text("not hdf5", encoding="utf-8")

    assert is_teleopto_h5(missing_key_path) is False
    assert is_teleopto_h5(text_path) is False
    assert is_teleopto_h5(tmp_path / "missing.h5") is False


def test_find_first_teleopto_h5_uses_detector_contract(tmp_path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    not_teleopto = tmp_path / "plain.h5"
    with h5py.File(not_teleopto, "w") as handle:
        handle.create_dataset("signal", data=np.arange(3, dtype=float))
    match_path = nested / "recording.h5"
    with h5py.File(match_path, "w") as handle:
        handle.create_dataset("d1", data=np.arange(3, dtype=float))
        handle.create_dataset("d2", data=np.arange(3, dtype=float))
        handle.create_dataset("num", data=np.asarray([10.0, 10.0]))
        handle.create_dataset("st1", data=np.asarray([0.0]))
        handle.create_dataset("str", data=np.asarray([b"Signal", b"Control"]))

    assert find_first_teleopto_h5(tmp_path) == match_path
    assert find_first_teleopto_h5(not_teleopto) is None


def test_parse_teleopto_h5_arrays_matches_file_semantics() -> None:
    datasets = {
        "d1": np.asarray([1.0, 1.1, 1.2, 1.3]),
        "d2": np.asarray([0.0, 0.0, 5.0, 0.0]),
        "num": np.asarray([0.0, 10.0]),
        "st1": np.asarray([0.4]),
        "str": np.asarray([b"Signal", b"", b"TTL"]),
        "ct1": np.asarray([0.15]),
    }

    session = parse_teleopto_h5_arrays(datasets, session_id="arrays")
    photometry = session.signals["photometry"]

    assert isinstance(photometry, PhotometryRecording)
    assert session.session_id == "arrays"
    assert photometry.channel_names == ("Signal", "TTL")
    assert photometry.reference_channel is None
    assert photometry.metadata["sampling_rate_hz"] == pytest.approx(10.0)
    assert photometry.metadata["sampling_rate_source"] == "num[1]"
    assert photometry.metadata["event_label_scheme"] == "teleopto_native"
    by_label = {event.label: event.start_s for event in session.events}
    assert by_label["ct1"] == pytest.approx(0.15)
    assert by_label["TTL_ttl"] == pytest.approx(0.2)


def test_parse_teleopto_h5_arrays_records_st1_sampling_rate_source() -> None:
    datasets = {
        "d1": np.asarray([1.0, 1.1, 1.2, 1.3]),
        "num": np.asarray([0.0]),
        "st1": np.asarray([0.4]),
        "str": np.asarray([b"Signal"]),
    }

    photometry = parse_teleopto_h5_arrays(datasets).signals["photometry"]

    assert photometry.metadata["sampling_rate_hz"] == pytest.approx(10.0)
    assert photometry.metadata["sampling_rate_source"] == "st1_duration"


def test_read_teleopto_h5_rejects_nonfinite_events(tmp_path) -> None:
    path = tmp_path / "bad-teleopto.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("d1", data=np.asarray([1.0, 1.1, 1.2]))
        handle.create_dataset("num", data=np.asarray([0.0, 10.0]))
        handle.create_dataset("st1", data=np.asarray([0.3]))
        handle.create_dataset("str", data=np.asarray([b"Signal"]))
        handle.create_dataset("ct1", data=np.asarray([0.1, np.nan]))

    with pytest.raises(ValueError, match="Teleopto H5 event channel ct1"):
        read_teleopto_h5(path)


def test_read_teleopto_h5_rejects_multidimensional_primary_signal(tmp_path) -> None:
    path = tmp_path / "bad-primary-teleopto.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("d1", data=np.asarray([[1.0, 1.1], [1.2, 1.3]]))
        handle.create_dataset("d2", data=np.asarray([0.0, 0.0]))
        handle.create_dataset("num", data=np.asarray([2.0, 10.0]))
        handle.create_dataset("st1", data=np.asarray([0.2]))
        handle.create_dataset("str", data=np.asarray([b"Signal", b"", b"TTL"]))

    with pytest.raises(ValueError, match="Teleopto H5 d1 must be one-dimensional"):
        read_teleopto_h5(path)


def test_read_teleopto_h5_rejects_multidimensional_event_channel(tmp_path) -> None:
    path = tmp_path / "bad-event-teleopto.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("d1", data=np.asarray([1.0, 1.1, 1.2]))
        handle.create_dataset("d2", data=np.asarray([0.0, 0.0, 0.0]))
        handle.create_dataset("num", data=np.asarray([3.0, 10.0]))
        handle.create_dataset("st1", data=np.asarray([0.3]))
        handle.create_dataset("str", data=np.asarray([b"Signal", b"", b"TTL"]))
        handle.create_dataset("ct1", data=np.asarray([[0.1, 0.2]]))

    with pytest.raises(
        ValueError,
        match="Teleopto H5 event channel ct1 must be one-dimensional",
    ):
        read_teleopto_h5(path)


def test_is_tdt_block_detects_matching_block_pairs(tmp_path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "block.tsq").write_bytes(b"")
    (tmp_path / "b" / "block.tev").write_bytes(b"")

    assert is_tdt_block(tmp_path) is False

    (tmp_path / "a" / "block.tev").write_bytes(b"")
    assert is_tdt_block(tmp_path) is True

    nested = tmp_path / "tank" / "subject"
    nested.mkdir(parents=True)
    (nested / "BLOCK.TSQ").write_bytes(b"")
    (nested / "BLOCK.TEV").write_bytes(b"")
    assert is_tdt_block(tmp_path / "tank") is True

    not_a_dir = tmp_path / "not_a_dir.tsq"
    not_a_dir.write_bytes(b"")
    assert is_tdt_block(not_a_dir) is False


def test_is_tdt_block_detects_sev_stream_blocks(tmp_path) -> None:
    block_dir = tmp_path / "tank" / "subject"
    block_dir.mkdir(parents=True)
    (block_dir / "BLOCK.TSQ").write_bytes(b"")
    (block_dir / "Tank_Subject_x465A_ch1.sev").write_bytes(b"")

    assert is_tdt_block(tmp_path / "tank") is True

    sev_only_dir = tmp_path / "sev_only"
    sev_only_dir.mkdir()
    (sev_only_dir / "Tank_Subject_x405A_ch1.sev").write_bytes(b"")

    assert is_tdt_block(sev_only_dir) is True


def test_find_tdt_block_directories_returns_exact_blocks(tmp_path) -> None:
    first = tmp_path / "tank" / "subject_a"
    first.mkdir(parents=True)
    (first / "BLOCK.TSQ").write_bytes(b"")
    (first / "BLOCK.TEV").write_bytes(b"")

    second = tmp_path / "tank" / "subject_b"
    second.mkdir()
    (second / "Tank_Subject_x465A_ch1.sev").write_bytes(b"")

    hidden = tmp_path / ".hidden" / "subject_c"
    hidden.mkdir(parents=True)
    (hidden / "BLOCK.TSQ").write_bytes(b"")
    (hidden / "BLOCK.TEV").write_bytes(b"")

    assert find_tdt_block_directories(tmp_path) == [
        first.resolve(),
        second.resolve(),
    ]
    assert find_tdt_block_directories(
        tmp_path,
        include_hidden_dirs=True,
    ) == [
        hidden.resolve(),
        first.resolve(),
        second.resolve(),
    ]


def test_resolve_tdt_block_path_requires_one_exact_block(tmp_path) -> None:
    first = tmp_path / "tank" / "subject_a"
    first.mkdir(parents=True)
    (first / "BLOCK.TSQ").write_bytes(b"")
    (first / "BLOCK.TEV").write_bytes(b"")

    assert resolve_tdt_block_path(tmp_path / "tank") == first.resolve()
    assert resolve_tdt_block_path(first) == first.resolve()

    second = tmp_path / "tank" / "subject_b"
    second.mkdir()
    (second / "BLOCK.TSQ").write_bytes(b"")
    (second / "BLOCK.TEV").write_bytes(b"")

    with pytest.raises(ValueError, match="Multiple TDT blocks"):
        resolve_tdt_block_path(tmp_path / "tank")

    with pytest.raises(ValueError, match="No TDT block"):
        resolve_tdt_block_path(tmp_path / "missing")


def test_read_tdt_photometry_block_uses_optional_module() -> None:
    streams = SimpleNamespace(
        x465A=SimpleNamespace(data=np.asarray([1.0, 1.1, 1.2]), fs=100.0, start_time=0.0),
        x405A=SimpleNamespace(data=np.asarray([0.5, 0.4, 0.3]), fs=100.0),
    )
    epocs = SimpleNamespace(Cue=SimpleNamespace(onset=np.asarray([0.25])))
    fake_tdt = SimpleNamespace(
        read_block=lambda *_args, **_kwargs: SimpleNamespace(streams=streams, epocs=epocs)
    )

    session = read_tdt_photometry_block(
        "tank/block",
        signal_store="x465A",
        reference_store="x405A",
        event_stores=["Cue"],
        tdt_module=fake_tdt,
    )
    photometry = session.signals["photometry"]

    assert isinstance(photometry, PhotometryRecording)
    assert photometry.signal_channel == "x465A"
    assert photometry.reference_channel == "x405A"
    assert photometry.metadata["sampling_rate_hz"] == pytest.approx(100.0)
    assert photometry.metadata["sampling_rate_source"] == "streams.x465A.fs"
    assert photometry.metadata["stream_start_s"] == pytest.approx(0.0)
    assert photometry.metadata["stream_start_source"] == "streams.x465A.start_time"
    assert photometry.metadata["channel_inference"] == "explicit_store"
    assert photometry.metadata["reference_channel_inference"] == "explicit_store"
    assert session.metadata["sampling_rate_hz"] == pytest.approx(100.0)
    assert session.metadata["sampling_rate_source"] == "streams.x465A.fs"
    assert session.metadata["stream_start_source"] == "streams.x465A.start_time"
    assert session.events.events[0].label == "Cue"


def test_read_tdt_photometry_block_rejects_missing_explicit_event_store() -> None:
    streams = SimpleNamespace(
        x465A=SimpleNamespace(data=np.asarray([1.0, 1.1, 1.2]), fs=100.0, start_time=0.0)
    )
    fake_tdt = SimpleNamespace(
        read_block=lambda *_args, **_kwargs: SimpleNamespace(
            streams=streams,
            epocs=SimpleNamespace(Cue=SimpleNamespace(onset=np.asarray([0.25]))),
        )
    )

    with pytest.raises(ValueError, match="TDT event store 'Missing' was not found"):
        read_tdt_photometry_block(
            "tank/block",
            signal_store="x465A",
            event_stores=["Missing"],
            tdt_module=fake_tdt,
        )


@pytest.mark.parametrize(
    ("kwargs", "exc_type", "message"),
    [
        (
            {"signal_store": " x465A"},
            ValueError,
            "TDT signal_store must not contain surrounding whitespace",
        ),
        (
            {"reference_store": " x405A"},
            ValueError,
            "TDT reference_store must not contain surrounding whitespace",
        ),
        (
            {"event_stores": [" Cue"]},
            ValueError,
            r"TDT event_stores\[0\] must not contain surrounding whitespace",
        ),
        ({"event_stores": [1]}, TypeError, r"TDT event_stores\[0\] must be a string"),
        (
            {"event_stores": "Cue"},
            TypeError,
            "TDT event_stores must be a sequence of strings, not a string",
        ),
    ],
)
def test_read_tdt_photometry_block_rejects_unclean_explicit_store_selectors(
    kwargs: dict[str, object],
    exc_type: type[Exception],
    message: str,
) -> None:
    fake_tdt = SimpleNamespace(
        read_block=lambda *_args, **_kwargs: pytest.fail(
            "invalid explicit TDT selectors must fail before read_block"
        )
    )

    with pytest.raises(exc_type, match=message):
        read_tdt_photometry_block("tank/block", tdt_module=fake_tdt, **kwargs)


def test_read_tdt_photometry_block_rejects_malformed_explicit_event_store() -> None:
    streams = SimpleNamespace(
        x465A=SimpleNamespace(data=np.asarray([1.0, 1.1, 1.2]), fs=100.0, start_time=0.0)
    )
    fake_tdt = SimpleNamespace(
        read_block=lambda *_args, **_kwargs: SimpleNamespace(
            streams=streams,
            epocs=SimpleNamespace(Cue=SimpleNamespace(label="Cue")),
        )
    )

    with pytest.raises(ValueError, match="no onset or data timestamps"):
        read_tdt_photometry_block(
            "tank/block",
            signal_store="x465A",
            event_stores=["Cue"],
            tdt_module=fake_tdt,
        )


def test_read_tdt_photometry_block_rejects_missing_start_time() -> None:
    # A signal stream without start_time is corrupt metadata: failing fast beats
    # silently aligning every event to 0.0.
    streams = SimpleNamespace(
        x465A=SimpleNamespace(data=np.asarray([1.0, 1.1, 1.2]), fs=100.0),
        x405A=SimpleNamespace(data=np.asarray([0.5, 0.4, 0.3]), fs=100.0, start_time=0.0),
    )
    fake_tdt = SimpleNamespace(
        read_block=lambda *_args, **_kwargs: SimpleNamespace(
            streams=streams, epocs=SimpleNamespace()
        )
    )

    with pytest.raises(ValueError, match="start_time"):
        read_tdt_photometry_block(
            "tank/block",
            signal_store="x465A",
            reference_store="x405A",
            tdt_module=fake_tdt,
        )


def test_read_tdt_photometry_block_accepts_zero_start_time() -> None:
    # A legitimately present, finite 0.0 (stream starts at recording onset) is
    # valid and must not be rejected.
    streams = SimpleNamespace(
        x465A=SimpleNamespace(data=np.asarray([1.0, 1.1, 1.2]), fs=100.0, start_time=0.0),
        x405A=SimpleNamespace(data=np.asarray([0.5, 0.4, 0.3]), fs=100.0, start_time=0.0),
    )
    fake_tdt = SimpleNamespace(
        read_block=lambda *_args, **_kwargs: SimpleNamespace(
            streams=streams, epocs=SimpleNamespace()
        )
    )

    session = read_tdt_photometry_block(
        "tank/block",
        signal_store="x465A",
        reference_store="x405A",
        tdt_module=fake_tdt,
    )

    assert session.signals["photometry"].metadata["stream_start_s"] == pytest.approx(0.0)


def test_read_tdt_photometry_block_accepts_sev_only_stream_shape() -> None:
    data = SimpleNamespace(
        x465A=SimpleNamespace(data=np.asarray([1.0, 1.1, 1.2]), fs=100.0),
        x405A=SimpleNamespace(data=np.asarray([0.5, 0.4, 0.3]), fs=100.0),
    )
    fake_tdt = SimpleNamespace(read_block=lambda *_args, **_kwargs: data)

    session = read_tdt_photometry_block("tank/sev", tdt_module=fake_tdt)
    photometry = session.signals["photometry"]

    assert isinstance(photometry, PhotometryRecording)
    assert photometry.signal_channel == "x465A"
    assert photometry.reference_channel == "x405A"
    assert photometry.metadata["stream_start_s"] == pytest.approx(0.0)
    assert photometry.metadata["stream_start_source"] == "tdt.read_sev.t1_default"
    assert session.metadata["stream_start_source"] == "tdt.read_sev.t1_default"
    assert len(session.events.events) == 0
    np.testing.assert_allclose(photometry.series.values[:, 0], [1.0, 1.1, 1.2])
    np.testing.assert_allclose(photometry.series.values[:, 1], [0.5, 0.4, 0.3])


def test_read_tdt_photometry_block_prefers_official_wavelength_stores() -> None:
    streams = SimpleNamespace(
        _405A=SimpleNamespace(data=np.asarray([0.5, 0.4, 0.3]), fs=100.0),
        _465A=SimpleNamespace(
            data=np.asarray([1.0, 1.1, 1.2]),
            fs=100.0,
            start_time=0.05,
        ),
        Fi1r=SimpleNamespace(data=np.arange(18, dtype=float), fs=600.0),
    )
    epocs = SimpleNamespace(Cam1=SimpleNamespace(onset=np.asarray([0.1, 0.2])))
    fake_tdt = SimpleNamespace(
        read_block=lambda *_args, **_kwargs: SimpleNamespace(streams=streams, epocs=epocs)
    )

    session = read_tdt_photometry_block("tank/block", tdt_module=fake_tdt)
    photometry = session.signals["photometry"]

    assert isinstance(photometry, PhotometryRecording)
    assert photometry.signal_channel == "_465A"
    assert photometry.reference_channel == "_405A"
    assert photometry.metadata["stores"] == ["_465A", "_405A", "Fi1r"]
    assert photometry.metadata["channel_inference"] == "wavelength_tokens"
    assert photometry.metadata["reference_channel_inference"] == "wavelength_tokens"
    assert photometry.metadata["stream_start_s"] == pytest.approx(0.05)
    assert photometry.metadata["sampling_rate_hz"] == pytest.approx(100.0)
    assert photometry.metadata["sampling_rate_source"] == "streams._465A.fs"
    assert session.metadata["sampling_rate_hz"] == pytest.approx(100.0)
    assert session.metadata["sampling_rate_source"] == "streams._465A.fs"
    assert photometry.series.sample_rate_hz == pytest.approx(100.0)
    np.testing.assert_allclose(photometry.series.values[:, 0], [1.0, 1.1, 1.2])
    np.testing.assert_allclose(photometry.series.values[:, 1], [0.5, 0.4, 0.3])
    assert [event.label for event in session.events] == ["Cam1", "Cam1"]
    np.testing.assert_allclose([event.start_s for event in session.events], [0.05, 0.15])


def test_read_tdt_photometry_block_flags_storage_order_signal() -> None:
    streams = SimpleNamespace(
        RandomStore=SimpleNamespace(data=np.asarray([1.0, 1.1, 1.2]), fs=100.0, start_time=0.0),
        OtherStore=SimpleNamespace(data=np.asarray([0.5, 0.4, 0.3]), fs=100.0),
    )
    fake_tdt = SimpleNamespace(
        read_block=lambda *_args, **_kwargs: SimpleNamespace(
            streams=streams,
            epocs=SimpleNamespace(),
        )
    )

    session = read_tdt_photometry_block("tank/block", tdt_module=fake_tdt)
    photometry = session.signals["photometry"]

    assert isinstance(photometry, PhotometryRecording)
    assert photometry.signal_channel == "RandomStore"
    assert photometry.metadata["channel_inference"] == "storage_order"
