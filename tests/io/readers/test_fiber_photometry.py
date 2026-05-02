from __future__ import annotations

from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from xpkg.io.readers import (
    read_doric_photometry,
    read_neurophotometrics_csv,
    read_pmat_events_csv,
    read_pmat_photometry_csv,
    read_rwd_ofrs_session,
    read_tdt_photometry_block,
    read_teleopto_h5,
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
    np.testing.assert_allclose(photometry.timeline.timestamps_s, [0.0, 0.1])
    assert isinstance(events, EventTable)
    assert events.events[0].label == "cue"
    assert events.events[0].duration_s == pytest.approx(0.15)


def test_read_neurophotometrics_csv_preserves_roi_channels_and_flags(tmp_path) -> None:
    path = tmp_path / "npm.csv"
    path.write_text(
        "\n".join(
            [
                "FrameCounter,Timestamp,Flags,Region0G,Region0R",
                "0,0.0,1,0.2,0.1",
                "1,0.1,2,0.3,0.2",
            ]
        ),
        encoding="utf-8",
    )

    session = read_neurophotometrics_csv(path, reference_column="Region0R")
    photometry = session.signals["photometry"]

    assert isinstance(session, RecordingSession)
    assert isinstance(photometry, PhotometryRecording)
    assert photometry.channel_names == ("Region0G", "Region0R")
    assert photometry.reference_channel == "Region0R"
    assert "flags" in session.signals
    assert session.metadata["frame_counter"] == [0, 1]


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
    assert [event.label for event in session.events] == ["brush", "brush_offset"]
    np.testing.assert_allclose([event.start_s for event in session.events], [0.033333, 0.066666])


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


def test_read_teleopto_h5_extracts_channels_and_ttl(tmp_path) -> None:
    path = tmp_path / "teleopto.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("d1", data=np.asarray([1.0, 1.1, 1.2, 1.3]))
        handle.create_dataset("d2", data=np.asarray([0.0, 0.0, 5.0, 0.0]))
        handle.create_dataset("num", data=np.asarray([0.0, 10.0]))
        handle.create_dataset("st1", data=np.asarray([0.4]))
        handle.create_dataset("str", data=np.asarray([b"Signal", b"", b"TTL"]))
        handle.create_dataset("ct1", data=np.asarray([0.15]))

    session = read_teleopto_h5(path)
    photometry = session.signals["photometry"]

    assert isinstance(photometry, PhotometryRecording)
    assert photometry.channel_names == ("Signal", "TTL")
    assert [event.label for event in session.events] == ["ct1", "TTL_ttl"]
    np.testing.assert_allclose([event.start_s for event in session.events], [0.15, 0.2])


def test_read_tdt_photometry_block_uses_optional_module() -> None:
    streams = SimpleNamespace(
        x465A=SimpleNamespace(data=np.asarray([1.0, 1.1, 1.2]), fs=100.0),
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
    assert session.events.events[0].label == "Cue"
