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


def _write_rwd_fluorescence(
    session_dir, *, metadata_line: str | None, timestamps: list[float]
) -> None:
    session_dir.mkdir()
    rows = ["TimeStamp,Events,CH1-410,CH1-470,"]
    for index, stamp in enumerate(timestamps):
        rows.append(f"{stamp:.6f},,{1.0 + 0.1 * index},{2.0 + 0.1 * index},")
    lines = ([metadata_line] if metadata_line is not None else []) + rows
    (session_dir / "Fluorescence.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    np.testing.assert_allclose(photometry.timeline.timestamps_s, [0.0, 2.0, 4.0, 6.0])


def test_read_rwd_ofrs_subsecond_without_fps_infers_milliseconds(tmp_path) -> None:
    _write_rwd_fluorescence(tmp_path / "sub", metadata_line=None, timestamps=[0.0, 0.4, 0.8, 1.2])
    session = read_rwd_ofrs_session(tmp_path / "sub")
    photometry = session.signals["photometry"]
    assert photometry.metadata["time_scale"] == pytest.approx(0.001)
    assert photometry.metadata["sampling_rate_hz"] == pytest.approx(2500.0, rel=1e-6)


def test_read_rwd_ofrs_ambiguous_spacing_without_fps_raises(tmp_path) -> None:
    _write_rwd_fluorescence(tmp_path / "amb", metadata_line=None, timestamps=[0.0, 2.0, 4.0, 6.0])
    with pytest.raises(ValueError, match="ambiguous"):
        read_rwd_ofrs_session(tmp_path / "amb")


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
    assert photometry.metadata["channel_inference"] == "wavelength_tokens"


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
    assert photometry.metadata["stream_start_s"] == pytest.approx(0.05)
    assert photometry.series.sample_rate_hz == pytest.approx(100.0)
    np.testing.assert_allclose(photometry.series.values[:, 0], [1.0, 1.1, 1.2])
    np.testing.assert_allclose(photometry.series.values[:, 1], [0.5, 0.4, 0.3])
    assert [event.label for event in session.events] == ["Cam1", "Cam1"]
    np.testing.assert_allclose([event.start_s for event in session.events], [0.05, 0.15])
