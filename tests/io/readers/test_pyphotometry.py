from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from xpkg.io.readers import (
    find_first_pyphotometry_csv,
    find_first_pyphotometry_ppd_file,
    is_pyphotometry_csv,
    is_pyphotometry_ppd_file,
    read_pyphotometry_csv,
    read_pyphotometry_ppd,
)
from xpkg.model import PhotometryRecording, RecordingSession, TimeSeries


def _write_ppd(
    path: Path,
    *,
    fs: float | None = 100.0,
    n_samples: int = 10,
    n_channels: int = 2,
    volts_per_division: float | None = None,
    odd_payload_byte: bool = False,
    extra_payload_words: int = 0,
) -> None:
    header: dict[str, object] = {"n_analog_channels": n_channels}
    if fs is not None:
        header["sampling_rate"] = fs
    if volts_per_division is not None:
        header["volts_per_division"] = volts_per_division
    header_bytes = json.dumps(header).encode("utf-8")

    rows = np.zeros((n_samples, n_channels), dtype=np.uint16)
    for sample in range(n_samples):
        for channel in range(n_channels):
            analog = 100 * (channel + 1) + sample
            digital = channel == 0 and sample in {2, 5}
            rows[sample, channel] = (analog << 1) | int(digital)
    payload = rows.ravel().astype("<u2").tobytes()
    if extra_payload_words:
        payload += np.arange(extra_payload_words, dtype="<u2").tobytes()
    if odd_payload_byte:
        payload += b"\x00"

    path.write_bytes(struct.pack("<H", len(header_bytes)) + header_bytes + payload)


def test_read_pyphotometry_ppd_returns_session_signals_and_events(tmp_path: Path) -> None:
    path = tmp_path / "recording.ppd"
    _write_ppd(path, fs=50.0, n_samples=6)

    session = read_pyphotometry_ppd(path)

    assert isinstance(session, RecordingSession)
    photometry = session.signals["photometry"]
    digital = session.signals["digital"]
    assert isinstance(photometry, PhotometryRecording)
    assert isinstance(digital, TimeSeries)
    assert photometry.signal_channel == "analog_1"
    assert photometry.reference_channel == "analog_2"
    assert photometry.channel_names == ("analog_1", "analog_2")
    assert digital.channel_names == ("digital_1", "digital_2")
    assert photometry.series.sample_rate_hz == pytest.approx(50.0)
    assert photometry.metadata["sampling_rate_hz"] == pytest.approx(50.0)
    assert photometry.metadata["sampling_rate_source"] == "header.sampling_rate"
    assert photometry.metadata["event_label_scheme"] == "digital_channels"
    assert session.metadata["sampling_rate_hz"] == pytest.approx(50.0)
    assert session.metadata["sampling_rate_source"] == "header.sampling_rate"
    assert session.metadata["event_label_scheme"] == "digital_channels"
    np.testing.assert_allclose(photometry.timeline.timestamps_s[:3], [0.0, 0.02, 0.04])
    np.testing.assert_allclose(photometry.series.values[:, 0], [100, 101, 102, 103, 104, 105])
    assert [event.label for event in session.events] == ["digital_1", "digital_1"]
    np.testing.assert_allclose([event.start_s for event in session.events], [0.04, 0.1])


def test_read_pyphotometry_ppd_applies_volts_per_division(tmp_path: Path) -> None:
    path = tmp_path / "scaled.ppd"
    _write_ppd(path, fs=100.0, volts_per_division=0.001)

    session = read_pyphotometry_ppd(path)
    photometry = session.signals["photometry"]
    assert isinstance(photometry, PhotometryRecording)

    assert photometry.series.channels[0].unit == "V"
    np.testing.assert_allclose(photometry.series.values[:2, 0], [0.1, 0.101])


def test_read_pyphotometry_ppd_rejects_nonpositive_volts_per_division(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad-scale.ppd"
    _write_ppd(path, fs=100.0, volts_per_division=0.0)

    with pytest.raises(ValueError, match="volts_per_division must be positive"):
        read_pyphotometry_ppd(path)


def test_read_pyphotometry_ppd_rejects_missing_sample_rate(tmp_path: Path) -> None:
    path = tmp_path / "missing_fs.ppd"
    _write_ppd(path, fs=None)

    with pytest.raises(ValueError, match="Sampling rate"):
        read_pyphotometry_ppd(path)


def test_read_pyphotometry_ppd_rejects_odd_payload_byte(tmp_path: Path) -> None:
    path = tmp_path / "odd.ppd"
    _write_ppd(path, fs=10.0, n_samples=3, odd_payload_byte=True)

    with pytest.raises(ValueError, match="payload byte length"):
        read_pyphotometry_ppd(path)


def test_read_pyphotometry_ppd_rejects_incomplete_old_layout_sample(
    tmp_path: Path,
) -> None:
    path = tmp_path / "incomplete-old.ppd"
    _write_ppd(path, fs=10.0, n_samples=3, extra_payload_words=1)

    with pytest.raises(ValueError, match="word count"):
        read_pyphotometry_ppd(path)


def test_read_pyphotometry_ppd_supports_v11_pulsed_baseline_layout(tmp_path: Path) -> None:
    path = tmp_path / "pulsed.ppd"
    header = {
        "sampling_rate": 20.0,
        "n_analog_channels": 2,
        "mode": "pulsed",
        "version": "1.1",
        "volts_per_division": 0.001,
    }
    header_bytes = json.dumps(header).encode("utf-8")
    rows = np.asarray(
        [
            [(100 << 1) | 0, 20 << 1, (200 << 1) | 0, 40 << 1],
            [(110 << 1) | 1, 30 << 1, (210 << 1) | 0, 50 << 1],
        ],
        dtype=np.uint16,
    )
    path.write_bytes(struct.pack("<H", len(header_bytes)) + header_bytes + rows.tobytes())

    session = read_pyphotometry_ppd(path)
    photometry = session.signals["photometry"]

    assert isinstance(photometry, PhotometryRecording)
    np.testing.assert_allclose(photometry.series.values[:, 0], [0.08, 0.08])
    np.testing.assert_allclose(photometry.series.values[:, 1], [0.16, 0.16])
    assert "raw_led_on" in session.signals
    assert "raw_baseline" in session.signals
    assert [event.label for event in session.events] == ["digital_1"]


def test_read_pyphotometry_ppd_rejects_incomplete_pulsed_layout_sample(
    tmp_path: Path,
) -> None:
    path = tmp_path / "incomplete-pulsed.ppd"
    header = {
        "sampling_rate": 20.0,
        "n_analog_channels": 2,
        "mode": "pulsed",
        "version": "1.1",
    }
    header_bytes = json.dumps(header).encode("utf-8")
    rows = np.asarray([[(100 << 1) | 0, 20 << 1, (200 << 1) | 0]], dtype=np.uint16)
    path.write_bytes(struct.pack("<H", len(header_bytes)) + header_bytes + rows.tobytes())

    with pytest.raises(ValueError, match="pulsed-mode payload word count"):
        read_pyphotometry_ppd(path)


def test_is_pyphotometry_ppd_file_detects_ppd_header_envelope(tmp_path: Path) -> None:
    valid = tmp_path / "recording.ppd"
    _write_ppd(valid)

    legacy = tmp_path / "legacy.ppd"
    legacy_header = b"\xff" * 34
    legacy.write_bytes(struct.pack("<H", len(legacy_header)) + legacy_header + b"\x00\x00")

    wrong_suffix = tmp_path / "recording.bin"
    _write_ppd(wrong_suffix)

    incomplete = tmp_path / "incomplete.ppd"
    incomplete.write_bytes(struct.pack("<H", 12) + b'{"fs"')

    malformed = tmp_path / "malformed.ppd"
    malformed_header = b'{"sampling_rate":'
    malformed.write_bytes(struct.pack("<H", len(malformed_header)) + malformed_header)

    assert is_pyphotometry_ppd_file(valid) is True
    assert is_pyphotometry_ppd_file(legacy) is True
    assert is_pyphotometry_ppd_file(wrong_suffix) is False
    assert is_pyphotometry_ppd_file(incomplete) is False
    assert is_pyphotometry_ppd_file(malformed) is False
    assert is_pyphotometry_ppd_file(tmp_path / "missing.ppd") is False


def test_find_first_pyphotometry_ppd_file_uses_detector_contract(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    wrong_suffix = tmp_path / "recording.bin"
    _write_ppd(wrong_suffix)
    match_path = nested / "recording.ppd"
    _write_ppd(match_path)

    assert find_first_pyphotometry_ppd_file(tmp_path) == match_path
    assert find_first_pyphotometry_ppd_file(wrong_suffix) is None


def test_is_pyphotometry_csv_detects_analog_columns(tmp_path: Path) -> None:
    spaced_path = tmp_path / "spaced.csv"
    spaced_path.write_text(
        "Analog1, Analog2, Digital1\n100,200,0\n",
        encoding="utf-8",
    )
    underscore_path = tmp_path / "underscore.csv"
    underscore_path.write_text(
        "analog_1,digital_1\n100,0\n",
        encoding="utf-8",
    )
    plain_path = tmp_path / "plain.csv"
    plain_path.write_text("time,signal\n0.0,1.0\n", encoding="utf-8")

    assert is_pyphotometry_csv(spaced_path) is True
    assert is_pyphotometry_csv(underscore_path) is True
    assert is_pyphotometry_csv(plain_path) is False
    assert is_pyphotometry_csv(tmp_path / "missing.csv") is False


def test_find_first_pyphotometry_csv_uses_detector_contract(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    plain_path = tmp_path / "plain.csv"
    plain_path.write_text("time,signal\n0.0,1.0\n", encoding="utf-8")
    match_path = nested / "recording.csv"
    match_path.write_text(
        "Analog1,Analog2,Digital1\n100,200,0\n",
        encoding="utf-8",
    )

    assert find_first_pyphotometry_csv(tmp_path) == match_path
    assert find_first_pyphotometry_csv(plain_path) is None


def test_read_pyphotometry_csv_uses_json_settings_and_digital_events(tmp_path: Path) -> None:
    path = tmp_path / "recording.csv"
    path.write_text(
        "Analog1,Analog2,Digital1,Digital2\n100,200,0,0\n101,201,1,0\n",
        encoding="utf-8",
    )
    path.with_suffix(".json").write_text(
        json.dumps({"sampling_rate": 10.0, "volts_per_division": 0.001}),
        encoding="utf-8",
    )

    session = read_pyphotometry_csv(path)
    photometry = session.signals["photometry"]

    assert isinstance(session, RecordingSession)
    assert isinstance(photometry, PhotometryRecording)
    assert photometry.series.channels[0].unit == "V"
    assert photometry.metadata["sampling_rate_hz"] == pytest.approx(10.0)
    assert photometry.metadata["sampling_rate_source"] == "header.sampling_rate"
    assert photometry.metadata["event_label_scheme"] == "digital_channels"
    assert session.metadata["event_label_scheme"] == "digital_channels"
    np.testing.assert_allclose(photometry.series.values[:, 0], [0.1, 0.101])
    assert [event.label for event in session.events] == ["digital_1"]


def test_read_pyphotometry_csv_rejects_missing_explicit_settings_path(
    tmp_path: Path,
) -> None:
    path = tmp_path / "recording.csv"
    path.write_text(
        "Analog1,Analog2\n100,200\n101,201\n",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="pyPhotometry settings_path was not found"):
        read_pyphotometry_csv(
            path,
            settings_path=tmp_path / "missing-settings.json",
            sample_rate_hz=10.0,
            volts_per_division=0.001,
        )


def test_read_pyphotometry_csv_rejects_invalid_explicit_volts_per_division(
    tmp_path: Path,
) -> None:
    path = tmp_path / "recording.csv"
    path.write_text(
        "Analog1,Analog2\n100,200\n101,201\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="volts_per_division must be positive"):
        read_pyphotometry_csv(
            path,
            sample_rate_hz=10.0,
            volts_per_division=0.0,
        )


def test_read_pyphotometry_csv_tolerates_real_spaced_header(tmp_path: Path) -> None:
    # pyPhotometry's GUI writes the header as ", ".join(channels), so real exports
    # carry a space after each delimiter. The reader must still resolve channels.
    path = tmp_path / "spaced.csv"
    path.write_text(
        "Analog1, Analog2, Digital1, Digital2\n100,200,0,0\n101,201,1,0\n",
        encoding="utf-8",
    )
    path.with_suffix(".json").write_text(
        json.dumps({"sampling_rate": 10.0}),
        encoding="utf-8",
    )

    session = read_pyphotometry_csv(path)
    photometry = session.signals["photometry"]

    assert isinstance(photometry, PhotometryRecording)
    assert photometry.channel_names == ("analog_1", "analog_2")
    np.testing.assert_allclose(photometry.series.values[:, 0], [100, 101])
    assert [event.label for event in session.events] == ["digital_1"]


def test_read_pyphotometry_csv_rejects_nonbinary_digital_values(
    tmp_path: Path,
) -> None:
    path = tmp_path / "recording.csv"
    path.write_text(
        "Analog1,Analog2,Digital1\n100,200,0\n101,201,2\n",
        encoding="utf-8",
    )
    path.with_suffix(".json").write_text(
        json.dumps({"sampling_rate": 10.0}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="digital values must be 0 or 1"):
        read_pyphotometry_csv(path)
