from __future__ import annotations

import numpy as np
import pytest

from xpkg.io.readers import read_ephys_csv
from xpkg.model import EphysRecording


def _write(path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_read_ephys_csv_single_sweep_normalizes_units(tmp_path) -> None:
    path = tmp_path / "cell01.csv"
    _write(
        path,
        [
            "time,electrode,monitor",
            "0.0,0.001,1.0",
            "0.0001,0.002,2.0",
            "0.0002,0.003,3.0",
        ],
    )

    recording = read_ephys_csv(
        path,
        units={"electrode": "V", "monitor": "nA"},
        channel_roles={"electrode": "electrode", "monitor": "stimulus_monitor"},
    )

    assert isinstance(recording, EphysRecording)
    assert recording.mode == "current_clamp"
    assert recording.channel_names == ("electrode", "monitor")
    assert recording.channel_unit("electrode") == "mV"
    assert recording.channel_unit("monitor") == "pA"

    np.testing.assert_allclose(recording.sweeps[0].series.values[:, 0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        recording.sweeps[0].series.values[:, 1], [1000.0, 2000.0, 3000.0]
    )

    log = {entry["channel"]: entry for entry in recording.conversion_log}
    assert log["electrode"]["from_unit"] == "V"
    assert log["electrode"]["to_unit"] == "mV"
    assert log["monitor"]["from_unit"] == "nA"
    assert log["monitor"]["to_unit"] == "pA"
    assert recording.provenance["source"]["type"] == "ephys_csv"
    assert recording.provenance["source"]["sha256"]


def test_read_ephys_csv_splits_sweeps_and_shares_layout(tmp_path) -> None:
    path = tmp_path / "fi.csv"
    _write(
        path,
        [
            "sweep,time,electrode",
            "0,0.0,-65.0",
            "0,0.0001,-64.5",
            "1,0.0,-65.0",
            "1,0.0001,-63.0",
        ],
    )

    recording = read_ephys_csv(
        path,
        units={"electrode": "mV"},
        channel_roles={"electrode": "electrode"},
    )

    assert recording.n_sweeps == 2
    assert recording.channel_names == ("electrode",)
    assert recording.mode == "current_clamp"
    assert [sweep.index for sweep in recording.sweeps] == [0, 1]


def test_read_ephys_csv_uses_sample_rate_when_no_time_column(tmp_path) -> None:
    path = tmp_path / "samples.csv"
    _write(path, ["electrode", "-65.0", "-64.0", "-63.0"])

    recording = read_ephys_csv(
        path,
        units={"electrode": "mV"},
        channel_roles={"electrode": "electrode"},
        sample_rate_hz=10_000.0,
    )

    assert recording.sample_rate_hz == pytest.approx(10_000.0)
    np.testing.assert_allclose(
        recording.sweeps[0].series.timeline.timestamps_s,
        [0.0, 0.0001, 0.0002],
    )


def test_read_ephys_csv_requires_timebase(tmp_path) -> None:
    path = tmp_path / "broken.csv"
    _write(path, ["electrode", "-65.0", "-64.0"])

    with pytest.raises(ValueError, match="time column or sample_rate_hz"):
        read_ephys_csv(path, channel_roles={"electrode": "electrode"})


def test_read_ephys_csv_rejects_unknown_role_channel(tmp_path) -> None:
    path = tmp_path / "bad_role.csv"
    _write(path, ["time,electrode", "0.0,-65.0", "0.0001,-64.0"])

    with pytest.raises(ValueError, match="unknown channel"):
        read_ephys_csv(path, channel_roles={"missing": "electrode"})


def test_read_ephys_csv_enforces_max_size(tmp_path) -> None:
    path = tmp_path / "big.csv"
    _write(path, ["time,electrode", "0.0,-65.0"])
    with pytest.raises(ValueError, match="exceeds max load size"):
        read_ephys_csv(path, max_mb=0.000001)


def test_read_ephys_csv_records_voltage_clamp_mode(tmp_path) -> None:
    path = tmp_path / "vc.csv"
    _write(
        path,
        [
            "time,electrode",
            "0.0,1.0",
            "0.0001,2.0",
        ],
    )

    recording = read_ephys_csv(
        path,
        units={"electrode": "pA"},
        channel_roles={"electrode": "electrode"},
    )

    assert recording.mode == "voltage_clamp"
