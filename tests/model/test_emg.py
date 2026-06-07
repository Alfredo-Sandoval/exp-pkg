from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from xpkg.model import EMGSignalData


def _valid_emg(**overrides: object) -> EMGSignalData:
    values: dict[str, Any] = {
        "sample_times_s": np.array([0.0, 0.001, 0.002], dtype=np.float64),
        "signals": np.array(
            [
                [0.1, 0.2],
                [0.3, 0.4],
                [0.5, 0.6],
            ],
            dtype=np.float64,
        ),
        "channel_names": ("Voltage.RTA", "Voltage.LBF"),
        "muscle_names": ("tibialis anterior", "biceps femoris"),
        "sides": ("right", "left"),
        "sample_rate_hz": 1000.0,
        "units": (("signal", "V"),),
        "processing_state": "raw",
        "provenance": (("source_path", "trial.csv"), ("reader", "test")),
    }
    values.update(overrides)
    return EMGSignalData(**values)


def test_emg_signal_data_accepts_valid_raw_signal() -> None:
    emg = _valid_emg(sides=("RIGHT", "Left"), processing_state="RAW")

    assert emg.sample_times_s.dtype == np.float64
    assert emg.signals.shape == (3, 2)
    assert emg.sides == ("right", "left")
    assert emg.processing_state == "raw"


def test_emg_signal_data_rejects_nonfinite_sample_times() -> None:
    with pytest.raises(ValueError, match="finite"):
        _valid_emg(sample_times_s=np.array([0.0, np.nan, 0.002]))


def test_emg_signal_data_rejects_nonmonotonic_sample_times() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        _valid_emg(sample_times_s=np.array([0.0, 0.002, 0.001]))


def test_emg_signal_data_rejects_bad_signal_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        _valid_emg(signals=np.array([0.1, 0.2, 0.3]))


def test_emg_signal_data_rejects_metadata_length_mismatch() -> None:
    with pytest.raises(ValueError, match="muscle_names length"):
        _valid_emg(muscle_names=("tibialis anterior",))


def test_emg_signal_data_rejects_duplicate_channel_names() -> None:
    with pytest.raises(ValueError, match="unique"):
        _valid_emg(channel_names=("Voltage.RTA", "Voltage.RTA"))


def test_emg_signal_data_rejects_invalid_side() -> None:
    with pytest.raises(ValueError, match="left, right, unknown, or bilateral"):
        _valid_emg(sides=("right", "ipsilateral"))


def test_emg_signal_data_rejects_invalid_processing_state() -> None:
    with pytest.raises(ValueError, match="processing_state"):
        _valid_emg(processing_state="normalized")


@pytest.mark.parametrize("sample_rate_hz", [0.0, -100.0, np.inf])
def test_emg_signal_data_rejects_nonfinite_or_nonpositive_sample_rate(
    sample_rate_hz: float,
) -> None:
    with pytest.raises(ValueError, match="sample_rate_hz"):
        _valid_emg(sample_rate_hz=sample_rate_hz)


def test_emg_signal_data_rejects_empty_provenance() -> None:
    with pytest.raises(ValueError, match="provenance"):
        _valid_emg(provenance=())
