from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from xpkg.io.readers import candidate_vicon_emg_channels, extract_vicon_emg
from xpkg.model import ViconAnalogData, ViconRecording


def _recording(*, analog: ViconAnalogData | None) -> ViconRecording:
    return ViconRecording(
        path=Path("trial.c3d"),
        source_type="c3d",
        fps=100,
        marker_names=("center",),
        source_marker_labels=("Subject:center",),
        positions=np.zeros((2, 1, 3), dtype=np.float64),
        marker_valid=np.ones((2, 1), dtype=bool),
        frame_offset=1,
        analog=analog,
    )


def _analog() -> ViconAnalogData:
    return ViconAnalogData(
        fps=1000,
        samples_per_frame=2,
        channel_names=(
            "Force.Fx1",
            "Voltage.RTA",
            "Voltage.RGAS",
            "Voltage.LBF",
            "Voltage.RVL",
            "Voltage.FootswitchA",
        ),
        channel_units=("N", "V", "V", "V", "V", "V"),
        values=np.array(
            [
                [1.0, 10.0, 20.0, 30.0, 40.0, 90.0],
                [2.0, 11.0, 21.0, 31.0, 41.0, 91.0],
                [3.0, 12.0, 22.0, 32.0, 42.0, 92.0],
                [4.0, 13.0, 23.0, 33.0, 43.0, 93.0],
            ],
            dtype=np.float64,
        ),
    )


def test_candidate_vicon_emg_channels_preserves_order_and_excludes_non_emg() -> None:
    assert candidate_vicon_emg_channels(
        [
            "Force.Fx1",
            "Voltage.RTA",
            "Moment.Mx1",
            "Voltage.RGAS",
            "Voltage.FootswitchA",
            "Voltage.LBF",
            "Voltage.RVL",
        ]
    ) == ("Voltage.RTA", "Voltage.RGAS", "Voltage.LBF", "Voltage.RVL")


def test_extract_vicon_emg_uses_explicit_mapping() -> None:
    recording = _recording(analog=_analog())

    emg = extract_vicon_emg(
        recording,
        {
            "right_ta": {
                "analog_channel": "Voltage.RTA",
                "muscle_name": "tibialis anterior",
                "side": "right",
            },
            "left_bf": ("Voltage.LBF", "biceps femoris", "left"),
        },
    )

    assert emg.channel_names == ("Voltage.RTA", "Voltage.LBF")
    assert emg.muscle_names == ("tibialis anterior", "biceps femoris")
    assert emg.sides == ("right", "left")
    assert emg.sample_rate_hz == 1000.0
    assert emg.units == (("signal", "V"),)
    assert emg.processing_state == "raw"
    np.testing.assert_allclose(emg.sample_times_s, np.array([0.0, 0.001, 0.002, 0.003]))
    np.testing.assert_allclose(
        emg.signals,
        np.array(
            [
                [10.0, 30.0],
                [11.0, 31.0],
                [12.0, 32.0],
                [13.0, 33.0],
            ],
            dtype=np.float64,
        ),
    )
    assert ("source_path", "trial.c3d") in emg.provenance
    assert ("reader", "extract_vicon_emg") in emg.provenance
    assert ("source_channels", "Voltage.RTA,Voltage.LBF") in emg.provenance


def test_extract_vicon_emg_requires_explicit_mapping() -> None:
    with pytest.raises(ValueError, match="explicit channel mapping"):
        extract_vicon_emg(_recording(analog=_analog()), {})


def test_extract_vicon_emg_fails_when_analog_is_missing() -> None:
    with pytest.raises(ValueError, match="no analog data"):
        extract_vicon_emg(
            _recording(analog=None),
            {"Voltage.RTA": ("tibialis anterior", "right")},
        )


def test_extract_vicon_emg_fails_when_channel_is_missing() -> None:
    with pytest.raises(KeyError, match="Voltage.Missing"):
        extract_vicon_emg(
            _recording(analog=_analog()),
            {"Voltage.Missing": ("tibialis anterior", "right")},
        )


def test_extract_vicon_emg_preserves_selected_channel_order() -> None:
    emg = extract_vicon_emg(
        _recording(analog=_analog()),
        {
            "Voltage.RVL": ("vastus lateralis", "right"),
            "Voltage.RGAS": ("gastrocnemius", "right"),
            "Voltage.RTA": ("tibialis anterior", "right"),
        },
    )

    assert emg.channel_names == ("Voltage.RVL", "Voltage.RGAS", "Voltage.RTA")
    np.testing.assert_allclose(emg.signals[0], np.array([40.0, 20.0, 10.0]))
