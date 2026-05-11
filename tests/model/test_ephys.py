from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from xpkg.model import (
    EphysRecording,
    RecordingMode,
    SignalChannel,
    StimulusEpoch,
    Sweep,
    SweepSet,
    Timeline,
    TimeSeries,
)
from xpkg.model.ephys import (
    current_scale_to_pA,
    detect_recording_mode,
    is_current_unit,
    is_voltage_unit,
    normalize_signal_units,
    voltage_scale_to_mV,
)


def _build_sweep(
    *,
    index: int = 0,
    n_samples: int = 100,
    sample_rate_hz: float = 10_000.0,
    channels: tuple[tuple[str, str], ...] = (("electrode", "mV"),),
    epochs: tuple[StimulusEpoch, ...] = (),
) -> Sweep:
    values = np.zeros((n_samples, len(channels)), dtype=np.float64)
    timeline = Timeline.from_sample_rate(
        n_samples=n_samples, sample_rate_hz=sample_rate_hz
    )
    series = TimeSeries(
        values=values,
        channels=tuple(SignalChannel(name=name, unit=unit) for name, unit in channels),
        timeline=timeline,
        name=f"sweep_{index}",
    )
    return Sweep(index=index, series=series, epochs=epochs)


def test_stimulus_epoch_validates_and_exposes_time_range() -> None:
    epoch = StimulusEpoch(
        index=0,
        kind="step",
        start_s=0.05,
        duration_s=0.20,
        level=-50.0,
        level_unit="pA",
    )

    assert epoch.kind == "step"
    assert epoch.end_s == pytest.approx(0.25)
    assert epoch.time_range.start_s == pytest.approx(0.05)
    assert epoch.time_range.end_s == pytest.approx(0.25)


def test_stimulus_epoch_rejects_negative_duration() -> None:
    with pytest.raises(ValueError, match="duration_s must be non-negative"):
        StimulusEpoch(index=0, kind="step", start_s=0.0, duration_s=-0.1)


def test_sweep_set_requires_consistent_channel_layout() -> None:
    sweep_a = _build_sweep(index=0, channels=(("electrode", "mV"),))
    sweep_b = _build_sweep(index=1, channels=(("monitor", "pA"),))
    with pytest.raises(ValueError, match="share channel layout"):
        SweepSet.from_sweeps([sweep_a, sweep_b])


def test_sweep_set_rejects_duplicate_indices() -> None:
    sweep_a = _build_sweep(index=0)
    sweep_b = _build_sweep(index=0)
    with pytest.raises(ValueError, match="indices must be unique"):
        SweepSet.from_sweeps([sweep_a, sweep_b])


def test_ephys_recording_exposes_summary_properties() -> None:
    epoch = StimulusEpoch(
        index=0, kind="step", start_s=0.0, duration_s=0.05, level=-50.0, level_unit="pA"
    )
    sweeps = SweepSet.from_sweeps(
        [
            _build_sweep(index=i, n_samples=50, epochs=(epoch,))
            for i in range(3)
        ]
    )
    recording = EphysRecording(
        sweeps=sweeps,
        mode="current_clamp",
        channel_roles={"electrode": "electrode"},
        conversion_log=({"channel": "electrode", "scale": 1000.0},),
        provenance={"source": {"type": "test"}},
    )

    assert recording.n_sweeps == 3
    assert recording.channel_names == ("electrode",)
    assert recording.electrode_channel == "electrode"
    assert recording.stimulus_monitor_channel is None
    assert recording.sample_rate_hz == pytest.approx(10_000.0)
    assert recording.duration_s == pytest.approx(3 * sweeps[0].duration_s)
    assert recording.channel_unit("electrode") == "mV"
    assert recording.conversion_log[0]["channel"] == "electrode"


def test_ephys_recording_rejects_unknown_role_channels() -> None:
    sweeps = SweepSet.from_sweeps([_build_sweep(index=0)])
    with pytest.raises(ValueError, match="unknown channels"):
        EphysRecording(
            sweeps=sweeps,
            channel_roles={"missing": "electrode"},
        )


def test_ephys_recording_rejects_invalid_mode() -> None:
    sweeps = SweepSet.from_sweeps([_build_sweep(index=0)])
    with pytest.raises(ValueError, match="mode must be"):
        EphysRecording(sweeps=sweeps, mode=cast(RecordingMode, "something_else"))


def test_recording_mode_detection_uses_electrode_unit() -> None:
    assert (
        detect_recording_mode(
            channel_roles={"IN0": "electrode"},
            channel_units={"IN0": "mV"},
        )
        == "current_clamp"
    )
    assert (
        detect_recording_mode(
            channel_roles={"IN0": "electrode"},
            channel_units={"IN0": "pA"},
        )
        == "voltage_clamp"
    )
    assert (
        detect_recording_mode(
            channel_roles={"IN0": "auxiliary"},
            channel_units={"IN0": "mV"},
        )
        == "unknown"
    )


def test_unit_helpers_recognize_voltage_and_current() -> None:
    assert is_voltage_unit("mV")
    assert is_voltage_unit("V")
    assert not is_voltage_unit("pA")
    assert is_current_unit("pA")
    assert is_current_unit("nA")
    assert not is_current_unit("mV")
    assert voltage_scale_to_mV("V") == pytest.approx(1000.0)
    assert current_scale_to_pA("nA") == pytest.approx(1000.0)


def test_normalize_signal_units_rescales_and_logs() -> None:
    values = np.array([[1.0, 0.001], [2.0, 0.002]], dtype=np.float64)
    channels = (
        SignalChannel(name="electrode", unit="V"),
        SignalChannel(name="monitor", unit="nA"),
    )
    scaled, normalized, log = normalize_signal_units(
        values,
        channels,
        roles={"electrode": "electrode", "monitor": "stimulus_monitor"},
    )

    np.testing.assert_allclose(scaled[:, 0], [1000.0, 2000.0])
    np.testing.assert_allclose(scaled[:, 1], [1.0, 2.0])
    assert normalized[0].unit == "mV"
    assert normalized[1].unit == "pA"
    assert normalized[0].metadata["original_unit"] == "V"
    by_channel = {entry["channel"]: entry for entry in log}
    assert by_channel["electrode"]["from_unit"] == "V"
    assert by_channel["electrode"]["to_unit"] == "mV"
    assert by_channel["monitor"]["scale"] == pytest.approx(1000.0)


def test_normalize_signal_units_passes_through_already_canonical_channels() -> None:
    values = np.ones((3, 1), dtype=np.float64)
    channels = (SignalChannel(name="electrode", unit="mV"),)
    scaled, normalized, log = normalize_signal_units(
        values, channels, roles={"electrode": "electrode"}
    )
    np.testing.assert_allclose(scaled, values)
    assert normalized[0].unit == "mV"
    assert log == []
