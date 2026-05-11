# Attribute and method names on FakeABF intentionally mirror pyABF's camelCase
# public surface so read_abf can call them through the same getattr paths.
# ruff: noqa: N802, N803, N815
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from xpkg.io.readers import read_abf
from xpkg.model import EphysRecording


@dataclass
class FakeSweepEpochs:
    types: list[str]
    levels: list[float]
    p1s: list[int]
    p2s: list[int]


@dataclass
class FakeABF:
    """Minimal stand-in matching the slice of pyABF's surface that read_abf uses."""

    sweep_data: dict[int, dict[int, np.ndarray]]
    times: np.ndarray
    adcNames: list[str]
    adcUnits: list[str]
    channelList: list[int]
    sweepCount: int
    dataRate: float
    sweepEpochsByChannel: dict[int, FakeSweepEpochs] = field(default_factory=dict)
    sweepUnitsC: str = ""
    abfVersionString: str = "ABF2 (synthetic)"
    protocol: str = "synthetic_protocol"
    protocolPath: str = "synthetic.pro"
    _current_sweep: int = 0
    _current_channel: int = 0

    def setSweep(self, *, sweepNumber: int, channel: int) -> None:
        self._current_sweep = int(sweepNumber)
        self._current_channel = int(channel)

    @property
    def sweepX(self) -> np.ndarray:
        return self.times

    @property
    def sweepY(self) -> np.ndarray:
        return self.sweep_data[self._current_sweep][self._current_channel]

    @property
    def sweepEpochs(self) -> FakeSweepEpochs:
        return self.sweepEpochsByChannel[self._current_channel]


@dataclass
class FakePyABFModule:
    abf: FakeABF
    __version__: str = "fake-2.0"

    def ABF(self, path: str) -> FakeABF:
        return self.abf


def _build_fake_abf(*, sample_rate_hz: float = 10_000.0, n_samples: int = 200) -> FakeABF:
    times = np.arange(n_samples, dtype=np.float64) / sample_rate_hz
    sweep_data: dict[int, dict[int, np.ndarray]] = {}
    for sweep_index in range(3):
        baseline = -0.065 + 0.001 * sweep_index  # volts
        electrode = baseline + 0.001 * np.sin(2 * np.pi * 5.0 * times)
        # Stimulus monitor in nA: -50 pA = -0.05 nA during a 50-sample step.
        monitor = np.zeros_like(times)
        monitor[50:150] = -0.05
        sweep_data[sweep_index] = {0: electrode, 1: monitor}
    epochs_for_electrode = FakeSweepEpochs(
        types=["Off", "Step", "Off"],
        levels=[0.0, -50.0, 0.0],
        p1s=[0, 50, 150],
        p2s=[50, 150, n_samples],
    )
    epochs_for_monitor = FakeSweepEpochs(
        types=["Off", "Step", "Off"],
        levels=[0.0, -50.0, 0.0],
        p1s=[0, 50, 150],
        p2s=[50, 150, n_samples],
    )
    return FakeABF(
        sweep_data=sweep_data,
        times=times,
        adcNames=["IN_0", "IN_1"],
        adcUnits=["V", "nA"],
        channelList=[0, 1],
        sweepCount=3,
        dataRate=sample_rate_hz,
        sweepEpochsByChannel={0: epochs_for_electrode, 1: epochs_for_monitor},
        sweepUnitsC="pA",
    )


def test_read_abf_normalizes_units_and_detects_current_clamp_mode(tmp_path) -> None:
    abf_path = tmp_path / "cell01.abf"
    abf_path.write_bytes(b"ABF synthetic placeholder")
    fake_module = FakePyABFModule(abf=_build_fake_abf())

    recording = read_abf(abf_path, pyabf_module=fake_module)

    assert isinstance(recording, EphysRecording)
    assert recording.mode == "current_clamp"
    assert recording.n_sweeps == 3
    assert recording.channel_names == ("IN_0", "IN_1")
    assert recording.channel_roles == {
        "IN_0": "electrode",
        "IN_1": "stimulus_monitor",
    }
    assert recording.channel_unit("IN_0") == "mV"
    assert recording.channel_unit("IN_1") == "pA"
    assert recording.sample_rate_hz == pytest.approx(10_000.0)

    # Conversion log: V -> mV (x1000) and nA -> pA (x1000) per sweep.
    by_channel = {
        (entry["sweep_index"], entry["channel"]): entry
        for entry in recording.conversion_log
    }
    assert by_channel[(0, "IN_0")]["scale"] == pytest.approx(1000.0)
    assert by_channel[(0, "IN_1")]["scale"] == pytest.approx(1000.0)

    # Provenance includes file hash and parser version.
    source = recording.provenance["source"]
    assert source["type"] == "abf"
    assert source["sha256"]
    assert recording.provenance["parser"] == {"name": "pyabf", "version": "fake-2.0"}
    assert recording.provenance["protocol"] == "synthetic_protocol"

    # First sweep electrode column should be in mV after scaling.
    sweep0 = recording.sweeps[0]
    np.testing.assert_allclose(sweep0.series.values[0, 0], -65.0, rtol=1e-9, atol=1e-9)
    # Stimulus monitor step is -0.05 nA -> -50 pA mid-sweep.
    np.testing.assert_allclose(sweep0.series.values[100, 1], -50.0, atol=1e-9)


def test_read_abf_captures_protocol_epochs_per_sweep(tmp_path) -> None:
    abf_path = tmp_path / "cell01.abf"
    abf_path.write_bytes(b"ABF synthetic placeholder")
    fake_module = FakePyABFModule(abf=_build_fake_abf())

    recording = read_abf(abf_path, pyabf_module=fake_module)

    sweep0 = recording.sweeps[0]
    assert [epoch.kind for epoch in sweep0.epochs] == ["Off", "Step", "Off"]
    step_epoch = sweep0.epochs[1]
    assert step_epoch.start_s == pytest.approx(50 / 10_000.0)
    assert step_epoch.duration_s == pytest.approx(100 / 10_000.0)
    assert step_epoch.level == pytest.approx(-50.0)
    assert step_epoch.level_unit == "pA"


def test_read_abf_respects_user_overrides(tmp_path) -> None:
    abf_path = tmp_path / "cell01.abf"
    abf_path.write_bytes(b"ABF synthetic placeholder")
    fake_module = FakePyABFModule(abf=_build_fake_abf())

    recording = read_abf(
        abf_path,
        channel_roles={"IN_1": "auxiliary"},
        pyabf_module=fake_module,
    )

    assert recording.channel_roles["IN_1"] == "auxiliary"
    assert recording.metadata["role_overrides"] == {"IN_1": "auxiliary"}


def test_read_abf_raises_when_file_missing(tmp_path) -> None:
    fake_module = FakePyABFModule(abf=_build_fake_abf())
    with pytest.raises(FileNotFoundError):
        read_abf(tmp_path / "missing.abf", pyabf_module=fake_module)
