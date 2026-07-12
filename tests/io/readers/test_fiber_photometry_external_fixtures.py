from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

from xpkg.io.readers import (
    read_doric_photometry,
    read_neurophotometrics_csv,
    read_pmat_events_csv,
    read_pmat_photometry_csv,
    read_pyphotometry_ppd,
    read_tdt_photometry_block,
)
from xpkg.model import PhotometryRecording, RecordingSession

pytestmark = pytest.mark.vendorfixtures


def _fixture_root() -> Path:
    raw_root = os.environ.get("XPKG_FIBER_FIXTURE_ROOT")
    if not raw_root:
        raise RuntimeError(
            "XPKG_FIBER_FIXTURE_ROOT is required for vendor-fixture tests. "
            "Run them through `make test-vendor`."
        )
    return Path(raw_root)


def _fixture(relative_path: str) -> Path:
    path = _fixture_root() / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Required fiber-photometry fixture does not exist: {path}")
    return path


def test_external_pyphotometry_ppd_fixture_loads() -> None:
    path = _fixture("pyphotometry/dopamine_data/P10V_16-2018-08-16-085115.ppd")

    session = read_pyphotometry_ppd(path)
    photometry = session.signal("photometry")

    assert isinstance(session, RecordingSession)
    assert isinstance(photometry, PhotometryRecording)
    assert photometry.series.n_samples > 1_000
    assert photometry.series.sample_rate_hz is not None
    assert photometry.series.sample_rate_hz > 0
    assert "digital" in session.signal_names


def test_external_pmat_csv_fixtures_load() -> None:
    photometry_path = _fixture("pmat_csv/Ca2Data .csv")
    events_path = _fixture("pmat_csv/BehData .csv")

    photometry = read_pmat_photometry_csv(photometry_path)
    events = read_pmat_events_csv(events_path)

    assert isinstance(photometry, PhotometryRecording)
    assert photometry.series.n_samples > 1_000
    assert photometry.reference_channel is not None
    assert len(events.events) > 0


def test_external_official_doric_fixture_loads() -> None:
    path = _fixture("doric/Console_Acq_0000.doric")

    session = read_doric_photometry(path)
    photometry = session.signal("photometry")

    assert isinstance(session, RecordingSession)
    assert isinstance(photometry, PhotometryRecording)
    assert photometry.series.n_samples > 0
    assert photometry.signal_channel in photometry.channel_names


def test_external_neurophotometrics_fixture_loads() -> None:
    path = _fixture("neurophotometrics/bl72bl82_12feb2024_fp.csv")

    session = read_neurophotometrics_csv(path)
    photometry = session.signal("photometry")

    assert isinstance(session, RecordingSession)
    assert isinstance(photometry, PhotometryRecording)
    assert photometry.series.n_samples > 1_000
    assert "G0" in photometry.channel_names
    assert "flags" in session.signal_names
    assert session.metadata["time_column"] == "SystemTimestamp"
    assert session.metadata["state_column"] == "LedState"


def test_external_pmat_tdt_block_loads_when_tdt_is_installed() -> None:
    if importlib.util.find_spec("tdt") is None:
        pytest.skip()
    path = _fixture("pmat_tdt/Photometry-161823")

    session = read_tdt_photometry_block(path)
    photometry = session.signal("photometry")

    assert isinstance(session, RecordingSession)
    assert isinstance(photometry, PhotometryRecording)
    assert photometry.series.n_samples > 0
    assert photometry.series.sample_rate_hz is not None
