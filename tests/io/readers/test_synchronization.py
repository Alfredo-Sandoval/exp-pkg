from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from xpkg.io.readers import read_synchronization_csv
from xpkg.model import AlignmentModel, SynchronizationMethod, Timebase


def test_read_synchronization_csv_returns_paired_typed_evidence(tmp_path: Path) -> None:
    path = tmp_path / "sync.csv"
    path.write_text(
        "pulse_id,source_time_s,target_time_s\np1,0.0,0.25\np2,10.0,10.35\n",
        encoding="utf-8",
    )

    alignment = read_synchronization_csv(
        path,
        source_timebase=Timebase(name="camera"),
        target_timebase=Timebase(name="daq"),
        model=AlignmentModel.AFFINE,
        method=SynchronizationMethod.PULSES,
    )

    assert alignment.name == "camera-to-daq"
    assert alignment.scale == pytest.approx(1.01)
    assert alignment.offset_s == pytest.approx(0.25)
    assert alignment.evidence[0].correspondence_id == "p1"
    assert alignment.metadata["source"]["path"] == str(path)


@settings(max_examples=24, deadline=None)
@given(
    source_times=st.lists(
        st.integers(min_value=-1000, max_value=1000),
        min_size=2,
        max_size=12,
        unique=True,
    ),
    scale=st.integers(min_value=1, max_value=5),
    offset=st.integers(min_value=-100, max_value=100),
)
def test_synchronization_csv_affine_fit_round_trips_generated_correspondences(
    tmp_path: Path,
    source_times: list[int],
    scale: int,
    offset: int,
) -> None:
    path = tmp_path / "generated-sync.csv"
    rows = ["source_time_s,target_time_s"]
    rows.extend(f"{value},{(value * scale) + offset}" for value in source_times)
    path.write_text("\n".join(rows), encoding="utf-8")

    alignment = read_synchronization_csv(
        path,
        source_timebase=Timebase(name="source"),
        target_timebase=Timebase(name="target"),
        model=AlignmentModel.AFFINE,
        method=SynchronizationMethod.TIMESTAMPS,
    )

    assert alignment.scale == pytest.approx(scale)
    assert alignment.offset_s == pytest.approx(offset)
    assert alignment.residual_s == pytest.approx(0.0, abs=1e-9)
    for value in source_times:
        assert alignment.map_time(value) == pytest.approx((value * scale) + offset)


def test_synchronization_csv_requires_both_clock_columns(tmp_path: Path) -> None:
    path = tmp_path / "one-clock.csv"
    path.write_text("source_time_s\n0.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="target time column"):
        read_synchronization_csv(
            path,
            source_timebase=Timebase(name="source"),
            target_timebase=Timebase(name="target"),
        )
