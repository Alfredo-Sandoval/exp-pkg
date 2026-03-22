from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from posetta.io.readers import read_pose_node_names, read_pose_track, resolve_pose_node_indices
from posetta.io.readers.dlc import read_node_names, read_track, resolve_node_indices


def _make_dlc_dataframe(*, include_likelihood: bool = True) -> pd.DataFrame:
    tuples: list[tuple[str, str, str]] = [
        ("model", "HIP", "x"),
        ("model", "HIP", "y"),
        ("model", "KNEE", "x"),
        ("model", "KNEE", "y"),
    ]
    data = np.array(
        [
            [1.0, 2.0, 11.0, 12.0],
            [3.0, 4.0, 13.0, 14.0],
            [5.0, 6.0, np.nan, 16.0],
        ],
        dtype=np.float64,
    )
    if include_likelihood:
        tuples.extend(
            [
                ("model", "HIP", "likelihood"),
                ("model", "KNEE", "likelihood"),
            ]
        )
        data = np.column_stack(
            [
                data,
                np.array([0.9, 0.7, 0.5], dtype=np.float64),
                np.array([0.8, 0.6, np.nan], dtype=np.float64),
            ]
        )

    columns = pd.MultiIndex.from_tuples(tuples)
    return pd.DataFrame(
        data,
        index=pd.Index([0, 1, 2], name="frame"),
        columns=columns,
    )


def _write_dlc_csv(path: Path, *, include_likelihood: bool = True) -> None:
    _make_dlc_dataframe(include_likelihood=include_likelihood).to_csv(path)


def _write_dlc_h5(path: Path, *, include_likelihood: bool = True) -> None:
    _make_dlc_dataframe(include_likelihood=include_likelihood).to_hdf(path, key="df")


@pytest.mark.parametrize(
    ("file_type", "writer"),
    [("csv", _write_dlc_csv), ("h5", _write_dlc_h5), ("hdf5", _write_dlc_h5)],
)
def test_read_track_returns_expected_dlc_arrays(
    tmp_path: Path,
    file_type: str,
    writer: Callable[..., None],
) -> None:
    path = tmp_path / f"tracking.{file_type}"
    writer(path)

    track = read_track(path, file_type=file_type, track_index=0)

    expected_coords = np.array(
        [
            [[1.0, 2.0], [11.0, 12.0]],
            [[3.0, 4.0], [13.0, 14.0]],
            [[5.0, 6.0], [np.nan, 16.0]],
        ],
        dtype=np.float64,
    )
    expected_scores = np.array(
        [
            [0.9, 0.8],
            [0.7, 0.6],
            [0.5, np.nan],
        ],
        dtype=np.float64,
    )
    expected_instance_score = np.array([0.85, 0.65, 0.5], dtype=np.float64)

    assert track.coords.shape == (3, 2, 2)
    assert track.scores.shape == (3, 2)
    assert track.instance_score.shape == (3,)
    assert track.coords.dtype == np.float64
    assert track.scores.dtype == np.float64
    assert track.instance_score.dtype == np.float64
    assert track.node_names == ("HIP", "KNEE")
    np.testing.assert_allclose(track.coords, expected_coords, equal_nan=True)
    np.testing.assert_allclose(track.scores, expected_scores, equal_nan=True)
    np.testing.assert_allclose(track.instance_score, expected_instance_score, equal_nan=True)


@pytest.mark.parametrize(
    ("file_type", "writer"),
    [("csv", _write_dlc_csv), ("h5", _write_dlc_h5)],
)
def test_read_node_names_and_resolve_indices_for_dlc(
    tmp_path: Path,
    file_type: str,
    writer: Callable[..., None],
) -> None:
    path = tmp_path / f"tracking.{file_type}"
    writer(path)

    assert read_node_names(path, file_type=file_type) == ["HIP", "KNEE"]
    assert resolve_node_indices(
        path,
        file_type=file_type,
        target_names=["KNEE", "HIP", "KNEE"],
    ) == [1, 0]


@pytest.mark.parametrize(
    ("file_type", "writer"),
    [("csv", _write_dlc_csv), ("h5", _write_dlc_h5)],
)
def test_read_track_rejects_nonzero_dlc_track_index(
    tmp_path: Path,
    file_type: str,
    writer: Callable[..., None],
) -> None:
    path = tmp_path / f"tracking.{file_type}"
    writer(path)

    with pytest.raises(ValueError, match="track_index must be 0"):
        read_track(path, file_type=file_type, track_index=1)


@pytest.mark.parametrize(
    ("file_type", "writer"),
    [("csv", _write_dlc_csv), ("h5", _write_dlc_h5)],
)
def test_read_track_requires_dlc_likelihood_columns(
    tmp_path: Path,
    file_type: str,
    writer: Callable[..., None],
) -> None:
    path = tmp_path / f"tracking.{file_type}"
    writer(path, include_likelihood=False)

    with pytest.raises(ValueError, match="likelihood"):
        read_track(path, file_type=file_type, track_index=0)


def test_generic_pose_reader_dispatches_to_dlc_and_sleap(tmp_path: Path) -> None:
    from tests.io.readers.test_sleap_analysis_h5 import _write_sleap_analysis_h5

    sleap_path = tmp_path / "analysis.h5"
    _write_sleap_analysis_h5(sleap_path)
    dlc_path = tmp_path / "tracking.csv"
    _write_dlc_csv(dlc_path)

    sleap_track = read_pose_track(
        sleap_path,
        software="SLEAP",
        file_type="hdf5",
        track_index=1,
    )
    dlc_track = read_pose_track(
        dlc_path,
        software="DLC",
        file_type="csv",
        track_index=0,
    )

    assert sleap_track.coords.shape == (10, 4, 2)
    assert dlc_track.coords.shape == (3, 2, 2)
    assert read_pose_node_names(dlc_path, software="DLC", file_type="csv") == ["HIP", "KNEE"]
    assert resolve_pose_node_indices(
        sleap_path,
        software="SLEAP",
        file_type="h5",
        target_names=["TOE", "HIP"],
    ) == [3, 0]


def test_generic_pose_reader_rejects_unsupported_software_file_type_combo(tmp_path: Path) -> None:
    path = tmp_path / "tracking.csv"
    _write_dlc_csv(path)

    with pytest.raises(ValueError, match="Unsupported SLEAP file_type"):
        read_pose_track(path, software="SLEAP", file_type="csv")
