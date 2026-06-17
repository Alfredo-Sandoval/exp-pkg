from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from xpkg.io.readers import read_pose_node_names, read_pose_track, resolve_pose_node_indices
from xpkg.io.readers.pose.dlc import read_node_names, read_track, resolve_node_indices


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


def test_read_dlc_h5_rejects_pickled_object_nodes(tmp_path: Path) -> None:
    path = tmp_path / "tracking.h5"
    with pytest.warns(pd.errors.PerformanceWarning):
        pd.DataFrame({"payload": [{"unsafe": "object"}]}).to_hdf(path, key="df")

    with pytest.raises(ValueError, match="object/pickled PyTables data"):
        read_track(path, file_type="h5", track_index=0)


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
    assert track.metadata["source"] == {
        "type": "dlc_h5" if file_type in {"h5", "hdf5"} else "dlc_csv",
        "path": str(path),
    }
    assert track.metadata["software"] == "DLC"
    assert track.metadata["file_type"] == file_type
    assert track.metadata["track_index"] == 0
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


def test_generic_pose_reader_dispatches_to_dlc_lightning_pose_and_sleap(tmp_path: Path) -> None:
    from tests.factories import write_sleap_analysis_h5

    sleap_path = tmp_path / "analysis.h5"
    write_sleap_analysis_h5(sleap_path)
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
    lightning_pose_track = read_pose_track(
        dlc_path,
        software="LightningPose",
        file_type="csv",
        track_index=0,
    )

    assert sleap_track.coords.shape == (10, 4, 2)
    assert dlc_track.coords.shape == (3, 2, 2)
    assert lightning_pose_track.coords.shape == (3, 2, 2)
    assert sleap_track.metadata["source"] == {
        "type": "sleap_analysis_h5",
        "path": str(sleap_path),
    }
    assert dlc_track.metadata["source"] == {"type": "dlc_csv", "path": str(dlc_path)}
    assert lightning_pose_track.metadata["source"] == {
        "type": "lightning_pose_csv",
        "path": str(dlc_path),
    }
    assert lightning_pose_track.metadata["software"] == "LIGHTNING_POSE"
    assert read_pose_node_names(dlc_path, software="DLC", file_type="csv") == ["HIP", "KNEE"]
    assert read_pose_node_names(dlc_path, software="lightning-pose", file_type="csv") == [
        "HIP",
        "KNEE",
    ]
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

    with pytest.raises(ValueError, match="Unsupported Lightning Pose file_type"):
        read_pose_track(path, software="LightningPose", file_type="h5")


# --- Real-format contract tests (byte-faithful to DeepLabCut's writer) -------


def _write_real_dlc_single_animal(csv_path: Path, h5_path: Path) -> None:
    # Mirror DeepLabCut's create_df_from_prediction / save_data writers exactly:
    # a DLC_resnet50_* scorer, a (scorer, bodyparts, coords=x/y/likelihood)
    # MultiIndex, a nameless integer index, empty-cell occlusions, and the real
    # HDF key "df_with_missing" stored as a PyTables table.
    scorer = "DLC_resnet50_DemoMay1shuffle1_50000"
    columns = pd.MultiIndex.from_product(
        [[scorer], ["snout", "tailbase"], ["x", "y", "likelihood"]],
        names=["scorer", "bodyparts", "coords"],
    )
    data = np.array(
        [
            [10.0, 20.0, 0.99, 30.0, 40.0, 0.95],
            [11.0, 21.0, 0.98, np.nan, 41.0, np.nan],  # tailbase occluded on frame 1
        ],
        dtype=np.float64,
    )
    frame = pd.DataFrame(data, columns=columns, index=range(2))  # nameless index
    frame.to_csv(csv_path)
    frame.to_hdf(h5_path, key="df_with_missing", format="table", mode="w")


def test_read_track_reads_real_dlc_single_animal_csv_and_h5(tmp_path: Path) -> None:
    csv_path = tmp_path / "vidDLC_resnet50_DemoMay1shuffle1_50000.csv"
    h5_path = tmp_path / "vidDLC_resnet50_DemoMay1shuffle1_50000.h5"
    _write_real_dlc_single_animal(csv_path, h5_path)

    for path, file_type in ((csv_path, "csv"), (h5_path, "h5")):
        track = read_pose_track(path, software="DLC", file_type=file_type)
        assert track.node_names == ("snout", "tailbase")
        assert track.coords.shape == (2, 2, 2)
        np.testing.assert_allclose(track.coords[0], [[10.0, 20.0], [30.0, 40.0]])
        # Empty CSV cell / NaN HDF value -> NaN coordinate (occluded keypoint).
        assert np.isnan(track.coords[1, 1, 0])


def test_read_track_rejects_real_multi_animal_dlc_csv(tmp_path: Path) -> None:
    # maDLC inserts an "individuals" header row (4 header rows). The single-animal
    # reader must fail loud rather than silently mis-parse it.
    scorer = "DLC_resnet50_DemoMay1shuffle1_50000"
    columns = pd.MultiIndex.from_product(
        [[scorer], ["mouse1", "mouse2"], ["snout", "tailbase"], ["x", "y", "likelihood"]],
        names=["scorer", "individuals", "bodyparts", "coords"],
    )
    data = np.arange(2 * 12, dtype=np.float64).reshape(2, 12)
    frame = pd.DataFrame(data, columns=columns, index=range(2))
    csv_path = tmp_path / "maDLC.csv"
    frame.to_csv(csv_path)

    with pytest.raises(ValueError):
        read_pose_track(csv_path, software="DLC", file_type="csv")
