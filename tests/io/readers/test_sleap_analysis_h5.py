from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from xpkg.io.readers.sleap_analysis_h5 import (
    read_node_names,
    read_track,
    read_track_count,
    read_track_names,
    resolve_node_indices,
)


def _write_sleap_analysis_h5(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[bytes]]:
    n_tracks = 2
    n_nodes = 4
    n_frames = 10

    tracks = np.zeros((n_tracks, 2, n_nodes, n_frames), dtype=np.float64)
    point_scores = np.zeros((n_tracks, n_nodes, n_frames), dtype=np.float64)
    instance_scores = np.zeros((n_tracks, n_frames), dtype=np.float64)

    for track_idx in range(n_tracks):
        for node_idx in range(n_nodes):
            for frame_idx in range(n_frames):
                x = 100.0 * track_idx + 10.0 * node_idx + frame_idx
                y = -x
                tracks[track_idx, 0, node_idx, frame_idx] = x
                tracks[track_idx, 1, node_idx, frame_idx] = y
                point_scores[track_idx, node_idx, frame_idx] = (
                    0.1 * track_idx + 0.01 * node_idx + 0.001 * frame_idx
                )
        for frame_idx in range(n_frames):
            instance_scores[track_idx, frame_idx] = 0.5 + 0.1 * track_idx + 0.01 * frame_idx

    # Ensure NaN propagation can be validated.
    tracks[1, 0, 2, 5] = np.nan

    node_names = [b"HIP", b"KNEE", b"ANKLE", b"TOE"]
    track_names = [b"track_0", b"track_1"]

    with h5py.File(path, "w") as handle:
        handle.create_dataset("tracks", data=tracks)
        handle.create_dataset("point_scores", data=point_scores)
        handle.create_dataset("instance_scores", data=instance_scores)
        handle.create_dataset("node_names", data=np.asarray(node_names, dtype="S"))
        handle.create_dataset("track_names", data=np.asarray(track_names, dtype="S"))

    return tracks, point_scores, instance_scores, node_names


def test_read_node_names_decodes_utf8_bytes(tmp_path: Path) -> None:
    path = tmp_path / "analysis.h5"
    _write_sleap_analysis_h5(path)

    names = read_node_names(path)

    assert names == ["HIP", "KNEE", "ANKLE", "TOE"]


def test_read_track_count_and_names(tmp_path: Path) -> None:
    path = tmp_path / "analysis.h5"
    _write_sleap_analysis_h5(path)

    assert read_track_count(path) == 2
    assert read_track_names(path) == ["track_0", "track_1"]


@pytest.mark.parametrize("track_index", [0, 1])
def test_read_track_returns_expected_values_and_shapes(tmp_path: Path, track_index: int) -> None:
    path = tmp_path / "analysis.h5"
    tracks, point_scores, instance_scores, _node_names = _write_sleap_analysis_h5(path)

    track = read_track(path, track_index=track_index)

    expected_coords = np.stack(
        [tracks[track_index, 0].T, tracks[track_index, 1].T],
        axis=-1,
    )
    expected_scores = point_scores[track_index].T
    expected_instance = instance_scores[track_index]

    assert track.coords.shape == (10, 4, 2)
    assert track.scores.shape == (10, 4)
    assert track.instance_score.shape == (10,)
    assert track.coords.dtype == np.float64
    assert track.scores.dtype == np.float64
    assert track.instance_score.dtype == np.float64
    assert track.node_names == ("HIP", "KNEE", "ANKLE", "TOE")
    np.testing.assert_allclose(track.coords, expected_coords, equal_nan=True)
    np.testing.assert_allclose(track.scores, expected_scores)
    np.testing.assert_allclose(track.instance_score, expected_instance)


def test_read_track_raises_index_error_for_out_of_range_track(tmp_path: Path) -> None:
    path = tmp_path / "analysis.h5"
    _write_sleap_analysis_h5(path)

    with pytest.raises(IndexError, match="track_index=2 out of range"):
        read_track(path, track_index=2)


def test_resolve_node_indices_maps_names_and_raises_for_missing(tmp_path: Path) -> None:
    path = tmp_path / "analysis.h5"
    _write_sleap_analysis_h5(path)

    assert resolve_node_indices(path, ["TOE", "HIP", "TOE"]) == [3, 0]

    with pytest.raises(KeyError, match="ELBOW"):
        resolve_node_indices(path, ["HIP", "ELBOW"])


def test_read_track_preserves_nan_coordinates(tmp_path: Path) -> None:
    path = tmp_path / "analysis.h5"
    _write_sleap_analysis_h5(path)

    track = read_track(path, track_index=1)

    assert np.isnan(track.coords[5, 2, 0])
