from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from xpkg._core.json_utils import write_json
from xpkg.io.readers import (
    read_pose_node_names,
    read_pose_track,
    resolve_pose_node_indices,
)
from xpkg.io.readers.mediapipe_pose_landmarks import (
    MEDIAPIPE_POSE_LANDMARK_NAMES,
    MEDIAPIPE_POSE_LANDMARKS_JSON_SCHEMA,
    read_node_names,
    read_track,
    resolve_node_indices,
)


def _pose_landmarks(
    *,
    x_shift: float = 0.0,
    y_shift: float = 0.0,
    visibility: float = 0.9,
    presence: float | None = 0.8,
) -> list[dict[str, float]]:
    landmarks: list[dict[str, float]] = []
    for index, _node_name in enumerate(MEDIAPIPE_POSE_LANDMARK_NAMES):
        entry: dict[str, float] = {
            "x": 0.05 + (index * 0.01) + x_shift,
            "y": 0.10 + (index * 0.005) + y_shift,
            "z": -0.01 * index,
            "visibility": visibility,
        }
        if presence is not None:
            entry["presence"] = presence
        landmarks.append(entry)
    return landmarks


def _write_mediapipe_pose_landmarks_json(
    path: Path,
    *,
    image_width: int = 200,
    image_height: int = 100,
    frames: list[dict[str, object]] | None = None,
) -> None:
    if frames is None:
        frames = [
            {"frame_index": 0, "pose_landmarks": _pose_landmarks()},
            {"frame_index": 1, "pose_landmarks": _pose_landmarks(x_shift=0.02, y_shift=0.01)},
        ]

    write_json(
        path,
        {
            "schema": MEDIAPIPE_POSE_LANDMARKS_JSON_SCHEMA,
            "image_width": image_width,
            "image_height": image_height,
            "frames": frames,
        },
    )


def test_read_track_returns_pixel_space_arrays_and_scores(tmp_path: Path) -> None:
    json_path = tmp_path / "pose_landmarks.json"
    _write_mediapipe_pose_landmarks_json(
        json_path,
        frames=[
            {"frame_index": 0, "pose_landmarks": _pose_landmarks(visibility=0.9, presence=0.8)},
            {"frame_index": 2, "pose_landmarks": _pose_landmarks(x_shift=0.1, visibility=0.7)},
        ],
    )

    track = read_track(json_path, track_index=0)

    assert track.node_names == MEDIAPIPE_POSE_LANDMARK_NAMES
    assert track.coords.shape == (3, len(MEDIAPIPE_POSE_LANDMARK_NAMES), 2)
    assert track.scores.shape == (3, len(MEDIAPIPE_POSE_LANDMARK_NAMES))
    np.testing.assert_allclose(track.coords[0, 0], np.array([10.0, 10.0]))
    np.testing.assert_allclose(track.coords[2, 0], np.array([30.0, 10.0]))
    assert np.isnan(track.coords[1]).all()
    np.testing.assert_allclose(track.scores[0, 0], 0.8)
    np.testing.assert_allclose(track.scores[2, 0], 0.7)
    np.testing.assert_allclose(track.instance_score[[0, 2]], np.array([0.8, 0.7]))
    assert np.isnan(track.instance_score[1])


def test_read_node_names_and_resolve_indices_for_mediapipe(tmp_path: Path) -> None:
    json_path = tmp_path / "pose_landmarks.json"
    _write_mediapipe_pose_landmarks_json(json_path)

    assert read_node_names(json_path) == list(MEDIAPIPE_POSE_LANDMARK_NAMES)
    assert resolve_node_indices(
        json_path,
        target_names=["left_wrist", "nose", "left_wrist"],
    ) == [15, 0]


def test_read_track_rejects_nonzero_mediapipe_track_index(tmp_path: Path) -> None:
    json_path = tmp_path / "pose_landmarks.json"
    _write_mediapipe_pose_landmarks_json(json_path)

    with pytest.raises(ValueError, match="track_index must be 0"):
        read_track(json_path, track_index=1)


def test_generic_pose_reader_dispatches_to_mediapipe(tmp_path: Path) -> None:
    json_path = tmp_path / "pose_landmarks.json"
    _write_mediapipe_pose_landmarks_json(json_path)

    track = read_pose_track(
        json_path,
        software="MEDIAPIPE",
        file_type="json",
        track_index=0,
    )

    assert track.coords.shape == (2, len(MEDIAPIPE_POSE_LANDMARK_NAMES), 2)
    assert read_pose_node_names(
        json_path,
        software="MEDIAPIPE",
        file_type="json",
    ) == list(MEDIAPIPE_POSE_LANDMARK_NAMES)
    assert resolve_pose_node_indices(
        json_path,
        software="MEDIAPIPE",
        file_type="json",
        target_names=["right_ankle", "nose"],
    ) == [28, 0]
