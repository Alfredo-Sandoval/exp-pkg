from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.factories import pose_landmarks, write_mediapipe_pose_landmarks_json
from xpkg.io.readers import (
    read_pose_node_names,
    read_pose_track,
    resolve_pose_node_indices,
)
from xpkg.io.readers.pose.mediapipe_pose_landmarks import (
    MEDIAPIPE_POSE_LANDMARK_NAMES,
    read_node_names,
    read_track,
    resolve_node_indices,
)


def test_read_track_returns_pixel_space_arrays_and_scores(tmp_path: Path) -> None:
    json_path = tmp_path / "pose_landmarks.json"
    write_mediapipe_pose_landmarks_json(
        json_path,
        frames=[
            {"frame_index": 0, "pose_landmarks": pose_landmarks(visibility=0.9, presence=0.8)},
            {"frame_index": 2, "pose_landmarks": pose_landmarks(x_shift=0.1, visibility=0.7)},
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
    assert track.metadata["source"] == {
        "type": "mediapipe_pose_landmarks_json",
        "path": str(json_path),
    }
    assert track.metadata["software"] == "MEDIAPIPE"
    assert track.metadata["file_type"] == "json"
    assert track.metadata["track_index"] == 0


def test_read_node_names_and_resolve_indices_for_mediapipe(tmp_path: Path) -> None:
    json_path = tmp_path / "pose_landmarks.json"
    write_mediapipe_pose_landmarks_json(json_path)

    assert read_node_names(json_path) == list(MEDIAPIPE_POSE_LANDMARK_NAMES)
    assert resolve_node_indices(
        json_path,
        target_names=["left_wrist", "nose", "left_wrist"],
    ) == [15, 0]


def test_read_track_rejects_nonzero_mediapipe_track_index(tmp_path: Path) -> None:
    json_path = tmp_path / "pose_landmarks.json"
    write_mediapipe_pose_landmarks_json(json_path)

    with pytest.raises(ValueError, match="track_index must be 0"):
        read_track(json_path, track_index=1)


def test_generic_pose_reader_dispatches_to_mediapipe(tmp_path: Path) -> None:
    json_path = tmp_path / "pose_landmarks.json"
    write_mediapipe_pose_landmarks_json(json_path)

    track = read_pose_track(
        json_path,
        software="MEDIAPIPE",
        file_type="json",
        track_index=0,
    )

    assert track.coords.shape == (2, len(MEDIAPIPE_POSE_LANDMARK_NAMES), 2)
    assert track.metadata["source"] == {
        "type": "mediapipe_pose_landmarks_json",
        "path": str(json_path),
    }
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


# --- Real-format contract tests ---------------------------------------------
# MediaPipe's NormalizedLandmark.visibility and .presence are Optional[float]
# and may be omitted from a faithful dump, so the reader must not require them.


def test_read_track_accepts_landmark_without_visibility(tmp_path: Path) -> None:
    landmarks = pose_landmarks(visibility=0.9, presence=0.8)
    del landmarks[0]["visibility"]  # real field is optional / may be unset
    json_path = tmp_path / "no_visibility.json"
    write_mediapipe_pose_landmarks_json(
        json_path, frames=[{"frame_index": 0, "pose_landmarks": landmarks}]
    )

    track = read_track(json_path, track_index=0)

    # With visibility absent, the score falls back to presence alone.
    np.testing.assert_allclose(track.scores[0, 0], 0.8)


def test_read_track_landmark_without_any_confidence_scores_nan(tmp_path: Path) -> None:
    landmarks = pose_landmarks(visibility=0.9, presence=None)
    del landmarks[0]["visibility"]  # neither visibility nor presence present
    json_path = tmp_path / "no_confidence.json"
    write_mediapipe_pose_landmarks_json(
        json_path, frames=[{"frame_index": 0, "pose_landmarks": landmarks}]
    )

    track = read_track(json_path, track_index=0)

    assert np.isnan(track.scores[0, 0])
