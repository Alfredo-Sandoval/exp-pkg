from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.factories import (
    pose_landmarks,
    write_dummy_video,
    write_mediapipe_pose_landmarks_json,
)


def test_convert_mediapipe_pose_landmarks_json_builds_labels(tmp_path: Path) -> None:
    from xpkg.io.converters.mediapipe_import import convert_mediapipe_pose_landmarks_json

    json_path = tmp_path / "pose_landmarks.json"
    write_mediapipe_pose_landmarks_json(
        json_path,
        image_width=16,
        image_height=12,
        frames=[
            {"frame_index": 0, "pose_landmarks": pose_landmarks()},
            {"frame_index": 1, "pose_landmarks": []},
            {"frame_index": 2, "pose_landmarks": pose_landmarks(x_shift=0.02, y_shift=0.01)},
        ],
    )
    video_path = tmp_path / "session.avi"
    write_dummy_video(video_path, frame_count=3)

    result = convert_mediapipe_pose_landmarks_json(
        json_path.as_posix(),
        video_path.as_posix(),
        skeleton_name="mediapipe_pose",
    )

    assert result.metadata["source"] == "mediapipe_pose_landmarks_json_import"
    assert result.metadata["source_json"] == json_path.name
    assert result.metadata["source_video"] == video_path.name

    labels = result.labels
    assert len(labels.skeletons[0].keypoint_names) == 33
    assert ("left_shoulder", "left_elbow") in labels.skeletons[0].links_by_names()
    assert len(labels.videos) == 1
    assert len(labels.labeled_frames) == 2

    first_frame = next(frame for frame in labels.labeled_frames if frame.frame_idx == 0)
    points = first_frame.instances[0].point_records(copy=False)
    assert float(points["x"][0]) == pytest.approx(0.8)
    assert float(points["y"][0]) == pytest.approx(1.2)


def test_convert_mediapipe_pose_landmarks_json_applies_threshold(tmp_path: Path) -> None:
    from xpkg.io.converters.mediapipe_import import convert_mediapipe_pose_landmarks_json

    json_path = tmp_path / "pose_landmarks.json"
    low_confidence_landmarks = pose_landmarks()
    low_confidence_landmarks[0]["visibility"] = 0.2
    write_mediapipe_pose_landmarks_json(
        json_path,
        image_width=16,
        image_height=12,
        frames=[{"frame_index": 0, "pose_landmarks": low_confidence_landmarks}],
    )
    video_path = tmp_path / "session.avi"
    write_dummy_video(video_path, frame_count=1)

    result = convert_mediapipe_pose_landmarks_json(
        json_path,
        video_path,
        likelihood_threshold=0.5,
    )

    labels = result.labels
    points = labels.labeled_frames[0].instances[0].point_records(copy=False)
    assert int(np.count_nonzero(~np.isnan(points["x"]))) == 32


def test_convert_mediapipe_pose_landmarks_json_rejects_video_size_mismatch(
    tmp_path: Path,
) -> None:
    from xpkg.io.converters.mediapipe_import import convert_mediapipe_pose_landmarks_json

    json_path = tmp_path / "pose_landmarks.json"
    write_mediapipe_pose_landmarks_json(
        json_path,
        image_width=20,
        image_height=12,
    )
    video_path = tmp_path / "session.avi"
    write_dummy_video(video_path, frame_count=2)

    with pytest.raises(ValueError, match="does not match video size"):
        convert_mediapipe_pose_landmarks_json(
            json_path,
            video_path,
        )


def test_import_mediapipe_pose_landmarks_json_project_imports_sequence_into_project(
    tmp_path: Path,
) -> None:
    from xpkg.model import Labels
    from xpkg.project import current_project_state_path, load_project_session
    from xpkg.project.store.imports import import_mediapipe_pose_landmarks_json_project

    json_path = tmp_path / "pose_landmarks.json"
    write_mediapipe_pose_landmarks_json(
        json_path,
        image_width=16,
        image_height=12,
    )
    video_path = tmp_path / "session.avi"
    write_dummy_video(video_path, frame_count=2)
    project = tmp_path / "Imported MediaPipe Project"

    state_path = import_mediapipe_pose_landmarks_json_project(
        json_path,
        video_path,
        project,
    )

    assert state_path == current_project_state_path(project)
    pose_metadata = load_project_session(project).poses[0].metadata
    assert pose_metadata["source"] == "mediapipe_pose_landmarks_json_import"
    assert pose_metadata["source_json"] == json_path.name

    labels = Labels.load_file(project.as_posix())
    assert len(labels.skeletons[0].keypoint_names) == 33
    assert len(labels.videos) == 1
    assert len(labels.labeled_frames) == 2
