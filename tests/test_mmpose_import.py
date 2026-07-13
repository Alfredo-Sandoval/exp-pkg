from __future__ import annotations

from pathlib import Path

import numpy as np

from tests.factories import write_dummy_video, write_mmpose_topdown_json


def test_convert_mmpose_topdown_json_builds_labels_with_links(tmp_path: Path) -> None:
    from xpkg.io.converters.mmpose_import import convert_mmpose_topdown_json

    json_path = write_mmpose_topdown_json(tmp_path / "results_session.json")
    video_path = tmp_path / "session.avi"
    write_dummy_video(video_path, frame_count=3)

    result = convert_mmpose_topdown_json(
        json_path,
        video_path,
        skeleton_name="toy_subject",
    )

    assert result.metadata["source"] == "mmpose_topdown_json_import"
    assert result.metadata["source_json"] == json_path.name
    assert result.metadata["source_video"] == video_path.name

    labels = result.labels
    assert len(labels.skeletons[0].keypoint_names) == 3
    assert labels.skeletons[0].links_ids == [(0, 1), (1, 2)]
    assert len(labels.videos) == 1
    assert len(labels.labeled_frames) == 3

    third_frame = next(frame for frame in labels.labeled_frames if frame.frame_idx == 2)
    points = third_frame.instances[0].point_records(copy=False)
    np.testing.assert_allclose(points["x"][:2], np.array([12.0, 32.0]))
    assert float(points["x"][2]) == 52.0


def test_convert_mmpose_topdown_json_supports_instance_slots(tmp_path: Path) -> None:
    from xpkg.io.converters.mmpose_import import convert_mmpose_topdown_json

    json_path = write_mmpose_topdown_json(tmp_path / "results_session.json")
    video_path = tmp_path / "session.avi"
    write_dummy_video(video_path, frame_count=3)

    result = convert_mmpose_topdown_json(
        json_path,
        video_path,
        instance_index=1,
        likelihood_threshold=0.5,
    )

    labels = result.labels
    frame_indices = [frame.frame_idx for frame in labels.labeled_frames]
    assert frame_indices == [0, 2]

    third_frame = next(frame for frame in labels.labeled_frames if frame.frame_idx == 2)
    points = third_frame.instances[0].point_records(copy=False)
    assert int(np.count_nonzero(~np.isnan(points["x"]))) == 2


def test_import_mmpose_topdown_json_project_imports_sequence_into_project(
    tmp_path: Path,
) -> None:
    from xpkg.project import (
        current_project_state_path,
        load_project_labels,
        load_project_session,
    )
    from xpkg.project.store.imports import import_mmpose_topdown_json_project

    json_path = write_mmpose_topdown_json(tmp_path / "results_session.json")
    video_path = tmp_path / "session.avi"
    write_dummy_video(video_path, frame_count=3)
    project = tmp_path / "Imported MMPose Project"

    state_path = import_mmpose_topdown_json_project(
        json_path,
        video_path,
        project,
        skeleton_name="toy_subject",
    )

    assert state_path == current_project_state_path(project)
    pose_metadata = load_project_session(project).poses[0].metadata
    assert pose_metadata["source"] == "mmpose_topdown_json_import"
    assert pose_metadata["source_json"] == json_path.name

    labels = load_project_labels(project)
    assert len(labels.skeletons[0].keypoint_names) == 3
    assert labels.skeletons[0].links_ids == [(0, 1), (1, 2)]
    assert len(labels.videos) == 1
    assert len(labels.labeled_frames) == 3
