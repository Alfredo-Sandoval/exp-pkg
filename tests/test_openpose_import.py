from __future__ import annotations

from pathlib import Path

import numpy as np

from tests.io.readers.test_openpose import _write_openpose_json_sequence
from tests.test_sleap_h5_import import _write_dummy_video


def test_convert_openpose_json_builds_multi_person_archive(tmp_path: Path) -> None:
    from xpkg.io.archive_format import read_archive
    from xpkg.io.converters.openpose_import import convert_openpose_json
    from xpkg.model import Labels

    json_dir = _write_openpose_json_sequence(tmp_path / "openpose_json")
    video_path = tmp_path / "session.avi"
    _write_dummy_video(video_path, frame_count=3)
    out_path = tmp_path / "openpose.xpkg"

    result = convert_openpose_json(
        json_dir,
        video_path,
        out_path,
        skeleton_name="body25",
    )

    assert result.archive_path == out_path
    payload = read_archive(out_path, lazy=False)
    assert payload["metadata"]["source"] == "openpose_json_import"
    assert payload["metadata"]["source_json_dir"] == json_dir.as_posix()
    assert payload["metadata"]["pose_model"] == "BODY_25"

    labels = Labels.load_file(out_path.as_posix())
    assert len(labels.skeletons[0].keypoint_names) == 25
    assert len(labels.skeletons[0].links_ids) == 24
    assert ("Neck", "MidHip") in labels.skeletons[0].links_by_names()
    assert ("LAnkle", "LBigToe") in labels.skeletons[0].links_by_names()
    assert len(labels.videos) == 1
    assert len(labels.labeled_frames) == 2

    first_frame = next(frame for frame in labels.labeled_frames if frame.frame_idx == 0)
    second_frame = next(frame for frame in labels.labeled_frames if frame.frame_idx == 1)
    assert len(first_frame.instances) == 2
    assert len(second_frame.instances) == 1


def test_convert_openpose_json_applies_likelihood_threshold(tmp_path: Path) -> None:
    from xpkg.io.converters.openpose_import import convert_openpose_json
    from xpkg.model import Labels

    json_dir = _write_openpose_json_sequence(tmp_path / "openpose_json")
    video_path = tmp_path / "session.avi"
    _write_dummy_video(video_path, frame_count=3)
    out_path = tmp_path / "openpose_thresholded.xpkg"

    convert_openpose_json(
        json_dir,
        video_path,
        out_path,
        likelihood_threshold=0.5,
    )

    labels = Labels.load_file(out_path.as_posix())
    second_frame = next(frame for frame in labels.labeled_frames if frame.frame_idx == 1)
    points = second_frame.instances[0].get_points_array(copy=False, full=True)
    assert int(np.count_nonzero(~np.isnan(points["x"]))) == 23


def test_import_openpose_json_workspace_imports_sequence_into_workspace(tmp_path: Path) -> None:
    from xpkg.io.project_workspace import (
        current_project_snapshot_path,
        import_openpose_json_workspace,
    )
    from xpkg.io.workspace_snapshot_backend import read_workspace_snapshot_payload
    from xpkg.model import Labels

    json_dir = _write_openpose_json_sequence(tmp_path / "openpose_json")
    video_path = tmp_path / "session.avi"
    _write_dummy_video(video_path, frame_count=3)
    workspace = tmp_path / "Imported OpenPose Project"

    snapshot_path = import_openpose_json_workspace(
        json_dir,
        video_path,
        workspace,
        skeleton_name="body25",
    )

    assert snapshot_path == current_project_snapshot_path(workspace)
    payload = read_workspace_snapshot_payload(snapshot_path)
    assert payload["metadata"]["source"] == "openpose_json_import"
    assert payload["metadata"]["pose_model"] == "BODY_25"

    labels = Labels.load_file(workspace.as_posix())
    assert len(labels.skeletons[0].keypoint_names) == 25
    assert len(labels.skeletons[0].links_ids) == 24
    assert len(labels.videos) == 1
    assert len(labels.labeled_frames) == 2
