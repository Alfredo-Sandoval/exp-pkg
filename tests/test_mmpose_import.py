from __future__ import annotations

from pathlib import Path

import numpy as np

from tests.io.readers.test_mmpose import _write_mmpose_topdown_json
from tests.test_sleap_h5_import import _write_dummy_video


def test_convert_mmpose_topdown_json_builds_archive_with_links(tmp_path: Path) -> None:
    from xpkg.io.archive_format import read_archive
    from xpkg.io.converters.mmpose_import import convert_mmpose_topdown_json
    from xpkg.model import Labels

    json_path = _write_mmpose_topdown_json(tmp_path / "results_session.json")
    video_path = tmp_path / "session.avi"
    _write_dummy_video(video_path, frame_count=3)
    out_path = tmp_path / "mmpose.xpkg"

    result = convert_mmpose_topdown_json(
        json_path,
        video_path,
        out_path,
        skeleton_name="toy_mouse",
    )

    assert result.archive_path == out_path
    payload = read_archive(out_path, lazy=False)
    assert payload["metadata"]["source"] == "mmpose_topdown_json_import"
    assert payload["metadata"]["source_json"] == json_path.as_posix()
    assert payload["metadata"]["source_video"] == video_path.as_posix()

    labels = Labels.load_file(out_path.as_posix())
    assert len(labels.skeletons[0].keypoint_names) == 3
    assert labels.skeletons[0].links_ids == [(0, 1), (1, 2)]
    assert len(labels.videos) == 1
    assert len(labels.labeled_frames) == 3

    third_frame = next(frame for frame in labels.labeled_frames if frame.frame_idx == 2)
    points = third_frame.instances[0].get_points_array(copy=False, full=True)
    np.testing.assert_allclose(points["x"][:2], np.array([12.0, 32.0]))
    assert float(points["x"][2]) == 52.0


def test_convert_mmpose_topdown_json_supports_instance_slots(tmp_path: Path) -> None:
    from xpkg.io.converters.mmpose_import import convert_mmpose_topdown_json
    from xpkg.model import Labels

    json_path = _write_mmpose_topdown_json(tmp_path / "results_session.json")
    video_path = tmp_path / "session.avi"
    _write_dummy_video(video_path, frame_count=3)
    out_path = tmp_path / "mmpose_instance_1.xpkg"

    convert_mmpose_topdown_json(
        json_path,
        video_path,
        out_path,
        instance_index=1,
        likelihood_threshold=0.5,
    )

    labels = Labels.load_file(out_path.as_posix())
    frame_indices = [frame.frame_idx for frame in labels.labeled_frames]
    assert frame_indices == [0, 2]

    third_frame = next(frame for frame in labels.labeled_frames if frame.frame_idx == 2)
    points = third_frame.instances[0].get_points_array(copy=False, full=True)
    assert int(np.count_nonzero(~np.isnan(points["x"]))) == 2


def test_import_mmpose_topdown_json_workspace_imports_sequence_into_workspace(
    tmp_path: Path,
) -> None:
    from xpkg.io.project_workspace import (
        current_project_snapshot_path,
        import_mmpose_topdown_json_workspace,
    )
    from xpkg.io.workspace_snapshot_backend import read_workspace_snapshot_payload
    from xpkg.model import Labels

    json_path = _write_mmpose_topdown_json(tmp_path / "results_session.json")
    video_path = tmp_path / "session.avi"
    _write_dummy_video(video_path, frame_count=3)
    workspace = tmp_path / "Imported MMPose Project"

    snapshot_path = import_mmpose_topdown_json_workspace(
        json_path,
        video_path,
        workspace,
        skeleton_name="toy_mouse",
    )

    assert snapshot_path == current_project_snapshot_path(workspace)
    payload = read_workspace_snapshot_payload(snapshot_path)
    assert payload["metadata"]["source"] == "mmpose_topdown_json_import"
    assert payload["metadata"]["source_json"] == json_path.as_posix()

    labels = Labels.load_file(workspace.as_posix())
    assert len(labels.skeletons[0].keypoint_names) == 3
    assert labels.skeletons[0].links_ids == [(0, 1), (1, 2)]
    assert len(labels.videos) == 1
    assert len(labels.labeled_frames) == 3
