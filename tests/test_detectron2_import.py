from __future__ import annotations

from pathlib import Path

import numpy as np

from tests.io.readers.test_detectron2_coco import _write_detectron2_coco_fixture


def test_convert_detectron2_coco_builds_image_sequence_archive(tmp_path: Path) -> None:
    from xpkg.io.archive_format import read_archive
    from xpkg.io.converters.detectron2_import import convert_detectron2_coco
    from xpkg.model import Labels

    predictions_path, dataset_json_path, image_root = _write_detectron2_coco_fixture(tmp_path)
    out_path = tmp_path / "detectron2.xpkg"

    result = convert_detectron2_coco(
        predictions_path,
        dataset_json_path,
        image_root,
        out_path,
        skeleton_name="mouse_keypoints",
        likelihood_threshold=0.5,
    )

    assert result.archive_path == out_path
    payload = read_archive(out_path, lazy=False)
    assert payload["metadata"]["source"] == "detectron2_coco_import"
    assert payload["metadata"]["source_category_id"] == 1
    assert payload["metadata"]["source_category_name"] == "mouse"

    labels = Labels.load_file(out_path.as_posix())
    assert len(labels.videos) == 1
    assert len(labels.videos[0].image_filenames) == 3
    assert len(labels.skeletons) == 1
    assert labels.skeletons[0].keypoint_names == ["nose", "tail"]
    assert labels.skeletons[0].links_ids == [(0, 1)]
    assert len(labels.labeled_frames) == 2

    frame_zero = next(frame for frame in labels.labeled_frames if frame.frame_idx == 0)
    assert len(frame_zero.instances) == 1
    frame_one = next(frame for frame in labels.labeled_frames if frame.frame_idx == 1)
    assert len(frame_one.instances) == 2

    primary_points = frame_one.instances[0].get_points_array(copy=False, full=True)
    assert np.isfinite(primary_points["x"]).sum() == 1
    assert float(primary_points["x"][0]) == 10.0

    secondary_points = frame_one.instances[1].get_points_array(copy=False, full=True)
    assert np.isfinite(secondary_points["x"]).sum() == 1
    assert float(secondary_points["x"][0]) == 12.0


def test_import_detectron2_coco_workspace_copies_sequence_media(tmp_path: Path) -> None:
    from xpkg.formats import current_project_snapshot_path, workspace_media_root
    from xpkg.io.project_workspace import import_detectron2_coco_workspace
    from xpkg.model import Labels

    predictions_path, dataset_json_path, image_root = _write_detectron2_coco_fixture(tmp_path)
    workspace = tmp_path / "Imported Detectron2"

    snapshot_path = import_detectron2_coco_workspace(
        predictions_path,
        dataset_json_path,
        image_root,
        workspace,
        likelihood_threshold=0.5,
    )

    assert snapshot_path == current_project_snapshot_path(workspace)
    loaded = Labels.load_file(workspace.as_posix())
    assert len(loaded.videos) == 1
    assert len(loaded.labeled_frames) == 2

    media_root = workspace_media_root(workspace).resolve()
    frame_paths = [Path(path).resolve() for path in loaded.videos[0].image_filenames]
    assert len(frame_paths) == 3
    assert all(path.parent.parent == media_root for path in frame_paths)
    assert all(path.is_file() for path in frame_paths)
