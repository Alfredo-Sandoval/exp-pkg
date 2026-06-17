from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.factories import write_mmpose_topdown_json
from xpkg._core.json_utils import write_json
from xpkg.io.readers import read_pose_node_names, read_pose_track, resolve_pose_node_indices
from xpkg.io.readers.pose.mmpose import read_node_names, read_track, resolve_node_indices


def test_read_track_uses_instance_slot_semantics_for_mmpose(tmp_path: Path) -> None:
    json_path = write_mmpose_topdown_json(tmp_path / "results_session.json")

    track_zero = read_track(json_path, track_index=0)
    track_one = read_track(json_path, track_index=1)

    assert track_zero.node_names == ("nose", "mid_back", "tail_base")
    assert track_zero.coords.shape == (3, 3, 2)
    assert track_zero.scores.shape == (3, 3)
    assert track_zero.instance_score.shape == (3,)
    np.testing.assert_allclose(track_zero.coords[0, 0], np.array([10.0, 20.0]))
    np.testing.assert_allclose(track_zero.coords[2, 2], np.array([52.0, 62.0]))

    np.testing.assert_allclose(track_one.coords[0, 0], np.array([110.0, 120.0]))
    assert np.isnan(track_one.coords[1]).all()
    assert np.isnan(track_one.scores[1]).all()
    np.testing.assert_allclose(track_one.coords[2, 1], np.array([132.0, 142.0]))
    np.testing.assert_allclose(track_one.instance_score[[0, 2]], np.array([0.55, 0.52]))
    assert np.isnan(track_one.instance_score[1])


def test_mmpose_node_names_and_generic_dispatch_work(tmp_path: Path) -> None:
    json_path = write_mmpose_topdown_json(tmp_path / "results_session.json")

    assert read_node_names(json_path) == ["nose", "mid_back", "tail_base"]
    assert resolve_node_indices(json_path, ["tail_base", "nose", "tail_base"]) == [2, 0]

    track = read_pose_track(
        json_path,
        software="MMPose",
        file_type="json",
        track_index=0,
    )
    assert track.coords.shape == (3, 3, 2)
    assert read_pose_node_names(
        json_path,
        software="MMPose",
        file_type="json",
    ) == ["nose", "mid_back", "tail_base"]
    assert resolve_pose_node_indices(
        json_path,
        software="MMPose",
        file_type="json",
        target_names=["mid_back", "nose"],
    ) == [1, 0]


def test_mmpose_reader_rejects_image_style_json(tmp_path: Path) -> None:
    json_path = tmp_path / "results_image.json"
    write_json(
        json_path,
        {
            "meta_info": {
                "dataset_name": "toy_subject",
                "keypoint_id2name": {0: "nose"},
                "skeleton_links": [],
            },
            "instance_info": [
                {
                    "keypoints": [[1.0, 2.0]],
                    "keypoint_scores": [0.9],
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="Only video-style MMPose demo JSON is supported"):
        read_track(json_path, track_index=0)


def test_read_track_rejects_non_integer_frame_id(tmp_path: Path) -> None:
    # A non-integer frame_id must fail loud, not truncate (int(1.5) -> 1) and
    # silently remap to the wrong frame.
    json_path = tmp_path / "results_bad_frame.json"
    write_json(
        json_path,
        {
            "meta_info": {
                "dataset_name": "toy_subject",
                "keypoint_id2name": {"0": "nose"},
                "skeleton_links": [],
            },
            "instance_info": [
                {
                    "frame_id": 1.5,
                    "instances": [{"keypoints": [[1.0, 2.0]], "keypoint_scores": [0.9]}],
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="frame_id must be an integer"):
        read_track(json_path, track_index=0)


def test_read_track_reads_real_demo_json_shape(tmp_path: Path) -> None:
    # Byte-faithful to topdown_demo_with_mmdet.py --save-predictions: string
    # keypoint_id2name keys (JSON coerces ints to strings), 1-based frame_id,
    # double-nested bbox, and numpy-bearing meta_info dumped by json_tricks.
    json_path = tmp_path / "results_real.json"
    write_json(
        json_path,
        {
            "meta_info": {
                "dataset_name": "coco",
                "num_keypoints": 2,
                "keypoint_id2name": {"0": "nose", "1": "left_eye"},
                "keypoint_name2id": {"nose": 0, "left_eye": 1},
                "skeleton_links": [[0, 1]],
                "num_skeleton_links": 1,
                # json_tricks serializes meta numpy arrays as __ndarray__ objects.
                "sigmas": {"__ndarray__": [0.026, 0.025], "dtype": "float32", "shape": [2]},
            },
            "instance_info": [
                {
                    "frame_id": 1,
                    "instances": [
                        {
                            "keypoints": [[10.0, 20.0], [30.0, 40.0]],
                            "keypoint_scores": [0.95, 0.85],
                            "bbox": [[5.0, 5.0, 64.0, 48.0]],
                            "bbox_score": 0.9,
                        }
                    ],
                }
            ],
        },
    )

    track = read_track(json_path, track_index=0)

    assert track.node_names == ("nose", "left_eye")
    np.testing.assert_allclose(track.coords[0], [[10.0, 20.0], [30.0, 40.0]])
    np.testing.assert_allclose(track.scores[0], [0.95, 0.85])
