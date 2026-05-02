from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from xpkg.core.json_utils import write_json
from xpkg.io.readers import read_pose_node_names, read_pose_track, resolve_pose_node_indices
from xpkg.io.readers.mmpose import read_node_names, read_track, resolve_node_indices


def _mmpose_instance(
    *,
    base: float,
    scores: list[float],
) -> dict[str, object]:
    keypoints = [
        [base + 0.0, base + 10.0],
        [base + 20.0, base + 30.0],
        [base + 40.0, base + 50.0],
    ]
    return {
        "keypoints": keypoints,
        "keypoint_scores": scores,
        "bbox": [base - 5.0, base - 5.0, 64.0, 48.0],
        "bbox_score": float(np.mean(scores)),
    }


def _write_mmpose_topdown_json(path: Path) -> Path:
    write_json(
        path,
        {
            "meta_info": {
                "dataset_name": "toy_subject",
                "num_keypoints": 3,
                "keypoint_id2name": {
                    0: "nose",
                    1: "mid_back",
                    2: "tail_base",
                },
                "keypoint_name2id": {
                    "nose": 0,
                    "mid_back": 1,
                    "tail_base": 2,
                },
                "skeleton_links": [[0, 1], [1, 2]],
                "num_skeleton_links": 2,
            },
            "instance_info": [
                {
                    "frame_id": 1,
                    "instances": [
                        _mmpose_instance(base=10.0, scores=[0.95, 0.85, 0.75]),
                        _mmpose_instance(base=110.0, scores=[0.65, 0.55, 0.45]),
                    ],
                },
                {
                    "frame_id": 2,
                    "instances": [
                        _mmpose_instance(base=11.0, scores=[0.90, 0.80, 0.70]),
                    ],
                },
                {
                    "frame_id": 3,
                    "instances": [
                        _mmpose_instance(base=12.0, scores=[0.88, 0.78, 0.60]),
                        _mmpose_instance(base=112.0, scores=[0.62, 0.52, 0.42]),
                    ],
                },
            ],
        },
    )
    return path


def test_read_track_uses_instance_slot_semantics_for_mmpose(tmp_path: Path) -> None:
    json_path = _write_mmpose_topdown_json(tmp_path / "results_session.json")

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
    json_path = _write_mmpose_topdown_json(tmp_path / "results_session.json")

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
