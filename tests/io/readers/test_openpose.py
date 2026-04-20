from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from xpkg.io.readers import read_pose_node_names, read_pose_track, resolve_pose_node_indices
from xpkg.io.readers.openpose import (
    read_node_names,
    read_sequence,
    read_track,
    resolve_node_indices,
)


def _make_body25_person(
    *,
    base: float,
    missing_nodes: set[int] | None = None,
    low_confidence_nodes: dict[int, float] | None = None,
) -> list[float]:
    missing = missing_nodes or set()
    low_conf = low_confidence_nodes or {}
    values: list[float] = []
    for node_idx in range(25):
        if node_idx in missing:
            values.extend([0.0, 0.0, 0.0])
            continue

        score = low_conf.get(node_idx, 0.95 - 0.01 * node_idx)
        x_val = base + float(node_idx)
        y_val = -(base + float(node_idx))
        values.extend([x_val, y_val, score])
    return values


def _write_openpose_json_sequence(path: Path) -> Path:
    path.mkdir(parents=True)
    payloads = [
        {
            "version": 1.1,
            "people": [
                {"pose_keypoints_2d": _make_body25_person(base=10.0)},
                {"pose_keypoints_2d": _make_body25_person(base=110.0, missing_nodes={4})},
            ],
        },
        {
            "version": 1.1,
            "people": [
                {
                    "pose_keypoints_2d": _make_body25_person(
                        base=20.0,
                        missing_nodes={2},
                        low_confidence_nodes={0: 0.2},
                    )
                }
            ],
        },
        {
            "version": 1.1,
            "people": [],
        },
    ]
    for frame_idx, payload in enumerate(payloads):
        frame_path = path / f"frame_{frame_idx:012d}_keypoints.json"
        frame_path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_read_sequence_decodes_people_and_empty_frames(tmp_path: Path) -> None:
    json_dir = _write_openpose_json_sequence(tmp_path / "openpose_json")

    sequence = read_sequence(json_dir)

    assert len(sequence.frames) == 3
    assert sequence.node_names[0] == "Nose"
    assert sequence.node_names[8] == "MidHip"
    assert len(sequence.frames[0].people) == 2
    assert len(sequence.frames[1].people) == 1
    assert len(sequence.frames[2].people) == 0
    assert np.isnan(sequence.frames[0].people[1].coords[4, 0])
    assert np.isnan(sequence.frames[1].people[0].coords[2, 0])


def test_read_track_uses_person_slot_semantics_for_openpose(tmp_path: Path) -> None:
    json_dir = _write_openpose_json_sequence(tmp_path / "openpose_json")

    track_zero = read_track(json_dir, track_index=0)
    track_one = read_track(json_dir, track_index=1)

    assert track_zero.coords.shape == (3, 25, 2)
    assert track_zero.scores.shape == (3, 25)
    assert track_zero.instance_score.shape == (3,)
    assert track_zero.node_names[1] == "Neck"
    assert np.isfinite(track_zero.coords[0, 0, 0])
    assert np.isnan(track_zero.coords[1, 2, 0])

    assert np.isfinite(track_one.coords[0, 0, 0])
    assert np.isnan(track_one.coords[1]).all()
    assert np.isnan(track_one.coords[2]).all()
    assert np.count_nonzero(track_one.scores[1]) == 0
    assert np.count_nonzero(track_one.scores[2]) == 0


def test_openpose_node_names_and_generic_dispatch_work(tmp_path: Path) -> None:
    json_dir = _write_openpose_json_sequence(tmp_path / "openpose_json")

    assert read_node_names(json_dir)[:3] == ["Nose", "Neck", "RShoulder"]
    assert resolve_node_indices(json_dir, ["MidHip", "Nose", "MidHip"]) == [8, 0]

    track = read_pose_track(
        json_dir,
        software="OPENPOSE",
        file_type="json",
        track_index=0,
    )
    assert track.coords.shape == (3, 25, 2)
    assert read_pose_node_names(json_dir, software="OPENPOSE", file_type="json")[19] == "LBigToe"
    assert resolve_pose_node_indices(
        json_dir,
        software="OPENPOSE",
        file_type="json",
        target_names=["RHeel", "LHeel"],
    ) == [24, 21]


def test_openpose_reader_rejects_non_body25_vectors(tmp_path: Path) -> None:
    json_dir = tmp_path / "openpose_json"
    json_dir.mkdir()
    payload = {
        "version": 1.1,
        "people": [{"pose_keypoints_2d": [1.0, 2.0, 0.9] * 18}],
    }
    (json_dir / "frame_000000000000_keypoints.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="BODY_25"):
        read_sequence(json_dir)
