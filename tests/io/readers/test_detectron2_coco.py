from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from xpkg.io.readers.detectron2_coco import (
    read_node_names,
    read_sequence,
    read_track,
    read_track_count,
    resolve_node_indices,
)


def _write_png(path: Path, *, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((12, 16, 3), value, dtype=np.uint8)
    ok = cv2.imwrite(path.as_posix(), image)
    assert ok


def _write_detectron2_coco_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    image_root = tmp_path / "images"
    _write_png(image_root / "session" / "frame_000002.png", value=10)
    _write_png(image_root / "session" / "frame_000010.png", value=30)
    _write_png(image_root / "session" / "frame_000030.png", value=50)

    dataset_json_path = tmp_path / "dataset.json"
    dataset_json_path.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 30, "file_name": "session/frame_000030.png"},
                    {"id": 10, "file_name": "session/frame_000010.png"},
                    {"id": 2, "file_name": "session/frame_000002.png"},
                ],
                "annotations": [],
                "categories": [
                    {
                        "id": 1,
                        "name": "mouse",
                        "keypoints": ["nose", "tail"],
                        "skeleton": [[1, 2]],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    predictions_path = tmp_path / "coco_instances_results.json"
    predictions_path.write_text(
        json.dumps(
            [
                {
                    "image_id": 10,
                    "category_id": 1,
                    "bbox": [1.0, 2.0, 3.0, 4.0],
                    "score": 0.95,
                    "keypoints": [10.0, 20.0, 0.9, 30.0, 40.0, 0.3],
                },
                {
                    "image_id": 2,
                    "category_id": 1,
                    "bbox": [5.0, 6.0, 7.0, 8.0],
                    "score": 0.85,
                    "keypoints": [11.0, 21.0, 0.8, 31.0, 41.0, 0.7],
                },
                {
                    "image_id": 10,
                    "category_id": 1,
                    "bbox": [9.0, 10.0, 11.0, 12.0],
                    "score": 0.60,
                    "keypoints": [12.0, 22.0, 0.6, 32.0, 42.0, 0.4],
                },
            ]
        ),
        encoding="utf-8",
    )

    return predictions_path, dataset_json_path, image_root


def test_read_sequence_decodes_categories_and_empty_frames(tmp_path: Path) -> None:
    predictions_path, dataset_json_path, image_root = _write_detectron2_coco_fixture(tmp_path)

    sequence = read_sequence(predictions_path, dataset_json_path, image_root)

    assert len(sequence.categories) == 1
    assert sequence.categories[0].name == "mouse"
    assert sequence.categories[0].node_names == ("nose", "tail")
    assert sequence.categories[0].skeleton_links == ((0, 1),)

    assert [frame.image_id for frame in sequence.frames] == [2, 10, 30]
    assert [frame.frame_index for frame in sequence.frames] == [0, 1, 2]
    assert sequence.frames[0].image_path.name == "frame_000002.png"
    assert len(sequence.frames[0].detections) == 1
    assert len(sequence.frames[1].detections) == 2
    assert len(sequence.frames[2].detections) == 0
    np.testing.assert_allclose(
        sequence.frames[0].detections[0].keypoints,
        np.array([[11.0, 21.0, 0.8], [31.0, 41.0, 0.7]], dtype=np.float64),
    )


def test_read_sequence_orders_frames_by_filename_not_image_id(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    _write_png(image_root / "session" / "frame_000001.png", value=10)
    _write_png(image_root / "session" / "frame_000002.png", value=20)
    _write_png(image_root / "session" / "frame_000003.png", value=30)

    dataset_json_path = tmp_path / "dataset.json"
    dataset_json_path.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 30, "file_name": "session/frame_000001.png"},
                    {"id": 10, "file_name": "session/frame_000002.png"},
                    {"id": 20, "file_name": "session/frame_000003.png"},
                ],
                "annotations": [],
                "categories": [
                    {
                        "id": 1,
                        "name": "mouse",
                        "keypoints": ["nose"],
                        "skeleton": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    predictions_path = tmp_path / "coco_instances_results.json"
    predictions_path.write_text(
        json.dumps(
            [
                {"image_id": 30, "category_id": 1, "score": 0.9, "keypoints": [1.0, 1.0, 0.9]},
                {"image_id": 10, "category_id": 1, "score": 0.9, "keypoints": [2.0, 2.0, 0.9]},
                {"image_id": 20, "category_id": 1, "score": 0.9, "keypoints": [3.0, 3.0, 0.9]},
            ]
        ),
        encoding="utf-8",
    )

    sequence = read_sequence(predictions_path, dataset_json_path, image_root)

    assert [frame.image_id for frame in sequence.frames] == [30, 10, 20]
    assert [frame.image_path.name for frame in sequence.frames] == [
        "frame_000001.png",
        "frame_000002.png",
        "frame_000003.png",
    ]
    assert [frame.frame_index for frame in sequence.frames] == [0, 1, 2]


def test_read_track_uses_detection_slot_semantics(tmp_path: Path) -> None:
    predictions_path, dataset_json_path, image_root = _write_detectron2_coco_fixture(tmp_path)

    track_zero = read_track(
        predictions_path,
        dataset_json_path,
        image_root,
        track_index=0,
    )
    track_one = read_track(
        predictions_path,
        dataset_json_path,
        image_root,
        track_index=1,
    )

    assert read_track_count(predictions_path, dataset_json_path, image_root) == 2
    assert track_zero.coords.shape == (3, 2, 2)
    assert track_zero.node_names == ("nose", "tail")
    np.testing.assert_allclose(
        track_zero.coords[0],
        np.array([[11.0, 21.0], [31.0, 41.0]], dtype=np.float64),
    )
    np.testing.assert_allclose(
        track_zero.instance_score,
        np.array([0.85, 0.95, np.nan]),
        equal_nan=True,
    )

    np.testing.assert_allclose(
        track_one.coords[1],
        np.array([[12.0, 22.0], [32.0, 42.0]], dtype=np.float64),
    )
    assert np.isnan(track_one.coords[0]).all()
    assert np.isnan(track_one.coords[2]).all()


def test_detectron2_node_names_and_indices(tmp_path: Path) -> None:
    _predictions_path, dataset_json_path, _image_root = _write_detectron2_coco_fixture(tmp_path)

    assert read_node_names(dataset_json_path) == ["nose", "tail"]
    assert resolve_node_indices(
        dataset_json_path,
        target_names=["tail", "nose", "tail"],
    ) == [1, 0]


def test_detectron2_reader_rejects_missing_keypoints(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    _write_png(image_root / "frame.png", value=10)
    dataset_json_path = tmp_path / "dataset.json"
    dataset_json_path.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "frame.png"}],
                "annotations": [],
                "categories": [{"id": 1, "name": "mouse", "keypoints": ["nose"]}],
            }
        ),
        encoding="utf-8",
    )
    predictions_path = tmp_path / "results.json"
    predictions_path.write_text(
        json.dumps(
            [
                {
                    "image_id": 1,
                    "category_id": 1,
                    "score": 0.9,
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match="keypoints must be an array"):
        read_sequence(predictions_path, dataset_json_path, image_root)
