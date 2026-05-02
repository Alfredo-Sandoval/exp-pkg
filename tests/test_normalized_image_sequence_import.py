from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def _write_frame(path: Path, value: int) -> None:
    frame = np.full((12, 16, 3), value, dtype=np.uint8)
    ok = cv2.imwrite(path.as_posix(), frame)
    assert ok


def test_convert_normalized_image_sequence_annotations_builds_workspace_ready_labels(
    tmp_path: Path,
) -> None:
    from xpkg.io.converters.normalized_image_sequence_import import (
        convert_normalized_image_sequence_annotations,
    )

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    input_frame = source_dir / "frame_0000.png"
    _write_frame(input_frame, 25)

    annotations_path = source_dir / "normalized_annotations.json"
    annotations_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_key": "demo_dataset",
                "slice_key": "core_v1",
                "project_name": "demo_dataset_core_v1",
                "keypoint_names": ["nose", "tail"],
                "links": [[0, 1]],
                "frames": [
                    {
                        "sequence_id": "demo_sequence",
                        "image_path": input_frame.name,
                        "frame_id": "frame_0000",
                        "source_index": 0,
                        "instances": [
                            {
                                "keypoints": [[1.0, 2.0, 1], [3.0, 4.0, 1]],
                                "metadata": {"animal_id": "subject_a"},
                            }
                        ],
                    }
                ],
            }
        )
    )

    project_root = tmp_path / "workspace"
    result = convert_normalized_image_sequence_annotations(
        annotations_path,
        project_root,
    )

    copied_frame = project_root / "videos" / "demo_sequence" / "000000_frame_0000.png"
    assert result.project_root == project_root
    assert result.metadata == {
        "project_name": "demo_dataset_core_v1",
        "source": "normalized_image_sequence_import",
        "source_annotations": annotations_path.as_posix(),
        "dataset_key": "demo_dataset",
        "slice_key": "core_v1",
    }
    assert result.videos == [project_root / "videos" / "demo_sequence"]
    assert copied_frame.is_file()

    labels = result.labels
    assert len(labels.videos) == 1
    assert labels.videos[0].image_filenames == [copied_frame.as_posix()]
    assert len(labels.labeled_frames) == 1
    assert labels.skeletons[0].keypoint_names == ["nose", "tail"]
    assert labels.skeletons[0].links_ids == [(0, 1)]
