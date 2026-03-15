from __future__ import annotations

import json
from pathlib import Path

import cv2
import h5py
import numpy as np


def _write_frame(path: Path, value: int) -> None:
    frame = np.full((12, 16, 3), value, dtype=np.uint8)
    ok = cv2.imwrite(path.as_posix(), frame)
    assert ok


def test_convert_normalized_image_sequence_annotations_writes_bundle_and_manifest(
    tmp_path: Path,
) -> None:
    from posetta.io.converters.normalized_image_sequence_import import (
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
                                "metadata": {"animal_id": "mouse_a"},
                            }
                        ],
                    }
                ],
            }
        )
    )

    project_root = tmp_path / "archive"
    result = convert_normalized_image_sequence_annotations(
        annotations_path,
        project_root,
    )

    bundle_path = project_root / "archive.sta"
    copied_frame = project_root / "videos" / "demo_sequence" / "000000_frame_0000.png"
    assert result.project_root == project_root
    assert result.siesta_path == bundle_path
    assert copied_frame.is_file()

    with h5py.File(str(bundle_path), "r") as handle:
        raw = handle["project_metadata"].attrs["manifest_json"]
        if isinstance(raw, bytes | bytearray):
            raw = raw.decode("utf-8")
        manifest = json.loads(str(raw))

    entries = manifest["entries"]
    expected_sequence_dir = {
        str((project_root / "videos" / "demo_sequence").resolve()),
        "videos/demo_sequence",
    }
    assert any(
        entry.get("asset_type") == "video"
        and entry.get("path") in expected_sequence_dir
        and entry.get("metadata") == {
            "backend": "images",
            "frame_count": 1,
            "index": 0,
            "role": "image_sequence",
        }
        for entry in entries
    )
