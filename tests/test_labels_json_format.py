from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _write_test_frame(path: Path, value: int) -> None:
    frame = np.full((12, 16, 3), value, dtype=np.uint8)
    ok = cv2.imwrite(path.as_posix(), frame)
    assert ok


def test_labels_json_roundtrip_with_image_sequence(tmp_path: Path) -> None:
    from posetta.core.annotations import Instance, LabeledFrame, Point
    from posetta.formats import read_labels_json_payload
    from posetta.model import Labels, Video, build_keypoint_skeleton

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frame_paths = []
    for idx, value in enumerate((10, 20, 30), start=0):
        frame_path = frames_dir / f"frame_{idx:04d}.png"
        _write_test_frame(frame_path, value)
        frame_paths.append(frame_path.as_posix())

    skeleton = build_keypoint_skeleton(["nose", "tail"], name="mouse")
    video = Video.from_image_filenames(frame_paths)
    frame0 = LabeledFrame(
        video=video,
        frame_idx=0,
        instances=[
            Instance(
                skeleton=skeleton,
                init_points={
                    "nose": Point(1.0, 2.0, visible=True, complete=True),
                    "tail": Point(3.0, 4.0, visible=True, complete=True),
                },
            )
        ],
    )
    frame2 = LabeledFrame(
        video=video,
        frame_idx=2,
        instances=[
            Instance(
                skeleton=skeleton,
                init_points={
                    "nose": Point(5.0, 6.0, visible=True, complete=True),
                },
            )
        ],
    )

    labels = Labels(labeled_frames=[frame0, frame2], videos=[video], skeletons=[skeleton])
    labels.preferences = {"theme": "light"}
    labels.session = {"note": "roundtrip"}
    labels.provenance = {"source": "test"}

    json_path = tmp_path / "labels.json"
    labels.save_file(labels, json_path.as_posix())

    payload = read_labels_json_payload(json_path)
    assert payload["videos"]["image_filenames"] == [frame_paths]
    assert payload["frames"]["frame_index"] == [0, 2]

    loaded = Labels.load_file(json_path.as_posix())
    assert len(loaded.videos) == 1
    assert loaded.videos[0].backend == "images"
    assert loaded.videos[0].image_filenames == frame_paths
    assert loaded.preferences == {"theme": "light"}
    assert loaded.session == {"note": "roundtrip"}
    assert loaded.provenance == {"source": "test"}
    assert [lf.frame_idx for lf in loaded.labeled_frames] == [0, 2]
    assert np.allclose(
        loaded.labeled_frames[0].instances[0].numpy(),
        np.array([[1.0, 2.0], [3.0, 4.0]]),
    )
