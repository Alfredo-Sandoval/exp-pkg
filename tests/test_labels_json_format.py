from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _write_test_frame(path: Path, value: int) -> None:
    frame = np.full((12, 16, 3), value, dtype=np.uint8)
    ok = cv2.imwrite(path.as_posix(), frame)
    assert ok


def test_labels_json_roundtrip_with_image_sequence(tmp_path: Path) -> None:
    from xpkg.core.annotations import Instance, LabeledFrame, Point
    from xpkg.exchange import labels_from_json_payload, labels_to_json_payload
    from xpkg.formats import read_labels_json_payload
    from xpkg.model import Labels, Video, build_keypoint_skeleton

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frame_paths = []
    for idx, value in enumerate((10, 20, 30), start=0):
        frame_path = frames_dir / f"frame_{idx:04d}.png"
        _write_test_frame(frame_path, value)
        frame_paths.append(frame_path.as_posix())

    skeleton = build_keypoint_skeleton(["nose", "tail"], name="subject")
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

    loaded_from_exchange = labels_from_json_payload(labels_to_json_payload(labels))
    assert [lf.frame_idx for lf in loaded_from_exchange.labeled_frames] == [0, 2]
    assert loaded_from_exchange.videos[0].image_filenames == frame_paths

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


def test_labels_json_roundtrip_preserves_tracks_and_segmentation(tmp_path: Path) -> None:
    from xpkg.core.annotations import (
        ROI,
        Instance,
        LabeledFrame,
        Point,
        SegmentationMask,
        Track,
    )
    from xpkg.model import Labels, Video, build_keypoint_skeleton

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frame_path = frames_dir / "frame_0000.png"
    _write_test_frame(frame_path, 42)

    skeleton = build_keypoint_skeleton(["nose"], name="subject")
    video = Video.from_image_filenames([frame_path.as_posix()])
    track = Track(spawned_on=7, name="seg-track")
    frame = LabeledFrame(
        video=video,
        frame_idx=0,
        instances=[
            Instance(
                skeleton=skeleton,
                init_points={"nose": Point(1.0, 2.0, visible=True, complete=True)},
            )
        ],
        masks=[
            SegmentationMask.from_polygon(
                np.array([[0.0, 0.0], [5.0, 0.0], [5.0, 5.0]], dtype=np.float32),
                track=track,
                class_name="subject",
            )
        ],
        rois=[ROI(x1=1.0, y1=2.0, x2=8.0, y2=9.0, track=track, class_name="subject")],
    )

    labels = Labels(labeled_frames=[frame], videos=[video], skeletons=[skeleton], tracks=[track])
    json_path = tmp_path / "labels.json"
    labels.save_file(labels, json_path.as_posix())

    loaded = Labels.load_file(json_path.as_posix())
    assert [track.name for track in loaded.tracks] == ["seg-track"]
    assert len(loaded.labeled_frames[0].masks) == 1
    assert loaded.labeled_frames[0].masks[0].track is not None
    assert loaded.labeled_frames[0].masks[0].track.name == "seg-track"
    assert len(loaded.labeled_frames[0].rois) == 1
    assert loaded.labeled_frames[0].rois[0].track is not None
    assert loaded.labeled_frames[0].rois[0].track.name == "seg-track"
