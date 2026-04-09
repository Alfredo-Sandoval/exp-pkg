from __future__ import annotations

from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest


def _write_test_image(path: Path, value: int = 128) -> None:
    image = np.full((12, 16, 3), value, dtype=np.uint8)
    ok = cv2.imwrite(path.as_posix(), image)
    assert ok


def _make_single_frame_video(tmp_path: Path):
    from xpkg.model import Video

    frame_path = tmp_path / "frame.png"
    _write_test_image(frame_path)
    video = Video.from_image_filenames([frame_path.as_posix()])
    video.filename = frame_path.as_posix()
    return frame_path, video


def _make_labels(tmp_path: Path, *, x: float, y: float, visible: bool = True, frame_idx: int = 0):
    from xpkg.core.annotations import Instance, LabeledFrame, Point
    from xpkg.model import Labels, build_keypoint_skeleton

    _, video = _make_single_frame_video(tmp_path)
    skeleton = build_keypoint_skeleton(["nose"], name="mouse")
    frame = LabeledFrame(
        video=video,
        frame_idx=frame_idx,
        instances=[
            Instance(
                skeleton=skeleton,
                init_points={
                    "nose": Point(x, y, visible=visible, complete=True),
                },
            )
        ],
    )
    labels = Labels(labeled_frames=[frame], videos=[video], skeletons=[skeleton])
    labels.validate()
    return labels


def test_labels_save_file_defaults_to_xpkg_suffix(tmp_path: Path) -> None:
    from xpkg.model import Labels

    labels = _make_labels(tmp_path, x=1.0, y=2.0)
    raw_path = tmp_path / "archive"
    written_path = Labels.save_file(labels, raw_path.as_posix())

    assert written_path.endswith(".xpkg")
    assert Path(written_path).exists()


def test_labels_load_file_accepts_custom_archive_reader(tmp_path: Path) -> None:
    from xpkg.io.archive_format import read_archive, write_archive
    from xpkg.io.labels import serialization as label_serialization
    from xpkg.model import Labels

    labels = _make_labels(tmp_path, x=7.0, y=8.0)
    archive_path = tmp_path / "labels.xpkg"
    write_archive(archive_path, labels)

    calls: list[tuple[Path, bool]] = []

    def recording_reader(path: Path, *, lazy: bool = False):
        calls.append((path, lazy))
        return read_archive(path, lazy=lazy)

    loaded = label_serialization.labels_load_file(
        Labels,
        archive_path.as_posix(),
        read_archive_fn=recording_reader,
        supported_archive_suffixes=(".xpkg",),
        allow_json=False,
    )

    assert calls == [(archive_path, False)]
    assert loaded.path == archive_path


def test_labels_save_file_accepts_custom_archive_writer(tmp_path: Path) -> None:
    from xpkg.io.archive_format import read_archive, write_archive
    from xpkg.io.labels import serialization as label_serialization

    labels = _make_labels(tmp_path, x=2.0, y=5.0)
    raw_path = tmp_path / "custom_archive"
    calls: list[tuple[Path, dict[str, object] | None]] = []

    def recording_writer(path: Path, labels_obj, *, metadata=None, **kwargs):
        calls.append((path, metadata))
        write_archive(path, labels_obj, metadata=metadata, **kwargs)

    written_path = label_serialization.labels_save_file(
        labels,
        raw_path.as_posix(),
        metadata={"project_name": "demo"},
        write_archive_fn=recording_writer,
        supported_archive_suffixes=(".xpkg",),
        allow_json=False,
    )

    assert Path(written_path) == raw_path.with_suffix(".xpkg")
    assert calls == [(raw_path.with_suffix(".xpkg"), {"project_name": "demo"})]
    payload = read_archive(Path(written_path), lazy=False)
    assert payload["metadata"]["project_name"] == "demo"


def test_archive_labels_roundtrip_uses_explicit_visibility_dataset(tmp_path: Path) -> None:
    from xpkg.model import Labels

    labels = _make_labels(tmp_path, x=3.0, y=4.0, visible=False)
    archive_path = tmp_path / "labels.xpkg"
    labels.save_file(labels, archive_path.as_posix())

    with h5py.File(archive_path.as_posix(), "r") as handle:
        keypoints = np.asarray(handle["labels"]["data"]["keypoints"][...], dtype=np.float32)
        visibility = np.asarray(handle["labels"]["data"]["visibility"][...], dtype=np.uint8)
        track_id = np.asarray(handle["labels"]["data"]["track_id"][...], dtype=np.int32)

    assert np.isnan(keypoints[0, 0, 0, 2])
    assert int(visibility[0, 0, 0]) == 0
    assert int(track_id[0, 0]) == -1

    loaded = Labels.load_file(archive_path.as_posix())
    pts = loaded.labeled_frames[0].instances[0].get_points_array(copy=False, full=True)
    assert float(pts["x"][0]) == 3.0
    assert float(pts["y"][0]) == 4.0
    assert bool(pts["visible"][0]) is False
    assert bool(pts["complete"][0]) is True


def test_update_labels_xpkg_preserves_predictions_by_default(tmp_path: Path) -> None:
    from xpkg.compat import (
        PredictionAppendItem,
        SerializerPredictedInstance,
        read_xpkg,
        update_labels_xpkg,
        write_xpkg,
    )

    initial_labels = _make_labels(tmp_path, x=1.0, y=2.0)
    updated_labels = _make_labels(tmp_path, x=9.0, y=10.0)
    archive_path = tmp_path / "project.xpkg"
    predictions = [
        PredictionAppendItem(
            video_index=0,
            frame_index=0,
            instances=[
                SerializerPredictedInstance(
                    keypoints=[(11.0, 12.0, 0.8)],
                    keypoint_scores=[0.8],
                    score=0.95,
                    track_id=7,
                )
            ],
        )
    ]

    write_xpkg(archive_path, initial_labels, predictions=predictions)
    update_labels_xpkg(archive_path, updated_labels)

    payload = read_xpkg(archive_path, lazy=False)
    label_keypoints = np.asarray(payload["labels"]["data"]["keypoints"], dtype=np.float32)
    prediction_scores = np.asarray(
        payload["predictions"]["data"]["keypoint_score"],
        dtype=np.float32,
    )
    prediction_track_ids = np.asarray(payload["predictions"]["data"]["track_id"], dtype=np.int32)

    assert float(label_keypoints[0, 0, 0, 0]) == 9.0
    assert float(label_keypoints[0, 0, 0, 1]) == 10.0
    assert float(prediction_scores[0, 0, 0]) == pytest.approx(0.8)
    assert int(prediction_track_ids[0, 0]) == 7


def test_read_xpkg_tolerates_missing_manifest_with_path_fallback(tmp_path: Path) -> None:
    from xpkg.compat import read_xpkg, write_xpkg

    labels = _make_labels(tmp_path, x=5.0, y=6.0)
    archive_path = tmp_path / "nometa.xpkg"
    write_xpkg(archive_path, labels)

    with h5py.File(archive_path.as_posix(), "r+") as handle:
        del handle["project_metadata"].attrs["manifest_json"]

    payload = read_xpkg(archive_path, lazy=False)

    assert payload["metadata"]["manifest"] is None
    videos_info = payload["metadata"]["videos"]
    assert videos_info["resolved_exists"] == [True]
    assert videos_info["resolved_paths"][0].endswith("frame.png")
