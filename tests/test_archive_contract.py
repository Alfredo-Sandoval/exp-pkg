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


def test_append_predictions_xpkg_roundtrip_uses_canonical_compat_surface(tmp_path: Path) -> None:
    from xpkg.compat import (
        PredictionAppendItem,
        SerializerPredictedInstance,
        append_predictions_xpkg,
        read_xpkg,
        write_xpkg,
    )

    labels = _make_labels(tmp_path, x=1.0, y=2.0)
    archive_path = tmp_path / "append.xpkg"
    write_xpkg(archive_path, labels, predictions=[])

    appended = append_predictions_xpkg(
        archive_path,
        [
            PredictionAppendItem(
                video_index=0,
                frame_index=0,
                instances=[
                    SerializerPredictedInstance(
                        keypoints=[(11.0, 12.0, 0.75)],
                        keypoint_scores=[0.75],
                        score=0.95,
                        track_id=5,
                    )
                ],
            )
        ],
        allow_max_inst_growth=True,
    )

    payload = read_xpkg(archive_path, lazy=False)
    frames = np.asarray(payload["predictions"]["frames"]["num_instances"], dtype=np.int32)
    keypoints = np.asarray(payload["predictions"]["data"]["keypoints"], dtype=np.float32)
    track_ids = np.asarray(payload["predictions"]["data"]["track_id"], dtype=np.int32)

    assert appended == 1
    assert int(payload["predictions"]["attrs"]["committed_length"]) == 1
    assert int(frames[0]) == 1
    assert float(keypoints[0, 0, 0, 0]) == 11.0
    assert float(keypoints[0, 0, 0, 1]) == 12.0
    assert float(keypoints[0, 0, 0, 2]) == pytest.approx(0.75)
    assert int(track_ids[0, 0]) == 5


def test_merge_predictions_xpkg_roundtrip_uses_canonical_compat_surface(tmp_path: Path) -> None:
    from xpkg.compat import (
        PredictionAppendItem,
        SerializerPredictedInstance,
        merge_predictions_xpkg,
        read_xpkg,
        write_xpkg,
    )

    labels = _make_labels(tmp_path, x=1.0, y=2.0)
    archive_path = tmp_path / "merge.xpkg"
    write_xpkg(
        archive_path,
        labels,
        predictions=[
            PredictionAppendItem(
                video_index=0,
                frame_index=0,
                instances=[
                    SerializerPredictedInstance(
                        keypoints=[(3.0, 4.0, 0.5)],
                        keypoint_scores=[0.5],
                        score=0.8,
                        track_id=1,
                    )
                ],
            )
        ],
    )

    merged = merge_predictions_xpkg(
        archive_path,
        [
            PredictionAppendItem(
                video_index=0,
                frame_index=0,
                instances=[
                    SerializerPredictedInstance(
                        keypoints=[(13.0, 14.0, 0.9)],
                        keypoint_scores=[0.9],
                        score=0.97,
                        track_id=9,
                    )
                ],
            )
        ],
        allow_max_inst_growth=True,
    )

    payload = read_xpkg(archive_path, lazy=False)
    frames = np.asarray(payload["predictions"]["frames"]["num_instances"], dtype=np.int32)
    keypoints = np.asarray(payload["predictions"]["data"]["keypoints"], dtype=np.float32)
    instance_scores = np.asarray(
        payload["predictions"]["data"]["instance_score"],
        dtype=np.float32,
    )
    track_ids = np.asarray(payload["predictions"]["data"]["track_id"], dtype=np.int32)

    assert merged == 1
    assert int(payload["predictions"]["attrs"]["committed_length"]) == 1
    assert int(frames[0]) == 2
    assert float(keypoints[0, 0, 0, 0]) == 3.0
    assert float(keypoints[0, 1, 0, 0]) == 13.0
    assert float(instance_scores[0, 0]) == pytest.approx(0.8)
    assert float(instance_scores[0, 1]) == pytest.approx(0.97)
    assert int(track_ids[0, 0]) == 1
    assert int(track_ids[0, 1]) == 9


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


def test_archive_metadata_field_helpers_roundtrip_mapping_payload(tmp_path: Path) -> None:
    from xpkg.compat import (
        load_archive_metadata_field,
        save_archive_metadata_field,
        write_xpkg,
    )

    labels = _make_labels(tmp_path, x=8.0, y=9.0)
    archive_path = tmp_path / "metadata.xpkg"
    write_xpkg(archive_path, labels)

    save_archive_metadata_field(
        archive_path,
        "session_json",
        {"active_frame_idx": 7},
    )

    assert load_archive_metadata_field(archive_path, "session_json") == {"active_frame_idx": 7}


def test_summarize_xpkg_uses_committed_prediction_length_when_metadata_count_is_stale(
    tmp_path: Path,
) -> None:
    from xpkg.compat import (
        PredictionAppendItem,
        SerializerPredictedInstance,
        summarize_xpkg,
        write_xpkg,
    )

    labels = _make_labels(tmp_path, x=1.0, y=2.0)
    archive_path = tmp_path / "summary.xpkg"
    write_xpkg(
        archive_path,
        labels,
        predictions=[
            PredictionAppendItem(
                video_index=0,
                frame_index=0,
                instances=[
                    SerializerPredictedInstance(
                        keypoints=[(3.0, 4.0, 0.5)],
                        keypoint_scores=[0.5],
                        score=0.8,
                        track_id=1,
                    )
                ],
            )
        ],
    )

    with h5py.File(archive_path.as_posix(), "r+") as handle:
        handle["project_metadata"].attrs["n_predictions_committed"] = 0

    summary = summarize_xpkg(archive_path)

    assert summary.prediction_frames == 1


def test_video_stub_and_build_prediction_stub_cover_generic_prediction_setup() -> None:
    from xpkg.model import VideoStub, build_prediction_stub

    video = VideoStub(filename="clip.mp4", frames=4, height=32, width=24)
    labels = build_prediction_stub(["nose", "tail"], video, skeleton_name="demo-predict")

    assert video.last_frame_idx == 3
    assert video.image_filenames == []
    with pytest.raises(RuntimeError, match="metadata-only"):
        video.get_frame(0)
    assert labels.videos[0] == video
    assert labels.skeletons[0].name == "demo-predict"
    assert [kp.name for kp in labels.keypoints] == ["nose", "tail"]
