from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest

from xpkg.io.labels.model import Labels
from xpkg.io.pose_hdf5 import export_pose_h5
from xpkg.media.video import Video
from xpkg.pose.annotations.frames import LabeledFrame
from xpkg.pose.annotations.instances import Instance, PredictedInstance
from xpkg.pose.annotations.points import Point, PredictedPoint
from xpkg.pose.skeleton import Keypoint, Skeleton


def _make_video(tmp_path: Path, name: str) -> Video:
    frame_path = tmp_path / name
    ok = cv2.imwrite(frame_path.as_posix(), np.full((12, 16, 3), 128, dtype=np.uint8))
    assert ok
    return Video.from_image_filenames([frame_path.as_posix()])


@pytest.fixture
def sample_skeleton() -> Skeleton:
    keypoints = [
        Keypoint(id=0, name="nose"),
        Keypoint(id=1, name="left_ear"),
        Keypoint(id=2, name="right_ear"),
        Keypoint(id=3, name="tail"),
    ]
    return Skeleton(name="test_skeleton", keypoints=keypoints, links_ids=[(0, 3)])


@pytest.fixture
def sample_video(tmp_path: Path) -> Video:
    return _make_video(tmp_path, "sample.png")


@pytest.fixture
def sample_labels(sample_skeleton: Skeleton, sample_video: Video) -> Labels:
    labels = Labels(skeletons=[sample_skeleton], videos=[sample_video])

    for frame_idx in [0, 5, 10, 15]:
        points: Mapping[str | Keypoint, Point] = {
            "nose": Point(100.0 + frame_idx, 200.0, visible=True, complete=True),
            "left_ear": Point(80.0 + frame_idx, 180.0, visible=True, complete=True),
            "right_ear": Point(120.0 + frame_idx, 180.0, visible=True, complete=True),
            "tail": Point(100.0 + frame_idx, 300.0, visible=True, complete=True),
        }
        instance = Instance(skeleton=sample_skeleton, init_points=points)
        labels.append(LabeledFrame(video=sample_video, frame_idx=frame_idx, instances=[instance]))

    labels.update_cache()
    return labels


def test_export_pose_h5_creates_required_datasets(sample_labels: Labels, tmp_path) -> None:
    output_path = tmp_path / "analysis.h5"

    result = export_pose_h5(sample_labels, output_path)

    assert result.exists()
    with h5py.File(output_path, "r") as f:
        assert f.attrs["format"] == "xpkg_pose_analysis_v1"
        assert "node_names" in f
        assert "track_names" in f
        assert "tracks" in f
        assert "track_occupancy" in f
        assert "frame_indices" in f


def test_export_pose_h5_tracks_shape_and_node_names(sample_labels: Labels, tmp_path) -> None:
    output_path = tmp_path / "analysis.h5"

    export_pose_h5(sample_labels, output_path)

    with h5py.File(output_path, "r") as f:
        tracks = f["tracks"][:]
        names = [n.decode() if isinstance(n, bytes) else n for n in f["node_names"][:]]
        assert tracks.shape[1] == 4
        assert tracks.shape[2] == 2
        assert "nose" in names
        assert "tail" in names


def test_export_pose_h5_excludes_confidence_when_disabled(sample_labels: Labels, tmp_path) -> None:
    output_path = tmp_path / "analysis.h5"

    export_pose_h5(sample_labels, output_path, include_confidence=False)

    with h5py.File(output_path, "r") as f:
        assert "confidence" not in f


def test_export_pose_h5_skips_confidence_for_missing_predicted_points(
    sample_skeleton: Skeleton,
    sample_video: Video,
    tmp_path,
) -> None:
    labels = Labels(skeletons=[sample_skeleton], videos=[sample_video])
    frame = LabeledFrame(video=sample_video, frame_idx=0)
    pred = PredictedInstance(skeleton=sample_skeleton, frame=frame)
    pred["nose"] = PredictedPoint(x=11.0, y=22.0, visible=True, score=0.9)
    pred["left_ear"] = PredictedPoint(x=np.nan, y=np.nan, visible=True, score=0.8)
    frame.instances.append(pred)
    labels.append(frame)
    labels.update_cache()

    output_path = tmp_path / "analysis_missing_prediction.h5"
    export_pose_h5(labels, output_path, include_confidence=True)

    with h5py.File(output_path, "r") as f:
        tracks = f["tracks"][:]
        confidence = f["confidence"][:]

    assert tracks[0, 0, :, 0].tolist() == [11.0, 22.0]
    assert confidence[0, 0, 0] == pytest.approx(0.9)
    assert np.isnan(tracks[0, 1, :, 0]).all()
    assert confidence[0, 1, 0] == pytest.approx(0.0)


def test_export_pose_h5_video_filter_sizes_by_selected_video(
    sample_skeleton: Skeleton,
    sample_video: Video,
    tmp_path,
) -> None:
    second_video = _make_video(tmp_path, "sample_b.png")
    labels = Labels(skeletons=[sample_skeleton], videos=[sample_video, second_video])

    points_a: Mapping[str | Keypoint, Point] = {
        "nose": Point(100.0, 200.0, visible=True, complete=True),
        "left_ear": Point(80.0, 180.0, visible=True, complete=True),
        "right_ear": Point(120.0, 180.0, visible=True, complete=True),
        "tail": Point(100.0, 300.0, visible=True, complete=True),
    }
    points_b: Mapping[str | Keypoint, Point] = {
        "nose": Point(110.0, 210.0, visible=True, complete=True),
        "left_ear": Point(90.0, 190.0, visible=True, complete=True),
        "right_ear": Point(130.0, 190.0, visible=True, complete=True),
        "tail": Point(110.0, 310.0, visible=True, complete=True),
    }

    labels.append(
        LabeledFrame(
            video=sample_video,
            frame_idx=0,
            instances=[Instance(skeleton=sample_skeleton, init_points=points_a)],
        )
    )
    labels.append(
        LabeledFrame(
            video=second_video,
            frame_idx=10000,
            instances=[Instance(skeleton=sample_skeleton, init_points=points_b)],
        )
    )
    labels.update_cache()

    output_path = tmp_path / "analysis_video0.h5"
    export_pose_h5(labels, output_path, video_index=0)

    with h5py.File(output_path, "r") as f:
        assert int(f.attrs["n_frames"]) == 1
        assert f["tracks"].shape[0] == 1
        np.testing.assert_array_equal(f["frame_indices"][:], np.array([0], dtype=np.int32))
        assert int(np.asarray(f["track_occupancy"], dtype=np.int32).sum()) == 1


def test_export_pose_h5_video_filter_raises_for_empty_selection(
    sample_labels: Labels, tmp_path
) -> None:
    output_path = tmp_path / "analysis_missing_video.h5"

    with pytest.raises(ValueError, match="video_index=99"):
        export_pose_h5(sample_labels, output_path, video_index=99)
