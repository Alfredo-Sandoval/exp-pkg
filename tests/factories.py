"""Shared factories for synthetic test inputs.

Test modules import these instead of reaching into other test modules for
their private helpers. Keep factories deterministic, and import xpkg model
types lazily so collecting one test module does not eagerly import the
whole package surface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import pandas as pd

from xpkg._core.json_utils import write_json


def video_writer_fourcc(code: str) -> int:
    fourcc_fn = getattr(cv2, "VideoWriter_fourcc", None)
    if not callable(fourcc_fn):
        raise RuntimeError("OpenCV build does not expose VideoWriter_fourcc")
    return int(fourcc_fn(*code))


def write_dummy_video(path: Path, *, frame_count: int = 2) -> None:
    """Write a tiny decodable MJPG video with the requested frame count."""
    writer = cv2.VideoWriter(path.as_posix(), video_writer_fourcc("MJPG"), 5.0, (16, 12))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    try:
        for idx in range(frame_count):
            frame = np.full((12, 16, 3), (idx + 1) * 20 % 256, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def write_test_image(path: Path, value: int = 128) -> None:
    image = np.full((12, 16, 3), value, dtype=np.uint8)
    if not cv2.imwrite(path.as_posix(), image):
        raise RuntimeError(f"Could not write test image {path}")


def make_single_frame_video(tmp_path: Path):
    from xpkg.model import Video

    frame_path = tmp_path / "frame.png"
    write_test_image(frame_path)
    video = Video.from_image_filenames([frame_path.as_posix()])
    video.filename = frame_path.as_posix()
    return frame_path, video


def write_test_video(path: Path) -> None:
    writer = cv2.VideoWriter(path.as_posix(), video_writer_fourcc("MJPG"), 5.0, (16, 12))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    for value in (32, 64, 96):
        frame = np.full((12, 16, 3), value, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    if not path.exists():
        raise RuntimeError(f"Video was not written: {path}")


def make_media_labels(video_path: Path, *, x: float, y: float):
    from xpkg.model import Labels, Video, build_keypoint_skeleton
    from xpkg.pose.annotations import Instance, LabeledFrame, Point

    video = Video.from_filename(video_path.as_posix())
    skeleton = build_keypoint_skeleton(["nose"], name="subject")
    frame = LabeledFrame(
        video=video,
        frame_idx=0,
        instances=[
            Instance(
                skeleton=skeleton,
                init_points={"nose": Point(x, y, visible=True, complete=True)},
            )
        ],
    )
    labels = Labels(labeled_frames=[frame], videos=[video], skeletons=[skeleton])
    labels.validate()
    return labels


def make_labels(tmp_path: Path, *, x: float, y: float):
    from xpkg.model import Labels, build_keypoint_skeleton
    from xpkg.pose.annotations import Instance, LabeledFrame, Point

    _, video = make_single_frame_video(tmp_path)
    skeleton = build_keypoint_skeleton(["nose"], name="subject")
    frame = LabeledFrame(
        video=video,
        frame_idx=0,
        instances=[
            Instance(
                skeleton=skeleton,
                init_points={"nose": Point(x, y, visible=True, complete=True)},
            )
        ],
    )
    labels = Labels(labeled_frames=[frame], videos=[video], skeletons=[skeleton])
    labels.validate()
    return labels


def sample_dlc_dataframe(*, x_offset: float = 0.0) -> pd.DataFrame:
    columns = pd.MultiIndex.from_product(
        [["demo"], ["nose", "tail"], ["x", "y", "likelihood"]],
        names=["scorer", "bodyparts", "coords"],
    )
    return pd.DataFrame(
        [
            [10.0 + x_offset, 20.0, 0.95, 30.0 + x_offset, 40.0, 0.90],
            [11.0 + x_offset, 21.0, 0.85, 31.0 + x_offset, 41.0, 0.80],
        ],
        columns=columns,
    )


def write_sample_dlc_h5(path: Path, *, x_offset: float = 0.0) -> pd.DataFrame:
    df = sample_dlc_dataframe(x_offset=x_offset)
    df.to_hdf(path, key="df")
    return df


def write_sample_dlc_csv(path: Path, *, x_offset: float = 0.0) -> pd.DataFrame:
    df = sample_dlc_dataframe(x_offset=x_offset)
    df.to_csv(path)
    return df


def write_sleap_analysis_h5(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[bytes]]:
    n_tracks = 2
    n_nodes = 4
    n_frames = 10

    tracks = np.zeros((n_tracks, 2, n_nodes, n_frames), dtype=np.float64)
    point_scores = np.zeros((n_tracks, n_nodes, n_frames), dtype=np.float64)
    instance_scores = np.zeros((n_tracks, n_frames), dtype=np.float64)

    for track_idx in range(n_tracks):
        for node_idx in range(n_nodes):
            for frame_idx in range(n_frames):
                x = 100.0 * track_idx + 10.0 * node_idx + frame_idx
                y = -x
                tracks[track_idx, 0, node_idx, frame_idx] = x
                tracks[track_idx, 1, node_idx, frame_idx] = y
                point_scores[track_idx, node_idx, frame_idx] = (
                    0.1 * track_idx + 0.01 * node_idx + 0.001 * frame_idx
                )
        for frame_idx in range(n_frames):
            instance_scores[track_idx, frame_idx] = 0.5 + 0.1 * track_idx + 0.01 * frame_idx

    # Ensure NaN propagation can be validated.
    tracks[1, 0, 2, 5] = np.nan

    node_names = [b"HIP", b"KNEE", b"ANKLE", b"TOE"]
    track_names = [b"track_0", b"track_1"]

    with h5py.File(path, "w") as handle:
        handle.create_dataset("tracks", data=tracks)
        handle.create_dataset("point_scores", data=point_scores)
        handle.create_dataset("instance_scores", data=instance_scores)
        handle.create_dataset("node_names", data=np.asarray(node_names, dtype="S"))
        handle.create_dataset("track_names", data=np.asarray(track_names, dtype="S"))

    return tracks, point_scores, instance_scores, node_names


def mmpose_instance(
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


def write_mmpose_topdown_json(path: Path) -> Path:
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
                        mmpose_instance(base=10.0, scores=[0.95, 0.85, 0.75]),
                        mmpose_instance(base=110.0, scores=[0.65, 0.55, 0.45]),
                    ],
                },
                {
                    "frame_id": 2,
                    "instances": [
                        mmpose_instance(base=11.0, scores=[0.90, 0.80, 0.70]),
                    ],
                },
                {
                    "frame_id": 3,
                    "instances": [
                        mmpose_instance(base=12.0, scores=[0.88, 0.78, 0.60]),
                        mmpose_instance(base=112.0, scores=[0.62, 0.52, 0.42]),
                    ],
                },
            ],
        },
    )
    return path


def pose_landmarks(
    *,
    x_shift: float = 0.0,
    y_shift: float = 0.0,
    visibility: float = 0.9,
    presence: float | None = 0.8,
) -> list[dict[str, float]]:
    from xpkg.io.readers.pose.mediapipe_pose_landmarks import MEDIAPIPE_POSE_LANDMARK_NAMES

    landmarks: list[dict[str, float]] = []
    for index, _node_name in enumerate(MEDIAPIPE_POSE_LANDMARK_NAMES):
        entry: dict[str, float] = {
            "x": 0.05 + (index * 0.01) + x_shift,
            "y": 0.10 + (index * 0.005) + y_shift,
            "z": -0.01 * index,
            "visibility": visibility,
        }
        if presence is not None:
            entry["presence"] = presence
        landmarks.append(entry)
    return landmarks


def write_mediapipe_pose_landmarks_json(
    path: Path,
    *,
    image_width: int = 200,
    image_height: int = 100,
    frames: list[dict[str, Any]] | None = None,
) -> None:
    from xpkg.io.readers.pose.mediapipe_pose_landmarks import (
        MEDIAPIPE_POSE_LANDMARKS_JSON_SCHEMA,
    )

    if frames is None:
        frames = [
            {"frame_index": 0, "pose_landmarks": pose_landmarks()},
            {"frame_index": 1, "pose_landmarks": pose_landmarks(x_shift=0.02, y_shift=0.01)},
        ]

    write_json(
        path,
        {
            "schema": MEDIAPIPE_POSE_LANDMARKS_JSON_SCHEMA,
            "image_width": image_width,
            "image_height": image_height,
            "frames": frames,
        },
    )
