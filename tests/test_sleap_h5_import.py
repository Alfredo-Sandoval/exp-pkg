from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from tests.io.readers.test_sleap_analysis_h5 import _write_sleap_analysis_h5


def _video_writer_fourcc(code: str) -> int:
    fourcc_fn = getattr(cv2, "VideoWriter_fourcc", None)
    if not callable(fourcc_fn):
        raise RuntimeError("OpenCV build does not expose VideoWriter_fourcc")
    return int(fourcc_fn(*code))


def _write_dummy_video(path: Path, *, frame_count: int) -> None:
    writer = cv2.VideoWriter(path.as_posix(), _video_writer_fourcc("MJPG"), 5.0, (16, 12))
    assert writer.isOpened()
    try:
        for idx in range(frame_count):
            frame = np.full((12, 16, 3), idx, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def test_convert_sleap_h5_builds_multi_track_labels(tmp_path: Path) -> None:
    from xpkg.io.converters.sleap_import import convert_sleap_h5

    tracking_path = tmp_path / "analysis.h5"
    _write_sleap_analysis_h5(tracking_path)
    video_path = tmp_path / "session.avi"
    _write_dummy_video(video_path, frame_count=10)

    result = convert_sleap_h5(
        tracking_path,
        video_path,
        skeleton_name="subject",
    )

    assert result.metadata["source"] == "sleap_h5_import"
    assert result.metadata["source_h5"] == tracking_path.as_posix()
    assert result.metadata["source_video"] == video_path.as_posix()

    labels = result.labels
    assert len(labels.videos) == 1
    assert len(labels.labeled_frames) == 10

    first_frame = next(frame for frame in labels.labeled_frames if frame.frame_idx == 0)
    assert sorted(
        inst.track.id for inst in first_frame.instances if inst.track is not None
    ) == [0, 1]

    frame_five = next(frame for frame in labels.labeled_frames if frame.frame_idx == 5)
    track_one = next(inst for inst in frame_five.instances if inst.track and inst.track.id == 1)
    points = track_one.get_points_array(copy=False, full=True)
    assert int(np.count_nonzero(~np.isnan(points["x"]))) == 3
