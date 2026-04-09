from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def _video_writer_fourcc(code: str) -> int:
    fourcc_fn = getattr(cv2, "VideoWriter_fourcc", None)
    if not callable(fourcc_fn):
        raise RuntimeError("OpenCV build does not expose VideoWriter_fourcc")
    return int(fourcc_fn(*code))


def _write_sample_dlc_h5(path: Path) -> pd.DataFrame:
    columns = pd.MultiIndex.from_product(
        [["demo"], ["nose", "tail"], ["x", "y", "likelihood"]],
        names=["scorer", "bodyparts", "coords"],
    )
    df = pd.DataFrame(
        [
            [10.0, 20.0, 0.95, 30.0, 40.0, 0.90],
            [11.0, 21.0, 0.85, 31.0, 41.0, 0.80],
        ],
        columns=columns,
    )
    df.to_hdf(path, key="df")
    return df


def _write_dummy_video(path: Path) -> None:
    fourcc = _video_writer_fourcc("MJPG")
    writer = cv2.VideoWriter(path.as_posix(), fourcc, 5.0, (16, 12))
    assert writer.isOpened()
    try:
        for idx in range(2):
            frame = np.full((12, 16, 3), (idx + 1) * 20, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def test_convert_dlc_h5_project_builds_multi_video_bundle(tmp_path: Path) -> None:
    from xpkg.io.archive_format import read_archive
    from xpkg.io.converters.dlc_import import convert_dlc_h5_project
    from xpkg.model import Labels

    recording_dir = tmp_path / "session-0"
    (recording_dir / "tracking").mkdir(parents=True)
    (recording_dir / "alpha_view").mkdir()
    (recording_dir / "beta_view").mkdir()
    tracking_path = recording_dir / "tracking" / "session-0-tracking.h5"
    df = _write_sample_dlc_h5(tracking_path)
    video_a = recording_dir / "alpha_view" / "session-0-leftCam.avi"
    video_b = recording_dir / "beta_view" / "session-0-underGlass.avi"
    _write_dummy_video(video_a)
    _write_dummy_video(video_b)

    result = convert_dlc_h5_project(
        tracking_path,
        [video_a, video_b],
        recording_dir,
    )

    assert result.bundle_path == recording_dir / "session-0.xpkg"
    payload = read_archive(result.bundle_path, lazy=False)
    assert payload["labels"]["videos"]["filenames"] == [
        "alpha_view/session-0-leftCam.avi",
        "beta_view/session-0-underGlass.avi",
    ]

    labels = Labels.load_file(result.bundle_path.as_posix())
    assert len(labels.videos) == 2
    assert len(labels.labeled_frames) == len(df) * 2
    counts = Counter(Path(frame.video.filename or "").name for frame in labels.labeled_frames)
    assert counts == {
        "session-0-leftCam.avi": len(df),
        "session-0-underGlass.avi": len(df),
    }
