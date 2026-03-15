from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


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
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(path.as_posix(), fourcc, 5.0, (16, 12))
    assert writer.isOpened()
    try:
        for idx in range(2):
            frame = np.full((12, 16, 3), (idx + 1) * 20, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def test_convert_dlc_h5_project_builds_multi_video_bundle(tmp_path: Path) -> None:
    from posetta.io.converters.dlc_import import convert_dlc_h5_project
    from posetta.io.siesta_format import read_siesta
    from posetta.model import Labels

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
        bundle_extension=".siesta",
    )

    assert result.siesta_path == recording_dir / "session-0.siesta"
    payload = read_siesta(result.siesta_path, lazy=False)
    assert payload["labels"]["videos"]["filenames"] == [
        "alpha_view/session-0-leftCam.avi",
        "beta_view/session-0-underGlass.avi",
    ]

    labels = Labels.load_file(result.siesta_path.as_posix())
    assert len(labels.videos) == 2
    assert len(labels.labeled_frames) == len(df) * 2
    counts = Counter(Path(frame.video.filename or "").name for frame in labels.labeled_frames)
    assert counts == {
        "session-0-leftCam.avi": len(df),
        "session-0-underGlass.avi": len(df),
    }
