from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def test_video_image_sequence_and_writer_roundtrip(tmp_path: Path) -> None:
    from xpkg.media.video import Video, gui_playback_backend_for_path, write_video

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()

    frame_paths: list[str] = []
    for idx in range(3):
        frame = np.zeros((12, 16, 3), dtype=np.uint8)
        frame[..., 0] = idx * 20
        frame[..., 1] = 50
        frame[..., 2] = 200
        path = frames_dir / f"{idx:03d}.png"
        ok = cv2.imwrite(path.as_posix(), frame)
        assert ok
        frame_paths.append(path.as_posix())

    video = Video.from_image_filenames(frame_paths)
    assert video.backend == "images"
    assert video.frames == 3
    assert video.width == 16
    assert video.height == 12
    assert gui_playback_backend_for_path("clip.mp4") == "opencv"

    loaded = video.get_frame(1)
    assert loaded.shape == (12, 16, 3)

    out_path = tmp_path / "out.avi"
    write_video(out_path.as_posix(), video, frames=[0, 1, 2], fps=5)
    assert out_path.exists()
    assert out_path.stat().st_size > 0

    decoded = Video.from_filename(out_path.as_posix())
    assert decoded.backend == "opencv"
    assert decoded.width == 16
    assert decoded.height == 12
    assert decoded.frames >= 1
    roundtrip = decoded.get_frame(0)
    assert roundtrip.shape == (12, 16, 3)
