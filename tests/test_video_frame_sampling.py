from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest


def _write_video(path: Path, *, frame_count: int = 10) -> Path:
    from xpkg.media.video import VideoWriterOpenCV

    with VideoWriterOpenCV(path.as_posix(), 16, 16, fps=5.0) as writer:
        for idx in range(frame_count):
            writer.add_frame(np.full((16, 16, 3), idx * 10, dtype=np.uint8), bgr=True)
    return path


def test_probe_video_path_reads_metadata(tmp_path: Path) -> None:
    from xpkg.media import probe_video_path

    video_path = _write_video(tmp_path / "sample.avi")

    metadata = probe_video_path(video_path)

    assert metadata.path == video_path
    assert metadata.frame_count >= 10
    assert metadata.width == 16
    assert metadata.height == 16
    assert metadata.fps > 0


def test_read_frame_indices_returns_requested_frames(tmp_path: Path) -> None:
    from xpkg.media import read_frame_indices

    video_path = _write_video(tmp_path / "sample.avi")

    frames = read_frame_indices(video_path, frame_indices=[0, 3, 7])

    assert sorted(frames) == [0, 3, 7]
    assert all(frame.shape == (16, 16, 3) for frame in frames.values())


def test_extract_frame_indices_writes_and_skips_existing(tmp_path: Path) -> None:
    from xpkg.media import extract_frame_indices

    video_path = _write_video(tmp_path / "sample.avi")
    output_dir = tmp_path / "frames"

    first = extract_frame_indices(
        video_path,
        frame_indices=[0, 5, 9],
        output_dir=output_dir,
        skip_existing=False,
    )
    second = extract_frame_indices(
        video_path,
        frame_indices=[0, 5, 9],
        output_dir=output_dir,
        skip_existing=True,
    )

    assert first == second
    for frame_idx, frame_path in first.items():
        assert frame_path == output_dir / f"frame_{frame_idx:06d}.jpg"
        assert frame_path.is_file()
        image = cv2.imread(frame_path.as_posix())
        assert image is not None
        assert image.shape == (16, 16, 3)


def test_frame_sampling_rejects_negative_and_missing_indices(tmp_path: Path) -> None:
    from xpkg.media import read_frame_indices

    video_path = _write_video(tmp_path / "sample.avi", frame_count=2)

    with pytest.raises(ValueError, match="non-negative"):
        read_frame_indices(video_path, frame_indices=[-1])
    with pytest.raises(RuntimeError, match="Failed to extract requested video frames"):
        read_frame_indices(video_path, frame_indices=[5])
