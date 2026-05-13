from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest


def _write_video(path: Path, *, frame_count: int = 10, width: int = 32, height: int = 24) -> Path:
    from xpkg.media.video import VideoWriterOpenCV

    with VideoWriterOpenCV(path.as_posix(), height, width, fps=10.0) as writer:
        for idx in range(frame_count):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[..., 0] = (idx * 7) % 255
            frame[..., 1] = (idx * 11) % 255
            frame[..., 2] = (idx * 13) % 255
            writer.add_frame(frame, bgr=True)
    return path


def _write_image_sequence(path: Path, *, frame_count: int = 4) -> list[str]:
    path.mkdir()
    filenames: list[str] = []
    for idx in range(frame_count):
        frame = np.full((12, 14, 3), idx * 20, dtype=np.uint8)
        filename = path / f"{idx:03d}.png"
        assert cv2.imwrite(filename.as_posix(), frame)
        filenames.append(filename.as_posix())
    return filenames


def test_video_color_mode_swaps_channels_and_is_case_insensitive(tmp_path: Path) -> None:
    from xpkg.media.video import Video

    video_path = _write_video(tmp_path / "sample.avi")

    with (
        Video.from_filename(video_path.as_posix(), color="bgr") as bgr_video,
        Video.from_filename(video_path.as_posix(), color="RGB") as rgb_video,
    ):
        bgr_frame = bgr_video.get_frame(1)
        rgb_frame = rgb_video.get_frame(1)

    np.testing.assert_array_equal(bgr_frame[..., 0], rgb_frame[..., 2])
    np.testing.assert_array_equal(bgr_frame[..., 2], rgb_frame[..., 0])


def test_video_get_frame_rejects_invalid_indices(tmp_path: Path) -> None:
    from xpkg.media.video import Video

    video_path = _write_video(tmp_path / "sample.avi", frame_count=3)

    with Video.from_filename(video_path.as_posix()) as video:
        with pytest.raises(IndexError):
            video.get_frame(-1)
        with pytest.raises(IndexError):
            video.get_frame(99)


def test_opencv_backend_random_access_matches_iteration(tmp_path: Path) -> None:
    from xpkg.media.reader_backends import OpenCVBackend

    video_path = _write_video(tmp_path / "sample.avi", frame_count=8)
    backend = OpenCVBackend(video_path.as_posix())
    expected = list(backend.iter_frames())

    frame_6 = backend.get_frame(6)
    frame_2 = backend.get_frame(2)
    backend.close()

    np.testing.assert_array_equal(frame_6, expected[6])
    np.testing.assert_array_equal(frame_2, expected[2])


def test_image_sequence_backend_preserves_indices_and_bounds(tmp_path: Path) -> None:
    from xpkg.media.reader_backends import ImageSequenceBackend

    filenames = _write_image_sequence(tmp_path / "frames", frame_count=4)
    backend = ImageSequenceBackend(filenames)

    assert backend.frames == 4
    assert backend.width == 14
    assert backend.height == 12
    np.testing.assert_array_equal(backend.get_frame(3), np.full((12, 14, 3), 60, dtype=np.uint8))
    with pytest.raises(IndexError):
        backend.get_frame(-1)
    with pytest.raises(IndexError):
        backend.get_frame(4)
    backend.close()


def test_pyav_backend_stride_matches_filtered_iteration(tmp_path: Path) -> None:
    pytest.importorskip("av")
    from xpkg.media.reader_backends import PyAVBackend

    video_path = _write_video(tmp_path / "sample.avi", frame_count=9)
    baseline = PyAVBackend(video_path.as_posix())
    expected = [frame for idx, frame in enumerate(baseline.iter_frames()) if idx % 3 == 0]
    baseline.close()

    backend = PyAVBackend(video_path.as_posix())
    actual = list(backend.iter_frames_stride(3))
    backend.close()

    assert len(actual) == len(expected)
    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(got, want)


class _FakeDecordFrame:
    def __init__(self, frame_rgb: np.ndarray) -> None:
        self._frame_rgb = frame_rgb

    def asnumpy(self) -> np.ndarray:
        return self._frame_rgb


class _FakeDecordReader:
    def __init__(self, _filename: str, *, ctx: object) -> None:
        self.ctx = ctx
        self.frames = [
            np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8),
            np.array([[[11, 12, 13], [14, 15, 16]]], dtype=np.uint8),
            np.array([[[21, 22, 23], [24, 25, 26]]], dtype=np.uint8),
        ]

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> _FakeDecordFrame:
        return _FakeDecordFrame(self.frames[int(idx)])

    def get_avg_fps(self) -> float:
        return 25.0


def test_decord_gpu_backend_converts_rgb_to_bgr_and_strides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from xpkg.media import backend_utils
    from xpkg.media.reader_backends import DecordGpuBackend

    monkeypatch.setattr(
        backend_utils,
        "require_decord_gpu_api",
        lambda: (object(), _FakeDecordReader, lambda device_idx: ("gpu", device_idx)),
    )
    video_path = _write_video(tmp_path / "sample.avi", frame_count=1)

    backend = DecordGpuBackend(video_path.as_posix())
    frame = backend.get_frame(1)
    strided = list(backend.iter_frames_stride(2))
    backend.close()

    assert backend.fps == pytest.approx(25.0)
    np.testing.assert_array_equal(frame, np.array([[[13, 12, 11], [16, 15, 14]]], dtype=np.uint8))
    assert [int(item[0, 0, 0]) for item in strided] == [3, 23]


def test_video_total_frames_rejects_invalid_contract_values() -> None:
    from xpkg.media import video_total_frames

    class _Video:
        def __init__(self, frames: Any) -> None:
            self.frames = frames

    assert video_total_frames(_Video(35)) == 35
    with pytest.raises(TypeError, match="not bool"):
        video_total_frames(_Video(True))
    with pytest.raises(ValueError, match="non-negative"):
        video_total_frames(_Video(-1))
