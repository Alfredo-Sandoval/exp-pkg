from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest


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
    assert gui_playback_backend_for_path("clip.mp4") == "pyav"

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


def test_video_opens_image_sequence_directory_and_selected_frames(tmp_path: Path) -> None:
    from xpkg.media.video import Video, gui_playback_backend_for_path

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for idx in range(4):
        frame = np.full((8, 10, 3), idx, dtype=np.uint8)
        assert cv2.imwrite((frames_dir / f"{idx:03d}.png").as_posix(), frame)

    video = Video.from_filename(frames_dir.as_posix())

    assert video.backend == "images"
    assert video.filename == frames_dir.as_posix()
    assert video.frames == 4
    assert gui_playback_backend_for_path(frames_dir.as_posix()) == "images"
    assert [idx for idx, _frame in video.iter_selected_frames([2, 0])] == [2, 0]


def test_video_iter_frames_stride_and_batches_share_source_indices(tmp_path: Path) -> None:
    from xpkg.media.video import Video, VideoWriterOpenCV

    video_path = tmp_path / "sample.avi"
    with VideoWriterOpenCV(video_path.as_posix(), 8, 10, 5.0) as writer:
        for idx in range(5):
            writer.add_frame(np.full((8, 10, 3), idx * 20, dtype=np.uint8), bgr=True)

    video = Video.from_filename(video_path.as_posix())

    assert len(list(video.iter_frames_stride(2))) == 3
    batches = list(video.iter_frame_batches_stride(batch_size=2, stride=2))
    assert [indices for indices, _frames in batches] == [[0, 2], [4]]
    with pytest.raises(ValueError, match="stride"):
        list(video.iter_frames_stride(0))


def test_get_frames_safely_rejects_heterogeneous_image_sequence_shapes(tmp_path: Path) -> None:
    from xpkg.media.video import Video

    paths: list[str] = []
    for idx, shape in enumerate(((4, 4, 3), (6, 5, 3))):
        frame = np.full(shape, idx, dtype=np.uint8)
        path = tmp_path / f"frame_{idx}.png"
        assert cv2.imwrite(path.as_posix(), frame)
        paths.append(path.as_posix())

    video = Video.from_image_filenames(paths)

    assert [frame.shape for frame in video.get_frames_list([0, 1])] == [(4, 4, 3), (6, 5, 3)]
    with pytest.raises(ValueError, match="shared shape"):
        video.get_frames_safely([0, 1])


def test_video_facade_reexports_split_media_modules() -> None:
    from xpkg.media.readers import Video as ReaderVideo
    from xpkg.media.transforms import resize_image
    from xpkg.media.video import Video, VideoWriter, augment_background
    from xpkg.media.writers import VideoWriter as WriterVideoWriter

    assert Video is ReaderVideo
    assert VideoWriter is WriterVideoWriter
    assert callable(resize_image)
    assert callable(augment_background)


def test_video_writer_factory_validates_requested_ffmpeg_codec(monkeypatch, tmp_path: Path) -> None:
    from xpkg.media import writers
    from xpkg.media.video import VideoWriterImageio, build_video_writer

    class _DummyImageioWriter:
        def append_data(self, _img: np.ndarray) -> None:
            return None

        def close(self) -> None:
            return None

    monkeypatch.setattr(writers, "ffmpeg_encoders", lambda: {"libx264"})
    monkeypatch.setattr(writers.iio, "get_writer", lambda *args, **kwargs: _DummyImageioWriter())

    writer = build_video_writer(
        (tmp_path / "out.mp4").as_posix(),
        8,
        10,
        5.0,
        backend="imageio",
        codec="libx264",
    )

    assert isinstance(writer, VideoWriterImageio)
    assert writer.codec == "libx264"
    assert writer.output_params == ["-preset", "superfast", "-crf", "21"]
    with pytest.raises(RuntimeError, match="not available"):
        build_video_writer(
            (tmp_path / "bad.mp4").as_posix(),
            8,
            10,
            5.0,
            backend="imageio",
            codec="missing_codec",
        )


def test_explicit_pyav_backend_requires_optional_extra_when_missing(tmp_path: Path) -> None:
    from xpkg.media import media_backend_status
    from xpkg.media.video import Video

    pyav_status = media_backend_status("pyav")
    if pyav_status.available:
        pytest.skip("pyav is installed in this environment")

    video_path = tmp_path / "not-a-real-video.mp4"
    video_path.write_bytes(b"not a real video")

    with pytest.raises(ImportError, match=r"exp-pkg\[media-rich\]"):
        Video.from_filename(video_path.as_posix(), backend="pyav")
