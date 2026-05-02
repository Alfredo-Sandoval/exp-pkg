"""Video writer utilities for canonical numpy media arrays."""

from __future__ import annotations

from pathlib import Path
from threading import Event
from typing import Any

import cv2
import imageio.v2 as iio
import numpy as np

from xpkg._core.colors import bgr_to_rgb, ensure_bgr, ensure_three_channels
from xpkg._core.path_registry import ensure_dir, resolve_path
from xpkg.media.transforms import augment_background, resize_image

__all__ = [
    "VideoWriter",
    "VideoWriterImageio",
    "VideoWriterOpenCV",
    "write_video",
]


def _video_writer_fourcc(code: str) -> int:
    fourcc_fn = getattr(cv2, "VideoWriter_fourcc", None)
    if not callable(fourcc_fn):
        raise RuntimeError("OpenCV build does not expose VideoWriter_fourcc")
    return int(fourcc_fn(code[0], code[1], code[2], code[3]))


class VideoWriterOpenCV:
    """OpenCV-backed encoder used for AVI output."""

    def __init__(
        self,
        filename: str,
        height: int,
        width: int,
        fps: float,
        *,
        fourcc: str | None = None,
        is_color: bool = True,
    ):
        ext = Path(filename).suffix.lower()
        code = fourcc or ("MJPG" if ext == ".avi" else "mp4v")
        if len(code) != 4:
            raise ValueError(f"fourcc must be exactly 4 characters, got {code!r}")
        writer = cv2.VideoWriter(
            filename,
            _video_writer_fourcc(code),
            fps,
            (int(width), int(height)),
            bool(is_color),
        )
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f"Failed to open OpenCV VideoWriter for {filename} (fourcc={code})")
        self._writer = writer

    def add_frame(self, image: np.ndarray, *, bgr: bool = False) -> None:
        self._writer.write(ensure_bgr(image, input_is_bgr=bgr))

    def close(self) -> None:
        self._writer.release()


class VideoWriterImageio:
    """ImageIO-backed encoder used for non-AVI outputs."""

    def __init__(self, filename: str, height: int, width: int, fps: float, **kwargs: Any):
        del height, width
        codec = kwargs.get("codec")
        writer_kwargs: dict[str, Any] = {"fps": fps}
        if codec is not None:
            writer_kwargs["codec"] = str(codec)
        self._writer = iio.get_writer(filename, **writer_kwargs)

    def add_frame(self, image: np.ndarray, *, bgr: bool = False) -> None:
        frame = ensure_three_channels(image)
        if bgr:
            frame = bgr_to_rgb(frame)
        self._writer.append_data(frame)

    def close(self) -> None:
        self._writer.close()


class VideoWriter:
    """Factory wrapper that selects the canonical writer for an output path."""

    @staticmethod
    def builder(
        filename: str,
        height: int,
        width: int,
        fps: float,
        backend: str = "auto",
        **kwargs: Any,
    ) -> VideoWriterOpenCV | VideoWriterImageio:
        ext = Path(filename).suffix.lower()
        chosen_backend = backend.strip().lower() if backend else "auto"
        if chosen_backend == "auto":
            chosen_backend = "opencv" if ext == ".avi" else "imageio"
        if chosen_backend == "opencv":
            return VideoWriterOpenCV(filename, height, width, fps, **kwargs)
        if chosen_backend == "imageio":
            return VideoWriterImageio(filename, height, width, fps, **kwargs)
        raise ValueError(f"Unknown video writer backend: {backend}")


def write_video(
    filename: str,
    video: Any,
    frames: list[int],
    fps: int = 15,
    scale: float = 1.0,
    background: str | None = None,
    in_queue: Any | None = None,
    out_queue: Any | None = None,
    intermediate_threads: Any | None = None,
    progress_callback: Any | None = None,
    stop_event: Event | None = None,
) -> None:
    """Write selected frames from `video` into a new video file."""
    del in_queue, out_queue, intermediate_threads
    if fps <= 0:
        raise ValueError("fps must be positive")
    if not frames:
        return

    filename_path = resolve_path(filename)
    ensure_dir(filename_path.parent)

    first_frame = video.get_frame(int(frames[0]))
    prepared_first = resize_image(
        augment_background(np.expand_dims(first_frame, axis=0), background)[0],
        scale,
    )
    height, width = prepared_first.shape[:2]
    writer = VideoWriter.builder(str(filename_path), height=height, width=width, fps=float(fps))

    try:
        total = len(frames)
        for index, frame_idx in enumerate(frames, start=1):
            if stop_event is not None and stop_event.is_set():
                break
            if index == 1:
                frame = prepared_first
            else:
                raw_frame = video.get_frame(int(frame_idx))
                frame = resize_image(
                    augment_background(np.expand_dims(raw_frame, axis=0), background)[0],
                    scale,
                )
            writer.add_frame(frame, bgr=True)
            if progress_callback is not None:
                progress_callback(index, total)
    finally:
        writer.close()
