"""Optional PyAV-backed media readers."""

from __future__ import annotations

import importlib
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from xpkg.media.backends import require_media_backend

__all__ = ["PyAVVideoReader"]


class PyAVVideoReader:
    """PyAV-backed reader that preserves xpkg's numpy frame contract."""

    def __init__(self, filename: str, *, grayscale: bool = False):
        require_media_backend("pyav")
        self._av = importlib.import_module("av")
        self.filename = Path(filename).as_posix()
        self.grayscale = bool(grayscale)
        self.width = 0
        self.height = 0
        self.frames = 0
        self.fps = 30.0
        self.channels = 1 if self.grayscale else 3
        self.last_frame_idx = 0
        self._container: Any | None = None
        self._probe()
        self._container = self._open_container()

    def _open_container(self) -> Any:
        return self._av.open(self.filename)

    @staticmethod
    def _video_stream(container: Any) -> Any:
        streams = list(container.streams.video)
        if not streams:
            raise RuntimeError("PyAV could not find a video stream")
        return streams[0]

    @staticmethod
    def _fps_from_stream(stream: Any) -> float:
        rate = getattr(stream, "average_rate", None) or getattr(stream, "base_rate", None)
        if isinstance(rate, Fraction):
            value = float(rate)
        elif rate is not None:
            value = float(rate)
        else:
            value = 0.0
        return value if value > 0 else 30.0

    def _frame_to_array(self, frame: Any) -> np.ndarray:
        if self.grayscale:
            gray = frame.to_ndarray(format="gray")
            return np.asarray(gray, dtype=np.uint8)[..., np.newaxis]
        return np.asarray(frame.to_ndarray(format="bgr24"), dtype=np.uint8)

    def _probe(self) -> None:
        container = self._open_container()
        try:
            stream = self._video_stream(container)
            declared_frames = int(getattr(stream, "frames", 0) or 0)
            self.fps = self._fps_from_stream(stream)
            first_frame: np.ndarray | None = None
            decoded_count = 0
            for frame in container.decode(stream):
                decoded_count += 1
                if first_frame is None:
                    first_frame = self._frame_to_array(frame)
                if declared_frames > 0:
                    break
            if first_frame is None:
                raise RuntimeError(f"Cannot decode video: {self.filename}")
            self.height, self.width = first_frame.shape[:2]
            self.frames = declared_frames if declared_frames > 0 else decoded_count
            self.last_frame_idx = max(0, self.frames - 1)
        finally:
            container.close()

    def get_frame(self, idx: int, *, approximate: bool = False) -> np.ndarray:
        """Return one frame by zero-based index."""
        del approximate
        if idx < 0 or idx > self.last_frame_idx:
            raise IndexError(f"Frame index out of range: {idx}")

        container = self._open_container()
        try:
            stream = self._video_stream(container)
            for frame_index, frame in enumerate(container.decode(stream)):
                if frame_index == idx:
                    return self._frame_to_array(frame)
        finally:
            container.close()
        raise IndexError(f"Frame index out of range: {idx}")

    def iter_frames(self):
        """Yield frames sequentially from the PyAV decoder."""
        container = self._open_container()
        try:
            stream = self._video_stream(container)
            for frame in container.decode(stream):
                yield self._frame_to_array(frame)
        finally:
            container.close()

    def close(self) -> None:
        if self._container is not None:
            self._container.close()
            self._container = None
