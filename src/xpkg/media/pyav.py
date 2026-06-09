"""Optional PyAV-backed media readers."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import numpy as np

from xpkg._core.colors import bgr_to_gray, rgb_to_bgr
from xpkg.media.backends import require_media_backend

__all__ = ["PyAVCursorState", "PyAVVideoReader", "open_pyav_container", "require_pyav"]


@dataclass(slots=True)
class PyAVCursorState:
    next_exact_idx: int = 0
    exact_cursor_valid: bool = True


def require_pyav() -> Any:
    """Ensure PyAV is installed and return the module."""
    require_media_backend("pyav")
    return importlib.import_module("av")


def open_pyav_container(filename: str) -> Any:
    """Open a PyAV container for the requested file."""
    return require_pyav().open(filename)


class PyAVVideoReader:
    """PyAV-backed reader that preserves xpkg's numpy frame contract."""

    _FORWARD_SEEK_THRESHOLD = 100

    def __init__(
        self,
        filename: str,
        *,
        grayscale: bool = False,
        container: Any | None = None,
        shared_lock: Lock | None = None,
        cursor_state: PyAVCursorState | None = None,
        release_callback: Any | None = None,
    ):
        self._lock = shared_lock or Lock()
        self.filename = Path(filename).as_posix()
        self.grayscale = bool(grayscale)
        self._release_cb = release_callback
        self._owns_container = container is None
        self._cursor_state = cursor_state or PyAVCursorState()
        self._frame_count_exact = False
        self._frame0_cache: np.ndarray | None = None
        self._seek_prefetched_frame: Any | None = None

        path = Path(filename)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {filename}")
        if not path.is_file():
            raise ValueError(f"Not a file: {filename}")

        self._container = container or open_pyav_container(filename)
        self._stream = self._video_stream(self._container)
        with self._lock:
            self._width = self._stream.width
            self._height = self._stream.height
            stream_frame_count = int(getattr(self._stream, "frames", 0) or 0)
            if stream_frame_count > 0:
                self._frames = stream_frame_count
                self._frame_count_exact = True
            else:
                self._frames = self._estimate_frame_count()
            self._fps = self._fps_from_stream(self._stream)
            self._container.seek(0, stream=self._stream)
            frame = next(self._container.decode(self._stream), None)
            if frame is None:
                if self._owns_container:
                    self._container.close()
                raise RuntimeError(f"Cannot decode video: {filename}")
            self._width = frame.width
            self._height = frame.height
            if self._frames == 1:
                self._frame0_cache = rgb_to_bgr(frame.to_ndarray(format="rgb24"))
            self._container.seek(0, stream=self._stream)
            self._cursor_state.next_exact_idx = 0
            self._cursor_state.exact_cursor_valid = True

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

    def _estimate_frame_count(self) -> int:
        fps = self._fps_from_stream(self._stream)
        duration = getattr(self._stream, "duration", None)
        time_base = getattr(self._stream, "time_base", None)
        if duration is not None and time_base is not None:
            return max(1, int(float(duration * time_base) * fps))
        container_duration = getattr(self._container, "duration", None)
        if container_duration is not None:
            av = require_pyav()
            return max(1, int(float(container_duration) / float(av.time_base) * fps))
        cap = cv2.VideoCapture(self.filename)
        try:
            if cap is not None and cap.isOpened():
                raw_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if raw_count > 0:
                    return raw_count
        finally:
            if cap is not None:
                cap.release()
        return 1

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def frames(self) -> int:
        return self._frames

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def channels(self) -> int:
        return 1 if self.grayscale else 3

    @property
    def last_frame_idx(self) -> int:
        return max(0, self._frames - 1)

    def _validate_frame_index(self, idx: int) -> None:
        if idx < 0:
            raise IndexError("Frame index out of range")
        if self._frame_count_exact and idx >= self._frames:
            raise IndexError(f"Frame index out of range: {idx}")

    def _target_pts_for_index(self, idx: int) -> int:
        stream = self._stream
        if stream is None:
            raise RuntimeError("Stream is None")
        if self._frames <= 1 or not stream.duration:
            return 0
        return int(idx * stream.duration / self._frames)

    def _reset_exact_cursor(self) -> None:
        if self._container is None or self._stream is None:
            raise RuntimeError("Backend closed")
        self._container.seek(0, stream=self._stream)
        self._cursor_state.next_exact_idx = 0
        self._cursor_state.exact_cursor_valid = True

    def _seek_exact_cursor_near(self, idx: int) -> None:
        if self._container is None or self._stream is None:
            raise RuntimeError("Backend closed")
        self._container.seek(
            self._target_pts_for_index(idx),
            stream=self._stream,
            any_frame=False,
            backward=True,
        )
        first = next(self._container.decode(self._stream), None)
        if first is None:
            self._reset_exact_cursor()
            return
        if first.pts is not None and self._stream.duration and self._frames > 1:
            landed_idx = round(first.pts * self._frames / self._stream.duration)
            landed_idx = max(0, min(landed_idx, self._frames - 1))
        else:
            landed_idx = 0
        if landed_idx > idx:
            self._reset_exact_cursor()
            return
        self._cursor_state.next_exact_idx = landed_idx
        self._cursor_state.exact_cursor_valid = True
        self._seek_prefetched_frame = first

    def _should_keyframe_seek(self, idx: int) -> bool:
        if not self._cursor_state.exact_cursor_valid:
            return True
        if idx < self._cursor_state.next_exact_idx:
            return True
        gap = idx - self._cursor_state.next_exact_idx
        return gap > self._FORWARD_SEEK_THRESHOLD

    def _decode_exact_frame(self, idx: int) -> Any:
        if self._container is None or self._stream is None:
            raise RuntimeError("Backend closed")
        if self._should_keyframe_seek(idx):
            if idx == 0:
                self._reset_exact_cursor()
            else:
                self._seek_exact_cursor_near(idx)
                prefetched = self._seek_prefetched_frame
                self._seek_prefetched_frame = None
                if prefetched is not None:
                    current_idx = self._cursor_state.next_exact_idx
                    self._cursor_state.next_exact_idx += 1
                    if current_idx == idx:
                        if not self._frame_count_exact and idx >= self._frames:
                            self._frames = idx + 1
                        return prefetched
        for frame in self._container.decode(self._stream):
            current_idx = self._cursor_state.next_exact_idx
            self._cursor_state.next_exact_idx += 1
            if current_idx == idx:
                if not self._frame_count_exact and idx >= self._frames:
                    self._frames = idx + 1
                return frame
        if not self._frame_count_exact:
            self._frames = max(1, self._cursor_state.next_exact_idx)
            self._frame_count_exact = True
            raise IndexError(f"Frame index out of range: {idx}")
        raise RuntimeError(f"Failed to decode frame {idx} from {self.filename}")

    def _decode_approximate_frame(self, idx: int) -> Any:
        if self._container is None or self._stream is None:
            raise RuntimeError("Backend closed")
        self._cursor_state.exact_cursor_valid = False
        self._container.seek(
            self._target_pts_for_index(idx),
            stream=self._stream,
            any_frame=False,
            backward=True,
        )
        frame = next(self._container.decode(self._stream), None)
        if frame is None:
            raise RuntimeError(f"Failed to read frame {idx} from {self.filename}")
        return frame

    def _to_output_channels(self, frame_bgr: np.ndarray) -> np.ndarray:
        if not self.grayscale:
            return frame_bgr
        return bgr_to_gray(frame_bgr)[..., np.newaxis]

    def _frame_to_bgr(self, frame: Any) -> np.ndarray:
        return rgb_to_bgr(frame.to_ndarray(format="rgb24"))

    def _decode_frame_bgr(self, idx: int, *, approximate: bool) -> np.ndarray:
        with self._lock:
            if approximate:
                frame = self._decode_approximate_frame(idx)
            else:
                frame = self._decode_exact_frame(idx)
            return self._frame_to_bgr(frame)

    def get_frame(self, idx: int, *, approximate: bool = False) -> np.ndarray:
        """Return one frame by zero-based index."""
        self._validate_frame_index(idx)
        if idx == 0 and self._frame0_cache is not None:
            frame_bgr = self._frame0_cache
        else:
            frame_bgr = self._decode_frame_bgr(idx, approximate=approximate)
            if not self._frame_count_exact and idx >= self._frames:
                self._frames = idx + 1
        return self._to_output_channels(frame_bgr)

    def _iter_exact_frame_items(self):
        with self._lock:
            if self._container is None or self._stream is None:
                raise RuntimeError("Backend closed")
            self._reset_exact_cursor()
            for frame in self._container.decode(self._stream):
                frame_idx = self._cursor_state.next_exact_idx
                self._cursor_state.next_exact_idx += 1
                yield frame_idx, frame
            self._frames = max(1, self._cursor_state.next_exact_idx)
            self._frame_count_exact = True

    def iter_frames(self):
        """Yield frames sequentially from the PyAV decoder."""
        for _frame_idx, frame in self._iter_exact_frame_items():
            yield self._to_output_channels(self._frame_to_bgr(frame))

    def iter_frames_stride(self, stride: int):
        """Yield exact sampled frames without materializing skipped output frames."""
        stride_value = int(stride)
        if stride_value <= 0:
            raise ValueError("stride must be > 0")
        if stride_value == 1:
            yield from self.iter_frames()
            return
        for frame_idx, frame in self._iter_exact_frame_items():
            if frame_idx % stride_value == 0:
                yield self._to_output_channels(self._frame_to_bgr(frame))

    def iter_frame_batches_stride(self, batch_size: int, stride: int):
        """Yield strided frames in Python list batches."""
        batch_size_value = int(batch_size)
        if batch_size_value <= 0:
            raise ValueError("batch_size must be > 0")
        batch: list[np.ndarray] = []
        for frame in self.iter_frames_stride(stride):
            batch.append(frame)
            if len(batch) == batch_size_value:
                yield batch
                batch = []
        if batch:
            yield batch

    def close(self) -> None:
        release_cb = self._release_cb
        with self._lock:
            if self._container is None:
                return
            if self._owns_container:
                self._container.close()
            self._container = None
            self._stream = None
            self._frame0_cache = None
            self._cursor_state.next_exact_idx = 0
            self._cursor_state.exact_cursor_valid = False
            self._release_cb = None
        if release_cb is not None and not self._owns_container:
            release_cb()
