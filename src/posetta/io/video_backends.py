"""Video backends for posetta."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Iterator
from importlib import import_module
from threading import Lock
from typing import Any, Protocol

import cv2
import numpy as np

from posetta.config import settings
from posetta.core.colors import bgr_to_gray, rgb_to_bgr
from posetta.core.logging_utils import get_logger
from posetta.io.images import read_bgr as _read_bgr

logger = get_logger(__name__)


def _require_pyav() -> Any:
    """Ensure PyAV is installed and return the ``av`` module."""
    if importlib.util.find_spec("av") is None:
        raise ImportError("PyAV (av) is not installed")
    return import_module("av")


def _open_pyav_container(filename: str):
    """Open a PyAV container using runtime ffmpeg settings."""
    av = _require_pyav()
    cfg = settings.video.ffmpeg
    options: dict[str, str] = {}
    hwaccel = str(cfg.hwaccel_playback or "").strip()
    if hwaccel and hwaccel.lower() != "auto":
        options["hwaccel"] = hwaccel
        if hwaccel.lower() == "vaapi" and cfg.vaapi_device:
            options["vaapi_device"] = str(cfg.vaapi_device)
    timeout = float(cfg.timeout_sec) if cfg.timeout_sec else None
    if options:
        return av.open(filename, options=options, timeout=timeout)
    if timeout is not None:
        return av.open(filename, timeout=timeout)
    return av.open(filename)


class VideoBackend(Protocol):
    """Protocol for video playback backends."""

    @property
    def width(self) -> int: ...

    @property
    def height(self) -> int: ...

    @property
    def frames(self) -> int: ...

    @property
    def fps(self) -> float: ...

    @property
    def channels(self) -> int: ...

    def get_frame(self, idx: int) -> np.ndarray:
        """Return frame at index `idx`."""
        ...

    def iter_frames(self) -> Iterator[np.ndarray]:
        """Yield frames sequentially."""
        ...

    def close(self) -> None:
        """Release resources."""
        ...


class OpenCVBackend:
    """OpenCV-based video backend.

    Notes:
    - `frames` is sourced from `CAP_PROP_FRAME_COUNT`, which can be inaccurate on
      some compressed/VFR streams.
    - `get_frame(idx)` uses sequential decode for exact frame selection instead of
      keyframe-based random seeking.
    """

    def __init__(self, filename: str, grayscale: bool = False):
        from pathlib import Path

        self._lock = Lock()
        self.filename = filename
        self.grayscale = grayscale

        path = Path(filename)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {filename}")
        if not path.is_file():
            raise ValueError(f"Not a file: {filename}")

        with self._lock:
            self._cap = cv2.VideoCapture(filename)

        if self._cap is None or not self._cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {filename}")

        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._frames = n if n > 0 else 1
        fps_val = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self._fps = fps_val if fps_val > 0 else 30.0
        self._fourcc = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        self._next_frame_idx = 0

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = self._cap.read()
        if not ok or frame is None:
            fourcc_str = self._fourcc_to_str(self._fourcc)
            self._cap.release()
            self._cap = None
            raise RuntimeError(
                f"Cannot decode video: {filename}\n"
                f"Container opened but codec failed. FourCC: {fourcc_str}\n"
                f"Ensure required codec is installed (H.264, MJPG typically work)."
            )

        self._height, self._width = frame.shape[:2]

        self._frame0_cache: np.ndarray | None = frame if self._frames == 1 else None

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._next_frame_idx = 0

    @staticmethod
    def _fourcc_to_str(code: int) -> str:
        """Convert FourCC int to readable string."""
        if code <= 0:
            return "unknown"
        chars = [chr((code >> (8 * i)) & 0xFF) for i in range(4)]
        return "".join(c if c.isprintable() else "?" for c in chars)

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

    def get_frame(self, idx: int) -> np.ndarray:
        if idx < 0:
            raise IndexError("Frame index out of range")

        if idx == 0 and self._frame0_cache is not None:
            frame = self._frame0_cache
        elif self._cap:
            with self._lock:
                if self._cap is None:
                    raise RuntimeError("Backend closed during get_frame")

                if idx < self._next_frame_idx:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self._next_frame_idx = 0

                frame = None
                while self._next_frame_idx <= idx:
                    ok, read_frame = self._cap.read()
                    if not ok or read_frame is None:
                        raise IndexError(f"Frame index out of range: {idx}")
                    current_idx = self._next_frame_idx
                    self._next_frame_idx += 1
                    if current_idx == idx:
                        frame = read_frame
                        break

            if frame is None:
                raise RuntimeError(f"Failed to read frame {idx} from {self.filename}")
        else:
            raise RuntimeError("Backend closed")

        if self.grayscale and frame.ndim == 3:
            frame = bgr_to_gray(frame)
            frame = frame[..., np.newaxis]

        return frame

    def iter_frames(self) -> Iterator[np.ndarray]:
        """Yield frames sequentially from the capture."""
        with self._lock:
            if self._cap is None:
                raise RuntimeError("Backend closed")
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._next_frame_idx = 0

        while True:
            with self._lock:
                if self._cap is None:
                    raise RuntimeError("Backend closed")
                ok, frame = self._cap.read()
            if not ok or frame is None:
                break
            self._next_frame_idx += 1
            if self.grayscale and frame.ndim == 3:
                frame = bgr_to_gray(frame)
                frame = frame[..., np.newaxis]
            yield frame

    def close(self) -> None:
        with self._lock:
            if self._cap:
                self._cap.release()
                self._cap = None
            self._next_frame_idx = 0


class PyAVBackend:
    """PyAV-based video backend.

    Provides better codec support and more accurate seeking than OpenCV.
    Returns BGR frames to match OpenCV backend convention.
    """

    def __init__(
        self,
        filename: str,
        grayscale: bool = False,
        *,
        container: Any | None = None,
        shared_lock: Lock | None = None,
        release_callback: Callable[[], None] | None = None,
    ):
        from pathlib import Path

        _require_pyav()

        self._lock = shared_lock or Lock()
        self.filename = filename
        self.grayscale = grayscale
        self._release_cb = release_callback
        self._frame0_cache: np.ndarray | None = None
        self._owns_container = container is None

        path = Path(filename)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {filename}")
        if not path.is_file():
            raise ValueError(f"Not a file: {filename}")

        self._container = container or _open_pyav_container(filename)
        self._stream = self._container.streams.video[0]

        with self._lock:
            self._width = self._stream.width
            self._height = self._stream.height
            self._frames = self._stream.frames or self._estimate_frame_count()
            self._fps = float(self._stream.average_rate or self._stream.base_rate or 30.0)

            self._container.seek(0, stream=self._stream)
            frame = next(self._container.decode(self._stream), None)
            if frame is None:
                codec_name = (
                    self._stream.codec_context.name if self._stream.codec_context else "unknown"
                )
                if self._owns_container:
                    self._container.close()
                raise RuntimeError(
                    f"Cannot decode video: {filename}\n"
                    f"Container opened but codec failed. Codec: {codec_name}\n"
                    f"Ensure required codec is installed."
                )

            self._width = frame.width
            self._height = frame.height

            if self._frames == 1:
                self._frame0_cache = rgb_to_bgr(frame.to_ndarray(format="rgb24"))

            self._container.seek(0, stream=self._stream)

    def _estimate_frame_count(self) -> int:
        """Estimate frame count from duration when not in container metadata."""
        if self._stream is None:
            raise RuntimeError(f"Cannot estimate frame count for {self.filename}: stream is None.")

        duration = self._stream.duration
        time_base = self._stream.time_base
        if duration is not None and time_base is not None:
            duration_sec = float(duration * time_base)
            fps = float(self._stream.average_rate or self._stream.base_rate or 30.0)
            return max(1, int(duration_sec * fps))

        if self._container is None:
            raise RuntimeError(
                f"Cannot estimate frame count for {self.filename}: container is None."
            )

        self._container.seek(0, stream=self._stream)
        count = 0
        for _frame in self._container.decode(self._stream):
            count += 1
        return max(1, count)

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

    def get_frame(self, idx: int, *, approximate: bool = False) -> np.ndarray:
        if idx == 0 and self._frame0_cache is not None:
            frame_bgr = self._frame0_cache
        else:
            with self._lock:
                if self._container is None:
                    raise RuntimeError("Backend closed")

                target_pts = 0
                if self._stream is None:
                    raise RuntimeError("Stream is None")

                if self._frames > 1 and self._stream.duration:
                    target_pts = int(idx * self._stream.duration / self._frames)

                if approximate:
                    self._container.seek(
                        target_pts,
                        stream=self._stream,
                        any_frame=False,
                        backward=True,
                    )
                    frame = next(self._container.decode(self._stream), None)
                else:
                    self._container.seek(target_pts, stream=self._stream)

                    frame = None
                    for f in self._container.decode(self._stream):
                        frame = f

                        if self._stream.duration and self._frames > 1 and f.pts is not None:
                            current_idx = int(f.pts * self._frames / self._stream.duration)
                            if current_idx >= idx:
                                break
                        else:
                            break

                if frame is None:
                    raise RuntimeError(f"Failed to read frame {idx} from {self.filename}")

                frame_bgr = rgb_to_bgr(frame.to_ndarray(format="rgb24"))

        if self.grayscale:
            frame_bgr = bgr_to_gray(frame_bgr)
            frame_bgr = frame_bgr[..., np.newaxis]

        return frame_bgr

    def iter_frames(self) -> Iterator[np.ndarray]:
        """Yield frames sequentially from the container."""
        with self._lock:
            if self._container is None or self._stream is None:
                raise RuntimeError("Backend closed")
            self._container.seek(0, stream=self._stream)
            for frame in self._container.decode(self._stream):
                frame_bgr = rgb_to_bgr(frame.to_ndarray(format="rgb24"))
                if self.grayscale:
                    frame_bgr = bgr_to_gray(frame_bgr)
                    frame_bgr = frame_bgr[..., np.newaxis]
                yield frame_bgr

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
            self._release_cb = None
        if release_cb is not None and not self._owns_container:
            release_cb()


class ImageSequenceBackend:
    """Backend for a sequence of image files."""

    def __init__(self, filenames: list[str], grayscale: bool = False):
        self.filenames = filenames
        self.grayscale = grayscale

        first = _read_bgr(filenames[0])
        if first is None:
            raise FileNotFoundError(f"Cannot read image: {filenames[0]}")
        self._height, self._width = first.shape[:2]
        self._frames = len(filenames)
        self._channels = 1 if grayscale else (1 if first.ndim == 2 else first.shape[-1])
        self._fps = 30.0

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
        return self._channels

    def get_frame(self, idx: int) -> np.ndarray:
        if idx < 0 or idx >= len(self.filenames):
            raise IndexError("Frame index out of range")

        frame = _read_bgr(self.filenames[idx])

        if frame is None:
            raise RuntimeError(f"Failed to read image frame {idx}")

        if self.grayscale and frame.ndim == 3:
            frame = bgr_to_gray(frame)
            frame = frame[..., np.newaxis]
        return frame

    def iter_frames(self) -> Iterator[np.ndarray]:
        """Yield frames sequentially from the image list."""
        for idx in range(len(self.filenames)):
            frame = _read_bgr(self.filenames[idx])
            if frame is None:
                raise RuntimeError(f"Failed to read image frame {idx}")
            if self.grayscale and frame.ndim == 3:
                frame = bgr_to_gray(frame)
                frame = frame[..., np.newaxis]
            yield frame

    def close(self) -> None:
        return
