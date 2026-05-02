"""Video readers for files, single images, and image sequences."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import numpy as np
from cattrs import Converter

from xpkg._core.colors import bgr_to_gray, bgr_to_rgb
from xpkg._core.logging_utils import get_logger
from xpkg._core.path_registry import resolve_path
from xpkg.media.images import read_bgr
from xpkg.media.pyav import PyAVVideoReader

logger = get_logger(__name__)

_RECOGNIZED_VIDEO_EXTS = (
    ".avi",
    ".mov",
    ".mp4",
    ".mkv",
    ".mpeg",
    ".mpg",
    ".wmv",
    ".m4v",
    ".webm",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
)

__all__ = [
    "SingleImageVideo",
    "Video",
    "VideoReader",
    "available_video_exts",
    "gui_playback_backend_for_path",
]


class SingleImageVideo:
    """Supported single-image extensions treated as one-frame videos."""

    EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def available_video_exts() -> list[str]:
    """Return normalized extensions treated as video-like media."""
    exts: list[str] = []
    for ext in _RECOGNIZED_VIDEO_EXTS:
        normalized = str(ext).strip().lower()
        if not normalized:
            continue
        if not normalized.startswith("."):
            normalized = f".{normalized}"
        exts.append(normalized)
    return list(dict.fromkeys(exts))


def gui_playback_backend_for_path(filename: str) -> str:
    """Return the canonical xpkg playback backend for a path."""
    ext = Path(filename).suffix.lower()
    if ext in SingleImageVideo.EXTS:
        return "images"
    return "opencv"


def _normalize_backend(backend: str, *, image_sequence: bool) -> str:
    if image_sequence:
        return "images"
    choice = backend.strip().lower().replace("_", "-") if backend else "auto"
    aliases = {"av": "pyav"}
    choice = aliases.get(choice, choice)
    if choice == "auto":
        return "opencv"
    if choice not in {"opencv", "pyav"}:
        raise ValueError(f"Unknown video backend: {backend}")
    return choice


def _load_image_sequence_frame(filename: str, *, grayscale: bool) -> np.ndarray:
    frame = read_bgr(filename)
    if frame is None:
        raise FileNotFoundError(f"Image file not found or unreadable: {filename}")
    if grayscale:
        gray = bgr_to_gray(frame)
        return gray[..., np.newaxis]
    return frame


class Video:
    """Frame reader for file-backed videos and ordered image sequences."""

    def __init__(
        self,
        filename: str | None = None,
        image_filenames: list[str] | None = None,
        grayscale: bool = False,
        backend: str = "auto",
    ):
        self._lock = Lock()
        self.filename: str | None = None
        self.id: str | None = None
        self.label: str | None = None
        self.sha256: str | None = None
        self._image_filenames: list[str] = []
        self.grayscale = bool(grayscale)
        self.backend = _normalize_backend(backend, image_sequence=image_filenames is not None)
        self.width = 0
        self.height = 0
        self.frames = 0
        self.fps = 0.0
        self.channels = 1 if self.grayscale else 3
        self.last_frame_idx = 0
        self._capture: cv2.VideoCapture | None = None
        self._pyav_reader: PyAVVideoReader | None = None

        if filename is not None:
            self._init_media(filename)
        elif image_filenames:
            self._init_image_sequence(image_filenames)
        else:
            raise ValueError("Video requires a filename or image filenames")

        self.last_frame_idx = max(0, self.frames - 1)

    @property
    def image_filenames(self) -> list[str]:
        return list(self._image_filenames)

    def _init_media(self, filename: str) -> None:
        path = resolve_path(filename)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")
        if not path.is_file():
            raise ValueError(f"Not a file: {path}")

        self.filename = path.as_posix()
        if self.backend == "pyav":
            self._init_pyav_media()
        else:
            self._init_opencv_media()

    def _init_opencv_media(self) -> None:
        if self.filename is None:
            raise RuntimeError("Cannot initialize OpenCV media without a filename")
        self.backend = "opencv"
        self._capture = cv2.VideoCapture(self.filename)
        if self._capture is None or not self._capture.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.filename}")

        ok, frame = self._capture.read()
        if not ok or frame is None:
            self.close()
            raise RuntimeError(f"Cannot decode video: {self.filename}")

        self.height, self.width = frame.shape[:2]
        frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frames = frame_count if frame_count > 0 else 1
        fps_val = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self.fps = fps_val if fps_val > 0 else 30.0
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def _init_pyav_media(self) -> None:
        if self.filename is None:
            raise RuntimeError("Cannot initialize PyAV media without a filename")
        self.backend = "pyav"
        self._pyav_reader = PyAVVideoReader(self.filename, grayscale=self.grayscale)
        self.height = self._pyav_reader.height
        self.width = self._pyav_reader.width
        self.frames = self._pyav_reader.frames
        self.fps = self._pyav_reader.fps
        self.channels = self._pyav_reader.channels

    def _init_image_sequence(self, image_filenames: list[str]) -> None:
        filenames = [resolve_path(name).as_posix() for name in image_filenames]
        if not filenames:
            raise ValueError("Image sequence is empty")

        first_frame = _load_image_sequence_frame(filenames[0], grayscale=self.grayscale)
        self.filename = None
        self._image_filenames = filenames
        self.backend = "images"
        self.height, self.width = first_frame.shape[:2]
        self.frames = len(filenames)
        self.fps = 1.0
        self.channels = 1 if self.grayscale else 3

    def _ensure_capture(self) -> cv2.VideoCapture:
        if self.filename is None:
            raise RuntimeError("Image-sequence videos do not use cv2 capture handles")
        if self._capture is None:
            self._capture = cv2.VideoCapture(self.filename)
            if self._capture is None or not self._capture.isOpened():
                raise FileNotFoundError(f"Cannot re-open video: {self.filename}")
        return self._capture

    def __len__(self) -> int:
        return self.frames

    def get_frame(self, idx: int, *, approximate: bool = False) -> np.ndarray:
        """Load the frame at `idx` and return it in BGR or grayscale format."""
        if idx < 0 or idx > self.last_frame_idx:
            raise IndexError(f"Frame index out of range: {idx}")

        with self._lock:
            if self._image_filenames:
                return _load_image_sequence_frame(
                    self._image_filenames[idx],
                    grayscale=self.grayscale,
                )
            if self._pyav_reader is not None:
                return self._pyav_reader.get_frame(idx, approximate=approximate)

            capture = self._ensure_capture()
            capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise IndexError(f"Frame index out of range: {idx}")
            if self.grayscale:
                gray = bgr_to_gray(frame)
                return gray[..., np.newaxis]
            return frame

    def get_frames_safely(self, indices: list[int]) -> tuple[list[int], np.ndarray]:
        """Batch frame getter used by converter pipelines."""
        frames: list[np.ndarray] = []
        loaded: list[int] = []
        for idx in indices:
            frame = self.get_frame(idx)
            frames.append(frame)
            loaded.append(idx)
        if not frames:
            return [], np.empty((0, self.height, self.width, self.channels), dtype=np.uint8)
        return loaded, np.stack(frames, axis=0)

    def iter_frames(self) -> Iterator[np.ndarray]:
        """Yield frames sequentially from the underlying media."""
        if self._image_filenames:
            for idx in range(self.frames):
                yield self.get_frame(idx)
            return
        if self._pyav_reader is not None:
            yield from self._pyav_reader.iter_frames()
            return

        with self._lock:
            capture = self._ensure_capture()
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while True:
            with self._lock:
                assert self._capture is not None
                ok, frame = self._capture.read()
            if not ok or frame is None:
                break
            if self.grayscale:
                gray = bgr_to_gray(frame)
                yield gray[..., np.newaxis]
            else:
                yield frame

    def close(self) -> None:
        with self._lock:
            if self._capture is not None:
                self._capture.release()
                self._capture = None
            if self._pyav_reader is not None:
                self._pyav_reader.close()
                self._pyav_reader = None

    @classmethod
    def from_media(
        cls,
        filename: str,
        grayscale: bool | None = None,
        backend: str = "auto",
    ) -> Video:
        return cls(
            filename=filename,
            grayscale=bool(grayscale) if grayscale is not None else False,
            backend=backend,
        )

    @classmethod
    def from_image_filenames(
        cls,
        filenames: list[str],
        grayscale: bool | None = None,
        backend: str = "images",
    ) -> Video:
        return cls(
            filename=None,
            image_filenames=filenames,
            grayscale=bool(grayscale) if grayscale is not None else False,
            backend=backend,
        )

    @classmethod
    def from_filename(cls, filename: str, **kwargs: Any) -> Video:
        ext = Path(filename).suffix.lower()
        if ext in SingleImageVideo.EXTS:
            return cls.from_image_filenames(
                [filename],
                grayscale=kwargs.get("grayscale"),
            )
        return cls.from_media(
            filename,
            grayscale=kwargs.get("grayscale"),
            backend=str(kwargs.get("backend", "auto")),
        )

    @staticmethod
    def cattr() -> Converter:
        conv = Converter()

        def _unstructure(video: Video) -> dict[str, Any]:
            if video.filename is not None:
                return {
                    "type": "media",
                    "filename": video.filename,
                    "width": video.width,
                    "height": video.height,
                    "frames": video.frames,
                }
            return {
                "type": "images",
                "filenames": list(video.image_filenames),
                "width": video.width,
                "height": video.height,
                "frames": video.frames,
            }

        def _structure(data: dict[str, Any], _type: Any) -> Video:
            if data.get("type") == "images" or data.get("filenames"):
                return Video.from_image_filenames(list(data["filenames"]))
            filename = data.get("filename")
            if not isinstance(filename, str) or not filename:
                raise ValueError("Missing filename for media video")
            return Video.from_filename(filename)

        conv.register_unstructure_hook(Video, _unstructure)
        conv.register_structure_hook(Video, _structure)
        return conv

    def __enter__(self) -> Video:
        return self

    def __exit__(self, _exc_type: Any, _exc_value: Any, _traceback: Any) -> None:
        self.close()


@dataclass
class VideoReader:
    """Iterate over frames from a video path in BGR or RGB channel order."""

    path: str
    color: str = "bgr"

    def __post_init__(self) -> None:
        self._video = Video.from_filename(self.path)
        self._is_rgb = self.color.lower() == "rgb"

    def __iter__(self) -> Iterator[np.ndarray]:
        for frame in self._video.iter_frames():
            if self._is_rgb:
                yield bgr_to_rgb(frame)
            else:
                yield frame

    @property
    def frames(self) -> int:
        return self._video.frames

    @property
    def fps(self) -> float:
        return self._video.fps

    @property
    def width(self) -> int:
        return self._video.width

    @property
    def height(self) -> int:
        return self._video.height

    def get_frame(self, index: int) -> np.ndarray:
        frame = self._video.get_frame(index)
        if self._is_rgb:
            return bgr_to_rgb(frame)
        return frame

    def close(self) -> None:
        self._video.close()

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
