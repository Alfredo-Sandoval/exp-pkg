"""Video wrappers and writer utilities used by labels and converters."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock
from typing import Any

import cv2
import imageio.v2 as iio
import numpy as np
from cattrs import Converter

from xpkg._core.colors import bgr_to_gray, bgr_to_rgb, ensure_bgr, ensure_three_channels
from xpkg._core.logging_utils import get_logger
from xpkg._core.path_registry import ensure_dir, resolve_path
from xpkg.media.images import read_bgr

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
    "VideoWriter",
    "VideoWriterImageio",
    "VideoWriterOpenCV",
    "augment_background",
    "available_video_exts",
    "gui_playback_backend_for_path",
    "resize_image",
    "resize_images",
    "write_video",
]


class SingleImageVideo:
    """Supported single-image extensions treated as one-frame videos."""

    EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _video_writer_fourcc(code: str) -> int:
    fourcc_fn = getattr(cv2, "VideoWriter_fourcc", None)
    if not callable(fourcc_fn):
        raise RuntimeError("OpenCV build does not expose VideoWriter_fourcc")
    return int(fourcc_fn(code[0], code[1], code[2], code[3]))


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
    choice = backend.strip().lower() if backend else "auto"
    if choice == "auto":
        return "opencv"
    if choice != "opencv":
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
        del approximate
        if idx < 0 or idx > self.last_frame_idx:
            raise IndexError(f"Frame index out of range: {idx}")

        with self._lock:
            if self._image_filenames:
                return _load_image_sequence_frame(
                    self._image_filenames[idx],
                    grayscale=self.grayscale,
                )

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


def resize_image(image: np.ndarray, scale: float) -> np.ndarray:
    """Resize one image by a scale factor."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    if scale == 1.0:
        return image
    height, width = image.shape[:2]
    new_height = max(1, int(round(height * scale)))
    new_width = max(1, int(round(width * scale)))
    if image.ndim == 3 and image.shape[2] == 1:
        resized = cv2.resize(image.squeeze(-1), (new_width, new_height))
        return resized[..., np.newaxis]
    return cv2.resize(image, (new_width, new_height))


def resize_images(images: np.ndarray, scale: float) -> np.ndarray:
    """Resize each image in `images` by `scale`."""
    if scale == 1.0:
        return images
    return np.stack([resize_image(image, scale) for image in images], axis=0)


def augment_background(images: np.ndarray, background: str | None) -> np.ndarray:
    """Fill with a solid color or keep original pixels."""
    if background is None or background == "original":
        return images

    fill_values = {"black": 0, "grey": 127, "white": 255}
    if background not in fill_values:
        valid = ", ".join(fill_values)
        raise ValueError(f"Invalid background color: {background}. Options include: {valid}")
    return np.full_like(images, fill_values[background])


def write_video(
    filename: str,
    video: Video,
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
