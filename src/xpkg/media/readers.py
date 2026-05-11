"""Video readers for files, single images, and image sequences."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from cattrs import Converter

from xpkg.media.backend_utils import normalize_file_video_backend
from xpkg.media.pyav import PyAVCursorState
from xpkg.media.reader_backends import (
    DecordGpuBackend,
    ImageSequenceBackend,
    OpenCVBackend,
    PyAVBackend,
    VideoBackend,
)

from .._core.colors import bgr_to_rgb
from .._core.path_registry import resolve_path

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
    "PyAVVideoResource",
    "SingleImageVideo",
    "Video",
    "VideoReader",
    "available_video_exts",
    "gui_playback_backend_for_path",
]


class SingleImageVideo:
    """Supported single-image extensions treated as one-frame videos."""

    EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True, slots=True)
class PyAVVideoResource:
    """Borrowed PyAV container state supplied by an application-level owner."""

    container: Any
    lock: Any
    cursor_state: PyAVCursorState
    release_callback: Callable[[], None] | None = None


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


def _image_sequence_dir_filenames(path: Path) -> list[str]:
    filenames = sorted(
        candidate.as_posix()
        for candidate in path.iterdir()
        if candidate.is_file() and candidate.suffix.lower() in SingleImageVideo.EXTS
    )
    if not filenames:
        raise ValueError(f"Image sequence directory contains no supported image files: {path}")
    return filenames


def gui_playback_backend_for_path(filename: str) -> str:
    """Return the canonical GUI playback backend for a path."""
    path = Path(filename)
    if path.is_dir():
        return "images"
    if path.suffix.lower() in SingleImageVideo.EXTS:
        return "opencv"
    return "pyav"


class Video:
    """Frame reader for file-backed videos and ordered image sequences."""

    def __init__(
        self,
        filename: str | None = None,
        image_filenames: list[str] | None = None,
        grayscale: bool = False,
        backend: str = "auto",
        color: str = "bgr",
        pyav_resource: PyAVVideoResource | None = None,
    ):
        self._lock = Lock()
        self.filename: str | None = filename
        self.id: str | None = None
        self.label: str | None = None
        self.sha256: str | None = None
        self._image_filenames: list[str] = [
            resolve_path(name).as_posix() for name in image_filenames or []
        ]
        self.grayscale = bool(grayscale)
        self.color = str(color)
        color_value = self.color.strip().lower()
        if color_value not in {"bgr", "rgb"}:
            raise ValueError(f"Unsupported color mode: {color!r}")
        self._is_rgb = color_value == "rgb"
        if self._image_filenames and str(backend).strip().lower() == "images":
            backend_choice = "images"
        else:
            backend_choice = normalize_file_video_backend(backend, label="backend")
        if backend_choice == "auto":
            backend_choice = "opencv"
        if pyav_resource is not None and backend_choice != "pyav":
            raise ValueError("pyav_resource requires backend='pyav'")
        self.backend = backend_choice
        self._backend: VideoBackend | None = None
        self.width = 0
        self.height = 0
        self.frames = 0
        self.fps = 0.0
        self.channels = 0
        self.last_frame_idx = 0

        if self._image_filenames:
            if self.filename is not None:
                path = resolve_path(self.filename)
                if not path.exists():
                    raise FileNotFoundError(f"Video file not found: {self.filename}")
                if not path.is_dir():
                    raise ValueError(
                        "Image-sequence videos with explicit filename must point to a directory: "
                        f"{self.filename}"
                    )
                self.filename = path.as_posix()
            self._backend = ImageSequenceBackend(self._image_filenames, grayscale=self.grayscale)
            self.backend = "images"
        elif self.filename is not None:
            path = resolve_path(self.filename)
            if not path.exists():
                raise FileNotFoundError(f"Video file not found: {path}")
            if not path.is_file():
                raise ValueError(f"Not a file: {path}")
            self.filename = path.as_posix()
            self._backend = self._init_backend(
                self.filename,
                self.backend,
                self.grayscale,
                pyav_resource=pyav_resource,
            )
        else:
            raise ValueError("Video requires a filename or image filenames")

        self.width = self._backend.width
        self.height = self._backend.height
        self.frames = self._backend.frames
        self.fps = self._backend.fps
        self.channels = self._backend.channels
        self.last_frame_idx = max(0, self.frames - 1)

    @staticmethod
    def _init_backend(
        filename: str,
        backend: str,
        grayscale: bool,
        *,
        pyav_resource: PyAVVideoResource | None = None,
    ) -> VideoBackend:
        choice = normalize_file_video_backend(backend, label="backend")
        if choice == "auto":
            choice = "opencv"
        if choice == "opencv":
            return OpenCVBackend(filename, grayscale=grayscale)
        if choice == "pyav":
            if pyav_resource is not None:
                return PyAVBackend(
                    filename,
                    grayscale=grayscale,
                    container=pyav_resource.container,
                    shared_lock=pyav_resource.lock,
                    cursor_state=pyav_resource.cursor_state,
                    release_callback=pyav_resource.release_callback,
                )
            return PyAVBackend(filename, grayscale=grayscale)
        if choice == "decord-gpu":
            return DecordGpuBackend(filename, grayscale=grayscale)
        raise ValueError(f"Unknown video backend: {backend}")

    @property
    def uses_pyav(self) -> bool:
        return isinstance(self._backend, PyAVBackend)

    @property
    def image_filenames(self) -> list[str]:
        return list(self._image_filenames)

    @property
    def path(self) -> str:
        if self.filename is not None:
            return self.filename
        if self._image_filenames:
            return self._image_filenames[0]
        raise RuntimeError("Video source path is unavailable.")

    def __len__(self) -> int:
        return self.frames

    def __iter__(self) -> Iterator[np.ndarray]:
        return self.iter_frames()

    def _ensure_backend(self) -> VideoBackend:
        if self._backend is None:
            raise RuntimeError("Video backend is closed")
        return self._backend

    def _format_frame_output(self, frame: np.ndarray) -> np.ndarray:
        if self._is_rgb:
            return bgr_to_rgb(frame)
        return frame

    def get_frame(self, idx: int, *, approximate: bool = False) -> np.ndarray:
        """Load the frame at `idx` in the configured output color order."""
        if idx < 0:
            raise IndexError("Frame index out of range")
        with self._lock:
            backend = self._ensure_backend()
            if isinstance(backend, PyAVBackend):
                return self._format_frame_output(backend.get_frame(idx, approximate=approximate))
            return self._format_frame_output(backend.get_frame(idx))

    def get_frames_list(self, indices: Sequence[int]) -> list[np.ndarray]:
        """Strict frame getter that preserves original per-frame shapes."""
        return [self.get_frame(int(idx)) for idx in indices]

    def get_frames_safely(self, indices: list[int]) -> tuple[list[int], np.ndarray]:
        """Batch frame getter used by converter pipelines."""
        frames = self.get_frames_list(indices)
        loaded = [int(idx) for idx in indices]
        if not frames:
            return [], np.empty((0, self.height, self.width, self.channels or 3), dtype=np.uint8)
        first_shape = frames[0].shape
        if any(frame.shape != first_shape for frame in frames[1:]):
            raise ValueError(
                "get_frames_safely requires frames with shared shape; "
                "use get_frames_list for heterogeneous sequences"
            )
        arr = np.stack(frames, axis=0)
        if arr.ndim == 3:
            arr = arr[..., np.newaxis]
        return loaded, arr

    def iter_frames(self) -> Iterator[np.ndarray]:
        """Yield frames sequentially from the underlying media."""
        with self._lock:
            backend = self._ensure_backend()
        for frame in backend.iter_frames():
            yield self._format_frame_output(frame)

    def iter_frames_stride(self, stride: int) -> Iterator[np.ndarray]:
        """Yield every `stride`th frame sequentially."""
        stride_value = int(stride)
        if stride_value <= 0:
            raise ValueError("stride must be > 0")
        with self._lock:
            backend = self._ensure_backend()
        for frame in backend.iter_frames_stride(stride_value):
            yield self._format_frame_output(frame)

    def iter_frame_batches_stride(
        self,
        batch_size: int,
        stride: int,
    ) -> Iterator[tuple[list[int], list[np.ndarray]]]:
        """Yield strided frame batches with source frame indices."""
        batch_size_value = int(batch_size)
        if batch_size_value <= 0:
            raise ValueError("batch_size must be > 0")
        stride_value = int(stride)
        if stride_value <= 0:
            raise ValueError("stride must be > 0")
        with self._lock:
            backend = self._ensure_backend()
        source_frame_idx = 0
        for frame_batch in backend.iter_frame_batches_stride(batch_size_value, stride_value):
            formatted_batch = [self._format_frame_output(frame) for frame in frame_batch]
            frame_indices = [
                source_frame_idx + (batch_offset * stride_value)
                for batch_offset in range(len(formatted_batch))
            ]
            source_frame_idx += len(formatted_batch) * stride_value
            yield frame_indices, formatted_batch

    def iter_torch_frame_batches_stride(
        self,
        batch_size: int,
        stride: int,
        *,
        color: str | None = None,
    ) -> Iterator[tuple[list[int], Any]]:
        """Yield Decord GPU torch batches with source frame indices."""
        batch_size_value = int(batch_size)
        if batch_size_value <= 0:
            raise ValueError("batch_size must be > 0")
        stride_value = int(stride)
        if stride_value <= 0:
            raise ValueError("stride must be > 0")
        with self._lock:
            backend = self._ensure_backend()
        if not isinstance(backend, DecordGpuBackend):
            raise TypeError("Torch frame batch decode requires the decord-gpu backend")
        source_frame_idx = 0
        for frame_batch in backend.iter_torch_frame_batches_stride(
            batch_size_value,
            stride_value,
            color=self.color if color is None else color,
        ):
            batch_length = int(frame_batch.shape[0])
            frame_indices = [
                source_frame_idx + (batch_offset * stride_value)
                for batch_offset in range(batch_length)
            ]
            source_frame_idx += batch_length * stride_value
            yield frame_indices, frame_batch

    def iter_dual_frame_batches_stride(
        self,
        batch_size: int,
        stride: int,
        *,
        color: str | None = None,
    ) -> Iterator[tuple[list[int], tuple[list[np.ndarray], Any]]]:
        """Yield Decord GPU numpy and torch batches with source frame indices."""
        batch_size_value = int(batch_size)
        if batch_size_value <= 0:
            raise ValueError("batch_size must be > 0")
        stride_value = int(stride)
        if stride_value <= 0:
            raise ValueError("stride must be > 0")
        with self._lock:
            backend = self._ensure_backend()
        if not isinstance(backend, DecordGpuBackend):
            raise TypeError("Dual frame batch decode requires the decord-gpu backend")
        source_frame_idx = 0
        for pose_frames, detector_frames in backend.iter_dual_frame_batches_stride(
            batch_size_value,
            stride_value,
            color=self.color if color is None else color,
        ):
            batch_length = len(pose_frames)
            frame_indices = [
                source_frame_idx + (batch_offset * stride_value)
                for batch_offset in range(batch_length)
            ]
            source_frame_idx += batch_length * stride_value
            yield frame_indices, (pose_frames, detector_frames)

    def iter_selected_dual_frame_batches(
        self,
        frame_indices: Sequence[int],
        batch_size: int,
        *,
        color: str | None = None,
    ) -> Iterator[tuple[list[int], tuple[list[np.ndarray], Any]]]:
        """Yield selected Decord GPU dual batches preserving requested indices."""
        requested = self._normalize_selected_frame_indices(frame_indices)
        if not requested:
            return
        with self._lock:
            backend = self._ensure_backend()
        if not isinstance(backend, DecordGpuBackend):
            raise TypeError("Dual frame batch decode requires the decord-gpu backend")
        for batch_indices, pose_frames, detector_frames in backend.iter_selected_dual_frame_batches(
            requested,
            batch_size,
            color=self.color if color is None else color,
        ):
            yield batch_indices, (pose_frames, detector_frames)

    @staticmethod
    def _normalize_selected_frame_indices(frame_indices: Sequence[int]) -> list[int]:
        requested: list[int] = []
        pending_source_indices: set[int] = set()
        for raw_idx in frame_indices:
            if isinstance(raw_idx, bool) or not isinstance(raw_idx, int | np.integer):
                raise TypeError("frame_indices must be integers")
            frame_idx = int(raw_idx)
            if frame_idx < 0:
                raise ValueError("frame_indices must be >= 0")
            if frame_idx in pending_source_indices:
                raise ValueError("frame_indices must be unique")
            requested.append(frame_idx)
            pending_source_indices.add(frame_idx)
        return requested

    def iter_selected_frames(
        self,
        frame_indices: Sequence[int],
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Yield selected frames in request order while streaming source frames once."""
        requested = self._normalize_selected_frame_indices(frame_indices)
        if not requested:
            return
        pending_source_indices = set(requested)
        buffered_frames: dict[int, np.ndarray] = {}
        next_request_pos = 0
        for source_frame_idx, frame in enumerate(self.iter_frames()):
            if source_frame_idx not in pending_source_indices:
                continue
            pending_source_indices.remove(source_frame_idx)
            buffered_frames[source_frame_idx] = frame
            while next_request_pos < len(requested):
                requested_frame_idx = requested[next_request_pos]
                buffered_frame = buffered_frames.get(requested_frame_idx)
                if buffered_frame is None:
                    break
                yield requested_frame_idx, buffered_frame
                del buffered_frames[requested_frame_idx]
                next_request_pos += 1
            if not pending_source_indices:
                return
        missing = ", ".join(str(idx) for idx in requested if idx in pending_source_indices)
        raise RuntimeError(f"Failed to decode requested video frames: {missing}")

    def close(self) -> None:
        with self._lock:
            if self._backend is not None:
                self._backend.close()
                self._backend = None

    @classmethod
    def from_media(
        cls,
        filename: str,
        grayscale: bool | None = None,
        backend: str = "auto",
        color: str = "bgr",
        pyav_resource: PyAVVideoResource | None = None,
    ) -> Video:
        return cls(
            filename=filename,
            grayscale=bool(grayscale) if grayscale is not None else False,
            backend=backend,
            color=color,
            pyav_resource=pyav_resource,
        )

    @classmethod
    def from_pyav_resource(
        cls,
        filename: str,
        resource: PyAVVideoResource,
        *,
        grayscale: bool | None = None,
        color: str = "bgr",
    ) -> Video:
        return cls.from_media(
            filename,
            grayscale=grayscale,
            backend="pyav",
            color=color,
            pyav_resource=resource,
        )

    @classmethod
    def from_image_filenames(
        cls,
        filenames: list[str],
        filename: str | None = None,
        grayscale: bool | None = None,
        backend: str = "auto",
        color: str = "bgr",
    ) -> Video:
        return cls(
            filename=filename,
            image_filenames=filenames,
            grayscale=bool(grayscale) if grayscale is not None else False,
            backend=backend,
            color=color,
        )

    @classmethod
    def from_filename(cls, filename: str, **kwargs: Any) -> Video:
        path = resolve_path(filename)
        backend = str(kwargs.get("backend", "auto"))
        color = str(kwargs.get("color", "bgr"))
        if path.is_dir():
            return cls.from_image_filenames(
                _image_sequence_dir_filenames(path),
                filename=path.as_posix(),
                grayscale=kwargs.get("grayscale"),
                backend=backend,
                color=color,
            )
        if path.suffix.lower() in SingleImageVideo.EXTS:
            return cls.from_image_filenames(
                [filename],
                grayscale=kwargs.get("grayscale"),
                backend=backend,
                color=color,
            )
        return cls.from_media(
            filename,
            grayscale=kwargs.get("grayscale"),
            backend=backend,
            color=color,
        )

    @staticmethod
    def cattr() -> Converter:
        conv = Converter()

        def _unstructure(video: Video) -> dict[str, Any]:
            if video.image_filenames:
                payload: dict[str, Any] = {
                    "type": "images",
                    "filenames": list(video.image_filenames),
                    "width": video.width,
                    "height": video.height,
                    "frames": video.frames,
                }
                if video.filename is not None:
                    payload["filename"] = video.filename
                return payload
            if video.filename is not None:
                return {
                    "type": "media",
                    "filename": video.filename,
                    "width": video.width,
                    "height": video.height,
                    "frames": video.frames,
                }
            raise ValueError("Video has no filename or image sequence")

        def _structure(data: dict[str, Any], _type: Any) -> Video:
            if data.get("type") == "images" or data.get("filenames"):
                image_filenames = data.get("filenames")
                if not isinstance(image_filenames, list):
                    raise ValueError("Missing filenames for image-sequence video")
                filename = data.get("filename")
                if filename is not None and not isinstance(filename, str):
                    raise ValueError("Image-sequence filename must be a string when provided")
                return Video.from_image_filenames(image_filenames, filename=filename)
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
        self._video = Video.from_filename(self.path, color=self.color)

    def __iter__(self) -> Iterator[np.ndarray]:
        yield from self._video.iter_frames()

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
        return self._video.get_frame(index)

    def close(self) -> None:
        self._video.close()

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
