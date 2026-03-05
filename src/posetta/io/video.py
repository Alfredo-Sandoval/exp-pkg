"""Video helpers and writer utilities.

This module intentionally avoids legacy backends (HDF5/ImgStore/Numpy).
It exposes a minimal `Video` wrapper, helpers, and a writer API used by
the GUI and dataset code.
"""

from __future__ import annotations

import contextlib
import importlib.util
import math
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Event, Lock, Thread
from time import perf_counter
from typing import Any

import cv2
import imageio.v2 as iio
import imageio_ffmpeg as _ff
import numpy as np
from cattrs import Converter

from posetta.config import settings
from posetta.config.definitions import get_video_io_defaults
from posetta.core.colors import bgr_to_rgb, ensure_bgr, ensure_three_channels
from posetta.core.logging_utils import get_logger
from posetta.core.path_registry import ensure_dir, resolve_path, usable_cpu_count
from posetta.io import video_writer_backend as _writer_backend

logger = get_logger(__name__)
_VIDEO_IO_DEFAULTS = get_video_io_defaults()

from posetta.io.framehub import frame_hub
from posetta.io.video_backends import (
    ImageSequenceBackend,
    OpenCVBackend,
    PyAVBackend,
    VideoBackend,
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
    "default_intermediate_target",
    "gui_playback_backend_for_path",
    "progress_feedback",
    "reader",
    "resize_image",
    "resize_images",
    "write_video",
]


class SingleImageVideo:
    """Supported single-image extensions (treated as 1-frame videos)."""

    EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def available_video_exts() -> list[str]:
    """Return normalized extensions we treat as videos (always leading dot, lowercase).

    Sourced from settings to avoid drift.
    """
    exts: list[str] = []
    for ext in settings.video.recognized_exts:
        normalized = ext.strip().lower()
        if not normalized:
            continue
        if not normalized.startswith("."):
            normalized = f".{normalized}"
        exts.append(normalized)
    return list(dict.fromkeys(exts))


def gui_playback_backend_for_path(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext in SingleImageVideo.EXTS:
        # Image-backed "videos" are one-frame sequences and don't need PyAV seeking.
        return "opencv"
    return "pyav"


class Video:
    """Minimal video wrapper using OpenCV.

    Provides:
    - width, height, frames
    - first_frame_idx, last_frame_idx
    - get_frame(idx) -> np.ndarray (BGR)
    - filename (str | None)

    Notes:
    - For compressed/VFR media, `frames` may be approximate depending on backend metadata.
    - Exact random access is backend-dependent; PyAV is preferred for highest seeking fidelity.
    """

    def __init__(
        self,
        filename: str | None = None,
        image_filenames: list[str] | None = None,
        grayscale: bool = False,
        backend: str = "auto",
        *,
        share_pyav_container: bool = True,
    ):
        self._lock = Lock()
        self.filename: str | None = filename
        self.id: str | None = None
        self.label: str | None = None
        self.sha256: str | None = None
        self._image_filenames: list[str] = list(image_filenames or [])
        self.grayscale: bool = bool(grayscale)
        backend_choice = backend.lower().strip()
        if backend_choice == "auto":
            backend_choice = "opencv"
        self.backend: str = backend_choice
        self._grayscale_hint = self.grayscale
        self._share_pyav_container = bool(share_pyav_container)

        self._backend: VideoBackend | None = None
        self.width: int = 0
        self.height: int = 0
        self.frames: int = 0
        self.fps: float = 0.0
        self.channels: int = 0
        self.last_frame_idx: int = 0

        if self.filename is not None:
            path = Path(self.filename)
            if not path.exists():
                raise FileNotFoundError(f"Video file not found: {self.filename}")
            if not path.is_file():
                raise ValueError(f"Not a file: {self.filename}")
            self._backend = self._init_backend(
                self.filename,
                self.backend,
                self.grayscale,
                share_pyav_container=self._share_pyav_container,
            )
        elif self._image_filenames:
            self._backend = ImageSequenceBackend(self._image_filenames, grayscale=self.grayscale)
            self.backend = "images"
        else:
            raise ValueError("Video requires a filename or image filenames")

        if self._backend is None:
            raise RuntimeError("Video backend initialization failed")
        self.width = self._backend.width
        self.height = self._backend.height
        self.frames = self._backend.frames
        self.fps = self._backend.fps
        self.channels = self._backend.channels

        self.last_frame_idx = max(0, self.frames - 1)

    @property
    def uses_pyav(self) -> bool:
        return isinstance(self._backend, PyAVBackend)

    def _init_backend(
        self,
        filename: str,
        backend: str,
        grayscale: bool,
        *,
        share_pyav_container: bool,
    ) -> VideoBackend:
        """Initialize the appropriate backend.

        OpenCV is the default backend. PyAV is available when explicitly requested.
        """
        choice = backend.lower().strip()

        if choice == "auto":
            choice = "opencv"

        if choice == "opencv":
            return OpenCVBackend(filename, grayscale=grayscale)

        if choice != "pyav":
            raise ValueError(f"Unknown video backend: {backend}")

        if importlib.util.find_spec("av") is None:
            raise ImportError("PyAV (av) is not installed")

        if not share_pyav_container:
            return PyAVBackend(filename, grayscale=grayscale)

        with contextlib.ExitStack() as stack:
            lease = frame_hub.borrow(filename)
            stack.callback(lease.release)
            pyav_backend = PyAVBackend(
                filename,
                grayscale=grayscale,
                container=lease.container,
                shared_lock=lease.lock,
                release_callback=lease.release,
            )
            stack.pop_all()
            return pyav_backend

    @property
    def image_filenames(self) -> list[str]:
        return list(self._image_filenames)

    def __len__(self) -> int:
        return self.frames

    def get_frame(self, idx: int, *, approximate: bool = False) -> np.ndarray:
        """Load the frame at `idx` and return it in BGR format."""
        with self._lock:
            if idx < 0:
                raise IndexError("Frame index out of range")

            if self._backend is None:
                if self.filename:
                    self._backend = self._init_backend(
                        self.filename,
                        self.backend,
                        self._grayscale_hint,
                        share_pyav_container=self._share_pyav_container,
                    )
                elif self._image_filenames:
                    self._backend = ImageSequenceBackend(
                        self._image_filenames, grayscale=self._grayscale_hint
                    )

            if self._backend is None:
                raise RuntimeError("Video backend is not initialized and could not be re-opened")

            if isinstance(self._backend, PyAVBackend):
                return self._backend.get_frame(idx, approximate=approximate)
            return self._backend.get_frame(idx)

    def get_frames_safely(self, indices: list[int]) -> tuple[list[int], np.ndarray]:
        """Batch frame getter used by writer threads (strict).

        Returns (loaded_indices, frames_array[B,H,W,C]). Raises on the first failure.
        """
        frames: list[np.ndarray] = []
        loaded: list[int] = []
        for i in indices:
            fr = self.get_frame(i)
            frames.append(fr)
            loaded.append(i)
        if not frames:
            return [], np.empty((0, self.height, self.width, self.channels or 3), dtype=np.uint8)
        arr = np.stack(frames, axis=0)
        if arr.ndim == 3:
            arr = arr[..., np.newaxis]
        return loaded, arr

    def iter_frames(self) -> Iterator[np.ndarray]:
        """Yield frames sequentially in BGR format."""
        with self._lock:
            if self._backend is None:
                if self.filename:
                    self._backend = self._init_backend(
                        self.filename,
                        self.backend,
                        self._grayscale_hint,
                        share_pyav_container=self._share_pyav_container,
                    )
                elif self._image_filenames:
                    self._backend = ImageSequenceBackend(
                        self._image_filenames, grayscale=self._grayscale_hint
                    )
        if self._backend is None:
            raise RuntimeError("Video backend is not initialized and could not be re-opened")
        yield from self._backend.iter_frames()

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
        *,
        share_pyav_container: bool = True,
    ) -> Video:
        return cls(
            filename=filename,
            grayscale=bool(grayscale) if grayscale is not None else False,
            backend=backend,
            share_pyav_container=share_pyav_container,
        )

    @classmethod
    def from_image_filenames(
        cls,
        filenames: list[str],
        grayscale: bool | None = None,
        backend: str = "auto",
        *,
        share_pyav_container: bool = True,
    ) -> Video:
        return cls(
            filename=None,
            image_filenames=filenames,
            grayscale=bool(grayscale) if grayscale is not None else False,
            backend=backend,
            share_pyav_container=share_pyav_container,
        )

    @classmethod
    def from_filename(cls, filename: str, **kwargs) -> Video:
        ext = Path(filename).suffix.lower()
        backend = kwargs.get("backend", "auto")
        share_pyav_container = bool(kwargs.get("share_pyav_container", True))
        if ext in SingleImageVideo.EXTS:
            return cls.from_image_filenames(
                [filename],
                grayscale=kwargs.get("grayscale"),
                backend=backend,
                share_pyav_container=share_pyav_container,
            )
        return cls.from_media(
            filename,
            grayscale=kwargs.get("grayscale"),
            backend=backend,
            share_pyav_container=share_pyav_container,
        )

    @staticmethod
    def cattr():
        conv = Converter()

        def _unstructure(v: Video):
            if v.filename is not None:
                return {
                    "type": "media",
                    "filename": v.filename,
                    "width": v.width,
                    "height": v.height,
                    "frames": v.frames,
                }
            return {
                "type": "images",
                "filenames": list(v.image_filenames),
                "width": v.width,
                "height": v.height,
                "frames": v.frames,
            }

        def _structure(d: dict, _t):
            t = d.get("type", "media")
            if t == "images" or (d.get("filenames")):
                return Video.from_image_filenames(d["filenames"])
            fn = d.get("filename")
            if not isinstance(fn, str) or not fn:
                raise ValueError("Missing filename for media video")
            return Video.from_filename(fn)

        conv.register_unstructure_hook(Video, _unstructure)
        conv.register_structure_hook(Video, _structure)
        return conv

    def __enter__(self) -> Video:
        return self

    def __exit__(self, _exc_type: Any, _exc_value: Any, _traceback: Any) -> None:
        self.close()


_sentinel = object()


class VideoWriter(ABC):
    """Abstract base for video encoders."""

    @abstractmethod
    def __init__(self, filename: str, height: int, width: int, fps: float, **kwargs):
        """Prepare the encoder for the target file/shape."""
        raise NotImplementedError

    @abstractmethod
    def add_frame(self, img: np.ndarray, bgr: bool = False):
        """Encode a single frame into the video."""
        raise NotImplementedError

    @abstractmethod
    def close(self):
        """Close the encoder and release native handles."""
        raise NotImplementedError

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, _exc_type: Any, _exc_value: Any, _traceback: Any) -> None:
        self.close()

    @staticmethod
    def builder(
        filename: str,
        height: int,
        width: int,
        fps: float,
        backend: str = "auto",
        **kwargs,
    ) -> VideoWriter:
        """Create a video writer with the specified backend or auto-detection.

        Args:
            filename: Output path.
            height: Video height.
            width: Video width.
            fps: Frames per second.
            backend: "auto", "ffmpeg", "imageio", or "opencv".
            **kwargs: Backend-specific arguments (e.g., crf, preset, pixelformat, codec,
                tune_nvenc).
        """
        ext = Path(filename).suffix.lower()
        _writer_backend.apply_ffmpeg_env()
        use_ffmpeg = VideoWriter.can_use_ffmpeg()

        if backend == "auto":
            if use_ffmpeg and ext not in (".avi", ".mkv"):
                if ext == ".mp4":
                    backend = "imageio"
                else:
                    backend = "imageio"
            else:
                backend = "opencv"

        if backend in ("imageio", "ffmpeg"):
            if not use_ffmpeg:
                raise RuntimeError(
                    "Requested ffmpeg backend but imageio-ffmpeg is unavailable. "
                    "Install imageio-ffmpeg or choose backend='opencv'."
                )
            if "codec" not in kwargs:
                prefer_codec = settings.video.writer.prefer_codec
                if prefer_codec:
                    kwargs["codec"] = str(prefer_codec)
                else:
                    platform_codec = None
                    encoders = _writer_backend.ffmpeg_encoders()
                    for name in _writer_backend.platform_preferred_encoders():
                        if name in encoders:
                            platform_codec = name
                            break
                    if platform_codec:
                        kwargs["codec"] = platform_codec
            if "codec" in kwargs:
                codec = str(kwargs["codec"])
                if codec not in _writer_backend.ffmpeg_encoders():
                    raise RuntimeError(
                        f"Requested FFmpeg encoder '{codec}' is not available. "
                        "Install an FFmpeg build with that encoder or update "
                        "settings.video.writer.prefer_codec."
                    )
            if "tune_nvenc" not in kwargs:
                kwargs["tune_nvenc"] = bool(settings.video.writer.tune_nvenc)
            return VideoWriterImageio(filename, height, width, fps, **kwargs)

        return VideoWriterOpenCV(filename, height, width, fps, **kwargs)

    @staticmethod
    def can_use_ffmpeg() -> bool:
        """Return True when imageio-ffmpeg is usable."""
        _writer_backend.apply_ffmpeg_env()
        v = _ff.get_ffmpeg_version()
        return bool(str(v))


class VideoWriterOpenCV(VideoWriter):
    """OpenCV-based encoder (MJPG)."""

    def __init__(
        self,
        filename: str,
        height: int,
        width: int,
        fps: float,
        fourcc: str | None = None,
        is_color: bool = True,
        **_kwargs,
    ):
        """Prepare the OpenCV writer.

        Args:
            filename: Output path.
            height: Image height.
            width: Image width.
            fps: Frame rate.
            fourcc: strictly 4-char string (e.g. 'MJPG', 'mp4v'). default 'MJPG'.
            is_color: If False, writer expects grayscale.
        """
        code = fourcc or "MJPG"
        self._lock = Lock()
        with self._lock:
            fcc = int(cv2.VideoWriter.fourcc(*code))
            self._writer = cv2.VideoWriter(filename, fcc, fps, (width, height), is_color)
            if not self._writer.isOpened():
                self._writer.release()
                raise RuntimeError(
                    f"Failed to open OpenCV VideoWriter for {filename} (fourcc={code})."
                )

    def add_frame(self, img: np.ndarray, bgr: bool = False):
        """Write `img` to the video, converting to BGR when needed."""
        img = ensure_bgr(img, input_is_bgr=bgr)
        with self._lock:
            self._writer.write(img)

    def close(self):
        """Release the OpenCV writer handle."""
        with self._lock:
            self._writer.release()


class VideoWriterImageio(VideoWriter):
    """FFmpeg encoder via imageio (prefers NVENC when available)."""

    def __init__(
        self,
        filename: str,
        height: int,
        width: int,
        fps: float,
        crf: int = 21,
        preset: str = "superfast",
        pixelformat: str = "yuv420p",
        codec: str | None = None,
        output_params: list[str] | None = None,
        tune_nvenc: bool = True,
        **_kwargs,
    ):
        """Prepare an imageio-ffmpeg writer with optional NVENC tuning.

        Args:
            filename: Output path.
            height: Video height.
            width: Video width.
            fps: Frames per second.
            crf: Constant Rate Factor (quality).
            preset: FFmpeg preset (e.g., "superfast").
            pixelformat: Output pixel format.
            codec: FFmpeg codec name.
            output_params: Additional FFmpeg output parameters.
            tune_nvenc: Whether to apply NVENC-specific tuning.
            **_kwargs: Ignored.
        """
        _writer_backend.apply_ffmpeg_env()
        self.filename = filename
        self.height = height
        self.width = width
        self.fps = fps
        self.crf = crf
        self.preset = preset

        if codec is not None:
            codec_name = str(codec)
            if codec_name not in _writer_backend.ffmpeg_encoders():
                raise RuntimeError(
                    f"Requested FFmpeg encoder '{codec_name}' is not available. "
                    "Install an FFmpeg build with that encoder or update "
                    "settings.video.writer.prefer_codec."
                )
        desired_codec = codec or self._auto_select_codec()
        self.codec = desired_codec
        final_output_params: list[str] = output_params or []
        self.output_params: list[str] = []

        nvenc_codecs = {"h264_nvenc", "hevc_nvenc", "av1_nvenc"}

        if not final_output_params and (desired_codec in nvenc_codecs) and bool(tune_nvenc):
            final_output_params = ["-preset", "p5"]
            sup = self._supported_nvenc_flags(desired_codec)

            def a(opt: str, *vals: str) -> None:
                if sup.get(opt):
                    final_output_params.extend([opt, *vals])

            a("-tune", "hq")
            a("-rc:v", "vbr")
            a("-cq", "22" if desired_codec != "av1_nvenc" else "28")
            for k in ("-spatial_aq", "-spatial-aq"):
                if sup.get(k):
                    a(k, "1")
                    break
            for k in ("-temporal_aq", "-temporal-aq"):
                if sup.get(k):
                    a(k, "1")
                    break
            for k in ("-look_ahead", "-rc-lookahead", "-rc_lookahead"):
                if sup.get(k):
                    a(k, "32")
                    break
            if desired_codec == "hevc_nvenc":
                a("-profile:v", "main")
            elif desired_codec == "h264_nvenc":
                a("-profile:v", "high")
        elif not final_output_params:
            final_output_params = ["-preset", preset, "-crf", str(crf)]

        self.output_params = list(final_output_params)

        self.writer = iio.get_writer(
            filename,
            fps=fps,
            codec=desired_codec,
            pixelformat=pixelformat,
            macro_block_size=1,
            output_params=final_output_params,
        )

    @staticmethod
    def _auto_select_codec() -> str:
        """Pick a preferred encoder supported by the bundled ffmpeg."""
        encoders = _writer_backend.ffmpeg_encoders()
        candidates = [*_writer_backend.platform_preferred_encoders(), "libx264"]
        for name in candidates:
            if name in encoders:
                return name
        raise RuntimeError(
            "No usable FFmpeg encoder found. Install an FFmpeg build with "
            "libx264 or set settings.video.writer.prefer_codec explicitly."
        )

    @staticmethod
    def _supported_nvenc_flags(codec: str) -> dict[str, bool]:
        """Probe `ffmpeg -h encoder=<codec>` for supported NVENC flags (cached)."""
        return _writer_backend.supported_nvenc_flags(codec)

    def add_frame(self, img, bgr: bool = False):
        """Append an RGB frame to the FFmpeg writer, converting from BGR when needed.

        Args:
            img: The image array to encode.
            bgr: If True, input is BGR; otherwise RGB is assumed.
        """
        img = ensure_three_channels(img)
        if bgr:
            img = bgr_to_rgb(img)
        self.writer.append_data(img)

    def close(self):
        """Close the FFmpeg writer and flush any buffered data."""
        self.writer.close()


def resize_image(img: np.ndarray, scale: float) -> np.ndarray:
    """Resize one image by a scale factor."""
    height, width, channels = img.shape
    new_height, new_width = int(height * scale), int(width * scale)

    if channels == 1:
        resized = cv2.resize(img.squeeze(-1), (new_width, new_height))
        return resized[..., np.newaxis]
    return cv2.resize(img, (new_width, new_height))


def resize_images(images: np.ndarray, scale: float) -> np.ndarray:
    """Resize each image in `images` by `scale`, skipping work when scale is 1."""
    if scale == 1.0:
        return images
    return np.stack([resize_image(img, scale) for img in images])


def augment_background(images: np.ndarray, background: str | None) -> np.ndarray:
    """Fill with a solid color or keep original pixels."""
    if background is None or background == "original":
        return images

    fill_values = {"black": 0, "grey": 127, "white": 255}
    if background not in fill_values:
        raise ValueError(
            f"Invalid background color: {background}. Options include: "
            f"{', '.join(fill_values.keys())}"
        )

    fill = fill_values[background]
    return np.full_like(images, fill)


def reader(
    out_q: Queue[Any],
    video: Video,
    frames: list[int],
    scale: float = 1.0,
    background: str | None = None,
    stop_event: Event | None = None,
):
    """Read frames in chunks and push batches to a queue."""
    background = background.lower() if background is not None else background
    cv2.setNumThreads(usable_cpu_count())

    total_count = len(frames)
    chunk_size = int(_VIDEO_IO_DEFAULTS["reader_chunk_size"])
    chunk_count = math.ceil(total_count / chunk_size)

    logger.info("Chunks: %d, chunk size: %d", chunk_count, chunk_size)
    i = 0
    for chunk_i in range(chunk_count):
        if stop_event is not None and stop_event.is_set():
            break
        frame_start = chunk_size * chunk_i
        frame_end = min(frame_start + chunk_size, total_count)
        frames_idx_chunk = frames[frame_start:frame_end]
        t0 = perf_counter()
        loaded_chunk_idxs, video_frame_images = video.get_frames_safely(frames_idx_chunk)
        if not loaded_chunk_idxs:
            i += 1
            continue
        video_frame_images = augment_background(
            images=video_frame_images,
            background=background,
        )
        video_frame_images = resize_images(images=video_frame_images, scale=scale)
        if logger.isEnabledFor(10):
            elapsed = perf_counter() - t0
            chunk_fps = len(loaded_chunk_idxs) / max(elapsed, 1e-9)
            logger.debug("reading chunk %d in %.3fs = %.1f fps", i, elapsed, chunk_fps)
        i += 1
        out_q.put((loaded_chunk_idxs, video_frame_images))
    out_q.put(_sentinel)


def writer(
    in_q: Queue[Any],
    progress_queue: Queue[Any],
    filename: str,
    fps: float,
):
    """Pop image batches and encode them into a video file."""

    cv2.setNumThreads(usable_cpu_count())

    writer_object = None
    total_frames_written = 0
    start_time = perf_counter()
    i = 0

    while True:
        data = in_q.get()

        if data is _sentinel:
            in_q.put(_sentinel)
            break

        if writer_object is None and len(data) > 0:
            height, width = data[0].shape[:2]
            writer_object = VideoWriter.builder(filename, height=height, width=width, fps=fps)

        t0 = perf_counter()

        if writer_object is None:
            continue
        for img in data:
            writer_object.add_frame(img, bgr=True)

        if logger.isEnabledFor(10):
            elapsed = perf_counter() - t0
            chunk_fps = len(data) / max(elapsed, 1e-9)
            logger.debug("writing chunk %d in %.3fs = %.1f fps", i, elapsed, chunk_fps)
        i += 1

        total_frames_written += len(data)
        total_elapsed = perf_counter() - start_time
        progress_queue.put((total_frames_written, total_elapsed))

    if writer_object is not None:
        writer_object.close()
    progress_queue.put((-1, perf_counter() - start_time))


def progress_feedback(
    progress_queue: Queue[Any],
    frames: list[int],
    progress_callback: Any | None = None,
):
    """Consume progress updates and route via callback; no GUI side-effects.

    Note: GUI dialogs must be implemented in the GUI layer by providing
    `progress_callback`.
    """
    while True:
        frames_complete, _elapsed = progress_queue.get()
        if frames_complete == -1:
            break
        if progress_callback is not None:
            progress_callback(frames_complete, len(frames))


def default_intermediate_target(in_queue: Queue[Any], out_queue: Queue[Any]):
    """Pass-through stage between reader and writer."""
    cv2.setNumThreads(usable_cpu_count())
    while True:
        data = in_queue.get()
        if data is _sentinel:
            in_queue.put(_sentinel)
            break
        _frame_indices, images = data
        out_queue.put(images)
    out_queue.put(_sentinel)


def write_video(
    filename: str,
    video: Video,
    frames: list[int],
    fps: int = 15,
    scale: float = 1.0,
    background: str | None = None,
    in_queue: Queue[Any] | None = None,
    out_queue: Queue[Any] | None = None,
    intermediate_threads: list[Thread] | None = None,
    progress_callback: Any | None = None,
    stop_event: Event | None = None,
):
    """Threaded writer with optional scaling and background fill."""

    filename_path = resolve_path(filename)
    ensure_dir(filename_path.parent)
    q1 = in_queue or Queue(maxsize=10)
    q2 = out_queue or Queue(maxsize=10)
    progress_queue: Queue[Any] = Queue()

    thread_read = Thread(
        target=reader,
        args=(q1, video, frames, scale, background, stop_event),
        name="siesta-video-reader",
    )
    thread_write = Thread(
        target=writer,
        args=(q2, progress_queue, str(filename_path), fps),
        name="siesta-video-writer",
    )

    thread_read.start()

    if intermediate_threads is None:
        intermediate_thread = Thread(
            target=default_intermediate_target,
            args=(q1, q2),
        )
        intermediate_threads = [intermediate_thread]

    for t in intermediate_threads:
        t.start()

    thread_write.start()

    progress_feedback(
        progress_queue,
        frames,
        progress_callback=progress_callback,
    )

    thread_read.join()
    for t in intermediate_threads:
        t.join()
    thread_write.join()


@dataclass
class VideoReader:
    """Iterate over frames from a video path using the shared Video wrapper.

    Args:
        path: input video path
        color: "bgr" (default) or "rgb" - select output channel order
    """

    path: str
    color: str = "bgr"

    def __post_init__(self) -> None:
        self._video = Video.from_filename(self.path)
        self._is_rgb = self.color.lower() == "rgb"

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield frames as uint8 arrays in requested color order (BGR or RGB)."""
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
