"""Concrete frame readers for xpkg's video contract."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from importlib import import_module
from pathlib import Path
from threading import Lock
from typing import Any, Protocol, cast

import cv2
import numpy as np

from xpkg._core.colors import bgr_to_gray, rgb_to_bgr
from xpkg.media import backend_utils
from xpkg.media.images import read_bgr
from xpkg.media.pyav import PyAVVideoReader

_CV2_ANY = cast(Any, cv2)


class VideoBackend(Protocol):
    """Protocol implemented by concrete video readers."""

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

    def get_frame(self, idx: int) -> np.ndarray: ...

    def iter_frames(self) -> Iterator[np.ndarray]: ...

    def iter_frames_stride(self, stride: int) -> Iterator[np.ndarray]: ...

    def iter_frame_batches_stride(self, batch_size: int, stride: int) -> Iterator[list[np.ndarray]]:
        ...

    def close(self) -> None: ...


def _validate_video_path(filename: str) -> None:
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {filename}")
    if not path.is_file():
        raise ValueError(f"Not a file: {filename}")


def validate_batched_read_args(batch_size: int, stride: int) -> tuple[int, int]:
    """Validate shared batched decode arguments."""
    batch_size_value = int(batch_size)
    if batch_size_value <= 0:
        raise ValueError("batch_size must be > 0")
    stride_value = int(stride)
    if stride_value <= 0:
        raise ValueError("stride must be > 0")
    return batch_size_value, stride_value


def chunk_frame_indices(*, frame_count: int, batch_size: int, stride: int) -> Iterator[list[int]]:
    """Yield exact source-frame index chunks for strided decode."""
    batch_indices: list[int] = []
    for frame_idx in range(0, frame_count, stride):
        batch_indices.append(frame_idx)
        if len(batch_indices) == batch_size:
            yield batch_indices
            batch_indices = []
    if batch_indices:
        yield batch_indices


def chunk_selected_frame_indices(
    frame_indices: Sequence[int],
    *,
    batch_size: int,
) -> Iterator[list[int]]:
    """Yield selected frame-index chunks preserving request order."""
    batch: list[int] = []
    for raw_frame_idx in frame_indices:
        batch.append(int(raw_frame_idx))
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def yield_frame_batches(
    frames: Iterator[np.ndarray],
    *,
    batch_size: int,
) -> Iterator[list[np.ndarray]]:
    """Group an existing frame iterator into batches."""
    batch: list[np.ndarray] = []
    for frame in frames:
        batch.append(frame)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _validate_decord_frame_array(array: np.ndarray) -> np.ndarray:
    if array.ndim != 3 or array.shape[-1] != 3:
        raise RuntimeError("decord GPU reader must return HxWx3 RGB frames")
    return array


def _validate_decord_batch_array(array: np.ndarray) -> np.ndarray:
    if array.ndim != 4 or array.shape[-1] != 3:
        raise RuntimeError("decord GPU reader must return NxHxWx3 RGB frame batches")
    return array


def _decord_frame_rgb(frame: Any) -> np.ndarray:
    array = frame.asnumpy()
    if not isinstance(array, np.ndarray):
        raise TypeError("decord GPU reader must return numpy RGB frames")
    return _validate_decord_frame_array(array)


def _decord_frame_bgr(frame: Any) -> np.ndarray:
    return rgb_to_bgr(_decord_frame_rgb(frame))


def _decord_batch_rgb(frame_batch: Any) -> np.ndarray:
    array = frame_batch.asnumpy()
    if not isinstance(array, np.ndarray):
        raise TypeError("decord GPU reader must return numpy RGB frame batches")
    return _validate_decord_batch_array(array)


def _decord_batch_bgr(frame_batch: Any) -> np.ndarray:
    return _decord_batch_rgb(frame_batch)[..., ::-1]


def _decord_batch_rgb_torch(frame_batch: Any) -> Any:
    torch = import_module("torch")
    to_dlpack = frame_batch.to_dlpack
    if not callable(to_dlpack):
        raise TypeError("decord GPU reader must expose to_dlpack() for torch batch decode")
    tensor = torch.utils.dlpack.from_dlpack(to_dlpack())
    if tensor.ndim != 4 or int(tensor.shape[-1]) != 3:
        raise RuntimeError(
            "decord GPU reader must return NxHxWx3 RGB torch frame batches; "
            f"got shape {tuple(tensor.shape)}"
        )
    if tensor.dtype is not torch.uint8:
        raise RuntimeError(
            f"decord GPU reader must return uint8 torch frame batches; got dtype {tensor.dtype}"
        )
    return tensor if tensor.is_contiguous() else tensor.contiguous()


def _decord_batch_bgr_torch(frame_batch: Any) -> Any:
    return _decord_batch_rgb_torch(frame_batch).flip(-1).contiguous()


def _normalize_color_mode(color: str) -> str:
    color_value = str(color).strip().lower()
    if color_value not in {"bgr", "rgb"}:
        raise ValueError(f"Unsupported color mode: {color!r}")
    return color_value


def _decord_batch_torch(frame_batch: Any, *, color: str) -> Any:
    if _normalize_color_mode(color) == "rgb":
        return _decord_batch_rgb_torch(frame_batch)
    return _decord_batch_bgr_torch(frame_batch)


def _frame_list_from_batch(frame_batch: np.ndarray) -> list[np.ndarray]:
    return [frame for frame in frame_batch]


def _coerce_color_batch(frame_batch_bgr: np.ndarray, *, grayscale: bool) -> list[np.ndarray]:
    if frame_batch_bgr.ndim != 4 or frame_batch_bgr.shape[-1] != 3:
        raise ValueError(
            f"Batched video decode expects NxHxWx3 frame arrays; got shape {frame_batch_bgr.shape}"
        )
    if not grayscale:
        return [frame for frame in frame_batch_bgr]
    return [bgr_to_gray(frame)[..., np.newaxis] for frame in frame_batch_bgr]


class OpenCVBackend:
    """OpenCV-based exact indexed reader."""

    def __init__(self, filename: str, grayscale: bool = False):
        self._lock = Lock()
        self.filename = filename
        self.grayscale = bool(grayscale)
        _validate_video_path(filename)
        self._cap = _CV2_ANY.VideoCapture(filename)
        if self._cap is None or not self._cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {filename}")
        self._width = int(self._cap.get(_CV2_ANY.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(_CV2_ANY.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(self._cap.get(_CV2_ANY.CAP_PROP_FRAME_COUNT))
        self._frames = frame_count if frame_count > 0 else 1
        fps_val = float(self._cap.get(_CV2_ANY.CAP_PROP_FPS) or 0.0)
        self._fps = fps_val if fps_val > 0 else 30.0
        self._fourcc = int(self._cap.get(_CV2_ANY.CAP_PROP_FOURCC))
        self._next_frame_idx = 0
        ok, frame = self._cap.read()
        if not ok or frame is None:
            fourcc_str = self._fourcc_to_str(self._fourcc)
            self._cap.release()
            self._cap = None
            raise RuntimeError(
                f"Cannot decode video: {filename}\n"
                f"Container opened but codec failed. FourCC: {fourcc_str}"
            )
        self._height, self._width = frame.shape[:2]
        self._frame0_cache = frame if self._frames == 1 else None
        self._cap.set(_CV2_ANY.CAP_PROP_POS_FRAMES, 0)

    @staticmethod
    def _fourcc_to_str(code: int) -> str:
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

    def _read_next_frame_locked(self) -> np.ndarray:
        if self._cap is None:
            raise RuntimeError("Backend closed")
        target_idx = self._next_frame_idx
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise IndexError(f"Frame index out of range: {target_idx}")
        self._next_frame_idx += 1
        return frame

    def _seek_and_read_exact_frame_locked(self, idx: int) -> np.ndarray:
        if self._cap is None:
            raise RuntimeError("Backend closed")
        if idx == self._next_frame_idx:
            return self._read_next_frame_locked()
        seek_ok = self._cap.set(_CV2_ANY.CAP_PROP_POS_FRAMES, float(idx))
        if not seek_ok:
            raise RuntimeError(f"OpenCV failed to seek to frame {idx} in {self.filename}")
        self._next_frame_idx = idx
        frame = self._read_next_frame_locked()
        position_after_read = round(float(self._cap.get(_CV2_ANY.CAP_PROP_POS_FRAMES)))
        actual_idx = max(0, position_after_read - 1)
        if actual_idx != idx:
            raise RuntimeError(
                "OpenCV failed exact frame seek "
                f"for {self.filename}: requested frame {idx}, landed on {actual_idx}"
            )
        return frame

    def _to_output_channels(self, frame: np.ndarray) -> np.ndarray:
        if not self.grayscale:
            return frame
        return bgr_to_gray(frame)[..., np.newaxis]

    def get_frame(self, idx: int) -> np.ndarray:
        if idx < 0:
            raise IndexError("Frame index out of range")
        if idx == 0 and self._frame0_cache is not None:
            frame = self._frame0_cache
        else:
            with self._lock:
                frame = self._seek_and_read_exact_frame_locked(idx)
        return self._to_output_channels(frame)

    def iter_frames(self) -> Iterator[np.ndarray]:
        with self._lock:
            if self._cap is None:
                raise RuntimeError("Backend closed")
            self._cap.set(_CV2_ANY.CAP_PROP_POS_FRAMES, 0)
            self._next_frame_idx = 0
        while True:
            with self._lock:
                if self._cap is None:
                    raise RuntimeError("Backend closed")
                ok, frame = self._cap.read()
            if not ok or frame is None:
                break
            self._next_frame_idx += 1
            yield self._to_output_channels(frame)

    def iter_frames_stride(self, stride: int) -> Iterator[np.ndarray]:
        stride_value = int(stride)
        if stride_value <= 0:
            raise ValueError("stride must be > 0")
        if stride_value == 1:
            yield from self.iter_frames()
            return
        for frame_idx in range(0, self.frames, stride_value):
            yield self.get_frame(frame_idx)

    def iter_frame_batches_stride(self, batch_size: int, stride: int) -> Iterator[list[np.ndarray]]:
        batch_size_value, stride_value = validate_batched_read_args(batch_size, stride)
        yield from yield_frame_batches(
            self.iter_frames_stride(stride_value),
            batch_size=batch_size_value,
        )

    def close(self) -> None:
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            self._next_frame_idx = 0


class DecordGpuBackend:
    """Decord GPU-backed exact indexed reader."""

    def __init__(self, filename: str, grayscale: bool = False):
        _validate_video_path(filename)
        _, video_reader_cls, gpu_ctx = backend_utils.require_decord_gpu_api()
        self.filename = filename
        self.grayscale = bool(grayscale)
        self._reader = video_reader_cls(filename, ctx=gpu_ctx(0))
        self._frames = len(self._reader)
        if self._frames <= 0:
            raise RuntimeError(f"decord GPU reader reported no frames for {filename}")
        frame0 = _decord_frame_bgr(self._reader[0])
        self._height, self._width = frame0.shape[:2]
        self._fps = float(self._reader.get_avg_fps() or 30.0)
        self._frame0_cache = frame0 if self._frames == 1 else None

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

    def _coerce_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.grayscale and frame.ndim == 3:
            return bgr_to_gray(frame)[..., np.newaxis]
        return frame

    def _decode_frame(self, idx: int) -> np.ndarray:
        if idx < 0 or idx >= self._frames:
            raise IndexError(f"Frame index out of range: {idx}")
        if self._reader is None:
            raise RuntimeError("Backend closed")
        if idx == 0 and self._frame0_cache is not None:
            return self._frame0_cache
        return _decord_frame_bgr(self._reader[idx])

    def _resolve_dual_frame_batch(
        self,
        raw_batch: Any,
        *,
        color: str,
    ) -> tuple[list[np.ndarray], Any]:
        if _normalize_color_mode(color) == "rgb":
            return _frame_list_from_batch(_decord_batch_rgb(raw_batch)), _decord_batch_rgb_torch(
                raw_batch
            )
        return (
            _coerce_color_batch(_decord_batch_bgr(raw_batch), grayscale=False),
            _decord_batch_bgr_torch(raw_batch),
        )

    def get_frame(self, idx: int) -> np.ndarray:
        return self._coerce_frame(self._decode_frame(idx))

    def iter_frames(self) -> Iterator[np.ndarray]:
        for frame_idx in range(self._frames):
            yield self.get_frame(frame_idx)

    def iter_frames_stride(self, stride: int) -> Iterator[np.ndarray]:
        stride_value = int(stride)
        if stride_value <= 0:
            raise ValueError("stride must be > 0")
        for frame_idx in range(0, self._frames, stride_value):
            yield self.get_frame(frame_idx)

    def iter_frame_batches_stride(self, batch_size: int, stride: int) -> Iterator[list[np.ndarray]]:
        batch_size_value, stride_value = validate_batched_read_args(batch_size, stride)
        for batch_indices in chunk_frame_indices(
            frame_count=self._frames,
            batch_size=batch_size_value,
            stride=stride_value,
        ):
            if len(batch_indices) == 1 and batch_indices[0] == 0 and self._frame0_cache is not None:
                frame_batch_bgr = np.expand_dims(self._frame0_cache, axis=0)
            else:
                if self._reader is None:
                    raise RuntimeError("Backend closed")
                frame_batch_bgr = _decord_batch_bgr(self._reader.get_batch(batch_indices))
            yield _coerce_color_batch(frame_batch_bgr, grayscale=self.grayscale)

    def iter_torch_frame_batches_stride(
        self,
        batch_size: int,
        stride: int,
        *,
        color: str = "bgr",
    ) -> Iterator[Any]:
        if self.grayscale:
            raise ValueError("Torch frame batch decode requires color frames")
        batch_size_value, stride_value = validate_batched_read_args(batch_size, stride)
        for batch_indices in chunk_frame_indices(
            frame_count=self._frames,
            batch_size=batch_size_value,
            stride=stride_value,
        ):
            if self._reader is None:
                raise RuntimeError("Backend closed")
            yield _decord_batch_torch(self._reader.get_batch(batch_indices), color=color)

    def iter_dual_frame_batches_stride(
        self,
        batch_size: int,
        stride: int,
        *,
        color: str = "bgr",
    ) -> Iterator[tuple[list[np.ndarray], Any]]:
        if self.grayscale:
            raise ValueError("Dual frame batch decode requires color frames")
        batch_size_value, stride_value = validate_batched_read_args(batch_size, stride)
        for batch_indices in chunk_frame_indices(
            frame_count=self._frames,
            batch_size=batch_size_value,
            stride=stride_value,
        ):
            if self._reader is None:
                raise RuntimeError("Backend closed")
            yield self._resolve_dual_frame_batch(self._reader.get_batch(batch_indices), color=color)

    def iter_selected_dual_frame_batches(
        self,
        frame_indices: Sequence[int],
        batch_size: int,
        *,
        color: str = "bgr",
    ) -> Iterator[tuple[list[int], list[np.ndarray], Any]]:
        if self.grayscale:
            raise ValueError("Dual frame batch decode requires color frames")
        batch_size_value, _ = validate_batched_read_args(batch_size, 1)
        for batch_indices in chunk_selected_frame_indices(
            frame_indices,
            batch_size=batch_size_value,
        ):
            for frame_idx in batch_indices:
                if frame_idx < 0 or frame_idx >= self._frames:
                    raise IndexError(f"Frame index out of range: {frame_idx}")
            if self._reader is None:
                raise RuntimeError("Backend closed")
            raw_batch = self._reader.get_batch(batch_indices)
            pose_frames, detector_frames = self._resolve_dual_frame_batch(raw_batch, color=color)
            yield list(batch_indices), pose_frames, detector_frames

    def close(self) -> None:
        self._reader = None
        self._frame0_cache = None


class PyAVBackend(PyAVVideoReader):
    """PyAV-backed reader implementing the shared backend protocol."""


class ImageSequenceBackend:
    """Backend for a sequence of image files."""

    def __init__(self, filenames: list[str], grayscale: bool = False):
        if not filenames:
            raise ValueError("Image sequence is empty")
        self.filenames = list(filenames)
        self.grayscale = bool(grayscale)
        first = read_bgr(filenames[0])
        if first is None:
            raise FileNotFoundError(f"Cannot read image: {filenames[0]}")
        self._height, self._width = first.shape[:2]
        self._frames = len(filenames)
        self._channels = 1 if grayscale else 3
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
        frame = read_bgr(self.filenames[idx])
        if frame is None:
            raise RuntimeError(f"Failed to read image frame {idx}")
        if self.grayscale and frame.ndim == 3:
            return bgr_to_gray(frame)[..., np.newaxis]
        return frame

    def iter_frames(self) -> Iterator[np.ndarray]:
        for idx in range(len(self.filenames)):
            yield self.get_frame(idx)

    def iter_frames_stride(self, stride: int) -> Iterator[np.ndarray]:
        stride_value = int(stride)
        if stride_value <= 0:
            raise ValueError("stride must be > 0")
        if stride_value == 1:
            yield from self.iter_frames()
            return
        for frame_idx in range(0, len(self.filenames), stride_value):
            yield self.get_frame(frame_idx)

    def iter_frame_batches_stride(self, batch_size: int, stride: int) -> Iterator[list[np.ndarray]]:
        batch_size_value, stride_value = validate_batched_read_args(batch_size, stride)
        yield from yield_frame_batches(
            self.iter_frames_stride(stride_value),
            batch_size=batch_size_value,
        )

    def close(self) -> None:
        return


__all__ = [
    "DecordGpuBackend",
    "ImageSequenceBackend",
    "OpenCVBackend",
    "PyAVBackend",
    "VideoBackend",
    "chunk_frame_indices",
    "chunk_selected_frame_indices",
    "validate_batched_read_args",
    "yield_frame_batches",
]
