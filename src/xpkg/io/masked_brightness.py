"""Low-level masked brightness trace extraction from video frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from xpkg.core.colors import bgr_to_gray


class VideoLike(Protocol):
    """Minimal video contract required for masked trace extraction."""

    width: int
    height: int
    frames: int
    fps: float

    def get_frame(self, idx: int, *, approximate: bool = False) -> np.ndarray:
        """Return one decoded frame."""


@dataclass(frozen=True, slots=True)
class MaskedBrightnessTrace:
    """Per-frame brightness statistics extracted from one segmentation mask."""

    frame_indices: np.ndarray
    times_seconds: np.ndarray
    mean_brightness: np.ndarray
    peak_brightness: np.ndarray
    max_brightness: np.ndarray
    background_mean: np.ndarray
    contrast_mean: np.ndarray
    contrast_peak: np.ndarray


def extract_masked_brightness_trace(
    video: VideoLike,
    *,
    mask: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float] | None = None,
    sample_rate_hz: float = 4.0,
    max_seconds: float = 120.0,
    ring_px: int = 6,
) -> MaskedBrightnessTrace:
    """Extract grayscale brightness statistics from a masked video region.

    This function intentionally stops at raw sampling. It does not decide when a
    cue light is on or off; higher-level policy layers should consume the trace.
    """

    mask_array = np.asarray(mask)
    if mask_array.ndim != 2:
        raise ValueError("mask must be a 2D array")
    if video.width <= 0 or video.height <= 0:
        raise ValueError("video width and height must be positive")
    if mask_array.shape != (video.height, video.width):
        raise ValueError(
            f"mask shape {mask_array.shape} does not match video frame shape {(video.height, video.width)}"
        )

    mask_bool = mask_array > 0
    if not np.any(mask_bool):
        raise ValueError("mask must contain at least one positive pixel")

    x0, y0, x1, y1 = _coerce_bbox(mask_bool, bbox_xyxy=bbox_xyxy, width=video.width, height=video.height)
    mask_crop = mask_bool[y0:y1, x0:x1]
    ring_mask = _ring_mask(mask_crop, ring_px=ring_px)

    frame_indices = _sample_frame_indices(video=video, sample_rate_hz=sample_rate_hz, max_seconds=max_seconds)
    if frame_indices.size == 0:
        return MaskedBrightnessTrace(
            frame_indices=np.empty(0, dtype=np.int32),
            times_seconds=np.empty(0, dtype=np.float32),
            mean_brightness=np.empty(0, dtype=np.float32),
            peak_brightness=np.empty(0, dtype=np.float32),
            max_brightness=np.empty(0, dtype=np.float32),
            background_mean=np.empty(0, dtype=np.float32),
            contrast_mean=np.empty(0, dtype=np.float32),
            contrast_peak=np.empty(0, dtype=np.float32),
        )

    mean_values: list[float] = []
    peak_values: list[float] = []
    max_values: list[float] = []
    background_values: list[float] = []
    contrast_mean_values: list[float] = []
    contrast_peak_values: list[float] = []

    for frame_index in frame_indices.tolist():
        frame = video.get_frame(int(frame_index))
        gray = _as_gray_2d(frame)
        crop = gray[y0:y1, x0:x1]
        inside = crop[mask_crop]
        background = crop[ring_mask] if np.any(ring_mask) else crop[~mask_crop]
        if background.size == 0:
            background = crop.reshape(-1)

        mean_value = float(np.mean(inside))
        peak_value = float(np.percentile(inside, 95.0))
        max_value = float(np.max(inside))
        background_mean = float(np.mean(background))

        mean_values.append(mean_value)
        peak_values.append(peak_value)
        max_values.append(max_value)
        background_values.append(background_mean)
        contrast_mean_values.append(mean_value - background_mean)
        contrast_peak_values.append(peak_value - background_mean)

    times_seconds = frame_indices.astype(np.float32) / float(video.fps)
    return MaskedBrightnessTrace(
        frame_indices=frame_indices.astype(np.int32, copy=False),
        times_seconds=times_seconds.astype(np.float32, copy=False),
        mean_brightness=np.asarray(mean_values, dtype=np.float32),
        peak_brightness=np.asarray(peak_values, dtype=np.float32),
        max_brightness=np.asarray(max_values, dtype=np.float32),
        background_mean=np.asarray(background_values, dtype=np.float32),
        contrast_mean=np.asarray(contrast_mean_values, dtype=np.float32),
        contrast_peak=np.asarray(contrast_peak_values, dtype=np.float32),
    )


def _sample_frame_indices(*, video: VideoLike, sample_rate_hz: float, max_seconds: float) -> np.ndarray:
    if video.frames <= 0:
        return np.empty(0, dtype=np.int32)
    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive")
    fps = float(video.fps)
    if fps <= 0.0:
        raise ValueError("video.fps must be positive")

    stride = max(1, int(round(fps / float(sample_rate_hz))))
    max_frames = int(video.frames)
    if max_seconds > 0.0:
        max_frames = min(max_frames, int(np.floor(float(max_seconds) * fps)) + 1)
    return np.arange(0, max_frames, stride, dtype=np.int32)


def _coerce_bbox(
    mask_bool: np.ndarray,
    *,
    bbox_xyxy: tuple[float, float, float, float] | None,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    if bbox_xyxy is None:
        nonzero_y, nonzero_x = np.nonzero(mask_bool)
        x0 = int(nonzero_x.min())
        y0 = int(nonzero_y.min())
        x1 = int(nonzero_x.max()) + 1
        y1 = int(nonzero_y.max()) + 1
        return x0, y0, x1, y1

    raw_x0, raw_y0, raw_x1, raw_y1 = bbox_xyxy
    x0 = max(0, int(np.floor(float(raw_x0))))
    y0 = max(0, int(np.floor(float(raw_y0))))
    x1 = min(width, int(np.ceil(float(raw_x1))))
    y1 = min(height, int(np.ceil(float(raw_y1))))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"invalid bbox_xyxy: {bbox_xyxy!r}")
    return x0, y0, x1, y1


def _ring_mask(mask_crop: np.ndarray, *, ring_px: int) -> np.ndarray:
    if ring_px <= 0:
        return ~mask_crop
    kernel_size = (2 * int(ring_px)) + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dilated = cv2.dilate(mask_crop.astype(np.uint8), kernel, iterations=1) > 0
    return dilated & ~mask_crop


def _as_gray_2d(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim == 2:
        return array.astype(np.float32, copy=False)
    if array.ndim == 3 and array.shape[2] == 1:
        return array[..., 0].astype(np.float32, copy=False)
    if array.ndim == 3 and array.shape[2] == 3:
        return bgr_to_gray(array).astype(np.float32, copy=False)
    raise ValueError(f"unsupported frame shape: {array.shape}")


__all__ = [
    "MaskedBrightnessTrace",
    "VideoLike",
    "extract_masked_brightness_trace",
]
