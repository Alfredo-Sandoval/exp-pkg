"""Deterministic raw-video probing and selected-frame extraction helpers."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .._core.path_registry import ensure_dir

SUPPORTED_VIDEO_SUFFIXES = frozenset({".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg"})


@dataclass(frozen=True, slots=True)
class VideoPathMetadata:
    path: Path
    frame_count: int
    fps: float
    width: int
    height: int


def is_supported_video_path(path: Path) -> bool:
    """Return whether ``path`` looks like a supported concrete video file."""
    return path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_SUFFIXES


def probe_video_path(video_path: Path) -> VideoPathMetadata:
    """Return deterministic metadata for a filesystem-backed video."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video {video_path}")
    metadata = VideoPathMetadata(
        path=video_path,
        frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        fps=float(capture.get(cv2.CAP_PROP_FPS)),
        width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    capture.release()
    return metadata


def _target_frame_set(frame_indices: Sequence[int]) -> set[int]:
    target = {int(frame_idx) for frame_idx in frame_indices}
    if any(frame_idx < 0 for frame_idx in target):
        raise ValueError("frame indices must be non-negative")
    return target


def select_frame_indices(
    total_frames: int,
    *,
    start_frame: int = 0,
    max_frames: int | None = None,
    frame_stride: int = 1,
) -> list[int]:
    """Resolve a deterministic list of source frame indices."""

    if total_frames < 1:
        raise ValueError("total_frames must be >= 1.")
    if start_frame < 0:
        raise ValueError("start_frame must be >= 0.")
    if start_frame >= total_frames:
        raise ValueError(
            f"start_frame={start_frame} is out of range for {total_frames} frame(s)."
        )
    if frame_stride < 1:
        raise ValueError("frame_stride must be >= 1.")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be >= 1 when provided.")
    selected = list(range(start_frame, total_frames, frame_stride))
    if max_frames is not None:
        selected = selected[:max_frames]
    if not selected:
        raise ValueError("Frame selection produced no frames.")
    return selected


def _stream_selected_frames(
    video_path: Path,
    *,
    frame_indices: Sequence[int],
) -> Iterator[tuple[int, np.ndarray]]:
    requested = _target_frame_set(frame_indices)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video {video_path}")
    frame_idx = 0
    while requested:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_idx in requested:
            yield frame_idx, frame
            requested.remove(frame_idx)
        frame_idx += 1
    capture.release()
    if requested:
        missing = ", ".join(str(idx) for idx in sorted(requested)[:10])
        raise RuntimeError(f"Failed to extract requested video frames: {missing}")


def read_frame_indices(
    video_path: Path,
    *,
    frame_indices: Sequence[int],
) -> dict[int, np.ndarray]:
    """Decode the selected frame indices into memory."""
    return {
        frame_idx: frame
        for frame_idx, frame in _stream_selected_frames(video_path, frame_indices=frame_indices)
    }


def extract_frame_indices(
    video_path: Path,
    *,
    frame_indices: Sequence[int],
    output_dir: Path,
    file_prefix: str = "frame",
    file_extension: str = ".jpg",
    skip_existing: bool = False,
) -> dict[int, Path]:
    """Write selected frame indices to disk and return their output paths."""
    ensure_dir(output_dir)
    requested = _target_frame_set(frame_indices)
    results = {
        frame_idx: output_dir / f"{file_prefix}_{frame_idx:06d}{file_extension}"
        for frame_idx in sorted(requested)
    }
    if skip_existing:
        requested = {idx for idx in requested if not results[idx].exists()}
        if not requested:
            return results
    for frame_idx, frame in _stream_selected_frames(video_path, frame_indices=sorted(requested)):
        output_path = results[frame_idx]
        if not cv2.imwrite(output_path.as_posix(), frame):
            raise RuntimeError(f"Failed to write extracted frame {output_path}")
    return results


__all__ = [
    "SUPPORTED_VIDEO_SUFFIXES",
    "VideoPathMetadata",
    "extract_frame_indices",
    "is_supported_video_path",
    "probe_video_path",
    "read_frame_indices",
    "select_frame_indices",
]
