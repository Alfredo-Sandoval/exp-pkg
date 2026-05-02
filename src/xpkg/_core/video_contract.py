"""Canonical frame-count contract for video-like objects."""

from __future__ import annotations

from typing import Protocol


class VideoWithFrames(Protocol):
    """Protocol for video-like objects that expose an integer frame count."""

    @property
    def frames(self) -> int: ...


def video_total_frames(video: VideoWithFrames) -> int:
    """Return normalized frame count from the canonical video contract."""
    frames_val = video.frames
    if isinstance(frames_val, bool):
        raise TypeError("Video.frames must be int-like, not bool")
    total_frames = int(frames_val)
    if total_frames < 0:
        raise ValueError("Video.frames must be non-negative")
    return total_frames


__all__ = ["VideoWithFrames", "video_total_frames"]
