"""Public video facade for media readers, writers, and transforms."""

from __future__ import annotations

from xpkg.media.readers import (
    SingleImageVideo,
    Video,
    VideoReader,
    available_video_exts,
    gui_playback_backend_for_path,
)
from xpkg.media.transforms import augment_background, resize_image, resize_images
from xpkg.media.writers import VideoWriter, VideoWriterImageio, VideoWriterOpenCV, write_video

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
