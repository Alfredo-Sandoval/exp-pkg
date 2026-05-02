"""Media primitives for images, videos, and image sequences."""

from __future__ import annotations

from xpkg.media.images import read_bgr, read_rgb, read_rgb_bytes
from xpkg.media.video import (
    SingleImageVideo,
    Video,
    VideoReader,
    VideoWriter,
    VideoWriterImageio,
    VideoWriterOpenCV,
    augment_background,
    available_video_exts,
    gui_playback_backend_for_path,
    resize_image,
    resize_images,
    write_video,
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
    "read_bgr",
    "read_rgb",
    "read_rgb_bytes",
    "resize_image",
    "resize_images",
    "write_video",
]
