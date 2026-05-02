"""Media primitives for images, videos, and image sequences."""

from __future__ import annotations

from xpkg.media.backends import (
    MediaBackendStatus,
    available_media_backends,
    media_backend_status,
    media_backend_status_by_name,
    missing_media_backends,
    require_media_backend,
)
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
    "MediaBackendStatus",
    "SingleImageVideo",
    "Video",
    "VideoReader",
    "VideoWriter",
    "VideoWriterImageio",
    "VideoWriterOpenCV",
    "augment_background",
    "available_video_exts",
    "available_media_backends",
    "gui_playback_backend_for_path",
    "media_backend_status",
    "media_backend_status_by_name",
    "missing_media_backends",
    "read_bgr",
    "read_rgb",
    "read_rgb_bytes",
    "require_media_backend",
    "resize_image",
    "resize_images",
    "write_video",
]
