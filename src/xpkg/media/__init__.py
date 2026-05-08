"""Media primitives for images, videos, and image sequences."""

from __future__ import annotations

from xpkg.media.backends import (
    HardwareAccelerationStatus,
    MediaBackendStatus,
    available_hardware_accelerators,
    available_media_backends,
    hardware_acceleration_status,
    media_backend_status,
    missing_hardware_accelerators,
    missing_media_backends,
    require_hardware_acceleration,
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
    "HardwareAccelerationStatus",
    "MediaBackendStatus",
    "SingleImageVideo",
    "Video",
    "VideoReader",
    "VideoWriter",
    "VideoWriterImageio",
    "VideoWriterOpenCV",
    "augment_background",
    "available_hardware_accelerators",
    "available_video_exts",
    "available_media_backends",
    "gui_playback_backend_for_path",
    "hardware_acceleration_status",
    "media_backend_status",
    "missing_media_backends",
    "missing_hardware_accelerators",
    "read_bgr",
    "read_rgb",
    "read_rgb_bytes",
    "require_hardware_acceleration",
    "require_media_backend",
    "resize_image",
    "resize_images",
    "write_video",
]
