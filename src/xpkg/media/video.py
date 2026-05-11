"""Public video facade for media readers, writers, and transforms."""

from __future__ import annotations

from xpkg.media.readers import (
    PyAVVideoResource,
    SingleImageVideo,
    Video,
    VideoReader,
    available_video_exts,
    gui_playback_backend_for_path,
)
from xpkg.media.transforms import augment_background, resize_image, resize_images
from xpkg.media.writers import (
    VideoWriter,
    VideoWriterImageio,
    VideoWriterOpenCV,
    build_video_writer,
    can_use_ffmpeg_writer,
    ffmpeg_encoders,
    platform_preferred_encoders,
    supported_nvenc_flags,
    write_video,
)

__all__ = [
    "PyAVVideoResource",
    "SingleImageVideo",
    "Video",
    "VideoReader",
    "VideoWriter",
    "VideoWriterImageio",
    "VideoWriterOpenCV",
    "augment_background",
    "available_video_exts",
    "build_video_writer",
    "can_use_ffmpeg_writer",
    "ffmpeg_encoders",
    "gui_playback_backend_for_path",
    "platform_preferred_encoders",
    "resize_image",
    "resize_images",
    "supported_nvenc_flags",
    "write_video",
]
