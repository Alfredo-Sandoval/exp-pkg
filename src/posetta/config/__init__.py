"""Minimal runtime configuration for the extracted Posetta package."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class FfmpegSettings:
    bin: str | None = None
    hwaccel_playback: str | None = None
    vaapi_device: str | None = None
    timeout_sec: float | None = None


@dataclass(slots=True)
class WriterSettings:
    prefer_codec: str | None = None
    tune_nvenc: bool = True


@dataclass(slots=True)
class VideoSettings:
    recognized_exts: tuple[str, ...] = (
        ".avi",
        ".mov",
        ".mp4",
        ".mkv",
        ".mpeg",
        ".mpg",
        ".wmv",
        ".m4v",
        ".webm",
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
    )
    ffmpeg: FfmpegSettings = field(default_factory=FfmpegSettings)
    writer: WriterSettings = field(default_factory=WriterSettings)


@dataclass(slots=True)
class LoggingSettings:
    level: str = "WARNING"
    file: str | None = None


@dataclass(slots=True)
class Settings:
    video: VideoSettings = field(default_factory=VideoSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)


settings = Settings()

__all__ = ["FfmpegSettings", "LoggingSettings", "Settings", "VideoSettings", "settings"]
