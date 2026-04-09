"""Minimal runtime configuration for xpkg."""

from __future__ import annotations

from dataclasses import dataclass, field


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
@dataclass(slots=True)
class LoggingSettings:
    level: str = "WARNING"
    file: str | None = None


@dataclass(slots=True)
class Settings:
    video: VideoSettings = field(default_factory=VideoSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)


settings = Settings()

__all__ = ["LoggingSettings", "Settings", "VideoSettings", "settings"]
