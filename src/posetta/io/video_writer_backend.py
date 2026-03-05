"""FFmpeg/backend plumbing for video writer selection."""

from __future__ import annotations

import os
import subprocess
import sys

import imageio_ffmpeg as _ff

from posetta.config import settings

_AUTO_CODEC: str | None = None
_NVENC_FLAGS_CACHE: dict[str, dict[str, bool]] = {}
_FFMPEG_ENCODER_CACHE: set[str] | None = None
_FFMPEG_EXE_HINT: str | None = None


def apply_ffmpeg_env() -> None:
    """Apply runtime ffmpeg overrides for imageio-ffmpeg."""
    global _FFMPEG_ENCODER_CACHE, _AUTO_CODEC, _NVENC_FLAGS_CACHE, _FFMPEG_EXE_HINT
    ffmpeg_bin = str(settings.video.ffmpeg.bin or "").strip()
    desired = ffmpeg_bin if ffmpeg_bin and ffmpeg_bin != "ffmpeg" else ""
    if desired != (_FFMPEG_EXE_HINT or ""):
        _FFMPEG_ENCODER_CACHE = None
        _AUTO_CODEC = None
        _NVENC_FLAGS_CACHE.clear()
        _FFMPEG_EXE_HINT = desired or None
    if desired:
        os.environ["IMAGEIO_FFMPEG_EXE"] = desired


def ffmpeg_encoders() -> set[str]:
    global _FFMPEG_ENCODER_CACHE
    if _FFMPEG_ENCODER_CACHE is not None:
        return _FFMPEG_ENCODER_CACHE

    apply_ffmpeg_env()
    exe = _ff.get_ffmpeg_exe()
    proc = subprocess.run([exe, "-hide_banner", "-encoders"], capture_output=True, check=False)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"ffmpeg -encoders failed ({proc.returncode}): {stderr}")
    text = proc.stdout.decode("utf-8", "replace")
    encoders: set[str] = set()
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        flags = parts[0]
        name = parts[1]
        if len(flags) >= 2 and flags[0] in {"V", "A", "S"}:
            encoders.add(name)
    if not encoders:
        raise RuntimeError("ffmpeg -encoders returned no parsable encoder names.")
    _FFMPEG_ENCODER_CACHE = encoders
    return encoders


def platform_preferred_encoders() -> list[str]:
    if sys.platform == "darwin":
        return ["h264_videotoolbox", "hevc_videotoolbox"]
    if sys.platform.startswith("linux"):
        return [
            "h264_nvenc",
            "hevc_nvenc",
            "av1_nvenc",
            "h264_qsv",
            "hevc_qsv",
            "h264_vaapi",
            "hevc_vaapi",
        ]
    return []


def select_platform_codec() -> str | None:
    encoders = ffmpeg_encoders()
    for name in platform_preferred_encoders():
        if name in encoders:
            return name
    return None


def require_ffmpeg_encoder(codec: str) -> None:
    """Raise if the requested encoder is not available."""
    if codec not in ffmpeg_encoders():
        raise RuntimeError(
            f"Requested FFmpeg encoder '{codec}' is not available. "
            "Install an FFmpeg build with that encoder or update "
            "settings.video.writer.prefer_codec."
        )


def auto_select_codec() -> str:
    global _AUTO_CODEC
    if _AUTO_CODEC is not None:
        return _AUTO_CODEC

    encoders = ffmpeg_encoders()
    candidates = [*platform_preferred_encoders(), "libx264"]
    for name in candidates:
        if name in encoders:
            _AUTO_CODEC = name
            return _AUTO_CODEC
    raise RuntimeError(
        "No usable FFmpeg encoder found. Install an FFmpeg build with "
        "libx264 or set settings.video.writer.prefer_codec explicitly."
    )


def supported_nvenc_flags(codec: str) -> dict[str, bool]:
    """Probe `ffmpeg -h encoder=<codec>` for supported NVENC flags (cached)."""
    if codec in _NVENC_FLAGS_CACHE:
        return _NVENC_FLAGS_CACHE[codec]

    tokens = [
        "-tune",
        "-rc:v",
        "-cq",
        "-spatial_aq",
        "-spatial-aq",
        "-temporal_aq",
        "-temporal-aq",
        "-look_ahead",
        "-rc-lookahead",
        "-rc_lookahead",
        "-profile:v",
    ]
    out = {t: False for t in tokens}
    apply_ffmpeg_env()
    exe = _ff.get_ffmpeg_exe()
    proc = subprocess.run(
        [exe, "-hide_banner", "-h", f"encoder={codec}"],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8").strip()
        raise RuntimeError(f"ffmpeg encoder help failed ({proc.returncode}): {stderr}")
    helptext = proc.stdout.decode("utf-8") + proc.stderr.decode("utf-8")
    lower_text = helptext.lower()
    for token in tokens:
        key = token.lstrip("-").replace("_", "-")
        out[token] = (key in lower_text) or (token in lower_text)

    _NVENC_FLAGS_CACHE[codec] = out
    return out
