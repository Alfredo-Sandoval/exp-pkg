"""Video writer utilities for canonical numpy media arrays."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from threading import Event
from typing import Any, Protocol, cast

import cv2
import imageio.v2 as iio
import imageio_ffmpeg as _ff
import numpy as np

from xpkg._core.colors import bgr_to_rgb, ensure_bgr, ensure_three_channels
from xpkg._core.path_registry import ensure_dir, resolve_path
from xpkg.media.transforms import augment_background, resize_image

__all__ = [
    "VideoWriter",
    "VideoWriterImageio",
    "VideoWriterOpenCV",
    "build_video_writer",
    "can_use_ffmpeg_writer",
    "ffmpeg_encoders",
    "platform_preferred_encoders",
    "supported_nvenc_flags",
    "write_video",
]

_NVENC_FLAGS_CACHE: dict[str, dict[str, bool]] = {}
_FFMPEG_ENCODER_CACHE: set[str] | None = None


class _FourCCFunction(Protocol):
    def __call__(self, c1: str, c2: str, c3: str, c4: str) -> int: ...


def _video_writer_fourcc(code: str) -> int:
    fourcc_fn = getattr(cv2, "VideoWriter_fourcc", None)
    if not callable(fourcc_fn):
        raise RuntimeError("OpenCV build does not expose VideoWriter_fourcc")
    return cast(_FourCCFunction, fourcc_fn)(code[0], code[1], code[2], code[3])


class VideoWriterOpenCV:
    """OpenCV-backed encoder used for AVI output."""

    def __init__(
        self,
        filename: str,
        height: int,
        width: int,
        fps: float,
        *,
        fourcc: str | None = None,
        is_color: bool = True,
    ):
        ext = Path(filename).suffix.lower()
        code = fourcc or ("MJPG" if ext == ".avi" else "mp4v")
        if len(code) != 4:
            raise ValueError(f"fourcc must be exactly 4 characters, got {code!r}")
        writer = cv2.VideoWriter(
            filename,
            _video_writer_fourcc(code),
            fps,
            (int(width), int(height)),
            bool(is_color),
        )
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f"Failed to open OpenCV VideoWriter for {filename} (fourcc={code})")
        self._writer = writer

    def add_frame(self, image: np.ndarray, *, bgr: bool = False) -> None:
        self._writer.write(ensure_bgr(image, input_is_bgr=bgr))

    def close(self) -> None:
        self._writer.release()

    def __enter__(self) -> VideoWriterOpenCV:
        return self

    def __exit__(self, _exc_type: Any, _exc_value: Any, _traceback: Any) -> None:
        self.close()


class VideoWriterImageio:
    """ImageIO-backed encoder used for non-AVI outputs."""

    _NVENC_CODECS = frozenset({"h264_nvenc", "hevc_nvenc", "av1_nvenc"})

    def __init__(
        self,
        filename: str,
        height: int,
        width: int,
        fps: float,
        *,
        crf: int = 21,
        preset: str = "superfast",
        pixelformat: str = "yuv420p",
        codec: str | None = None,
        output_params: list[str] | None = None,
        tune_nvenc: bool = True,
        **_kwargs: Any,
    ):
        del height, width
        desired_codec = self._resolve_codec(codec)
        self.codec = desired_codec
        self.output_params = self._resolve_output_params(
            desired_codec=desired_codec,
            output_params=output_params,
            tune_nvenc=tune_nvenc,
            preset=preset,
            crf=crf,
        )
        self._writer = iio.get_writer(
            filename,
            fps=fps,
            codec=desired_codec,
            pixelformat=pixelformat,
            macro_block_size=1,
            output_params=self.output_params,
        )

    @staticmethod
    def _resolve_codec(codec: str | None) -> str:
        if codec is None:
            return VideoWriterImageio._auto_select_codec()
        codec_name = str(codec)
        if codec_name not in ffmpeg_encoders():
            raise RuntimeError(
                f"Requested FFmpeg encoder '{codec_name}' is not available. "
                "Install an FFmpeg build with that encoder or choose a different codec."
            )
        return codec_name

    @classmethod
    def _resolve_output_params(
        cls,
        *,
        desired_codec: str,
        output_params: list[str] | None,
        tune_nvenc: bool,
        preset: str,
        crf: int,
    ) -> list[str]:
        if output_params:
            return list(output_params)
        if desired_codec in cls._NVENC_CODECS and bool(tune_nvenc):
            return cls._build_nvenc_output_params(desired_codec)
        return ["-preset", preset, "-crf", str(crf)]

    @staticmethod
    def _build_nvenc_output_params(desired_codec: str) -> list[str]:
        params = ["-preset", "p5"]
        supported = supported_nvenc_flags(desired_codec)
        _append_if_supported(params, supported, "-tune", "hq")
        _append_if_supported(params, supported, "-rc:v", "vbr")
        cq = "22" if desired_codec != "av1_nvenc" else "28"
        _append_if_supported(params, supported, "-cq", cq)
        _append_first_supported(params, supported, ("-spatial_aq", "-spatial-aq"), "1")
        _append_first_supported(params, supported, ("-temporal_aq", "-temporal-aq"), "1")
        _append_first_supported(
            params,
            supported,
            ("-look_ahead", "-rc-lookahead", "-rc_lookahead"),
            "32",
        )
        if desired_codec == "hevc_nvenc":
            _append_if_supported(params, supported, "-profile:v", "main")
        if desired_codec == "h264_nvenc":
            _append_if_supported(params, supported, "-profile:v", "high")
        return params

    @staticmethod
    def _auto_select_codec() -> str:
        encoders = ffmpeg_encoders()
        for name in [*platform_preferred_encoders(), "libx264"]:
            if name in encoders:
                return name
        raise RuntimeError(
            "No usable FFmpeg encoder found. Install an FFmpeg build with libx264 "
            "or request backend='opencv'."
        )

    def add_frame(self, image: np.ndarray, *, bgr: bool = False) -> None:
        frame = ensure_three_channels(image)
        if bgr:
            frame = bgr_to_rgb(frame)
        self._writer.append_data(frame)

    def close(self) -> None:
        self._writer.close()

    def __enter__(self) -> VideoWriterImageio:
        return self

    def __exit__(self, _exc_type: Any, _exc_value: Any, _traceback: Any) -> None:
        self.close()


class VideoWriter:
    """Factory wrapper that selects the canonical writer for an output path."""

    @staticmethod
    def builder(
        filename: str,
        height: int,
        width: int,
        fps: float,
        backend: str = "auto",
        **kwargs: Any,
    ) -> VideoWriterOpenCV | VideoWriterImageio:
        ext = Path(filename).suffix.lower()
        chosen_backend = backend.strip().lower() if backend else "auto"
        if chosen_backend == "auto":
            if can_use_ffmpeg_writer() and ext not in {".avi", ".mkv"}:
                chosen_backend = "imageio"
            else:
                chosen_backend = "opencv"
        if chosen_backend in {"opencv", "cv2"}:
            return VideoWriterOpenCV(filename, height, width, fps, **kwargs)
        if chosen_backend in {"imageio", "ffmpeg"}:
            if not can_use_ffmpeg_writer():
                raise RuntimeError(
                    "Requested ffmpeg backend but imageio-ffmpeg is unavailable. "
                    "Install imageio-ffmpeg or choose backend='opencv'."
                )
            return VideoWriterImageio(filename, height, width, fps, **kwargs)
        raise ValueError(f"Unknown video writer backend: {backend}")


def build_video_writer(
    filename: str,
    height: int,
    width: int,
    fps: float,
    backend: str = "auto",
    **kwargs: Any,
) -> VideoWriterOpenCV | VideoWriterImageio:
    """Create a video writer using the canonical backend-selection policy."""
    return VideoWriter.builder(filename, height, width, fps, backend=backend, **kwargs)


def can_use_ffmpeg_writer() -> bool:
    """Return true when imageio-ffmpeg can report a usable ffmpeg version."""
    version = _ff.get_ffmpeg_version()
    return bool(str(version))


def ffmpeg_encoders() -> set[str]:
    """Return FFmpeg encoder names exposed by imageio-ffmpeg's executable."""
    global _FFMPEG_ENCODER_CACHE
    if _FFMPEG_ENCODER_CACHE is not None:
        return _FFMPEG_ENCODER_CACHE
    exe = _ff.get_ffmpeg_exe()
    proc = subprocess.run([exe, "-hide_banner", "-encoders"], capture_output=True, check=False)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"ffmpeg -encoders failed ({proc.returncode}): {stderr}")
    encoders: set[str] = set()
    for line in proc.stdout.decode("utf-8", "replace").splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        flags, name = parts[0], parts[1]
        if len(flags) >= 2 and flags[0] in {"V", "A", "S"}:
            encoders.add(name)
    if not encoders:
        raise RuntimeError("ffmpeg -encoders returned no parsable encoder names.")
    _FFMPEG_ENCODER_CACHE = encoders
    return encoders


def platform_preferred_encoders() -> list[str]:
    """Return platform-preferred hardware encoders before CPU libx264."""
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


def supported_nvenc_flags(codec: str) -> dict[str, bool]:
    """Probe `ffmpeg -h encoder=<codec>` for supported NVENC flags."""
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
    out = {token: False for token in tokens}
    exe = _ff.get_ffmpeg_exe()
    proc = subprocess.run(
        [exe, "-hide_banner", "-h", f"encoder={codec}"],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"ffmpeg encoder help failed ({proc.returncode}): {stderr}")
    helptext = proc.stdout.decode("utf-8", "replace") + proc.stderr.decode("utf-8", "replace")
    lower_text = helptext.lower()
    for token in tokens:
        key = token.lstrip("-").replace("_", "-")
        out[token] = (key in lower_text) or (token in lower_text)
    _NVENC_FLAGS_CACHE[codec] = out
    return out


def _append_if_supported(
    output_params: list[str], supported: dict[str, bool], option: str, *values: str
) -> bool:
    if not supported.get(option):
        return False
    output_params.extend([option, *values])
    return True


def _append_first_supported(
    output_params: list[str],
    supported: dict[str, bool],
    options: tuple[str, ...],
    value: str,
) -> bool:
    for option in options:
        if _append_if_supported(output_params, supported, option, value):
            return True
    return False


def write_video(
    filename: str,
    video: Any,
    frames: list[int],
    fps: int = 15,
    scale: float = 1.0,
    background: str | None = None,
    progress_callback: Any | None = None,
    stop_event: Event | None = None,
) -> None:
    """Write selected frames from `video` into a new video file."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    if not frames:
        return

    filename_path = resolve_path(filename)
    ensure_dir(filename_path.parent)

    first_frame = video.get_frame(int(frames[0]))
    prepared_first = resize_image(
        augment_background(np.expand_dims(first_frame, axis=0), background)[0],
        scale,
    )
    height, width = prepared_first.shape[:2]
    writer = build_video_writer(str(filename_path), height=height, width=width, fps=float(fps))

    try:
        total = len(frames)
        for index, frame_idx in enumerate(frames, start=1):
            if stop_event is not None and stop_event.is_set():
                break
            if index == 1:
                frame = prepared_first
            else:
                raw_frame = video.get_frame(int(frame_idx))
                frame = resize_image(
                    augment_background(np.expand_dims(raw_frame, axis=0), background)[0],
                    scale,
                )
            writer.add_frame(frame, bgr=True)
            if progress_callback is not None:
                progress_callback(index, total)
    finally:
        writer.close()
