"""Shared helpers for file-backed media backend requests."""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from typing import Any

import numpy as np

from .._core.colors import rgb_to_bgr

FILE_VIDEO_BACKENDS = frozenset({"auto", "decord-gpu", "opencv", "pyav"})


def normalize_file_video_backend(value: Any, *, label: str) -> str:
    """Normalize an explicit file-video backend request."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    backend = value.strip().lower().replace("_", "-")
    if backend == "av":
        backend = "pyav"
    if backend not in FILE_VIDEO_BACKENDS:
        allowed = ", ".join(sorted(FILE_VIDEO_BACKENDS))
        raise ValueError(f"Unknown video backend {value!r}; {label} must be one of {{{allowed}}}")
    return backend


def require_decord_gpu_api() -> tuple[Any, Any, Any]:
    """Return Decord GPU API objects or raise an explicit import error."""
    if find_spec("decord") is None:
        raise ImportError("decord is not installed in the active environment")
    decord = import_module("decord")
    if "VideoReader" not in decord.__dict__ or "gpu" not in decord.__dict__:
        raise ImportError("decord GPU API is unavailable in the active environment")
    return decord, decord.__dict__["VideoReader"], decord.__dict__["gpu"]


def decord_frame_bgr(frame: Any) -> np.ndarray:
    """Convert a Decord RGB frame object to xpkg's BGR numpy contract."""
    return rgb_to_bgr(frame.asnumpy())


__all__ = [
    "FILE_VIDEO_BACKENDS",
    "decord_frame_bgr",
    "normalize_file_video_backend",
    "require_decord_gpu_api",
]
