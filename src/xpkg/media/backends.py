"""Optional media and deep-learning backend discovery."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec

__all__ = [
    "MediaBackendStatus",
    "available_media_backends",
    "media_backend_status",
    "media_backend_status_by_name",
    "missing_media_backends",
    "require_media_backend",
]


@dataclass(frozen=True, slots=True)
class MediaBackendStatus:
    """Installation status for a media or model-adjacent backend."""

    name: str
    modules: tuple[str, ...]
    role: str
    extra: str | None
    required: bool
    available: bool
    missing_modules: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _MediaBackendSpec:
    name: str
    modules: tuple[str, ...]
    role: str
    extra: str | None = None
    required: bool = False

    def status(self) -> MediaBackendStatus:
        missing = tuple(module for module in self.modules if find_spec(module) is None)
        return MediaBackendStatus(
            name=self.name,
            modules=self.modules,
            role=self.role,
            extra=self.extra,
            required=self.required,
            available=not missing,
            missing_modules=missing,
        )


_MEDIA_BACKENDS: tuple[_MediaBackendSpec, ...] = (
    _MediaBackendSpec(
        name="images",
        modules=("cv2",),
        role="single-image and image-sequence IO through the canonical numpy frame contract",
        required=True,
    ),
    _MediaBackendSpec(
        name="opencv",
        modules=("cv2",),
        role="baseline portable video decode, resize, and AVI writer support",
        required=True,
    ),
    _MediaBackendSpec(
        name="imageio",
        modules=("imageio", "imageio_ffmpeg"),
        role="baseline FFmpeg-backed writer support for non-AVI video outputs",
        required=True,
    ),
    _MediaBackendSpec(
        name="pyav",
        modules=("av",),
        role="rich FFmpeg container, stream, codec, metadata, and filter control",
        extra="media-rich",
    ),
    _MediaBackendSpec(
        name="torch",
        modules=("torch",),
        role="primary tensor and deep-learning runtime",
        extra="dl",
    ),
    _MediaBackendSpec(
        name="torchcodec",
        modules=("torch", "torchcodec"),
        role="PyTorch-native video/audio decode and encode for tensor pipelines",
        extra="dl",
    ),
    _MediaBackendSpec(
        name="torchvision",
        modules=("torch", "torchvision"),
        role="PyTorch vision transforms, image utilities, and model-adjacent operations",
        extra="dl",
    ),
    _MediaBackendSpec(
        name="onnxruntime",
        modules=("onnxruntime",),
        role="portable ONNX model inference runtime",
        extra="inference",
    ),
    _MediaBackendSpec(
        name="kornia",
        modules=("torch", "kornia"),
        role="differentiable computer-vision operations on PyTorch tensors",
        extra="vision",
    ),
)


def media_backend_status(*, include_unavailable: bool = True) -> tuple[MediaBackendStatus, ...]:
    """Return installation status for known xpkg media/model backends."""
    statuses = tuple(spec.status() for spec in _MEDIA_BACKENDS)
    if include_unavailable:
        return statuses
    return tuple(status for status in statuses if status.available)


def media_backend_status_by_name(name: str) -> MediaBackendStatus:
    """Return backend status by canonical backend name."""
    normalized = _normalize_backend_name(name)
    for status in media_backend_status():
        if status.name == normalized:
            return status
    known = ", ".join(status.name for status in media_backend_status())
    raise ValueError(f"Unknown media backend: {name}. Known backends: {known}")


def available_media_backends() -> tuple[str, ...]:
    """Return canonical backend names that are importable in the current environment."""
    return tuple(status.name for status in media_backend_status(include_unavailable=False))


def missing_media_backends() -> tuple[str, ...]:
    """Return canonical optional backend names that are not importable."""
    return tuple(
        status.name
        for status in media_backend_status()
        if not status.available and not status.required
    )


def require_media_backend(name: str) -> MediaBackendStatus:
    """Return backend status or raise an actionable ImportError."""
    status = media_backend_status_by_name(name)
    if status.available:
        return status
    missing = ", ".join(status.missing_modules)
    if status.extra:
        install_hint = f"Install `exp-pkg[{status.extra}]` to enable it."
    else:
        install_hint = "Reinstall the base exp-pkg environment to enable it."
    raise ImportError(
        f"Media backend `{status.name}` is unavailable. Missing module(s): {missing}. "
        f"{install_hint}"
    )


def _normalize_backend_name(name: str) -> str:
    normalized = str(name).strip().lower().replace("_", "-")
    aliases = {
        "av": "pyav",
        "onnx": "onnxruntime",
        "ort": "onnxruntime",
        "torch-codec": "torchcodec",
        "torch-codecs": "torchcodec",
    }
    return aliases.get(normalized, normalized)
