"""Optional media and deep-learning backend discovery."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec
from typing import Any

__all__ = [
    "HardwareAccelerationStatus",
    "MediaBackendStatus",
    "available_media_backends",
    "available_hardware_accelerators",
    "hardware_acceleration_status",
    "hardware_acceleration_status_by_name",
    "media_backend_status",
    "media_backend_status_by_name",
    "missing_media_backends",
    "missing_hardware_accelerators",
    "require_hardware_acceleration",
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
class HardwareAccelerationStatus:
    """Runtime status for optional hardware acceleration paths."""

    name: str
    role: str
    extra: str | None
    available: bool
    reason: str
    details: dict[str, str]


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


@dataclass(frozen=True, slots=True)
class _HardwareAccelerationSpec:
    name: str
    role: str
    detector: Callable[[], tuple[bool, str, dict[str, str]]]
    extra: str | None = None

    def status(self) -> HardwareAccelerationStatus:
        try:
            available, reason, details = self.detector()
        except Exception as exc:  # pragma: no cover - defensive boundary for optional stacks.
            available = False
            reason = f"{type(exc).__name__}: {exc}"
            details = {}
        return HardwareAccelerationStatus(
            name=self.name,
            role=self.role,
            extra=self.extra,
            available=available,
            reason=reason,
            details=details,
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
    _MediaBackendSpec(
        name="mlx",
        modules=("mlx",),
        role="Apple MLX tensor runtime for Metal-accelerated model pipelines",
        extra="mlx",
    ),
)

_HARDWARE_ACCELERATION: tuple[_HardwareAccelerationSpec, ...] = (
    _HardwareAccelerationSpec(
        name="mlx-metal",
        role="Apple MLX Metal acceleration for model/tensor pipelines",
        detector=lambda: _detect_mlx_metal(),
        extra="mlx",
    ),
    _HardwareAccelerationSpec(
        name="torch-cuda",
        role="NVIDIA CUDA acceleration through PyTorch tensors",
        detector=lambda: _detect_torch_cuda(),
        extra="nvidia",
    ),
    _HardwareAccelerationSpec(
        name="torchcodec-cuda",
        role="NVIDIA CUDA/NVDEC video tensor decoding through TorchCodec",
        detector=lambda: _detect_torchcodec_cuda(),
        extra="nvidia",
    ),
    _HardwareAccelerationSpec(
        name="ffmpeg-nvidia",
        role="NVIDIA NVDEC/NVENC support exposed by the host FFmpeg binary",
        detector=lambda: _detect_ffmpeg_nvidia(),
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


def hardware_acceleration_status(
    *, include_unavailable: bool = True
) -> tuple[HardwareAccelerationStatus, ...]:
    """Return runtime status for known xpkg hardware acceleration paths."""
    statuses = tuple(spec.status() for spec in _HARDWARE_ACCELERATION)
    if include_unavailable:
        return statuses
    return tuple(status for status in statuses if status.available)


def hardware_acceleration_status_by_name(name: str) -> HardwareAccelerationStatus:
    """Return hardware acceleration status by canonical accelerator name."""
    normalized = _normalize_hardware_name(name)
    for status in hardware_acceleration_status():
        if status.name == normalized:
            return status
    known = ", ".join(status.name for status in hardware_acceleration_status())
    raise ValueError(f"Unknown hardware accelerator: {name}. Known accelerators: {known}")


def available_hardware_accelerators() -> tuple[str, ...]:
    """Return canonical hardware acceleration names available on this host."""
    return tuple(
        status.name for status in hardware_acceleration_status(include_unavailable=False)
    )


def missing_hardware_accelerators() -> tuple[str, ...]:
    """Return canonical hardware acceleration names unavailable on this host."""
    return tuple(status.name for status in hardware_acceleration_status() if not status.available)


def require_hardware_acceleration(name: str) -> HardwareAccelerationStatus:
    """Return hardware acceleration status or raise an actionable RuntimeError."""
    status = hardware_acceleration_status_by_name(name)
    if status.available:
        return status
    if status.extra:
        install_hint = f"Install `exp-pkg[{status.extra}]` and verify host drivers."
    else:
        install_hint = "Verify the required host driver, runtime, or FFmpeg build."
    raise RuntimeError(
        f"Hardware accelerator `{status.name}` is unavailable. {status.reason}. {install_hint}"
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


def _normalize_hardware_name(name: str) -> str:
    normalized = str(name).strip().lower().replace("_", "-")
    aliases = {
        "cuda": "torch-cuda",
        "ffmpeg-cuda": "ffmpeg-nvidia",
        "mlx": "mlx-metal",
        "metal": "mlx-metal",
        "nvdec": "ffmpeg-nvidia",
        "nvenc": "ffmpeg-nvidia",
        "nvidia": "torch-cuda",
        "torchcodec": "torchcodec-cuda",
        "torch-codec": "torchcodec-cuda",
    }
    return aliases.get(normalized, normalized)


def _module_available(module: str) -> bool:
    return find_spec(module) is not None


def _detect_mlx_metal() -> tuple[bool, str, dict[str, str]]:
    if not _module_available("mlx"):
        return False, "Python module `mlx` is not installed", {}
    mx: Any = import_module("mlx.core")
    metal: Any = import_module("mlx.core.metal")

    if not bool(metal.is_available()):
        return False, "MLX is installed but Metal acceleration is unavailable", {}
    info = {str(key): str(value) for key, value in mx.device_info().items()}
    device = str(mx.default_device())
    info.setdefault("default_device", device)
    return True, "MLX Metal acceleration is available", info


def _detect_torch_cuda() -> tuple[bool, str, dict[str, str]]:
    if not _module_available("torch"):
        return False, "Python module `torch` is not installed", {}
    torch: Any = import_module("torch")

    if not bool(torch.cuda.is_available()):
        return False, "PyTorch is installed but CUDA is unavailable", {}
    device_count = int(torch.cuda.device_count())
    details = {
        "device_count": str(device_count),
        "torch_cuda_version": str(getattr(torch.version, "cuda", "") or ""),
    }
    for index in range(device_count):
        details[f"device_{index}"] = str(torch.cuda.get_device_name(index))
    return True, "PyTorch CUDA acceleration is available", details


def _detect_torchcodec_cuda() -> tuple[bool, str, dict[str, str]]:
    if not _module_available("torchcodec"):
        return False, "Python module `torchcodec` is not installed", {}
    torch_available, torch_reason, torch_details = _detect_torch_cuda()
    if not torch_available:
        return False, f"TorchCodec is installed but CUDA is unavailable: {torch_reason}", {}
    return True, "TorchCodec CUDA path is importable and PyTorch CUDA is available", torch_details


def _detect_ffmpeg_nvidia() -> tuple[bool, str, dict[str, str]]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False, "`ffmpeg` is not available on PATH", {}
    decoder_output = _ffmpeg_table(ffmpeg, "-decoders")
    encoder_output = _ffmpeg_table(ffmpeg, "-encoders")
    has_nvdecode = any(token in decoder_output.lower() for token in ("cuvid", "nvdec"))
    has_nvencode = "nvenc" in encoder_output.lower()
    details = {
        "ffmpeg": ffmpeg,
        "nvdec": str(has_nvdecode),
        "nvenc": str(has_nvencode),
    }
    if has_nvdecode or has_nvencode:
        return True, "FFmpeg exposes NVIDIA decode or encode support", details
    return False, "FFmpeg is installed but does not expose NVDEC/NVENC support", details


def _ffmpeg_table(ffmpeg: str, option: str) -> str:
    result = subprocess.run(
        [ffmpeg, "-hide_banner", option],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return f"{result.stdout}\n{result.stderr}"
