from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

import xpkg
from xpkg.media import (
    available_hardware_accelerators,
    available_media_backends,
    hardware_acceleration_status,
    media_backend_status,
    missing_hardware_accelerators,
    missing_media_backends,
    require_hardware_acceleration,
    require_media_backend,
)

ROOT = Path(__file__).resolve().parents[1]


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def test_media_backend_registry_reports_required_core_backends() -> None:
    statuses = {status.name: status for status in media_backend_status()}

    assert {"images", "opencv", "imageio"} <= set(statuses)
    assert statuses["images"].required is True
    assert statuses["opencv"].required is True
    assert statuses["imageio"].required is True
    assert statuses["opencv"].available is True
    assert statuses["imageio"].available is True
    assert "opencv" in available_media_backends()
    assert "imageio" in available_media_backends()


def test_media_backend_registry_keeps_heavy_stacks_optional() -> None:
    statuses = {status.name: status for status in media_backend_status()}

    assert statuses["pyav"].extra == "media-rich"
    assert statuses["decord"].extra == "media-dl"
    assert statuses["torch"].extra == "dl"
    assert statuses["torchcodec"].extra == "dl"
    assert statuses["torchvision"].extra == "dl"
    assert statuses["onnxruntime"].extra == "inference"
    assert statuses["kornia"].extra == "vision"
    assert statuses["mlx"].extra == "mlx"
    assert statuses["nvpkg"].extra is None
    assert statuses["nvpkg"].modules == ("nvpkg",)
    assert set(missing_media_backends()).isdisjoint({"images", "opencv", "imageio"})


def test_media_backend_lookup_normalizes_common_aliases() -> None:
    assert media_backend_status("av").name == "pyav"
    assert media_backend_status("decord-gpu").name == "decord"
    assert media_backend_status("onnx").name == "onnxruntime"
    assert media_backend_status("torch-codec").name == "torchcodec"


def test_hardware_acceleration_registry_reports_supported_paths() -> None:
    statuses = {status.name: status for status in hardware_acceleration_status()}

    assert {
        "mlx-metal",
        "torch-cuda",
        "torchcodec-cuda",
        "ffmpeg-nvidia",
        "opencv-cuda",
        "pyav-cuda",
        "decord-cuda",
        "dali-cuda",
    } <= set(statuses)
    assert statuses["mlx-metal"].extra == "mlx"
    assert statuses["torch-cuda"].extra == "nvidia"
    assert statuses["torchcodec-cuda"].extra == "nvidia"
    assert statuses["ffmpeg-nvidia"].extra is None
    assert statuses["opencv-cuda"].extra == "nvidia"
    assert statuses["pyav-cuda"].extra == "nvidia"
    assert statuses["decord-cuda"].extra == "nvidia"
    assert statuses["dali-cuda"].extra == "nvidia"
    assert set(available_hardware_accelerators()).isdisjoint(missing_hardware_accelerators())


def test_hardware_acceleration_lookup_normalizes_common_aliases() -> None:
    assert hardware_acceleration_status("mlx").name == "mlx-metal"
    assert hardware_acceleration_status("metal").name == "mlx-metal"
    assert hardware_acceleration_status("cuda").name == "torch-cuda"
    assert hardware_acceleration_status("nvidia").name == "torch-cuda"
    assert hardware_acceleration_status("nvenc").name == "ffmpeg-nvidia"
    assert hardware_acceleration_status("opencv_cuda").name == "opencv-cuda"
    assert hardware_acceleration_status("pyav_cuda").name == "pyav-cuda"
    assert hardware_acceleration_status("decord_cuda").name == "decord-cuda"
    assert hardware_acceleration_status("dali_cuda").name == "dali-cuda"
    assert hardware_acceleration_status("torch-codec").name == "torchcodec-cuda"


def test_require_media_backend_raises_actionable_error_for_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Known backends"):
        require_media_backend("not-a-backend")


def test_require_hardware_acceleration_raises_for_unknown_accelerator() -> None:
    with pytest.raises(ValueError, match="Known accelerators"):
        require_hardware_acceleration("not-an-accelerator")


def test_require_hardware_acceleration_raises_actionable_error_for_missing_accelerator() -> None:
    status = hardware_acceleration_status("torch-cuda")
    if status.available:
        pytest.skip()

    with pytest.raises(RuntimeError, match=r"exp-pkg\[nvidia\]"):
        require_hardware_acceleration("torch-cuda")


def test_require_hardware_acceleration_reports_nvpkg_install_command() -> None:
    status = hardware_acceleration_status("opencv-cuda")
    if status.available:
        pytest.skip()

    with pytest.raises(RuntimeError, match="nvpkg package install opencv_cuda"):
        require_hardware_acceleration("opencv-cuda")


def test_require_media_backend_raises_actionable_error_for_missing_backend() -> None:
    status = media_backend_status("pyav")
    if status.available:
        pytest.skip()

    with pytest.raises(ImportError, match=r"exp-pkg\[media-rich\]"):
        require_media_backend("pyav")


def test_require_media_backend_reports_external_nvpkg_install() -> None:
    status = media_backend_status("nvpkg")
    if status.available:
        pytest.skip()

    with pytest.raises(ImportError, match="Install nvpkg separately"):
        require_media_backend("nvpkg")


def test_media_surface_exports_backend_registry() -> None:
    assert callable(xpkg.media.hardware_acceleration_status)
    assert callable(xpkg.media.available_hardware_accelerators)
    assert callable(xpkg.media.missing_hardware_accelerators)
    assert callable(xpkg.media.require_hardware_acceleration)
    assert callable(xpkg.media.media_backend_status)
    assert callable(xpkg.media.available_media_backends)
    assert callable(xpkg.media.missing_media_backends)
    assert callable(xpkg.media.require_media_backend)


def test_pyproject_declares_media_and_deep_learning_extras() -> None:
    extras = _pyproject()["project"]["optional-dependencies"]

    assert extras["media-rich"] == ["av>=16,<17"]
    assert extras["dl"] == [
        "torch>=2.11,<2.12",
        "torchcodec>=0.11,<0.12",
        "torchvision>=0.26,<0.27",
    ]
    assert extras["inference"] == ["onnxruntime>=1.24,<2"]
    assert extras["mlx"] == ["mlx>=0.31,<1"]
    assert extras["nvidia"] == [
        "torch>=2.11,<2.12",
        "torchcodec>=0.11,<0.12",
        "torchvision>=0.26,<0.27",
    ]
    assert extras["vision"] == ["kornia>=0.8,<1", "torch>=2.11,<2.12"]
    assert extras["hardware-accel"] == [
        "mlx>=0.31,<1",
        "torch>=2.11,<2.12",
        "torchcodec>=0.11,<0.12",
        "torchvision>=0.26,<0.27",
    ]
    assert extras["media-dl"] == [
        "av>=16,<17",
        "decord>=0.6,<1; platform_machine == 'x86_64' or platform_machine == 'AMD64'",
        "kornia>=0.8,<1",
        "mlx>=0.31,<1",
        "onnxruntime>=1.24,<2",
        "torch>=2.11,<2.12",
        "torchcodec>=0.11,<0.12",
        "torchvision>=0.26,<0.27",
    ]
