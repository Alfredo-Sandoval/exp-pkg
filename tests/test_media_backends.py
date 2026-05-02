from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

import xpkg
from xpkg.media import (
    available_media_backends,
    media_backend_status,
    media_backend_status_by_name,
    missing_media_backends,
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
    assert statuses["torch"].extra == "dl"
    assert statuses["torchcodec"].extra == "dl"
    assert statuses["torchvision"].extra == "dl"
    assert statuses["onnxruntime"].extra == "inference"
    assert statuses["kornia"].extra == "vision"
    assert set(missing_media_backends()).isdisjoint({"images", "opencv", "imageio"})


def test_media_backend_lookup_normalizes_common_aliases() -> None:
    assert media_backend_status_by_name("av").name == "pyav"
    assert media_backend_status_by_name("onnx").name == "onnxruntime"
    assert media_backend_status_by_name("torch-codec").name == "torchcodec"


def test_require_media_backend_raises_actionable_error_for_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Known backends"):
        require_media_backend("not-a-backend")


def test_require_media_backend_raises_actionable_error_for_missing_backend() -> None:
    status = media_backend_status_by_name("pyav")
    if status.available:
        pytest.skip("pyav is installed in this environment")

    with pytest.raises(ImportError, match=r"exp-pkg\[media-rich\]"):
        require_media_backend("pyav")


def test_media_surface_exports_backend_registry() -> None:
    assert callable(xpkg.media.media_backend_status)
    assert callable(xpkg.media.media_backend_status_by_name)
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
    assert extras["vision"] == ["kornia>=0.8,<1", "torch>=2.11,<2.12"]
    assert extras["media-dl"] == [
        "av>=16,<17",
        "kornia>=0.8,<1",
        "onnxruntime>=1.24,<2",
        "torch>=2.11,<2.12",
        "torchcodec>=0.11,<0.12",
        "torchvision>=0.26,<0.27",
    ]
