"""Public package namespace for xpkg."""

from __future__ import annotations

import importlib

from xpkg.version import __version__

__all__ = ["__version__", "adapters", "api", "codecs", "compat", "formats", "model", "services"]


def __getattr__(name: str):
    if name in {"adapters", "api", "codecs", "compat", "formats", "model", "services"}:
        module = importlib.import_module(f"xpkg.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module 'xpkg' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
