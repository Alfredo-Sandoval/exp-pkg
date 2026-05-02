"""Public package namespace for xpkg."""

from __future__ import annotations

import importlib

from xpkg.version import __version__

__all__ = [
    "__version__",
    "api",
    "exchange",
    "formats",
    "model",
    "read_events_csv",
    "read_photometry_csv",
    "read_pyphotometry_ppd",
    "services",
]


def __getattr__(name: str):
    if name in {"api", "exchange", "formats", "model", "services"}:
        module = importlib.import_module(f"xpkg.{name}")
        globals()[name] = module
        return module
    if name in {"read_events_csv", "read_photometry_csv", "read_pyphotometry_ppd"}:
        module = importlib.import_module("xpkg.io.readers")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'xpkg' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
