"""Public package namespace for xpkg."""

from __future__ import annotations

import importlib

from xpkg.version import __version__

__all__ = [
    "__version__",
    "api",
    "adapters",
    "model",
    "pose",
    "read_doric_photometry",
    "read_events_csv",
    "read_neurophotometrics_csv",
    "read_photometry_csv",
    "read_pmat_events_csv",
    "read_pmat_photometry_csv",
    "read_pyphotometry_csv",
    "read_pyphotometry_ppd",
    "read_rwd_ofrs_session",
    "read_tdt_photometry_block",
    "read_teleopto_h5",
    "services",
    "project",
]


def __getattr__(name: str):
    if name in {"api", "adapters", "model", "pose", "services", "project"}:
        module = importlib.import_module(f"xpkg.{name}")
        globals()[name] = module
        return module
    reader_exports = {
        "read_doric_photometry",
        "read_events_csv",
        "read_neurophotometrics_csv",
        "read_photometry_csv",
        "read_pmat_events_csv",
        "read_pmat_photometry_csv",
        "read_pyphotometry_csv",
        "read_pyphotometry_ppd",
        "read_rwd_ofrs_session",
        "read_tdt_photometry_block",
        "read_teleopto_h5",
    }
    if name in reader_exports:
        module = importlib.import_module("xpkg.io.readers")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'xpkg' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
