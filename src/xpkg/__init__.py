"""Public package namespace for xpkg."""

from __future__ import annotations

from xpkg import adapters, api, formats, model, services
from xpkg.version import __version__

__all__ = ["__version__", "adapters", "api", "formats", "model", "services"]
