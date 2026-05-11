"""Top-level import surface for the xpkg runtime package.

The distribution is published as ``exp-pkg`` while downstream code imports the
runtime package as ``xpkg``. This namespace keeps the project, service, model,
adapter, media, and reader entry points lazy so importing ``xpkg`` does not
eagerly import optional IO stacks.

The canonical import locations are:

* :mod:`xpkg.project` for project lifecycle and metadata
* :mod:`xpkg.services` for the service-first ``ProjectService`` API
* :mod:`xpkg.model` for typed data classes
* :mod:`xpkg.json_utils` for JSON payload parsing and serialization
* :mod:`xpkg.readers` for format readers (``read_*`` functions)
* :mod:`xpkg.adapters` for exchange adapters
* :mod:`xpkg.media` for video/image IO
"""

from __future__ import annotations

import importlib

from xpkg.version import __version__

__all__ = [
    "__version__",
    "adapters",
    "json_utils",
    "media",
    "model",
    "pose",
    "project",
    "readers",
    "segmentation",
    "services",
]

_LAZY_SUBMODULES = frozenset(__all__) - {"__version__"}


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        module = importlib.import_module(f"xpkg.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module 'xpkg' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
