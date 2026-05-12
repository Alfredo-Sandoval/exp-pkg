"""In-memory exchange adapters for canonical xpkg objects.

This surface is intentionally separate from:

- ``xpkg.model`` for the canonical object graph
- ``xpkg.project`` for project and project artifacts

Use ``xpkg.adapters`` when another repo needs to transform xpkg objects into
JSON-friendly payloads, numpy arrays, or tabular structures without coupling
to project storage internals.

The :mod:`xpkg.adapters.primitives` submodule depends on the optional sibling
``primitives`` package. Its exports are loaded lazily so importing
``xpkg.adapters`` (or unrelated re-exports like :func:`vicon_recording_from_json_payload`)
does not require ``primitives`` to be installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from xpkg.adapters.vicon import (
    read_vicon_json_payload,
    vicon_recording_from_json_payload,
    vicon_recording_to_json_payload,
)
from xpkg.io.labels.export_ops import (
    labels_numpy,
    labels_to_dataframe,
)
from xpkg.io.labels.json_format import (
    labels_from_json_payload,
    labels_to_json_payload,
)

if TYPE_CHECKING:
    from xpkg.adapters.primitives import (
        labels_to_primitives_session,
        project_to_primitives_session,
    )

_PRIMITIVES_EXPORTS = frozenset({"labels_to_primitives_session", "project_to_primitives_session"})


def __getattr__(name: str) -> Any:
    if name in _PRIMITIVES_EXPORTS:
        from xpkg.adapters import primitives as _primitives_module

        return getattr(_primitives_module, name)
    raise AttributeError(f"module 'xpkg.adapters' has no attribute {name!r}")


__all__ = [
    "labels_from_json_payload",
    "labels_numpy",
    "labels_to_dataframe",
    "labels_to_json_payload",
    "labels_to_primitives_session",
    "project_to_primitives_session",
    "read_vicon_json_payload",
    "vicon_recording_from_json_payload",
    "vicon_recording_to_json_payload",
]
