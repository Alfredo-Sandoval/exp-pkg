"""In-memory exchange helpers for canonical xpkg objects.

This surface is intentionally separate from:

- ``xpkg.model`` for the canonical object graph
- ``xpkg.project`` for project and project artifacts

Use ``xpkg.adapters`` when another repo needs to transform xpkg objects into
JSON-friendly payloads, numpy arrays, or tabular structures without coupling
to project or archive internals.
"""

from __future__ import annotations

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

__all__ = [
    "labels_from_json_payload",
    "labels_numpy",
    "labels_to_dataframe",
    "labels_to_json_payload",
    "read_vicon_json_payload",
    "vicon_recording_from_json_payload",
    "vicon_recording_to_json_payload",
]
