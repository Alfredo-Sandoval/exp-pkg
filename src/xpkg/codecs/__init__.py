"""In-memory codecs for canonical xpkg objects.

This surface is intentionally separate from:

- ``xpkg.model`` for the canonical object graph
- ``xpkg.formats`` for workspace and project artifacts
- ``xpkg.compat`` for the low-level ``.xpkg`` edge seam kept for migration and
  private storage internals

Use ``xpkg.codecs`` when another repo needs to transform xpkg objects into
JSON-friendly payloads, numpy arrays, or tabular structures without coupling
to workspace or archive internals.
"""

from __future__ import annotations

from xpkg.codecs.labels import (
    labels_from_json_payload,
    labels_numpy,
    labels_to_dataframe,
    labels_to_json_payload,
)
from xpkg.codecs.vicon import (
    read_vicon_json_payload,
    vicon_recording_from_json_payload,
    vicon_recording_to_json_payload,
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
