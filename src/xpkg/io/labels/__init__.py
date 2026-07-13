"""Explicit labels JSON readers and writers."""

from xpkg.io.labels.json_format import write_labels_json
from xpkg.io.labels.serialization import read_labels_json

__all__ = ["read_labels_json", "write_labels_json"]
