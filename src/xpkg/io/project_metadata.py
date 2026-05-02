"""Compatibility wrapper for workspace metadata field helpers."""

from __future__ import annotations

from xpkg.workspace.metadata import (
    load_workspace_metadata_field,
    save_workspace_metadata_field,
)

__all__ = [
    "load_workspace_metadata_field",
    "save_workspace_metadata_field",
]
