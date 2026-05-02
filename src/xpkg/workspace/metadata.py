"""Workspace metadata helpers owned by the public workspace package."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from xpkg.io.project_workspace import load_workspace_metadata, save_workspace_metadata


def _require_field_name(field: str) -> str:
    normalized = str(field).strip()
    if not normalized:
        raise ValueError("metadata field name must be a non-empty string")
    return normalized


def _require_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    payload: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError(f"{name} keys must be strings")
        payload[key] = item
    return payload


def load_workspace_metadata_field(path: str | Path, field: str) -> dict[str, Any] | None:
    """Load one mapping-valued metadata field from the current workspace head."""
    metadata = load_workspace_metadata(path)
    if metadata is None:
        return None
    raw_value = metadata.get(_require_field_name(field))
    if raw_value is None:
        return None
    return _require_mapping(raw_value, name=f"workspace_metadata.{field}")


def save_workspace_metadata_field(
    path: str | Path,
    field: str,
    value: Mapping[str, Any],
    *,
    reason: str,
) -> Path:
    """Persist one mapping-valued metadata field onto the current workspace head."""
    attr_key = _require_field_name(field)
    metadata = load_workspace_metadata(path) or {}
    metadata[attr_key] = _require_mapping(value, name=f"workspace_metadata.{attr_key}")
    return save_workspace_metadata(path, metadata, reason=reason)


__all__ = [
    "load_workspace_metadata_field",
    "save_workspace_metadata_field",
]
