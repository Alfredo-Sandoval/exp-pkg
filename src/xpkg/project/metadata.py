"""Project metadata helpers owned by the public project package."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from xpkg.project.store import load_project_metadata, save_project_metadata


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


def load_project_metadata_field(path: str | Path, field: str) -> dict[str, Any] | None:
    """Load one mapping-valued metadata field from the current project head."""
    metadata = load_project_metadata(path)
    if metadata is None:
        return None
    raw_value = metadata.get(_require_field_name(field))
    if raw_value is None:
        return None
    return _require_mapping(raw_value, name=f"project_metadata.{field}")


def save_project_metadata_field(
    path: str | Path,
    field: str,
    value: Mapping[str, Any],
    *,
    reason: str,
) -> Path:
    """Persist one mapping-valued metadata field onto the current project head."""
    attr_key = _require_field_name(field)
    metadata = load_project_metadata(path) or {}
    metadata[attr_key] = _require_mapping(value, name=f"project_metadata.{attr_key}")
    return save_project_metadata(path, metadata, reason=reason)


__all__ = [
    "load_project_metadata_field",
    "save_project_metadata_field",
]
