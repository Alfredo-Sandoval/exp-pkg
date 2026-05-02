"""Mapping-valued metadata field persistence for archives and workspaces."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from xpkg._core.json_utils import parse_json
from xpkg.io.archive_format.shared import _serialize_json
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


def _decode_archive_metadata_attr(raw_value: Any, *, field: str) -> dict[str, Any] | None:
    if isinstance(raw_value, np.generic):
        raw_value = raw_value.item()
    if isinstance(raw_value, bytes | bytearray | np.bytes_):
        raw_value = bytes(raw_value).decode("utf-8")
    if raw_value is None:
        return None
    if not isinstance(raw_value, str):
        raw_value = str(raw_value)
    stripped = raw_value.strip()
    if not stripped:
        return None
    return _require_mapping(
        parse_json(stripped),
        name=f"project_metadata.{field}",
    )


def _archive_path(path: str | Path) -> Path:
    return Path(path).resolve()


def load_archive_metadata_field(path: str | Path, field: str) -> dict[str, Any] | None:
    """Load one mapping-valued metadata field from an archive."""
    archive_path = _archive_path(path)
    attr_key = _require_field_name(field)
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")
    try:
        with h5py.File(archive_path, "r") as handle:
            metadata_group = handle.get("project_metadata")
            if metadata_group is None:
                raise ValueError("archive is missing the /project_metadata group")
            if not isinstance(metadata_group, h5py.Group):
                raise TypeError("project_metadata must be an h5py Group")
            raw_value = metadata_group.attrs.get(attr_key)
    except OSError as exc:
        raise ValueError(f"archive is not a valid HDF5/.xpkg file: {archive_path}") from exc
    if raw_value is None:
        return None
    return _decode_archive_metadata_attr(raw_value, field=attr_key)


def save_archive_metadata_field(
    path: str | Path,
    field: str,
    value: Mapping[str, Any],
) -> None:
    """Persist one mapping-valued metadata field onto an existing archive."""
    archive_path = _archive_path(path)
    attr_key = _require_field_name(field)
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")
    payload = _require_mapping(value, name=f"project_metadata.{attr_key}")
    try:
        with h5py.File(archive_path, "r+") as handle:
            metadata_group = handle.get("project_metadata")
            if metadata_group is None:
                raise ValueError("archive is missing the /project_metadata group")
            if not isinstance(metadata_group, h5py.Group):
                raise TypeError("project_metadata must be an h5py Group")
            metadata_group.attrs[attr_key] = _serialize_json(payload)
            handle.flush()
    except OSError as exc:
        raise ValueError(f"archive is not a valid HDF5/.xpkg file: {archive_path}") from exc


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
    "load_archive_metadata_field",
    "load_workspace_metadata_field",
    "save_archive_metadata_field",
    "save_workspace_metadata_field",
]
