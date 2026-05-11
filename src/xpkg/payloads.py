"""Shared payload-boundary mapping helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def require_str_mapping(value: object, *, label: str) -> dict[str, Any]:
    """Require an internal payload mapping with string keys."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    out: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError(f"{label} must use string keys")
        out[key] = item
    return out


def mapping_or_empty(value: object, *, label: str) -> dict[str, Any]:
    """Return an empty mapping for None, otherwise require string keys."""
    if value is None:
        return {}
    return require_str_mapping(value, label=label)


def coerce_external_mapping_keys(value: object, *, label: str) -> dict[str, object]:
    """Normalize external ingress payload keys to strings once at the boundary."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return {_coerce_external_key(key): item for key, item in value.items()}


def _coerce_external_key(key: object) -> str:
    if isinstance(key, str):
        return key
    if isinstance(key, bytes):
        return key.decode("utf-8")
    return str(key)


__all__ = ["coerce_external_mapping_keys", "mapping_or_empty", "require_str_mapping"]
