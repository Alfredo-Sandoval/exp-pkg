"""Shared validation helpers for dependency-light metadata dataclasses."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any


def finite_float(value: Any, *, name: str) -> float:
    coerced = float(value)
    if not math.isfinite(coerced):
        raise ValueError(f"{name} must be finite, got {coerced!r}.")
    return coerced


def payload_mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return {str(key): item for key, item in value.items()}


def optional_text(value: object | None, *, name: str) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be a non-empty string when provided.")
    return text


def required_text(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be a non-empty string.")
    return text


def strict_text(value: object, *, name: str) -> str:
    """Require a string without normalizing whitespace; empty strings are valid."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if value != value.strip():
        raise ValueError(f"{name} must not contain surrounding whitespace.")
    return value


def strict_required_text(value: object, *, name: str) -> str:
    """Require a non-empty string without normalizing the supplied value."""
    text = strict_text(value, name=name)
    if not text:
        raise ValueError(f"{name} must be a non-empty string.")
    return text


def optional_bool(value: Any | None, *, name: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raise TypeError(f"{name} must be a boolean when provided.")


def optional_non_negative_int(value: Any | None, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer when provided.")
    coerced = int(value)
    if coerced < 0:
        raise ValueError(f"{name} must be non-negative when provided, got {coerced}.")
    return coerced


def text_tuple(value: Iterable[object] | None, *, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raise TypeError(f"{name} must be an iterable of strings, not a string.")
    result: list[str] = []
    for item in value:
        result.append(required_text(item, name=f"{name} item"))
    return tuple(result)


def metadata_dict(value: Mapping[str, Any] | None, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping or None.")
    normalized: dict[str, Any] = {}
    for key, item in value.items():
        key_text = required_text(key, name=f"{name} key")
        normalized[key_text] = item
    return normalized


def text_mapping(value: Mapping[str, object] | None, *, name: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping or None.")
    normalized: dict[str, str] = {}
    for key, item in value.items():
        key_text = required_text(key, name=f"{name} key")
        normalized[key_text] = required_text(item, name=f"{name} value")
    return normalized


__all__ = [
    "finite_float",
    "metadata_dict",
    "optional_bool",
    "optional_non_negative_int",
    "optional_text",
    "payload_mapping",
    "required_text",
    "strict_required_text",
    "strict_text",
    "text_mapping",
    "text_tuple",
]
