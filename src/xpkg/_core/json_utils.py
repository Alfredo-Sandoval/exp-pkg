"""JSON parsing and writing policy for xpkg payload files.

This module is the only place in the package that should call stdlib
``json.load/json.loads/json.dump/json.dumps`` directly. Other modules should
use this policy surface instead of ad hoc JSON handling.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from xpkg._core.path_registry import resolve_path


def parse_json(text: str | bytes | bytearray) -> object:
    """Parse JSON from a string or bytes payload.

    Args:
        text: JSON-encoded data as a ``str``, ``bytes``, or ``bytearray``.
            Byte inputs are decoded as UTF-8 before parsing.

    Returns:
        The parsed JSON value (dict, list, str, int, float, bool, or None).

    Raises:
        json.JSONDecodeError: If the input is not valid JSON.
        UnicodeDecodeError: If byte input cannot be decoded as UTF-8.
    """
    if isinstance(text, (bytes, bytearray)):
        text = bytes(text).decode("utf-8")
    return json.loads(text)


def parse_json_dict(text: str | bytes | bytearray) -> dict[str, Any]:
    """Parse JSON, enforcing an object (``{...}``) at the top level.

    Args:
        text: JSON-encoded data as a ``str``, ``bytes``, or ``bytearray``.

    Returns:
        A ``dict[str, Any]`` representing the top-level JSON object.

    Raises:
        json.JSONDecodeError: If the input is not valid JSON.
        TypeError: If the top-level value is not a JSON object, or if any key
            is not a string.
    """
    data = parse_json(text)
    if not isinstance(data, dict):
        raise TypeError("JSON payload must contain an object at the top level")
    out: dict[str, Any] = {}
    for key, value in data.items():
        if not isinstance(key, str):
            raise TypeError("JSON object keys must be strings")
        out[key] = value
    return out


def dump_json(
    payload: object,
    *,
    indent: int | None = 2,
    sort_keys: bool = False,
    ensure_ascii: bool = True,
    compact: bool = False,
    default: Callable[[Any], Any] | None = None,
) -> str:
    """Serialize a payload to a JSON string.

    Args:
        payload: The data to serialize.
        indent: Indentation level for pretty formatting. Use None for one-line output.
        sort_keys: Whether to sort object keys.
        ensure_ascii: Whether to escape non-ASCII characters.
        compact: When True, use compact separators and force one-line output.
        default: Optional serializer for non-JSON types. When omitted, unsupported types raise.
    """
    if compact:
        indent = None
        separators: tuple[str, str] | None = (",", ":")
    else:
        separators = None

    if separators is None:
        return json.dumps(
            payload,
            indent=indent,
            sort_keys=sort_keys,
            ensure_ascii=ensure_ascii,
            default=default,
        )

    return json.dumps(
        payload,
        indent=indent,
        sort_keys=sort_keys,
        ensure_ascii=ensure_ascii,
        separators=separators,
        default=default,
    )


def load_json(path: str | Path | os.PathLike[str]) -> object:
    """Load JSON data from a file path.

    Args:
        path: Path to the JSON file.

    Returns:
        The parsed JSON data.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path_obj = resolve_path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"JSON file not found: {path_obj}")
    return parse_json(path_obj.read_text(encoding="utf-8"))


def load_json_dict(path: str | Path | os.PathLike[str]) -> dict[str, Any]:
    """Load JSON data from a file path, enforcing an object at the top level.

    Args:
        path: Path to the JSON file.

    Returns:
        The parsed JSON object.

    Raises:
        FileNotFoundError: If the file does not exist.
        TypeError: If the JSON data is not an object.
    """
    path_obj = resolve_path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"JSON file not found: {path_obj}")
    data = parse_json(path_obj.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"JSON file must contain an object at the top level: {path_obj}")
    out: dict[str, Any] = {}
    for key, value in data.items():
        if not isinstance(key, str):
            raise TypeError(f"JSON object keys must be strings: {path_obj}")
        out[key] = value
    return out


def write_json(
    path: str | Path | os.PathLike[str],
    payload: object,
    *,
    indent: int | None = 2,
    sort_keys: bool = False,
    ensure_ascii: bool = True,
    trailing_newline: bool = True,
    compact: bool = False,
) -> None:
    """Write JSON data to a file path.

    Args:
        path: Path to write the JSON file.
        payload: Data to serialize as JSON.
        indent: Indentation level for pretty formatting. Use None for one-line output.
        sort_keys: Whether to sort object keys.
        ensure_ascii: Whether to escape non-ASCII characters.
        trailing_newline: Whether to add a trailing newline.
        compact: When True, use compact separators and force one-line output.
    """
    path_obj = resolve_path(path)
    text = dump_json(
        payload,
        indent=indent,
        sort_keys=sort_keys,
        ensure_ascii=ensure_ascii,
        compact=compact,
    )
    if trailing_newline and not text.endswith("\n"):
        text += "\n"
    path_obj.write_text(text, encoding="utf-8")
