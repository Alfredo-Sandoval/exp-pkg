"""Shared helpers and constants for siesta format serializer modules."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from posetta.core.json_utils import dump_json

_PROVENANCE_SCHEMA_VERSION = 1
_PROVENANCE_SENTINEL_OPERATION = "(truncated)"
_DEFAULT_PROVENANCE_MAX_BYTES = 524_288
_JOURNAL_SCHEMA_VERSION = 1
SIESTA_SCHEMA_NAME = "siesta"
SIESTA_SCHEMA_VERSION = "2.0.0"
CANONICAL_BUNDLE_SUFFIX = ".sta"
LEGACY_BUNDLE_SUFFIXES = (".siesta",)
SUPPORTED_BUNDLE_SUFFIXES = (CANONICAL_BUNDLE_SUFFIX, *LEGACY_BUNDLE_SUFFIXES)
LABEL_TRACK_ID_DATASET = "track_id"
LABEL_VISIBILITY_DATASET = "visibility"


_COERCE_PRIMITIVE_SENTINEL = object()
JsonPrimitive = str | int | float | bool | None


def _require_h5_group(
    container: h5py.File | h5py.Group,
    group_name: str,
    *,
    missing_message: str,
    type_message: str,
) -> h5py.Group:
    group_obj = container.get(group_name)
    if group_obj is None:
        raise ValueError(missing_message)
    if not isinstance(group_obj, h5py.Group):
        raise TypeError(type_message)
    return group_obj


def _require_project_metadata_group(
    container: h5py.File | h5py.Group,
    *,
    missing_message: str = "Missing project_metadata group in .siesta archive",
    type_message: str = "project_metadata must be an h5py Group",
) -> h5py.Group:
    return _require_h5_group(
        container,
        "project_metadata",
        missing_message=missing_message,
        type_message=type_message,
    )


def _coerce_primitive(value: Any) -> JsonPrimitive | object:
    """Coerce a single value into a JSON/HDF5 attribute-friendly primitive.

    Returns `_COERCE_PRIMITIVE_SENTINEL` when `value` does not match any supported primitive type.
    """
    coercers: tuple[tuple[type[Any], Callable[[Any], JsonPrimitive]], ...] = (
        (np.bool_, bool),
        (bool, lambda item: item),
        (str, lambda item: item),
        (int, lambda item: item),
        (float, lambda item: item),
        (np.integer, int),
        (np.floating, float),
        (Path, str),
    )
    if value is None:
        return None
    for value_type, coerce in coercers:
        if isinstance(value, value_type):
            return coerce(value)
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    return _COERCE_PRIMITIVE_SENTINEL


def _mapping_to_str_key_dict(data: Mapping[Any, Any], *, name: str) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in data.items():
        if not isinstance(key, str):
            raise TypeError(f"{name} keys must be strings")
        payload[key] = value
    return payload


def _json_ready(value: Any) -> Any:
    """Convert numpy/Path and datetime objects into JSON-friendly primitives."""
    primitive = _coerce_primitive(value)
    if primitive is not _COERCE_PRIMITIVE_SENTINEL:
        return primitive
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray | str):
        return [_json_ready(item) for item in value]
    return str(value)


def _serialize_json(value: Any) -> str:
    """Serialize a python object with deterministic key ordering."""
    return dump_json(_json_ready(value), sort_keys=True, compact=True, ensure_ascii=True)


def _compact_json(payload: Mapping[str, Any]) -> str:
    """Serialize a mapping using a compact JSON representation."""
    return dump_json(payload, sort_keys=True, compact=True, ensure_ascii=True)


def _now_utc_iso() -> str:
    """Return an ISO-8601 timestamp in timezone.utc with trailing Z."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _default_provenance_entry(operation: str, **extra: Any) -> dict[str, Any]:
    """Create a provenance entry stamped with the current timestamp."""
    entry: dict[str, Any] = {
        "operation": operation,
        "timestamp": _now_utc_iso(),
    }
    entry.update(extra)
    return entry


def _looks_like_int(text: str) -> bool:
    if not text:
        return False
    s = text.strip()
    if not s:
        return False
    if s[0] in "+-":
        if len(s) == 1:
            return False
        s = s[1:]
    return all(ch.isdigit() for ch in s)


def _coerce_int(value: Any, *, default: int | None = None) -> int | None:
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, np.floating | float):
        if not np.isfinite(value):
            return default
        return int(value)
    if isinstance(value, bytes | bytearray | np.bytes_):
        value = value.decode("utf-8")
    if isinstance(value, str) and _looks_like_int(value):
        return int(value)
    return default


def _normalize_predictions_committed_length(
    group: h5py.Group,
    *,
    total_rows: int,
    missing_default: int | None = None,
    enforce_upper_bound: bool = True,
    exceed_message: str = (
        "Committed length exceeds predictions dataset length; file may be corrupt"
    ),
) -> int:
    committed_raw = group.attrs.get("committed_length")
    if committed_raw is None:
        committed = total_rows if missing_default is None else int(missing_default)
    else:
        committed = _coerce_int(committed_raw)
        if committed is None:
            raise ValueError(
                "Predictions group committed_length attribute must be a non-negative integer"
            )

    if committed < 0:
        committed = 0
    if enforce_upper_bound and committed > total_rows:
        raise ValueError(exceed_message.format(committed=committed, total_rows=total_rows))
    return int(committed)


def _skeleton_keypoint_count(
    container: h5py.File | h5py.Group,
    *,
    default: int = 0,
) -> int:
    skeleton_group = container.get("skeleton")
    if not isinstance(skeleton_group, h5py.Group):
        return int(default)
    names_ds = skeleton_group.get("names")
    if not isinstance(names_ds, h5py.Dataset):
        return int(default)
    if not names_ds.shape:
        return int(default)
    return int(names_ds.shape[0])


def _normalize_run_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    if "id" in entry:
        raise ValueError("run_metadata key 'id' is no longer supported; use 'run_id'")
    if "created" in entry:
        raise ValueError("run_metadata key 'created' is no longer supported; use 'created_ns'")
    if "config" in entry:
        raise ValueError("run_metadata key 'config' is no longer supported; use 'config_json'")

    if "run_id" not in entry:
        raise ValueError("run_metadata must include run_id")
    run_id_val = entry["run_id"]
    run_id = int(run_id_val)

    if "created_ns" not in entry:
        created_ns = time.time_ns()
    else:
        created_val = entry["created_ns"]
        if isinstance(created_val, datetime):
            created_ns = int(created_val.timestamp() * 1_000_000_000)
        else:
            created_ns = int(created_val)

    if "config_json" not in entry:
        config_json = ""
    else:
        config_val = entry["config_json"]
        if isinstance(config_val, Mapping) or (
            isinstance(config_val, Sequence) and not isinstance(config_val, bytes | bytearray | str)
        ):
            config_json = _serialize_json(config_val)
        elif isinstance(config_val, bytes | bytearray | np.bytes_):
            config_json = config_val.decode("utf-8")
        else:
            config_json = str(config_val)

    return {
        "run_id": int(run_id),
        "created_ns": int(created_ns),
        "config_json": config_json,
    }


def _normalize_runs_entries(entries: Any) -> list[dict[str, Any]]:
    if entries is None:
        return []

    if isinstance(entries, Mapping):
        if "run_id" in entries and not isinstance(entries.get("run_id"), Mapping):
            candidates = [entries]
        else:
            candidates = list(entries.values())
    elif isinstance(entries, Sequence) and not isinstance(entries, bytes | bytearray | str):
        candidates = list(entries)
    else:
        raise TypeError("run metadata must be a mapping or a sequence of mappings")

    normalized = [_normalize_run_entry(item) for item in candidates if isinstance(item, Mapping)]
    normalized.sort(key=lambda entry: (entry["created_ns"], entry["run_id"]))
    return normalized
