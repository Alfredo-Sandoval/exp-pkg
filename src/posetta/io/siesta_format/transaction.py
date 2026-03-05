"""Serializer transaction helpers for .siesta append/write flows."""

from __future__ import annotations

import contextlib
import os
import socket
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py

from posetta.core.json_utils import dump_json, parse_json_dict
from posetta.core.logging_utils import get_logger
from posetta.io.siesta_format.shared import (
    _DEFAULT_PROVENANCE_MAX_BYTES,
    _JOURNAL_SCHEMA_VERSION,
    _PROVENANCE_SCHEMA_VERSION,
    _PROVENANCE_SENTINEL_OPERATION,
    _compact_json,
    _default_provenance_entry,
    _now_utc_iso,
    _serialize_json,
)

logger = get_logger(__name__)


class SiestaFileLock:
    """Advisory file lock for .siesta serializer."""

    def __init__(
        self,
        project_path: Path | str,
        *,
        timeout_seconds: float | None = None,
        poll_interval_seconds: float = 0.1,
        stale_after_seconds: float | None = None,
    ) -> None:
        self.project_path = Path(project_path)
        self.lock_path = self.project_path.with_suffix(self.project_path.suffix + ".lock")
        self.timeout_seconds = timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.stale_after_seconds = stale_after_seconds
        self._acquired = False
        if self.timeout_seconds is not None and self.timeout_seconds < 0:
            raise ValueError("timeout_seconds must be >= 0 when provided")
        if self.poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be > 0")
        if self.stale_after_seconds is not None and self.stale_after_seconds < 0:
            raise ValueError("stale_after_seconds must be >= 0 when provided")

    def __enter__(self) -> SiestaFileLock:
        self.acquire()
        return self

    def __exit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        self.release()

    def acquire(self) -> None:
        if self._acquired:
            return

        deadline = None
        if self.timeout_seconds is not None:
            deadline = time.monotonic() + float(self.timeout_seconds)

        while True:
            metadata = self._build_metadata()
            tmp_fd, tmp_name = tempfile.mkstemp(
                prefix=".siesta_lock_",
                dir=str(self.lock_path.parent),
            )
            tmp_path = Path(tmp_name)
            try:
                with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                    f.write(dump_json(metadata, indent=None))
                try:
                    os.link(tmp_name, str(self.lock_path))
                except FileExistsError as exc:
                    holder = self._read_lock_metadata()
                    if self._is_stale(holder):
                        self.lock_path.unlink(missing_ok=True)
                        continue

                    if deadline is not None and time.monotonic() < deadline:
                        time.sleep(self.poll_interval_seconds)
                        continue

                    raise FileExistsError(self._format_contention_message(holder)) from exc
                else:
                    self._acquired = True
                    logger.debug("Acquired lock for %s", self.project_path)
                    return
            finally:
                tmp_path.unlink(missing_ok=True)

    def _build_metadata(self) -> dict[str, Any]:
        return {
            "pid": os.getpid(),
            "tid": threading.get_ident(),
            "hostname": socket.gethostname(),
            "timestamp": time.time(),
        }

    def _read_lock_metadata(self) -> dict[str, Any]:
        try:
            raw = self.lock_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}

        try:
            return parse_json_dict(raw)
        except ValueError as exc:
            raise ValueError("Lock file metadata is corrupted JSON") from exc

    def _coerce_lock_timestamp(self, holder: Mapping[str, Any]) -> float | None:
        raw = holder.get("timestamp")
        if isinstance(raw, int | float):
            return float(raw)
        if isinstance(raw, str):
            text = raw.strip()
            if not text:
                return None
            try:
                return float(text)
            except ValueError:
                return None
        return None

    def _is_stale(self, holder: Mapping[str, Any]) -> bool:
        if self.stale_after_seconds is None:
            return False
        ts = self._coerce_lock_timestamp(holder)
        if ts is None:
            return False
        age = time.time() - ts
        return age > float(self.stale_after_seconds)

    def _format_contention_message(self, holder: Mapping[str, Any]) -> str:
        pid = holder.get("pid", "unknown")
        tid = holder.get("tid", "unknown")
        hostname = holder.get("hostname", "unknown")
        timestamp = holder.get("timestamp", "unknown")
        ts = self._coerce_lock_timestamp(holder)
        age = time.time() - ts if ts is not None else None
        if age is None:
            age_str = "unknown"
        else:
            age_str = f"{age:.3f}"
        return (
            f"Lock already exists: {self.lock_path} "
            f"(pid={pid}, tid={tid}, hostname={hostname}, timestamp={timestamp}, age_seconds={age_str})"
        )

    def release(self) -> None:
        if not self._acquired:
            return

        self.lock_path.unlink(missing_ok=True)
        logger.debug("Released lock for %s", self.project_path)
        self._acquired = False


def _load_journal_attr(group: h5py.Group) -> dict[str, Any]:
    """Read the journal attribute from an HDF5 group and coerce it into a mapping."""
    raw = group.attrs.get("journal")
    if isinstance(raw, bytes | bytearray):
        raw = raw.decode("utf-8")
    payload: dict[str, Any] = {}
    if raw:
        if isinstance(raw, str | bytes | bytearray):
            payload = parse_json_dict(raw)
        else:
            payload = {}

    payload.setdefault("schema_version", _JOURNAL_SCHEMA_VERSION)
    payload.setdefault("state", "idle")
    return payload


def _write_journal(group: h5py.Group, payload: Mapping[str, Any]) -> None:
    """Persist a JSON payload to the `journal` attribute."""
    group.attrs["journal"] = _compact_json(payload)


def _ensure_journal_attr(group: h5py.Group) -> dict[str, Any]:
    """Guarantee that the `journal` attribute exists and return its contents."""
    journal = _load_journal_attr(group)
    _write_journal(group, journal)
    return dict(journal)


def _append_provenance(
    metadata_group: h5py.Group,
    entry: Mapping[str, Any],
    *,
    max_bytes: int = _DEFAULT_PROVENANCE_MAX_BYTES,
) -> None:
    """Append a provenance event while enforcing the configured byte ceiling."""

    def _size(payload: dict[str, Any]) -> int:
        return len(_serialize_json(payload).encode("utf-8"))

    def _is_sentinel(evt: Any) -> bool:
        return isinstance(evt, Mapping) and evt.get("operation") == _PROVENANCE_SENTINEL_OPERATION

    raw = metadata_group.attrs.get("provenance")
    if isinstance(raw, bytes | bytearray):
        raw = raw.decode("utf-8")
    if not raw:
        payload: dict[str, Any] = {}
    elif isinstance(raw, str | bytes | bytearray):
        payload = parse_json_dict(raw)
    else:
        payload = {}

    events_raw = payload.get("events")
    if isinstance(events_raw, list):
        events = events_raw
    else:
        events = []

    payload["events"] = events
    payload.setdefault("schema_version", _PROVENANCE_SCHEMA_VERSION)

    events.append(dict(entry))

    metadata_group.attrs["provenance_max_bytes"] = int(max_bytes)

    current_size = _size(payload)
    if current_size > max_bytes:
        dropped = 0
        while current_size > max_bytes and len(events) > 1:
            drop_index = 1 if events and _is_sentinel(events[0]) else 0
            if drop_index >= len(events) - 1:
                break
            events.pop(drop_index)
            dropped += 1
            current_size = _size(payload)

        if current_size > max_bytes and (
            len(events) <= 1 or (len(events) == 2 and _is_sentinel(events[0]))
        ):
            raise ValueError("Cannot record provenance entry within the configured byte limit.")

        if events and _is_sentinel(events[0]):
            sentinel = events[0]
            sentinel["dropped"] = int(sentinel.get("dropped", 0)) + dropped
            sentinel["timestamp"] = _now_utc_iso()
        elif dropped:
            sentinel = _default_provenance_entry(
                _PROVENANCE_SENTINEL_OPERATION,
                dropped=dropped,
            )
            events.insert(0, sentinel)

        current_size = _size(payload)
        while current_size > max_bytes and len(events) > 2:
            events.pop(1)
            if events and _is_sentinel(events[0]):
                sentinel = events[0]
                sentinel["dropped"] = int(sentinel.get("dropped", 0)) + 1
                sentinel["timestamp"] = _now_utc_iso()
            current_size = _size(payload)

        if current_size > max_bytes:
            raise ValueError("Cannot record provenance entry within the configured byte limit.")

    serialized = _serialize_json(payload)
    metadata_group.attrs["provenance"] = serialized


def _get_journal_context(predictions_group: h5py.Group) -> tuple[h5py.File, h5py.Group]:
    """Return the owning file handle and metadata group for journal operations."""
    file_obj = predictions_group.file
    if not isinstance(file_obj, h5py.File):
        raise TypeError("Predictions group is detached from an HDF5 file")

    meta_group = file_obj["project_metadata"]
    if not isinstance(meta_group, h5py.Group):
        raise TypeError("project_metadata is not a group")

    _ensure_journal_attr(meta_group)
    return file_obj, meta_group


def _normalize_temp_paths(paths: Any) -> list[str]:
    """Return a list of stringified temp paths."""
    if paths is None:
        return []
    if isinstance(paths, str | Path):
        return [str(paths)]
    if isinstance(paths, Sequence) and not isinstance(paths, str | bytes | bytearray):
        return [str(item) for item in paths]
    return []


def _truncate_predictions_group(predictions_group: h5py.Group, *, length: int) -> None:
    """Resize predictions datasets along axis 0 down to ``length`` rows."""
    length = int(length)
    observed_lengths: dict[str, int] = {}

    def _shrink(dataset: h5py.Dataset) -> None:
        shape = dataset.shape
        if not shape:
            return
        if int(shape[0]) <= length:
            return
        new_shape = (length, *tuple(shape[1:]))
        dataset.resize(new_shape)

    frames_group = predictions_group["frames"]
    if isinstance(frames_group, h5py.Group):
        for name in ("video_index", "frame_index", "num_instances"):
            dataset = frames_group.get(name)
            if isinstance(dataset, h5py.Dataset):
                _shrink(dataset)
                observed_lengths[f"frames/{name}"] = (
                    int(dataset.shape[0]) if dataset.shape else length
                )

    data_group = predictions_group["data"]
    if isinstance(data_group, h5py.Group):
        for name in (
            "keypoints",
            "keypoint_score",
            "instance_score",
            "track_id",
            "deleted",
            "heatmaps",
        ):
            dataset = data_group.get(name)
            if isinstance(dataset, h5py.Dataset):
                _shrink(dataset)
                observed_lengths[f"data/{name}"] = (
                    int(dataset.shape[0]) if dataset.shape else length
                )

    if observed_lengths:
        unique_lengths = sorted(set(observed_lengths.values()))
        if len(unique_lengths) != 1 or unique_lengths[0] != length:
            details = ", ".join(
                f"{name}={value}" for name, value in sorted(observed_lengths.items())
            )
            raise ValueError(f"Predictions rollback length mismatch (expected {length}): {details}")


def _flush_file(handle: h5py.File, *, fsync: bool) -> None:
    """Flush the HDF5 file to disk; optionally fsync the underlying fd."""
    handle.flush()
    if fsync:
        vfd_handle = handle.id.get_vfd_handle()
        if vfd_handle is None:
            return
        os.fsync(vfd_handle)


def _journal_begin(
    predictions_group: h5py.Group,
    *,
    old_len: int,
    new_len: int,
    operation: str = "predictions.append",
    temp_paths: Sequence[Path | str] | Path | str | None = None,
) -> None:
    """Record a pending journal entry and flush it to disk."""

    file_obj, meta_group = _get_journal_context(predictions_group)

    journal = _ensure_journal_attr(meta_group)
    now_iso = _now_utc_iso()
    pending_entry = {
        "operation": operation,
        "old_len": int(old_len),
        "new_len": int(new_len),
        "temp_paths": _normalize_temp_paths(temp_paths),
        "started_at": now_iso,
        "updated_at": now_iso,
    }

    journal["schema_version"] = _JOURNAL_SCHEMA_VERSION
    journal["state"] = "pending"
    journal["pending"] = pending_entry
    journal["updated_at"] = now_iso
    if "last_committed" not in journal:
        journal["last_committed"] = int(old_len)

    _write_journal(meta_group, journal)
    _flush_file(file_obj, fsync=True)
    _flush_file(file_obj, fsync=True)


def _journal_commit(
    predictions_group: h5py.Group,
    *,
    committed_length: int | None = None,
) -> None:
    """Mark the journal entry as completed and flush metadata to disk."""

    file_obj, meta_group = _get_journal_context(predictions_group)

    journal = _ensure_journal_attr(meta_group)
    pending = journal.pop("pending", None)
    now_iso = _now_utc_iso()

    journal["schema_version"] = _JOURNAL_SCHEMA_VERSION
    journal["state"] = "idle"
    journal["updated_at"] = now_iso
    journal.pop("rollback", None)

    target_len: int | None = None
    if committed_length is not None:
        target_len = int(committed_length)
    elif isinstance(pending, Mapping) and "new_len" in pending:
        target_len = int(pending["new_len"])

    if isinstance(pending, Mapping):
        operation = pending.get("operation")
        if operation is not None:
            journal["last_operation"] = operation
        journal["completed_at"] = now_iso

    if target_len is not None:
        journal["last_committed"] = int(target_len)

    _write_journal(meta_group, journal)
    _flush_file(file_obj, fsync=True)
    _flush_file(file_obj, fsync=True)


def _journal_rollback(
    predictions_group: h5py.Group,
    *,
    error: str | None = None,
) -> None:
    """Rollback journal state and restore committed length."""

    file_obj, meta_group = _get_journal_context(predictions_group)
    journal = _ensure_journal_attr(meta_group)
    pending = journal.pop("pending", None)

    old_len: int | None = None
    new_len: int | None = None
    operation: str | None = None
    temp_paths: list[str] = []

    if isinstance(pending, Mapping):
        operation = pending.get("operation")
        if "old_len" in pending:
            old_len = int(pending["old_len"])
        if "new_len" in pending:
            new_len = int(pending["new_len"])
        temp_paths = _normalize_temp_paths(pending.get("temp_paths"))

    now_iso = _now_utc_iso()
    journal["schema_version"] = _JOURNAL_SCHEMA_VERSION
    journal["state"] = "rollback"
    journal["updated_at"] = now_iso
    journal["rollback"] = {
        "operation": operation,
        "old_len": old_len,
        "new_len": new_len,
        "temp_paths": temp_paths,
        "rolled_back_at": now_iso,
    }
    if error:
        journal["rollback"]["error"] = error
    _write_journal(meta_group, journal)

    if old_len is not None:
        _truncate_predictions_group(predictions_group, length=old_len)
        predictions_group.attrs["committed_length"] = int(old_len)
        meta_group.attrs["n_predictions_committed"] = int(old_len)

    _flush_file(file_obj, fsync=True)


class _JournalTransaction(contextlib.AbstractContextManager["_JournalTransaction"]):
    """Context manager that begins and resolves a journal entry automatically."""

    def __init__(
        self,
        predictions_group: h5py.Group,
        *,
        old_len: int,
        new_len: int,
        operation: str,
        temp_paths: Sequence[Path | str] | Path | str | None = None,
        commit_length: int | None = None,
        enabled: bool = True,
    ) -> None:
        self._group = predictions_group
        self._old_len = int(old_len)
        self._new_len = int(new_len)
        self._operation = operation
        self._temp_paths = temp_paths
        self._commit_length = commit_length if commit_length is not None else int(new_len)
        self._enabled = enabled

    def __enter__(self) -> _JournalTransaction:
        if self._enabled:
            _journal_begin(
                self._group,
                old_len=self._old_len,
                new_len=self._new_len,
                operation=self._operation,
                temp_paths=self._temp_paths,
            )
        return self

    def set_commit_length(self, length: int) -> None:
        self._commit_length = int(length)

    def __exit__(self, exc_type, exc, tb) -> bool:
        if not self._enabled:
            return False
        if exc_type is None:
            _journal_commit(self._group, committed_length=self._commit_length)
        else:
            _journal_rollback(self._group, error=str(exc) if exc else None)
        return False


__all__ = [
    "SiestaFileLock",
    "_JournalTransaction",
    "_append_provenance",
    "_ensure_journal_attr",
    "_flush_file",
    "_get_journal_context",
    "_journal_begin",
    "_journal_commit",
    "_journal_rollback",
    "_load_journal_attr",
    "_normalize_temp_paths",
    "_truncate_predictions_group",
    "_write_journal",
]
