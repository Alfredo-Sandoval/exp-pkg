"""Crash-safe committed-state store for the private ``.xpkg`` directory.

This module backs committed project state under ``.xpkg/``. Normal
callers should go through project APIs rather than treating this as a
standalone storage subsystem.
"""

from __future__ import annotations

import os
import secrets
import socket
import tempfile
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from .._core.hashing import sha256_bytes, sha256_file
from .._core.json_utils import dump_json, load_json_dict, parse_json_dict

_SUPPORTED_STORE_VERSION = 1
_STORE_FORMAT = "xpkg.project-durable-store"
_LEGACY_STORE_FORMAT = "xpkg.archive-store"
_SUPPORTED_STORE_FORMATS = {_STORE_FORMAT, _LEGACY_STORE_FORMAT}


class ProjectDurableStoreError(Exception):
    """Base exception for the private project durability layer."""


ArchiveStoreError = ProjectDurableStoreError


class StoreCorruptionError(ProjectDurableStoreError):
    """Raised when required store files are missing or checksum validation fails."""


class IncompatibleStoreVersionError(ProjectDurableStoreError):
    """Raised when the store_version is not supported by this xpkg build."""


class LockAcquisitionError(ProjectDurableStoreError):
    """Raised when the store lock cannot be acquired."""


class JournalStateError(ProjectDurableStoreError):
    """Raised for invalid journal transitions or inconsistent recovery states."""


class ChecksumError(ProjectDurableStoreError):
    """Raised when a checksum cannot be verified."""


def now_utc_iso() -> str:
    """Return an ISO-8601 UTC timestamp with trailing Z."""
    return datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")


def _canonical_json_bytes(payload: Any) -> bytes:
    text = dump_json(payload, indent=None, sort_keys=True, ensure_ascii=True, compact=True)
    return text.encode("utf-8")


def _compute_checksum(payload_without_checksum: dict[str, Any]) -> str:
    return f"sha256:{sha256_bytes(_canonical_json_bytes(payload_without_checksum))}"


def _verify_checksum(payload: dict[str, Any]) -> bool:
    expected = payload.get("checksum")
    if not isinstance(expected, str) or not expected.startswith("sha256:"):
        return False
    stripped = dict(payload)
    stripped.pop("checksum", None)
    return _compute_checksum(stripped) == expected


def _fsync_file(path: Path) -> None:
    with Path(path).open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_dir_best_effort(directory: Path) -> None:
    if os.name != "posix":
        return

    flags = getattr(os, "O_RDONLY", 0)
    odir = getattr(os, "O_DIRECTORY", 0)
    fd: int | None = None
    try:
        fd = os.open(str(directory), flags | odir)
        os.fsync(fd)
    except OSError:
        return
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass


def atomic_write_bytes(
    path: Path,
    data: bytes,
    *,
    fsync_file: bool = True,
    fsync_dir: bool = True,
) -> None:
    """Write bytes via temp file, flush, fsync, and atomic replace."""
    dst = Path(path)
    parent = dst.parent
    parent.mkdir(parents=True, exist_ok=True)

    tmp_handle = tempfile.NamedTemporaryFile(
        prefix=f".{dst.name}.",
        suffix=".tmp",
        dir=str(parent),
        delete=False,
    )
    tmp_path = Path(tmp_handle.name)
    try:
        tmp_handle.write(data)
        tmp_handle.flush()
        os.fsync(tmp_handle.fileno())
        tmp_handle.close()

        os.replace(tmp_path, dst)

        if fsync_file:
            _fsync_file(dst)
        if fsync_dir:
            _fsync_dir_best_effort(parent)
    finally:
        try:
            tmp_handle.close()
        except Exception:
            pass
        tmp_path.unlink(missing_ok=True)


def atomic_write_json(
    path: Path,
    payload: Any,
    *,
    fsync_file: bool = True,
    fsync_dir: bool = True,
) -> None:
    """Serialize JSON and write it atomically."""
    text = dump_json(payload, indent=2, sort_keys=True, ensure_ascii=True, compact=False)
    if not text.endswith("\n"):
        text += "\n"
    atomic_write_bytes(
        Path(path),
        text.encode("utf-8"),
        fsync_file=fsync_file,
        fsync_dir=fsync_dir,
    )


def atomic_copy_file(
    src: Path,
    dst: Path,
    *,
    fsync_file: bool = True,
    fsync_dir: bool = True,
    chunk_bytes: int = 1024 * 1024,
) -> None:
    """Copy a file into place via temp file, fsync, and atomic replace."""
    src_path = Path(src)
    dst_path = Path(dst)
    parent = dst_path.parent
    parent.mkdir(parents=True, exist_ok=True)

    tmp_handle = tempfile.NamedTemporaryFile(
        prefix=f".{dst_path.name}.",
        suffix=".tmp",
        dir=str(parent),
        delete=False,
    )
    tmp_path = Path(tmp_handle.name)
    try:
        with src_path.open("rb") as src_handle:
            while True:
                chunk = src_handle.read(chunk_bytes)
                if not chunk:
                    break
                tmp_handle.write(chunk)

        tmp_handle.flush()
        os.fsync(tmp_handle.fileno())
        tmp_handle.close()

        os.replace(tmp_path, dst_path)

        if fsync_file:
            _fsync_file(dst_path)
        if fsync_dir:
            _fsync_dir_best_effort(parent)
    finally:
        try:
            tmp_handle.close()
        except Exception:
            pass
        tmp_path.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class StorePaths:
    """Canonical path layout for committed project durability state."""

    root: Path

    @property
    def superblock_a(self) -> Path:
        return self.root / "superblock.a.json"

    @property
    def superblock_b(self) -> Path:
        return self.root / "superblock.b.json"

    @property
    def lock_file(self) -> Path:
        return self.root / "LOCK"

    @property
    def journal_dir(self) -> Path:
        return self.root / "journal"

    @property
    def active_journal(self) -> Path:
        return self.journal_dir / "active.json"

    @property
    def commits_dir(self) -> Path:
        return self.root / "commits"

    @property
    def objects_dir(self) -> Path:
        return self.root / "objects"

    def commit_dir(self, generation: int) -> Path:
        return self.commits_dir / f"{int(generation):012d}"

    def commit_json(self, generation: int) -> Path:
        return self.commit_dir(generation) / "commit.json"

    def object_path(self, object_id: str, *, ext: str) -> Path:
        normalized_ext = ext if ext.startswith(".") else f".{ext}"
        key = object_id.replace("obj_", "")
        a = key[:2] if len(key) >= 2 else "00"
        b = key[2:4] if len(key) >= 4 else "00"
        return self.objects_dir / a / b / f"{object_id}{normalized_ext}"


def read_journal(path: Path) -> dict[str, Any] | None:
    """Read the active journal file when present."""
    journal_path = Path(path)
    if not journal_path.exists():
        return None
    return load_json_dict(journal_path)


def write_journal(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write the active journal file."""
    atomic_write_json(Path(path), payload, fsync_file=True, fsync_dir=True)


def clear_journal(path: Path) -> None:
    """Remove the active journal file if it exists."""
    Path(path).unlink(missing_ok=True)


@dataclass(slots=True)
class RootEntry:
    object_id: str
    ext: str

    def to_dict(self) -> dict[str, str]:
        return {
            "object_id": str(self.object_id),
            "ext": str(self.ext),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, root_name: str | None = None) -> RootEntry:
        location = f"roots.{root_name}" if root_name else "root entry"
        object_id = str(payload.get("object_id", "")).strip()
        if not object_id:
            raise ValueError(f"{location} is missing object_id")
        ext = str(payload.get("ext", "")).strip()
        if not ext:
            raise ValueError(f"{location} is missing ext")
        return cls(object_id=object_id, ext=ext)


@dataclass(slots=True)
class Superblock:
    format: str
    store_version: int
    generation: int
    current_commit_id: str
    previous_commit_id: str | None
    last_clean_commit_id: str
    active_journal_txn_id: str | None
    created_at: str
    updated_at: str
    checksum: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": self.format,
            "store_version": int(self.store_version),
            "generation": int(self.generation),
            "current_commit_id": str(self.current_commit_id),
            "previous_commit_id": self.previous_commit_id,
            "last_clean_commit_id": str(self.last_clean_commit_id),
            "active_journal_txn_id": self.active_journal_txn_id,
            "created_at": str(self.created_at),
            "updated_at": str(self.updated_at),
        }
        if self.checksum:
            payload["checksum"] = self.checksum
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Superblock:
        return cls(
            format=str(payload.get("format", "")),
            store_version=int(payload.get("store_version", 0)),
            generation=int(payload.get("generation", 0)),
            current_commit_id=str(payload.get("current_commit_id", "")),
            previous_commit_id=payload.get("previous_commit_id"),
            last_clean_commit_id=str(payload.get("last_clean_commit_id", "")),
            active_journal_txn_id=payload.get("active_journal_txn_id"),
            created_at=str(payload.get("created_at", "")),
            updated_at=str(payload.get("updated_at", "")),
            checksum=str(payload.get("checksum")) if payload.get("checksum") else None,
        )

    def with_checksum(self) -> Superblock:
        payload = self.to_dict()
        payload.pop("checksum", None)
        self.checksum = _compute_checksum(payload)
        return self

    def checksum_valid(self) -> bool:
        return _verify_checksum(self.to_dict())


@dataclass(slots=True)
class Commit:
    commit_id: str
    generation: int
    parent_commit_id: str | None
    created_at: str
    reason: str
    created_by: dict[str, Any]
    roots: dict[str, RootEntry]
    checksum: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "commit_id": str(self.commit_id),
            "generation": int(self.generation),
            "parent_commit_id": self.parent_commit_id,
            "created_at": str(self.created_at),
            "reason": str(self.reason),
            "created_by": dict(self.created_by),
            "roots": {
                str(root_name): root_entry.to_dict()
                for root_name, root_entry in self.roots.items()
            },
        }
        if self.checksum:
            payload["checksum"] = self.checksum
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Commit:
        raw_roots = payload.get("roots", {}) or {}
        if not isinstance(raw_roots, Mapping):
            raise TypeError("commit.roots must be a mapping")
        roots: dict[str, RootEntry] = {}
        for root_name, root_entry in raw_roots.items():
            if not isinstance(root_entry, Mapping):
                raise TypeError(f"commit.roots.{root_name} must be a mapping")
            roots[str(root_name)] = RootEntry.from_dict(root_entry, root_name=str(root_name))
        return cls(
            commit_id=str(payload.get("commit_id", "")),
            generation=int(payload.get("generation", 0)),
            parent_commit_id=payload.get("parent_commit_id"),
            created_at=str(payload.get("created_at", "")),
            reason=str(payload.get("reason", "")),
            created_by=dict(payload.get("created_by", {}) or {}),
            roots=roots,
            checksum=str(payload.get("checksum")) if payload.get("checksum") else None,
        )

    def has_root(self, root_name: str) -> bool:
        return root_name in self.roots

    def root_entry(self, root_name: str) -> RootEntry:
        try:
            return self.roots[root_name]
        except KeyError as exc:
            raise KeyError(f"Commit missing roots.{root_name}") from exc

    def with_checksum(self) -> Commit:
        payload = self.to_dict()
        payload.pop("checksum", None)
        self.checksum = _compute_checksum(payload)
        return self

    def checksum_valid(self) -> bool:
        return _verify_checksum(self.to_dict())


@dataclass(slots=True)
class Journal:
    txn_id: str
    state: str
    intent: str
    base_commit_id: str
    target_generation: int
    started_at: str
    updated_at: str
    staged_commit_path: str | None = None
    new_object_id: str | None = None
    error: str | None = None
    checksum: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "txn_id": str(self.txn_id),
            "state": str(self.state),
            "intent": str(self.intent),
            "base_commit_id": str(self.base_commit_id),
            "target_generation": int(self.target_generation),
            "started_at": str(self.started_at),
            "updated_at": str(self.updated_at),
            "staged_commit_path": self.staged_commit_path,
            "new_object_id": self.new_object_id,
            "error": self.error,
        }
        if self.checksum:
            payload["checksum"] = self.checksum
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Journal:
        return cls(
            txn_id=str(payload.get("txn_id", "")),
            state=str(payload.get("state", "")),
            intent=str(payload.get("intent", "")),
            base_commit_id=str(payload.get("base_commit_id", "")),
            target_generation=int(payload.get("target_generation", 0)),
            started_at=str(payload.get("started_at", "")),
            updated_at=str(payload.get("updated_at", "")),
            staged_commit_path=payload.get("staged_commit_path"),
            new_object_id=payload.get("new_object_id"),
            error=payload.get("error"),
            checksum=str(payload.get("checksum")) if payload.get("checksum") else None,
        )

    def with_checksum(self) -> Journal:
        payload = self.to_dict()
        payload.pop("checksum", None)
        self.checksum = _compute_checksum(payload)
        return self

    def checksum_valid(self) -> bool:
        return _verify_checksum(self.to_dict())


class StoreLock:
    """Advisory hard-link lock for the private project durability root."""

    def __init__(
        self,
        store_root: Path,
        *,
        timeout_seconds: float | None = None,
        poll_interval_seconds: float = 0.1,
        stale_after_seconds: float | None = None,
    ) -> None:
        self.store_root = Path(store_root)
        self.lock_path = self.store_root / "LOCK"
        self.timeout_seconds = timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.stale_after_seconds = stale_after_seconds
        self._acquired = False

    def __enter__(self) -> StoreLock:
        self.acquire()
        return self

    def __exit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        self.release()

    def _metadata(self) -> dict[str, Any]:
        return {
            "pid": os.getpid(),
            "tid": threading.get_ident(),
            "hostname": socket.gethostname(),
            "timestamp": time.time(),
        }

    def _read_holder(self) -> dict[str, Any]:
        try:
            raw = self.lock_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        except OSError:
            return {}
        try:
            return parse_json_dict(raw)
        except Exception:
            return {}

    def _is_stale(self, holder: Mapping[str, Any]) -> bool:
        if self.stale_after_seconds is None:
            return False
        ts = holder.get("timestamp")
        if not isinstance(ts, int | float):
            return False
        return (time.time() - float(ts)) > float(self.stale_after_seconds)

    def acquire(self) -> None:
        if self._acquired:
            return

        deadline: float | None = None
        if self.timeout_seconds is not None:
            deadline = time.monotonic() + float(self.timeout_seconds)

        self.store_root.mkdir(parents=True, exist_ok=True)

        while True:
            fd, tmp_name = tempfile.mkstemp(
                prefix=".project_store_lock_",
                dir=str(self.store_root),
            )
            tmp_path = Path(tmp_name)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(dump_json(self._metadata(), indent=None, compact=True))
                try:
                    os.link(str(tmp_path), str(self.lock_path))
                except FileExistsError as err:
                    holder = self._read_holder()
                    if self._is_stale(holder):
                        self.lock_path.unlink(missing_ok=True)
                        continue
                    if deadline is not None and time.monotonic() < deadline:
                        time.sleep(self.poll_interval_seconds)
                        continue
                    raise LockAcquisitionError(
                        f"Store lock contention: {self.lock_path}"
                    ) from err
                else:
                    self._acquired = True
                    return
            finally:
                tmp_path.unlink(missing_ok=True)

    def release(self) -> None:
        if not self._acquired:
            return
        self.lock_path.unlink(missing_ok=True)
        self._acquired = False


@dataclass(slots=True)
class MountedHead:
    slot: Literal["a", "b"]
    superblock: Superblock


def _object_ext_for_path(path: Path, *, default: str = ".bin") -> str:
    suffix = Path(path).suffix.strip()
    if not suffix:
        return default
    return suffix if suffix.startswith(".") else f".{suffix}"


def _load_verified_json(path: Path) -> dict[str, Any]:
    payload = load_json_dict(path)
    if not _verify_checksum(payload):
        raise ChecksumError(f"Checksum invalid: {path}")
    return payload


def _load_superblock(path: Path) -> Superblock | None:
    if not path.exists():
        return None
    payload = _load_verified_json(path)
    sb = Superblock.from_dict(payload)
    if sb.store_version != _SUPPORTED_STORE_VERSION:
        raise IncompatibleStoreVersionError(
            f"Unsupported store_version={sb.store_version}; expected {_SUPPORTED_STORE_VERSION}"
        )
    if sb.format not in _SUPPORTED_STORE_FORMATS:
        raise StoreCorruptionError(f"Unexpected store format: {sb.format}")
    return sb


def _try_load_slot(
    path: Path,
    slot: Literal["a", "b"],
) -> tuple[MountedHead | None, Exception | None]:
    try:
        superblock = _load_superblock(path)
    except Exception as exc:
        return None, exc
    if superblock is None:
        return None, None
    return MountedHead(slot=slot, superblock=superblock), None


def _choose_head(paths: StorePaths) -> MountedHead | None:
    a_head, a_error = _try_load_slot(paths.superblock_a, "a")
    b_head, b_error = _try_load_slot(paths.superblock_b, "b")

    candidates = [head for head in (a_head, b_head) if head is not None]
    if not candidates:
        if a_error is not None:
            raise a_error
        if b_error is not None:
            raise b_error
        return None
    if len(candidates) == 1:
        return candidates[0]

    a = candidates[0]
    b = candidates[1]
    if a.superblock.generation > b.superblock.generation:
        return a
    if b.superblock.generation > a.superblock.generation:
        return b
    return a if a.superblock.updated_at >= b.superblock.updated_at else b


def _inactive_slot(active: Literal["a", "b"]) -> Literal["a", "b"]:
    return "b" if active == "a" else "a"


def _commit_root_entry(object_id: str, *, ext: str) -> RootEntry:
    return RootEntry(object_id=object_id, ext=ext)


def _normalize_root_name(root_name: str) -> str:
    normalized = str(root_name).strip().lower()
    if not normalized:
        raise ValueError("root_name must be a non-empty string")
    return normalized


def _object_ext_for_root(root_name: str, path: Path) -> str:
    _normalize_root_name(root_name)
    return _object_ext_for_path(path).lower()


def _generation_from_commit_id(commit_id: str) -> int | None:
    parts = str(commit_id).split("_", 2)
    if len(parts) < 2 or parts[0] != "c" or not parts[1].isdigit():
        return None
    return int(parts[1])


def put_object_file(
    paths: StorePaths,
    src_path: Path,
    *,
    ext: str,
) -> str:
    """Copy a file into the immutable content-addressed object store."""
    source = Path(src_path)
    digest = sha256_file(source)
    object_id = f"obj_{digest}"
    dst_path = paths.object_path(object_id, ext=ext)

    if dst_path.exists():
        return object_id

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_copy_file(source, dst_path, fsync_file=True, fsync_dir=True)
    return object_id


def get_object_file(paths: StorePaths, object_id: str, *, ext: str) -> Path:
    """Return the on-disk path for a stored object."""
    return paths.object_path(object_id, ext=ext)


def _commit_root_entries(
    paths: StorePaths,
    root_paths: Mapping[str, Path],
) -> dict[str, RootEntry]:
    if not root_paths:
        raise ValueError("root_paths must contain at least one entry")

    entries: dict[str, RootEntry] = {}
    for raw_root_name, raw_path in root_paths.items():
        root_name = _normalize_root_name(raw_root_name)
        candidate_path = Path(raw_path)
        if not candidate_path.is_file():
            raise FileNotFoundError(f"Root payload not found for {root_name}: {candidate_path}")
        object_ext = _object_ext_for_root(root_name, candidate_path)
        object_id = put_object_file(paths, candidate_path, ext=object_ext)
        entries[root_name] = _commit_root_entry(object_id, ext=object_ext)
    return entries


class ProjectDurableStore:
    """Crash-safe committed state store for a project's private ``.xpkg`` root."""

    def __init__(self, root: Path) -> None:
        self.paths = StorePaths(root=Path(root))

    @classmethod
    def create_from_roots(
        cls,
        store_root: Path,
        initial_roots: Mapping[str, Path],
        *,
        created_by: dict[str, Any] | None = None,
        reason: str = "init",
    ) -> ProjectDurableStore:
        store = cls(store_root)
        paths = store.paths
        paths.root.mkdir(parents=True, exist_ok=True)
        paths.journal_dir.mkdir(parents=True, exist_ok=True)
        paths.commits_dir.mkdir(parents=True, exist_ok=True)
        paths.objects_dir.mkdir(parents=True, exist_ok=True)

        with StoreLock(paths.root):
            commit_id = f"c_{1:012d}_{secrets.token_hex(4)}"
            commit = Commit(
                commit_id=commit_id,
                generation=1,
                parent_commit_id=None,
                created_at=now_utc_iso(),
                reason=reason,
                created_by=created_by or {},
                roots=_commit_root_entries(paths, initial_roots),
            ).with_checksum()

            commit_path = paths.commit_json(1)
            commit_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(commit_path, commit.to_dict(), fsync_file=True, fsync_dir=True)

            created_at = now_utc_iso()
            updated_at = now_utc_iso()
            sb = Superblock(
                format=_STORE_FORMAT,
                store_version=_SUPPORTED_STORE_VERSION,
                generation=1,
                current_commit_id=commit_id,
                previous_commit_id=None,
                last_clean_commit_id=commit_id,
                active_journal_txn_id=None,
                created_at=created_at,
                updated_at=updated_at,
            ).with_checksum()

            atomic_write_json(
                paths.superblock_a,
                sb.to_dict(),
                fsync_file=True,
                fsync_dir=True,
            )
            atomic_write_json(
                paths.superblock_b,
                sb.to_dict(),
                fsync_file=True,
                fsync_dir=True,
            )
            clear_journal(paths.active_journal)

        return store

    @classmethod
    def open(cls, store_root: Path) -> ProjectDurableStore:
        store = cls(store_root)
        store.recover()
        return store

    def recover(self) -> MountedHead:
        paths = self.paths
        if not paths.root.exists():
            raise FileNotFoundError(f"Store root not found: {paths.root}")

        with StoreLock(paths.root):
            head = _choose_head(paths)
            if head is None:
                raise StoreCorruptionError("No valid superblock found")

            journal_payload = read_journal(paths.active_journal)
            if journal_payload is None:
                return head

            journal = Journal.from_dict(journal_payload)
            if not journal.checksum_valid():
                raise ChecksumError("Journal checksum invalid")

            sb = head.superblock
            if journal.state in {"staging", "validating"}:
                sb.current_commit_id = sb.last_clean_commit_id
                sb.active_journal_txn_id = None
                sb.updated_at = now_utc_iso()
                sb.generation = sb.generation + 1
                sb.with_checksum()
                atomic_write_json(
                    paths.superblock_a,
                    sb.to_dict(),
                    fsync_file=True,
                    fsync_dir=True,
                )
                atomic_write_json(
                    paths.superblock_b,
                    sb.to_dict(),
                    fsync_file=True,
                    fsync_dir=True,
                )
                clear_journal(paths.active_journal)
                return MountedHead(slot="a", superblock=sb)

            if journal.state in {"committing", "cleanup"}:
                current_commit_path = self.commit_path_for_id(sb.current_commit_id)
                if current_commit_path is not None and current_commit_path.exists():
                    clear_journal(paths.active_journal)
                    return head

                sb.current_commit_id = sb.last_clean_commit_id
                sb.active_journal_txn_id = None
                sb.updated_at = now_utc_iso()
                sb.generation = sb.generation + 1
                sb.with_checksum()
                atomic_write_json(
                    paths.superblock_a,
                    sb.to_dict(),
                    fsync_file=True,
                    fsync_dir=True,
                )
                atomic_write_json(
                    paths.superblock_b,
                    sb.to_dict(),
                    fsync_file=True,
                    fsync_dir=True,
                )
                clear_journal(paths.active_journal)
                return MountedHead(slot="a", superblock=sb)

            raise JournalStateError(f"Unknown journal state: {journal.state}")

    def _head(self) -> MountedHead:
        head = _choose_head(self.paths)
        if head is None:
            raise StoreCorruptionError("No valid superblock found")
        return head

    def commit_path_for_id(self, commit_id: str) -> Path | None:
        """Resolve a commit id to its commit.json path."""
        commits_dir = self.paths.commits_dir
        if not commits_dir.exists():
            return None

        generation = _generation_from_commit_id(commit_id)
        if generation is not None:
            candidate = self.paths.commit_json(generation)
            if candidate.exists():
                try:
                    payload = _load_verified_json(candidate)
                except Exception:
                    return None
                if payload.get("commit_id") == commit_id:
                    return candidate

        for commit_json in commits_dir.glob("*/commit.json"):
            try:
                payload = _load_verified_json(commit_json)
            except Exception:
                continue
            if payload.get("commit_id") == commit_id:
                return commit_json
        return None

    def load_current_commit(self) -> Commit:
        """Load and verify the current commit referenced by the mounted head."""
        head = self._head()
        commit_json = self.commit_path_for_id(head.superblock.current_commit_id)
        if commit_json is None or not commit_json.exists():
            raise StoreCorruptionError("Current commit not found on disk")
        payload = _load_verified_json(commit_json)
        commit = Commit.from_dict(payload)
        if not commit.checksum_valid():
            raise ChecksumError("Commit checksum invalid")
        if commit.commit_id != head.superblock.current_commit_id:
            raise StoreCorruptionError(
                "Current commit id mismatch: "
                f"head={head.superblock.current_commit_id!r}, "
                f"commit={commit.commit_id!r}"
            )
        return commit

    def has_current_root(self, root_name: str) -> bool:
        commit = self.load_current_commit()
        return commit.has_root(_normalize_root_name(root_name))

    def current_root_entry(self, root_name: str) -> RootEntry:
        """Return the typed current commit entry for a named root."""
        commit = self.load_current_commit()
        normalized_root_name = _normalize_root_name(root_name)
        try:
            return commit.root_entry(normalized_root_name)
        except KeyError as exc:
            raise StoreCorruptionError(f"Commit missing roots.{normalized_root_name}") from exc

    def current_root_path(self, root_name: str) -> Path:
        """Return the current immutable payload path for a named commit root."""
        root_entry = self.current_root_entry(root_name)
        path = get_object_file(self.paths, root_entry.object_id, ext=root_entry.ext)
        if not path.exists():
            raise StoreCorruptionError(f"Commit root object missing: {path}")
        return path

    def commit_new_roots(
        self,
        root_paths: Mapping[str, Path],
        *,
        reason: str,
        created_by: dict[str, Any] | None = None,
    ) -> str:
        """Create a new immutable commit from staged named root payloads."""
        paths = self.paths

        with StoreLock(paths.root):
            head = self._head()
            sb = head.superblock
            new_gen = int(sb.generation) + 1
            base_commit_id = sb.current_commit_id

            txn_id = f"txn_{now_utc_iso().replace(':', '').replace('-', '')}_{secrets.token_hex(2)}"
            journal = Journal(
                txn_id=txn_id,
                state="staging",
                intent="commit.create",
                base_commit_id=base_commit_id,
                target_generation=new_gen,
                started_at=now_utc_iso(),
                updated_at=now_utc_iso(),
            ).with_checksum()
            write_journal(paths.active_journal, journal.to_dict())

            root_entries = _commit_root_entries(paths, root_paths)
            object_ids = [root_entry.object_id for root_entry in root_entries.values()]
            journal.new_object_id = object_ids[0] if object_ids else None
            journal.updated_at = now_utc_iso()
            journal.state = "validating"
            journal.with_checksum()
            write_journal(paths.active_journal, journal.to_dict())

            commit_id = f"c_{new_gen:012d}_{secrets.token_hex(4)}"
            commit = Commit(
                commit_id=commit_id,
                generation=new_gen,
                parent_commit_id=base_commit_id,
                created_at=now_utc_iso(),
                reason=reason,
                created_by=created_by or {},
                roots=root_entries,
            ).with_checksum()

            commit_path = paths.commit_json(new_gen)
            commit_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(commit_path, commit.to_dict(), fsync_file=True, fsync_dir=True)

            journal.state = "committing"
            journal.updated_at = now_utc_iso()
            journal.staged_commit_path = str(commit_path)
            journal.with_checksum()
            write_journal(paths.active_journal, journal.to_dict())

            new_sb = Superblock(
                format=_STORE_FORMAT,
                store_version=_SUPPORTED_STORE_VERSION,
                generation=new_gen,
                current_commit_id=commit_id,
                previous_commit_id=base_commit_id,
                last_clean_commit_id=commit_id,
                active_journal_txn_id=None,
                created_at=sb.created_at,
                updated_at=now_utc_iso(),
            ).with_checksum()

            slot = _inactive_slot(head.slot)
            if slot == "a":
                atomic_write_json(
                    paths.superblock_a,
                    new_sb.to_dict(),
                    fsync_file=True,
                    fsync_dir=True,
                )
            else:
                atomic_write_json(
                    paths.superblock_b,
                    new_sb.to_dict(),
                    fsync_file=True,
                    fsync_dir=True,
                )

            journal.state = "cleanup"
            journal.updated_at = now_utc_iso()
            journal.with_checksum()
            write_journal(paths.active_journal, journal.to_dict())

            clear_journal(paths.active_journal)
            return commit_id


ArchiveStore = ProjectDurableStore


__all__ = [
    "ArchiveStore",
    "ArchiveStoreError",
    "ChecksumError",
    "Commit",
    "IncompatibleStoreVersionError",
    "Journal",
    "JournalStateError",
    "LockAcquisitionError",
    "RootEntry",
    "StoreCorruptionError",
    "StorePaths",
    "Superblock",
    "ProjectDurableStore",
    "ProjectDurableStoreError",
    "atomic_write_json",
    "clear_journal",
    "get_object_file",
    "now_utc_iso",
    "put_object_file",
    "read_journal",
    "write_journal",
]
