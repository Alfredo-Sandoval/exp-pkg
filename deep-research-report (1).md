# APFS-Lite Durable Store Layer Files for xpkg

## Durability contract and failure model

The design goal is that a crash, power loss, or software bug cannot destroy the last committed human label state. The only acceptable loss is uncommitted edits since the last autosave boundary. This is achieved by combining (i) *atomic root pointer flips* and (ii) *append-only logging of edits* rather than in-place mutation of the last known-good state. Atomic replacement is implemented with `os.replace` (atomic on POSIX when successful, and only within a filesystem). citeturn5view1

The file durability boundary is explicitly “data is durable when flushed + fsynced,” using `os.fsync` (Unix `fsync()`, Windows `_commit()`). citeturn4view0 Because temporary-file semantics differ across platforms (especially Windows reopen/delete behavior), the implementation must ensure `delete=False` patterns and handle closure before rename/replace. citeturn2search0

Finally, because xpkg uses HDF5 as an underlying container for `.sta`, it is important not to assume “HDF5 will save us” under all I/O models: even SWMR has explicit filesystem semantics requirements (POSIX write semantics) and feature limitations. citeturn1search0turn1search1 The durable store layer therefore treats HDF5 payloads as *blobs* and keeps correctness in the store’s superblock/journal logic.

## On-disk store layout

The store is a directory bundle (works on macOS/Linux/Windows) with dual superblocks (A/B), a single active transaction journal, immutable commits, immutable objects, and a workspace oplog for crash recovery of unsaved edits.

Minimal layout:

```
<project>.siesta/
  superblock.a.json
  superblock.b.json
  LOCK
  journal/
    active.json
  commits/
    000000000001/
      commit.json
    000000000002/
      commit.json
  objects/
    ab/cd/obj_<sha256>.sta
  workspace/
    session-<uuid>.state.json
    session-<uuid>.oplog.jsonl
  snapshots/
    autosave-<timestamp>.json
```

The key atomicity trick is that only the superblock flip defines what is “current.” All other files are written as staged artifacts, fsynced, then referenced by the superblock. This mirrors the robust pattern enabled by atomic replacement via `os.replace`. citeturn5view1

The lock uses hard links (`os.link`) because it is a portable primitive on Unix and Windows. citeturn5view0

## Code files to add to xpkg

Below is the minimal file set to implement the store as an additive module under `src/xpkg/io/siesta_store/`, plus a small public wrapper in `xpkg.formats` and a focused test suite.

### Package entry points

Create the following files:

- `src/xpkg/io/siesta_store/__init__.py`
- `src/xpkg/io/siesta_store/errors.py`
- `src/xpkg/io/siesta_store/hashing.py`
- `src/xpkg/io/siesta_store/platform_io.py`
- `src/xpkg/io/siesta_store/paths.py`
- `src/xpkg/io/siesta_store/schema.py`
- `src/xpkg/io/siesta_store/lock.py`
- `src/xpkg/io/siesta_store/oplog.py`
- `src/xpkg/io/siesta_store/journal.py`
- `src/xpkg/io/siesta_store/object_store.py`
- `src/xpkg/io/siesta_store/store.py`
- `src/xpkg/formats/siesta_store.py`
- Tests: `tests/test_siesta_store_*.py` (listed later)

The code below is intentionally written to (a) be cross-platform, (b) anchor durability in `os.replace` + `os.fsync`, and (c) keep HDF5 payloads opaque blobs. `os.replace` provides cross-platform overwrite semantics and is atomic on POSIX when successful. citeturn5view1 `os.fsync` provides the durability boundary and maps to `_commit()` on Windows. citeturn4view0

### `src/xpkg/io/siesta_store/errors.py`

```python
from __future__ import annotations


class SiestaStoreError(Exception):
    """Base exception for siesta_store."""


class StoreCorruptionError(SiestaStoreError):
    """Raised when a superblock/commit/journal checksum fails or required files are missing."""


class IncompatibleStoreVersionError(SiestaStoreError):
    """Raised when the store_version is not supported by this xpkg build."""


class LockAcquisitionError(SiestaStoreError):
    """Raised when the store lock cannot be acquired."""


class JournalStateError(SiestaStoreError):
    """Raised for invalid journal transitions or inconsistent commit/journal states."""


class ChecksumError(SiestaStoreError):
    """Raised when a checksum cannot be verified."""
```

### `src/xpkg/io/siesta_store/hashing.py`

```python
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from xpkg.core.json_utils import dump_json


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    # Deterministic JSON serialization for checksums
    text = dump_json(payload, indent=None, sort_keys=True, ensure_ascii=True, compact=True)
    return text.encode("utf-8")


def compute_checksum(payload_without_checksum: dict[str, Any]) -> str:
    data = canonical_json_bytes(payload_without_checksum)
    return f"sha256:{sha256_bytes(data)}"


def verify_checksum(payload: dict[str, Any]) -> bool:
    expected = payload.get("checksum")
    if not isinstance(expected, str) or not expected.startswith("sha256:"):
        return False
    stripped = dict(payload)
    stripped.pop("checksum", None)
    return compute_checksum(stripped) == expected
```

### `src/xpkg/io/siesta_store/platform_io.py`

This is the durability shim. It centralizes the mechanics of “write temp → flush → fsync → atomic replace,” which is the critical pattern behind safe root flips. `os.replace` semantics are documented (including same-filesystem constraint). citeturn5view1 `os.fsync` semantics differ across Unix/Windows and should be treated as the durability boundary. citeturn4view0 Temporary file behavior differs on Windows, motivating explicit `delete=False` and closure before handing paths to other code. citeturn2search0

```python
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from xpkg.core.json_utils import dump_json


def _fsync_file(path: Path) -> None:
    # fsync a file by opening in binary read mode and syncing its fd
    with path.open("rb") as f:
        os.fsync(f.fileno())


def _fsync_dir_best_effort(directory: Path) -> None:
    # Directory fsync is a POSIX durability technique; not uniformly available on Windows.
    if os.name != "posix":
        return
    flags = getattr(os, "O_RDONLY", 0)
    odir = getattr(os, "O_DIRECTORY", 0)
    fd = None
    try:
        fd = os.open(str(directory), flags | odir)
        os.fsync(fd)
    except OSError:
        # Best effort only; correctness does not rely solely on this.
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
    path = Path(path)
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)

    # Use delete=False for Windows compatibility and to allow reopen/replace safely.
    tmp_handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.",
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

        os.replace(tmp_path, path)

        if fsync_file:
            _fsync_file(path)
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
    text = dump_json(payload, indent=2, sort_keys=True, ensure_ascii=True, compact=False)
    if not text.endswith("\n"):
        text += "\n"
    atomic_write_bytes(path, text.encode("utf-8"), fsync_file=fsync_file, fsync_dir=fsync_dir)


def atomic_copy_file(
    src: Path,
    dst: Path,
    *,
    fsync_file: bool = True,
    fsync_dir: bool = True,
    chunk_bytes: int = 1024 * 1024,
) -> None:
    src = Path(src)
    dst = Path(dst)
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
        with src.open("rb") as fsrc:
            while True:
                chunk = fsrc.read(chunk_bytes)
                if not chunk:
                    break
                tmp_handle.write(chunk)

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
```

### `src/xpkg/io/siesta_store/paths.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class StorePaths:
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

    @property
    def workspace_dir(self) -> Path:
        return self.root / "workspace"

    @property
    def snapshots_dir(self) -> Path:
        return self.root / "snapshots"

    def commit_dir(self, generation: int) -> Path:
        return self.commits_dir / f"{generation:012d}"

    def commit_json(self, generation: int) -> Path:
        return self.commit_dir(generation) / "commit.json"

    def object_path(self, object_id: str, *, ext: str) -> Path:
        # Spread objects to avoid huge directories
        key = object_id.replace("obj_", "")
        a = key[:2] if len(key) >= 2 else "00"
        b = key[2:4] if len(key) >= 4 else "00"
        return self.objects_dir / a / b / f"{object_id}{ext}"
```

### `src/xpkg/io/siesta_store/schema.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Mapping

from xpkg.io.siesta_store.hashing import compute_checksum, verify_checksum


def now_utc_iso() -> str:
    return datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")


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
    def from_dict(cls, payload: Mapping[str, Any]) -> "Superblock":
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

    def with_checksum(self) -> "Superblock":
        d = self.to_dict()
        d.pop("checksum", None)
        self.checksum = compute_checksum(d)
        return self

    def checksum_valid(self) -> bool:
        d = self.to_dict()
        return verify_checksum(d)


@dataclass(slots=True)
class Commit:
    commit_id: str
    generation: int
    parent_commit_id: str | None
    created_at: str
    reason: str
    created_by: dict[str, Any]
    roots: dict[str, Any]
    checksum: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "commit_id": str(self.commit_id),
            "generation": int(self.generation),
            "parent_commit_id": self.parent_commit_id,
            "created_at": str(self.created_at),
            "reason": str(self.reason),
            "created_by": dict(self.created_by),
            "roots": dict(self.roots),
        }
        if self.checksum:
            payload["checksum"] = self.checksum
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Commit":
        return cls(
            commit_id=str(payload.get("commit_id", "")),
            generation=int(payload.get("generation", 0)),
            parent_commit_id=payload.get("parent_commit_id"),
            created_at=str(payload.get("created_at", "")),
            reason=str(payload.get("reason", "")),
            created_by=dict(payload.get("created_by", {}) or {}),
            roots=dict(payload.get("roots", {}) or {}),
            checksum=str(payload.get("checksum")) if payload.get("checksum") else None,
        )

    def with_checksum(self) -> "Commit":
        d = self.to_dict()
        d.pop("checksum", None)
        self.checksum = compute_checksum(d)
        return self

    def checksum_valid(self) -> bool:
        return verify_checksum(self.to_dict())


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
    def from_dict(cls, payload: Mapping[str, Any]) -> "Journal":
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

    def with_checksum(self) -> "Journal":
        d = self.to_dict()
        d.pop("checksum", None)
        self.checksum = compute_checksum(d)
        return self

    def checksum_valid(self) -> bool:
        return verify_checksum(self.to_dict())
```

### `src/xpkg/io/siesta_store/lock.py`

Hard-link based advisory lock. This relies on `os.link`, which is available on Unix and Windows. citeturn5view0

```python
from __future__ import annotations

import os
import socket
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Mapping

from xpkg.core.json_utils import dump_json, parse_json_dict
from xpkg.io.siesta_store.errors import LockAcquisitionError


class StoreLock:
    """Advisory lock for the siesta_store directory bundle."""

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

    def __enter__(self) -> "StoreLock":
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
        return parse_json_dict(raw)

    def _is_stale(self, holder: Mapping[str, Any]) -> bool:
        if self.stale_after_seconds is None:
            return False
        ts = holder.get("timestamp")
        if not isinstance(ts, (int, float)):
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
            fd, tmp_name = tempfile.mkstemp(prefix=".siesta_lock_", dir=str(self.store_root))
            tmp_path = Path(tmp_name)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(dump_json(self._metadata(), indent=None, compact=True))
                try:
                    os.link(str(tmp_path), str(self.lock_path))
                except FileExistsError:
                    holder = self._read_holder()
                    if self._is_stale(holder):
                        self.lock_path.unlink(missing_ok=True)
                        continue
                    if deadline is not None and time.monotonic() < deadline:
                        time.sleep(self.poll_interval_seconds)
                        continue
                    raise LockAcquisitionError(f"Store lock contention: {self.lock_path}")
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
```

### `src/xpkg/io/siesta_store/oplog.py`

The oplog is JSONL append-only. Durability uses flush + `os.fsync`. citeturn4view0

```python
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator


def now_utc_iso() -> str:
    return datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")


@dataclass(slots=True)
class WorkspaceState:
    session_id: str
    base_commit_id: str
    last_seq: int


class OplogWriter:
    def __init__(
        self,
        oplog_path: Path,
        *,
        start_seq: int = 0,
        fsync_every_n: int = 16,
    ) -> None:
        self.oplog_path = Path(oplog_path)
        self.oplog_path.parent.mkdir(parents=True, exist_ok=True)
        self._seq = int(start_seq)
        self._fsync_every_n = max(int(fsync_every_n), 1)
        self._since_fsync = 0

        self._fh = self.oplog_path.open("a", encoding="utf-8", newline="\n")

    @property
    def seq(self) -> int:
        return self._seq

    def append(self, op: dict[str, Any]) -> int:
        self._seq += 1
        record = {"seq": self._seq, "ts": now_utc_iso(), **op}
        line = json.dumps(record, ensure_ascii=True, separators=(",", ":"))
        self._fh.write(line + "\n")
        self._fh.flush()

        self._since_fsync += 1
        if self._since_fsync >= self._fsync_every_n:
            os.fsync(self._fh.fileno())
            self._since_fsync = 0

        return self._seq

    def close(self) -> None:
        try:
            self._fh.flush()
            os.fsync(self._fh.fileno())
        except Exception:
            pass
        try:
            self._fh.close()
        except Exception:
            pass

    def __enter__(self) -> "OplogWriter":
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb) -> None:
        self.close()


def iter_oplog(path: Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            yield json.loads(text)
```

### `src/xpkg/io/siesta_store/journal.py`

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

from xpkg.core.json_utils import load_json_dict
from xpkg.io.siesta_store.platform_io import atomic_write_json


def read_journal(path: Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    return load_json_dict(path)


def write_journal(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(path, payload, fsync_file=True, fsync_dir=True)


def clear_journal(path: Path) -> None:
    path = Path(path)
    if path.exists():
        path.unlink()
```

### `src/xpkg/io/siesta_store/object_store.py`

Objects are immutable content-addressed blobs written via (copy → fsync → replace). Atomic replacement behavior is central. citeturn5view1

```python
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from xpkg.io.siesta_store.hashing import sha256_file
from xpkg.io.siesta_store.platform_io import atomic_copy_file
from xpkg.io.siesta_store.paths import StorePaths


def put_object_file(
    paths: StorePaths,
    src_path: Path,
    *,
    ext: str,
) -> str:
    src_path = Path(src_path)
    digest = sha256_file(src_path)
    object_id = f"obj_{digest}"
    dst_path = paths.object_path(object_id, ext=ext)

    if dst_path.exists():
        return object_id

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_copy_file(src_path, dst_path, fsync_file=True, fsync_dir=True)
    return object_id


def get_object_file(paths: StorePaths, object_id: str, *, ext: str) -> Path:
    return paths.object_path(object_id, ext=ext)
```

### `src/xpkg/io/siesta_store/store.py`

This is the core: superblock selection, commit creation, journaled commit flip, and recovery. Atomicity relies on `os.replace`. citeturn5view1 Durability relies on `os.fsync`. citeturn4view0

```python
from __future__ import annotations

import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from xpkg.core.json_utils import load_json_dict
from xpkg.io.siesta_store.errors import (
    ChecksumError,
    IncompatibleStoreVersionError,
    JournalStateError,
    StoreCorruptionError,
)
from xpkg.io.siesta_store.hashing import compute_checksum, verify_checksum
from xpkg.io.siesta_store.journal import clear_journal, read_journal, write_journal
from xpkg.io.siesta_store.lock import StoreLock
from xpkg.io.siesta_store.object_store import get_object_file, put_object_file
from xpkg.io.siesta_store.paths import StorePaths
from xpkg.io.siesta_store.platform_io import atomic_write_json
from xpkg.io.siesta_store.schema import Commit, Journal, Superblock, now_utc_iso


_SUPPORTED_STORE_VERSION = 1
_STORE_FORMAT = "xpkg.siesta-store"


@dataclass(slots=True)
class MountedHead:
    slot: Literal["a", "b"]
    superblock: Superblock


def _load_verified_json(path: Path) -> dict[str, Any]:
    payload = load_json_dict(path)
    if not verify_checksum(payload):
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
    if sb.format != _STORE_FORMAT:
        raise StoreCorruptionError(f"Unexpected store format: {sb.format}")
    return sb


def _choose_head(paths: StorePaths) -> MountedHead | None:
    a = _load_superblock(paths.superblock_a)
    b = _load_superblock(paths.superblock_b)
    if a is None and b is None:
        return None
    if a is None:
        return MountedHead(slot="b", superblock=b)  # type: ignore[arg-type]
    if b is None:
        return MountedHead(slot="a", superblock=a)

    # Choose highest generation; tie-breaker: updated_at lexicographic
    if a.generation > b.generation:
        return MountedHead(slot="a", superblock=a)
    if b.generation > a.generation:
        return MountedHead(slot="b", superblock=b)
    return MountedHead(slot="a", superblock=a) if a.updated_at >= b.updated_at else MountedHead(
        slot="b", superblock=b
    )


def _inactive_slot(active: Literal["a", "b"]) -> Literal["a", "b"]:
    return "b" if active == "a" else "a"


class SiestaStore:
    def __init__(self, root: Path) -> None:
        self.paths = StorePaths(root=Path(root))

    @classmethod
    def create_from_sta(
        cls,
        store_root: Path,
        initial_sta: Path,
        *,
        created_by: dict[str, Any] | None = None,
        reason: str = "init",
    ) -> "SiestaStore":
        store = cls(store_root)
        p = store.paths
        p.root.mkdir(parents=True, exist_ok=True)
        p.journal_dir.mkdir(parents=True, exist_ok=True)
        p.commits_dir.mkdir(parents=True, exist_ok=True)
        p.objects_dir.mkdir(parents=True, exist_ok=True)
        p.workspace_dir.mkdir(parents=True, exist_ok=True)
        p.snapshots_dir.mkdir(parents=True, exist_ok=True)

        with StoreLock(p.root):
            # Initial object
            obj_id = put_object_file(p, initial_sta, ext=".sta")
            commit_id = f"c_{1:012d}_{secrets.token_hex(4)}"
            commit = Commit(
                commit_id=commit_id,
                generation=1,
                parent_commit_id=None,
                created_at=now_utc_iso(),
                reason=reason,
                created_by=created_by or {},
                roots={"bundle": {"object_id": obj_id, "ext": ".sta"}},
            ).with_checksum()

            commit_path = p.commit_json(1)
            commit_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(commit_path, commit.to_dict(), fsync_file=True, fsync_dir=True)

            sb = Superblock(
                format=_STORE_FORMAT,
                store_version=_SUPPORTED_STORE_VERSION,
                generation=1,
                current_commit_id=commit_id,
                previous_commit_id=None,
                last_clean_commit_id=commit_id,
                active_journal_txn_id=None,
                created_at=now_utc_iso(),
                updated_at=now_utc_iso(),
            ).with_checksum()

            # Write both superblocks identically on init
            atomic_write_json(p.superblock_a, sb.to_dict(), fsync_file=True, fsync_dir=True)
            atomic_write_json(p.superblock_b, sb.to_dict(), fsync_file=True, fsync_dir=True)

            clear_journal(p.active_journal)

        return store

    @classmethod
    def open(cls, store_root: Path) -> "SiestaStore":
        store = cls(store_root)
        store.recover()
        return store

    def recover(self) -> MountedHead:
        p = self.paths
        if not p.root.exists():
            raise FileNotFoundError(f"Store root not found: {p.root}")

        with StoreLock(p.root):
            head = _choose_head(p)
            if head is None:
                raise StoreCorruptionError("No valid superblock found")

            journal_payload = read_journal(p.active_journal)
            if journal_payload is None:
                return head

            # If journal exists, we recover deterministically:
            # - staging/validating: superblock flip not guaranteed; mount last_clean
            # - committing/cleanup: if current_commit appears present, keep; else revert last_clean
            j = Journal.from_dict(journal_payload)
            if not j.checksum_valid():
                raise ChecksumError("Journal checksum invalid")

            sb = head.superblock
            if j.state in {"staging", "validating"}:
                # revert to last clean
                sb.current_commit_id = sb.last_clean_commit_id
                sb.previous_commit_id = sb.previous_commit_id
                sb.active_journal_txn_id = None
                sb.updated_at = now_utc_iso()
                sb.with_checksum()
                # Bring both superblocks into agreement via a new generation bump
                new_gen = sb.generation + 1
                sb.generation = new_gen
                sb.with_checksum()
                atomic_write_json(p.superblock_a, sb.to_dict(), fsync_file=True, fsync_dir=True)
                atomic_write_json(p.superblock_b, sb.to_dict(), fsync_file=True, fsync_dir=True)
                clear_journal(p.active_journal)
                return MountedHead(slot="a", superblock=sb)

            if j.state in {"committing", "cleanup"}:
                # Trust the superblock if it points to an existing commit file
                current_commit_path = self.commit_path_for_id(sb.current_commit_id)
                if current_commit_path is not None and current_commit_path.exists():
                    clear_journal(p.active_journal)
                    return head

                # Otherwise revert to last_clean
                sb.current_commit_id = sb.last_clean_commit_id
                sb.active_journal_txn_id = None
                sb.updated_at = now_utc_iso()
                sb.generation = sb.generation + 1
                sb.with_checksum()
                atomic_write_json(p.superblock_a, sb.to_dict(), fsync_file=True, fsync_dir=True)
                atomic_write_json(p.superblock_b, sb.to_dict(), fsync_file=True, fsync_dir=True)
                clear_journal(p.active_journal)
                return MountedHead(slot="a", superblock=sb)

            # Unknown journal state: fail closed
            raise JournalStateError(f"Unknown journal state: {j.state}")

    def _head(self) -> MountedHead:
        head = _choose_head(self.paths)
        if head is None:
            raise StoreCorruptionError("No valid superblock found")
        return head

    def commit_path_for_id(self, commit_id: str) -> Path | None:
        # Linear scan is acceptable for v0; replace with an index later if needed.
        commits_dir = self.paths.commits_dir
        if not commits_dir.exists():
            return None
        for commit_json in commits_dir.glob("*/commit.json"):
            try:
                payload = _load_verified_json(commit_json)
                if payload.get("commit_id") == commit_id:
                    return commit_json
            except Exception:
                continue
        return None

    def load_current_commit(self) -> Commit:
        head = self._head()
        commit_json = self.commit_path_for_id(head.superblock.current_commit_id)
        if commit_json is None or not commit_json.exists():
            raise StoreCorruptionError("Current commit not found on disk")
        payload = _load_verified_json(commit_json)
        c = Commit.from_dict(payload)
        if not c.checksum_valid():
            raise ChecksumError("Commit checksum invalid")
        return c

    def current_bundle_path(self) -> Path:
        commit = self.load_current_commit()
        root = commit.roots.get("bundle")
        if not isinstance(root, dict):
            raise StoreCorruptionError("Commit missing roots.bundle")
        object_id = str(root.get("object_id", ""))
        ext = str(root.get("ext", ".sta"))
        path = get_object_file(self.paths, object_id, ext=ext)
        if not path.exists():
            raise StoreCorruptionError(f"Bundle object missing: {path}")
        return path

    def commit_new_bundle(
        self,
        bundle_path: Path,
        *,
        reason: str,
        created_by: dict[str, Any] | None = None,
    ) -> str:
        p = self.paths
        with StoreLock(p.root):
            head = self._head()
            sb = head.superblock
            new_gen = int(sb.generation) + 1
            base_commit_id = sb.current_commit_id

            txn_id = f"txn_{now_utc_iso().replace(':','').replace('-','')}_{secrets.token_hex(2)}"
            journal = Journal(
                txn_id=txn_id,
                state="staging",
                intent="commit.create",
                base_commit_id=base_commit_id,
                target_generation=new_gen,
                started_at=now_utc_iso(),
                updated_at=now_utc_iso(),
            ).with_checksum()
            write_journal(p.active_journal, journal.to_dict())

            obj_id = put_object_file(p, bundle_path, ext=".sta")
            journal.new_object_id = obj_id
            journal.updated_at = now_utc_iso()
            journal.state = "validating"
            journal.with_checksum()
            write_journal(p.active_journal, journal.to_dict())

            commit_id = f"c_{new_gen:012d}_{secrets.token_hex(4)}"
            commit = Commit(
                commit_id=commit_id,
                generation=new_gen,
                parent_commit_id=base_commit_id,
                created_at=now_utc_iso(),
                reason=reason,
                created_by=created_by or {},
                roots={"bundle": {"object_id": obj_id, "ext": ".sta"}},
            ).with_checksum()

            commit_path = p.commit_json(new_gen)
            commit_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(commit_path, commit.to_dict(), fsync_file=True, fsync_dir=True)

            journal.state = "committing"
            journal.updated_at = now_utc_iso()
            journal.staged_commit_path = str(commit_path)
            journal.with_checksum()
            write_journal(p.active_journal, journal.to_dict())

            # Flip superblock (write to inactive slot)
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
                atomic_write_json(p.superblock_a, new_sb.to_dict(), fsync_file=True, fsync_dir=True)
            else:
                atomic_write_json(p.superblock_b, new_sb.to_dict(), fsync_file=True, fsync_dir=True)

            journal.state = "cleanup"
            journal.updated_at = now_utc_iso()
            journal.with_checksum()
            write_journal(p.active_journal, journal.to_dict())

            clear_journal(p.active_journal)

            return commit_id
```

### `src/xpkg/io/siesta_store/__init__.py`

```python
from __future__ import annotations

from xpkg.io.siesta_store.errors import (
    ChecksumError,
    IncompatibleStoreVersionError,
    JournalStateError,
    LockAcquisitionError,
    SiestaStoreError,
    StoreCorruptionError,
)
from xpkg.io.siesta_store.store import SiestaStore

__all__ = [
    "ChecksumError",
    "IncompatibleStoreVersionError",
    "JournalStateError",
    "LockAcquisitionError",
    "SiestaStore",
    "SiestaStoreError",
    "StoreCorruptionError",
]
```

### `src/xpkg/formats/siesta_store.py`

This mirrors the existing “formats” facade pattern already used for `.sta`.

```python
from __future__ import annotations

from pathlib import Path

from xpkg.io.siesta_store import SiestaStore


def create_store_from_sta(store_root: Path, initial_sta: Path) -> SiestaStore:
    return SiestaStore.create_from_sta(store_root=store_root, initial_sta=initial_sta)


def open_store(store_root: Path) -> SiestaStore:
    return SiestaStore.open(store_root)


__all__ = ["SiestaStore", "create_store_from_sta", "open_store"]
```

## Commit and recovery state machines

The commit state machine is intentionally narrow and auditable:

- **staging**: journal written, base commit known; nothing visible yet.
- **validating**: object written and durable; commit JSON written.
- **committing**: superblock flip is about to happen (the only step that changes “current”).
- **cleanup**: flip is done; journal cleanup remains.
- **idle**: no journal.

The atomicity of the “root flip” is grounded in the semantics of `os.replace` for replacement renames. citeturn5view1 Durability is grounded in explicit `os.fsync` behavior. citeturn4view0

Recovery is deterministic: if a journal exists and we cannot prove the flip completed cleanly, we revert to `last_clean_commit_id`. This is the same philosophical commitment as copy-on-write filesystems: never let in-progress mutation overwrite the last clean checkpoint.

## Tests to add

Create these test files under `tests/` (pytest is already configured in `pyproject.toml`):

- `tests/test_siesta_store_checksums.py`
- `tests/test_siesta_store_superblock_selection.py`
- `tests/test_siesta_store_journal_recovery.py`
- `tests/test_siesta_store_object_store.py`
- `tests/test_siesta_store_oplog.py`

Minimal examples:

### `tests/test_siesta_store_checksums.py`

```python
from __future__ import annotations

from xpkg.io.siesta_store.schema import Superblock, now_utc_iso


def test_superblock_checksum_roundtrip() -> None:
    sb = Superblock(
        format="xpkg.siesta-store",
        store_version=1,
        generation=1,
        current_commit_id="c_000000000001_deadbeef",
        previous_commit_id=None,
        last_clean_commit_id="c_000000000001_deadbeef",
        active_journal_txn_id=None,
        created_at=now_utc_iso(),
        updated_at=now_utc_iso(),
    ).with_checksum()

    assert sb.checksum is not None
    assert sb.checksum_valid()
```

### `tests/test_siesta_store_oplog.py`

```python
from __future__ import annotations

from pathlib import Path

from xpkg.io.siesta_store.oplog import OplogWriter, iter_oplog


def test_oplog_append_and_read(tmp_path: Path) -> None:
    p = tmp_path / "workspace" / "session.oplog.jsonl"
    with OplogWriter(p, fsync_every_n=1) as w:
        w.append({"op": "point.set", "x": 1.0, "y": 2.0})
        w.append({"op": "point.set", "x": 3.0, "y": 4.0})

    rows = list(iter_oplog(p))
    assert len(rows) == 2
    assert rows[0]["seq"] == 1
    assert rows[1]["seq"] == 2
```

## Cross-platform and filesystem caveats

This design works on macOS/Linux/Windows because it relies on primitives implemented across platforms:

- Hard-link creation with `os.link` (available on Unix and Windows). citeturn5view0  
- Atomic replacement rename with `os.replace` (documented semantics, including that it may fail across filesystems; atomicity is a POSIX requirement when successful). citeturn5view1  
- Durability boundary with `os.fsync` (Unix `fsync()`, Windows `_commit()`). citeturn4view0  
- Windows-safe temporary file handling (`NamedTemporaryFile` patterns differ; reopening by name and deletion semantics are OS-dependent). citeturn2search0  

Where this design intentionally *does not overpromise* is networked or sync-backed folders. Any system that violates POSIX write/rename semantics can defeat durability strategies (this is explicitly called out in HDF5 SWMR documentation as well, which requires POSIX write semantics). citeturn1search1turn1search0 The correct engineering stance is to treat those storage backends as “best effort” until validated empirically.

The code above is the minimal store substrate. Once it is in place, the next step is integrating it into xpkg’s `.sta` label save path (so label saves become “produce new `.sta` blob → commit flip” rather than “overwrite a single location”), and wiring the GUI to emit oplog operations for crash recovery.