from __future__ import annotations

import os
import socket
import tempfile
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from xpkg.core.json_utils import dump_json, parse_json_dict
from xpkg.io.archive_store.errors import LockAcquisitionError


class StoreLock:
    """Advisory hard-link lock for the archive_store directory root."""

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
            fd, tmp_name = tempfile.mkstemp(prefix=".sta_lock_", dir=str(self.store_root))
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
