from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def now_utc_iso() -> str:
    """Return an ISO-8601 UTC timestamp with trailing Z."""
    return datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")


@dataclass(slots=True)
class WorkspaceState:
    session_id: str
    base_commit_id: str
    last_seq: int


class OplogWriter:
    """Append-only JSONL oplog writer with periodic fsync."""

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

    def __enter__(self) -> OplogWriter:
        return self

    def __exit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        self.close()


def iter_oplog(path: Path) -> Iterator[dict[str, Any]]:
    """Yield decoded oplog rows from a JSONL file."""
    oplog_path = Path(path)
    if not oplog_path.exists():
        return
    with oplog_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            data = json.loads(text)
            if not isinstance(data, dict):
                raise TypeError("Oplog row must decode to a JSON object")
            out: dict[str, Any] = {}
            for key, value in data.items():
                if not isinstance(key, str):
                    raise TypeError("Oplog row keys must be strings")
                out[key] = value
            yield out
