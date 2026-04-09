from __future__ import annotations

from pathlib import Path
from typing import Any

from xpkg.core.json_utils import load_json_dict
from xpkg.io.archive_store.platform_io import atomic_write_json


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
