from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from xpkg._core.json_utils import dump_json


def _fsync_file(path: Path) -> None:
    """fsync a file by path."""
    with Path(path).open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_dir_best_effort(directory: Path) -> None:
    """Best-effort directory fsync on POSIX hosts."""
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
