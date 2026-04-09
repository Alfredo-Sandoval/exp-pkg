from __future__ import annotations

from pathlib import Path

from xpkg.io.archive_store.hashing import sha256_file
from xpkg.io.archive_store.paths import StorePaths
from xpkg.io.archive_store.platform_io import atomic_copy_file


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
