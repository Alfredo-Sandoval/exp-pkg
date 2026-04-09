from __future__ import annotations

from pathlib import Path

from xpkg.io.archive_store.errors import (
    ArchiveStoreError,
    ChecksumError,
    IncompatibleStoreVersionError,
    JournalStateError,
    LockAcquisitionError,
    StoreCorruptionError,
)
from xpkg.io.archive_store.store import ArchiveStore


def create_archive_store(store_root: Path, initial_archive: Path) -> ArchiveStore:
    """Create a durable store from an existing archive payload."""
    return ArchiveStore.create_from_archive(
        store_root=store_root,
        initial_archive=initial_archive,
    )


def create_xpkg_store(store_root: Path, initial_xpkg: Path) -> ArchiveStore:
    """Create a durable store from a canonical `.xpkg` payload."""
    return create_archive_store(store_root=store_root, initial_archive=initial_xpkg)


def open_archive_store(store_root: Path) -> ArchiveStore:
    """Open and recover an existing archive store."""
    return ArchiveStore.open(store_root)


__all__ = [
    "ArchiveStore",
    "ArchiveStoreError",
    "ChecksumError",
    "IncompatibleStoreVersionError",
    "JournalStateError",
    "LockAcquisitionError",
    "StoreCorruptionError",
    "create_archive_store",
    "create_xpkg_store",
    "open_archive_store",
]
