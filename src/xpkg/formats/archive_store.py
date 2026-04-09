from __future__ import annotations

from pathlib import Path

from xpkg.io.archive_store import ArchiveStore


def create_store_from_archive(store_root: Path, initial_archive: Path) -> ArchiveStore:
    """Create a durable archive store root from an existing archive payload."""
    return ArchiveStore.create_from_archive(
        store_root=store_root,
        initial_archive=initial_archive,
    )


def create_store_from_xpkg(store_root: Path, initial_xpkg: Path) -> ArchiveStore:
    """Canonical wrapper for creating a store from an existing `.xpkg` payload."""
    return create_store_from_archive(store_root=store_root, initial_archive=initial_xpkg)


def create_store_from_sta(store_root: Path, initial_sta: Path) -> ArchiveStore:
    """Legacy wrapper for creating a store from an existing `.sta` payload."""
    return create_store_from_archive(store_root=store_root, initial_archive=initial_sta)


def open_store(store_root: Path) -> ArchiveStore:
    """Open an existing archive store root and recover it if needed."""
    return ArchiveStore.open(store_root)


__all__ = [
    "ArchiveStore",
    "create_store_from_archive",
    "create_store_from_sta",
    "create_store_from_xpkg",
    "open_store",
]
