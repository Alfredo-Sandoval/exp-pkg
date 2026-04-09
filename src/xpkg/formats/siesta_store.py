from __future__ import annotations

from pathlib import Path

from xpkg.io.siesta_store import SiestaStore


def create_store_from_archive(store_root: Path, initial_archive: Path) -> SiestaStore:
    """Create a durable archive store root from an existing archive payload."""
    return SiestaStore.create_from_archive(
        store_root=store_root,
        initial_archive=initial_archive,
    )


def create_store_from_xpkg(store_root: Path, initial_xpkg: Path) -> SiestaStore:
    """Canonical wrapper for creating a store from an existing `.xpkg` payload."""
    return create_store_from_archive(store_root=store_root, initial_archive=initial_xpkg)


def create_store_from_sta(store_root: Path, initial_sta: Path) -> SiestaStore:
    """Legacy wrapper for creating a store from an existing `.sta` payload."""
    return create_store_from_archive(store_root=store_root, initial_archive=initial_sta)


def open_store(store_root: Path) -> SiestaStore:
    """Open an existing archive store root and recover it if needed."""
    return SiestaStore.open(store_root)


__all__ = [
    "SiestaStore",
    "create_store_from_archive",
    "create_store_from_sta",
    "create_store_from_xpkg",
    "open_store",
]
