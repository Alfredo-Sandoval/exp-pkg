from __future__ import annotations

from pathlib import Path

from posetta.io.siesta_store import SiestaStore


def create_store_from_archive(store_root: Path, initial_archive: Path) -> SiestaStore:
    """Create a durable siesta_store root from an existing `.siesta` archive."""
    return SiestaStore.create_from_archive(
        store_root=store_root,
        initial_archive=initial_archive,
    )


def create_store_from_sta(store_root: Path, initial_sta: Path) -> SiestaStore:
    """Compatibility wrapper for creating a store from an existing archive payload."""
    return create_store_from_archive(store_root=store_root, initial_archive=initial_sta)


def open_store(store_root: Path) -> SiestaStore:
    """Open an existing siesta_store root and recover it if needed."""
    return SiestaStore.open(store_root)


__all__ = [
    "SiestaStore",
    "create_store_from_archive",
    "create_store_from_sta",
    "open_store",
]
