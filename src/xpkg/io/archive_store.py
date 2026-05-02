"""Internal compatibility bridge for older durable-store imports.

Prefer :mod:`xpkg.io.workspace_durable_store` for new code.
"""

from __future__ import annotations

from xpkg.io.workspace_durable_store import (
    ArchiveStore,
    ArchiveStoreError,
    ChecksumError,
    IncompatibleStoreVersionError,
    JournalStateError,
    LockAcquisitionError,
    StoreCorruptionError,
)

__all__ = [
    "ArchiveStore",
    "ArchiveStoreError",
    "ChecksumError",
    "IncompatibleStoreVersionError",
    "JournalStateError",
    "LockAcquisitionError",
    "StoreCorruptionError",
]
