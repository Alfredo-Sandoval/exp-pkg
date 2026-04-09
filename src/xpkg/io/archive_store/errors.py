from __future__ import annotations


class ArchiveStoreError(Exception):
    """Base exception for archive_store."""


class StoreCorruptionError(ArchiveStoreError):
    """Raised when required store files are missing or checksum validation fails."""


class IncompatibleStoreVersionError(ArchiveStoreError):
    """Raised when the store_version is not supported by this xpkg build."""


class LockAcquisitionError(ArchiveStoreError):
    """Raised when the store lock cannot be acquired."""


class JournalStateError(ArchiveStoreError):
    """Raised for invalid journal transitions or inconsistent recovery states."""


class ChecksumError(ArchiveStoreError):
    """Raised when a checksum cannot be verified."""
