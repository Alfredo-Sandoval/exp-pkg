from __future__ import annotations


class SiestaStoreError(Exception):
    """Base exception for siesta_store."""


class StoreCorruptionError(SiestaStoreError):
    """Raised when required store files are missing or checksum validation fails."""


class IncompatibleStoreVersionError(SiestaStoreError):
    """Raised when the store_version is not supported by this xpkg build."""


class LockAcquisitionError(SiestaStoreError):
    """Raised when the store lock cannot be acquired."""


class JournalStateError(SiestaStoreError):
    """Raised for invalid journal transitions or inconsistent recovery states."""


class ChecksumError(SiestaStoreError):
    """Raised when a checksum cannot be verified."""
