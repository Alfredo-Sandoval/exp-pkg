from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    """Return the hex SHA-256 digest for a bytes payload."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    """Return the hex SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "sha256_bytes",
    "sha256_file",
]
