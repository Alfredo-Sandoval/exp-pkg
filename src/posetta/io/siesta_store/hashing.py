from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from posetta.core.json_utils import dump_json


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


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize JSON deterministically for checksum calculation."""
    text = dump_json(payload, indent=None, sort_keys=True, ensure_ascii=True, compact=True)
    return text.encode("utf-8")


def compute_checksum(payload_without_checksum: dict[str, Any]) -> str:
    """Compute a prefixed SHA-256 checksum for a JSON-serializable mapping."""
    return f"sha256:{sha256_bytes(canonical_json_bytes(payload_without_checksum))}"


def verify_checksum(payload: dict[str, Any]) -> bool:
    """Return True when the payload checksum matches its deterministic JSON bytes."""
    expected = payload.get("checksum")
    if not isinstance(expected, str) or not expected.startswith("sha256:"):
        return False
    stripped = dict(payload)
    stripped.pop("checksum", None)
    return compute_checksum(stripped) == expected
