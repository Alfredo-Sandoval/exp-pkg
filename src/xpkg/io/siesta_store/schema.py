from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from xpkg.io.siesta_store.hashing import compute_checksum, verify_checksum


def now_utc_iso() -> str:
    """Return an ISO-8601 UTC timestamp with trailing Z."""
    return datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")


@dataclass(slots=True)
class Superblock:
    format: str
    store_version: int
    generation: int
    current_commit_id: str
    previous_commit_id: str | None
    last_clean_commit_id: str
    active_journal_txn_id: str | None
    created_at: str
    updated_at: str
    checksum: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": self.format,
            "store_version": int(self.store_version),
            "generation": int(self.generation),
            "current_commit_id": str(self.current_commit_id),
            "previous_commit_id": self.previous_commit_id,
            "last_clean_commit_id": str(self.last_clean_commit_id),
            "active_journal_txn_id": self.active_journal_txn_id,
            "created_at": str(self.created_at),
            "updated_at": str(self.updated_at),
        }
        if self.checksum:
            payload["checksum"] = self.checksum
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Superblock:
        return cls(
            format=str(payload.get("format", "")),
            store_version=int(payload.get("store_version", 0)),
            generation=int(payload.get("generation", 0)),
            current_commit_id=str(payload.get("current_commit_id", "")),
            previous_commit_id=payload.get("previous_commit_id"),
            last_clean_commit_id=str(payload.get("last_clean_commit_id", "")),
            active_journal_txn_id=payload.get("active_journal_txn_id"),
            created_at=str(payload.get("created_at", "")),
            updated_at=str(payload.get("updated_at", "")),
            checksum=str(payload.get("checksum")) if payload.get("checksum") else None,
        )

    def with_checksum(self) -> Superblock:
        payload = self.to_dict()
        payload.pop("checksum", None)
        self.checksum = compute_checksum(payload)
        return self

    def checksum_valid(self) -> bool:
        return verify_checksum(self.to_dict())


@dataclass(slots=True)
class Commit:
    commit_id: str
    generation: int
    parent_commit_id: str | None
    created_at: str
    reason: str
    created_by: dict[str, Any]
    roots: dict[str, Any]
    checksum: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "commit_id": str(self.commit_id),
            "generation": int(self.generation),
            "parent_commit_id": self.parent_commit_id,
            "created_at": str(self.created_at),
            "reason": str(self.reason),
            "created_by": dict(self.created_by),
            "roots": dict(self.roots),
        }
        if self.checksum:
            payload["checksum"] = self.checksum
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Commit:
        return cls(
            commit_id=str(payload.get("commit_id", "")),
            generation=int(payload.get("generation", 0)),
            parent_commit_id=payload.get("parent_commit_id"),
            created_at=str(payload.get("created_at", "")),
            reason=str(payload.get("reason", "")),
            created_by=dict(payload.get("created_by", {}) or {}),
            roots=dict(payload.get("roots", {}) or {}),
            checksum=str(payload.get("checksum")) if payload.get("checksum") else None,
        )

    def with_checksum(self) -> Commit:
        payload = self.to_dict()
        payload.pop("checksum", None)
        self.checksum = compute_checksum(payload)
        return self

    def checksum_valid(self) -> bool:
        return verify_checksum(self.to_dict())


@dataclass(slots=True)
class Journal:
    txn_id: str
    state: str
    intent: str
    base_commit_id: str
    target_generation: int
    started_at: str
    updated_at: str
    staged_commit_path: str | None = None
    new_object_id: str | None = None
    error: str | None = None
    checksum: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "txn_id": str(self.txn_id),
            "state": str(self.state),
            "intent": str(self.intent),
            "base_commit_id": str(self.base_commit_id),
            "target_generation": int(self.target_generation),
            "started_at": str(self.started_at),
            "updated_at": str(self.updated_at),
            "staged_commit_path": self.staged_commit_path,
            "new_object_id": self.new_object_id,
            "error": self.error,
        }
        if self.checksum:
            payload["checksum"] = self.checksum
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Journal:
        return cls(
            txn_id=str(payload.get("txn_id", "")),
            state=str(payload.get("state", "")),
            intent=str(payload.get("intent", "")),
            base_commit_id=str(payload.get("base_commit_id", "")),
            target_generation=int(payload.get("target_generation", 0)),
            started_at=str(payload.get("started_at", "")),
            updated_at=str(payload.get("updated_at", "")),
            staged_commit_path=payload.get("staged_commit_path"),
            new_object_id=payload.get("new_object_id"),
            error=payload.get("error"),
            checksum=str(payload.get("checksum")) if payload.get("checksum") else None,
        )

    def with_checksum(self) -> Journal:
        payload = self.to_dict()
        payload.pop("checksum", None)
        self.checksum = compute_checksum(payload)
        return self

    def checksum_valid(self) -> bool:
        return verify_checksum(self.to_dict())
