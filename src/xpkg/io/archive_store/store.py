from __future__ import annotations

import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from xpkg.core.json_utils import load_json_dict
from xpkg.io.archive_format.shared import CANONICAL_ARCHIVE_SUFFIX
from xpkg.io.archive_store.errors import (
    ChecksumError,
    IncompatibleStoreVersionError,
    JournalStateError,
    StoreCorruptionError,
)
from xpkg.io.archive_store.hashing import verify_checksum
from xpkg.io.archive_store.journal import clear_journal, read_journal, write_journal
from xpkg.io.archive_store.lock import StoreLock
from xpkg.io.archive_store.object_store import get_object_file, put_object_file
from xpkg.io.archive_store.paths import StorePaths
from xpkg.io.archive_store.platform_io import atomic_write_json
from xpkg.io.archive_store.schema import Commit, Journal, RootEntry, Superblock, now_utc_iso

_SUPPORTED_STORE_VERSION = 1
_STORE_FORMAT = "xpkg.archive-store"


@dataclass(slots=True)
class MountedHead:
    slot: Literal["a", "b"]
    superblock: Superblock


def _object_ext_for_path(path: Path, *, default: str = ".xpkg") -> str:
    suffix = Path(path).suffix.strip()
    if not suffix:
        return default
    return suffix if suffix.startswith(".") else f".{suffix}"


def _load_verified_json(path: Path) -> dict[str, Any]:
    payload = load_json_dict(path)
    if not verify_checksum(payload):
        raise ChecksumError(f"Checksum invalid: {path}")
    return payload


def _load_superblock(path: Path) -> Superblock | None:
    if not path.exists():
        return None
    payload = _load_verified_json(path)
    sb = Superblock.from_dict(payload)
    if sb.store_version != _SUPPORTED_STORE_VERSION:
        raise IncompatibleStoreVersionError(
            f"Unsupported store_version={sb.store_version}; expected {_SUPPORTED_STORE_VERSION}"
        )
    if sb.format != _STORE_FORMAT:
        raise StoreCorruptionError(f"Unexpected store format: {sb.format}")
    return sb


def _try_load_slot(
    path: Path,
    slot: Literal["a", "b"],
) -> tuple[MountedHead | None, Exception | None]:
    try:
        superblock = _load_superblock(path)
    except Exception as exc:
        return None, exc
    if superblock is None:
        return None, None
    return MountedHead(slot=slot, superblock=superblock), None


def _choose_head(paths: StorePaths) -> MountedHead | None:
    a_head, a_error = _try_load_slot(paths.superblock_a, "a")
    b_head, b_error = _try_load_slot(paths.superblock_b, "b")

    candidates = [head for head in (a_head, b_head) if head is not None]
    if not candidates:
        if a_error is not None:
            raise a_error
        if b_error is not None:
            raise b_error
        return None
    if len(candidates) == 1:
        return candidates[0]

    a = candidates[0]
    b = candidates[1]
    if a.superblock.generation > b.superblock.generation:
        return a
    if b.superblock.generation > a.superblock.generation:
        return b
    return a if a.superblock.updated_at >= b.superblock.updated_at else b


def _inactive_slot(active: Literal["a", "b"]) -> Literal["a", "b"]:
    return "b" if active == "a" else "a"


def _commit_root_entry(object_id: str, *, ext: str) -> RootEntry:
    return RootEntry(object_id=object_id, ext=ext)


def _normalize_root_name(root_name: str) -> str:
    normalized = str(root_name).strip().lower()
    if not normalized:
        raise ValueError("root_name must be a non-empty string")
    return normalized


def _object_ext_for_root(root_name: str, path: Path) -> str:
    normalized_root = _normalize_root_name(root_name)
    default_ext = CANONICAL_ARCHIVE_SUFFIX if normalized_root == "archive" else ".bin"
    suffix = _object_ext_for_path(path, default=default_ext).lower()
    if normalized_root == "archive":
        return CANONICAL_ARCHIVE_SUFFIX
    return suffix


def _generation_from_commit_id(commit_id: str) -> int | None:
    parts = str(commit_id).split("_", 2)
    if len(parts) < 2 or parts[0] != "c" or not parts[1].isdigit():
        return None
    return int(parts[1])


def _commit_root_entries(
    paths: StorePaths,
    root_paths: Mapping[str, Path],
) -> dict[str, RootEntry]:
    if not root_paths:
        raise ValueError("root_paths must contain at least one entry")

    entries: dict[str, RootEntry] = {}
    for raw_root_name, raw_path in root_paths.items():
        root_name = _normalize_root_name(raw_root_name)
        candidate_path = Path(raw_path)
        if not candidate_path.is_file():
            raise FileNotFoundError(f"Root payload not found for {root_name}: {candidate_path}")
        object_ext = _object_ext_for_root(root_name, candidate_path)
        object_id = put_object_file(paths, candidate_path, ext=object_ext)
        entries[root_name] = _commit_root_entry(object_id, ext=object_ext)
    return entries


class ArchiveStore:
    """Mounted durable store backed by dual superblocks and immutable objects."""

    def __init__(self, root: Path) -> None:
        self.paths = StorePaths(root=Path(root))

    @classmethod
    def create_from_archive(
        cls,
        store_root: Path,
        initial_archive: Path,
        *,
        created_by: dict[str, Any] | None = None,
        reason: str = "init",
    ) -> ArchiveStore:
        return cls.create_from_roots(
            store_root,
            {"archive": initial_archive},
            created_by=created_by,
            reason=reason,
        )

    @classmethod
    def create_from_roots(
        cls,
        store_root: Path,
        initial_roots: Mapping[str, Path],
        *,
        created_by: dict[str, Any] | None = None,
        reason: str = "init",
    ) -> ArchiveStore:
        store = cls(store_root)
        paths = store.paths
        paths.root.mkdir(parents=True, exist_ok=True)
        paths.journal_dir.mkdir(parents=True, exist_ok=True)
        paths.commits_dir.mkdir(parents=True, exist_ok=True)
        paths.objects_dir.mkdir(parents=True, exist_ok=True)
        paths.workspace_dir.mkdir(parents=True, exist_ok=True)
        paths.snapshots_dir.mkdir(parents=True, exist_ok=True)

        with StoreLock(paths.root):
            commit_id = f"c_{1:012d}_{secrets.token_hex(4)}"
            commit = Commit(
                commit_id=commit_id,
                generation=1,
                parent_commit_id=None,
                created_at=now_utc_iso(),
                reason=reason,
                created_by=created_by or {},
                roots=_commit_root_entries(paths, initial_roots),
            ).with_checksum()

            commit_path = paths.commit_json(1)
            commit_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(commit_path, commit.to_dict(), fsync_file=True, fsync_dir=True)

            created_at = now_utc_iso()
            updated_at = now_utc_iso()
            sb = Superblock(
                format=_STORE_FORMAT,
                store_version=_SUPPORTED_STORE_VERSION,
                generation=1,
                current_commit_id=commit_id,
                previous_commit_id=None,
                last_clean_commit_id=commit_id,
                active_journal_txn_id=None,
                created_at=created_at,
                updated_at=updated_at,
            ).with_checksum()

            atomic_write_json(
                paths.superblock_a,
                sb.to_dict(),
                fsync_file=True,
                fsync_dir=True,
            )
            atomic_write_json(
                paths.superblock_b,
                sb.to_dict(),
                fsync_file=True,
                fsync_dir=True,
            )
            clear_journal(paths.active_journal)

        return store

    @classmethod
    def open(cls, store_root: Path) -> ArchiveStore:
        store = cls(store_root)
        store.recover()
        return store

    def recover(self) -> MountedHead:
        paths = self.paths
        if not paths.root.exists():
            raise FileNotFoundError(f"Store root not found: {paths.root}")

        with StoreLock(paths.root):
            head = _choose_head(paths)
            if head is None:
                raise StoreCorruptionError("No valid superblock found")

            journal_payload = read_journal(paths.active_journal)
            if journal_payload is None:
                return head

            journal = Journal.from_dict(journal_payload)
            if not journal.checksum_valid():
                raise ChecksumError("Journal checksum invalid")

            sb = head.superblock
            if journal.state in {"staging", "validating"}:
                sb.current_commit_id = sb.last_clean_commit_id
                sb.active_journal_txn_id = None
                sb.updated_at = now_utc_iso()
                sb.generation = sb.generation + 1
                sb.with_checksum()
                atomic_write_json(
                    paths.superblock_a,
                    sb.to_dict(),
                    fsync_file=True,
                    fsync_dir=True,
                )
                atomic_write_json(
                    paths.superblock_b,
                    sb.to_dict(),
                    fsync_file=True,
                    fsync_dir=True,
                )
                clear_journal(paths.active_journal)
                return MountedHead(slot="a", superblock=sb)

            if journal.state in {"committing", "cleanup"}:
                current_commit_path = self.commit_path_for_id(sb.current_commit_id)
                if current_commit_path is not None and current_commit_path.exists():
                    clear_journal(paths.active_journal)
                    return head

                sb.current_commit_id = sb.last_clean_commit_id
                sb.active_journal_txn_id = None
                sb.updated_at = now_utc_iso()
                sb.generation = sb.generation + 1
                sb.with_checksum()
                atomic_write_json(
                    paths.superblock_a,
                    sb.to_dict(),
                    fsync_file=True,
                    fsync_dir=True,
                )
                atomic_write_json(
                    paths.superblock_b,
                    sb.to_dict(),
                    fsync_file=True,
                    fsync_dir=True,
                )
                clear_journal(paths.active_journal)
                return MountedHead(slot="a", superblock=sb)

            raise JournalStateError(f"Unknown journal state: {journal.state}")

    def _head(self) -> MountedHead:
        head = _choose_head(self.paths)
        if head is None:
            raise StoreCorruptionError("No valid superblock found")
        return head

    def commit_path_for_id(self, commit_id: str) -> Path | None:
        """Resolve a commit id to its commit.json path."""
        commits_dir = self.paths.commits_dir
        if not commits_dir.exists():
            return None

        generation = _generation_from_commit_id(commit_id)
        if generation is not None:
            candidate = self.paths.commit_json(generation)
            if candidate.exists():
                try:
                    payload = _load_verified_json(candidate)
                except Exception:
                    return None
                if payload.get("commit_id") == commit_id:
                    return candidate

        for commit_json in commits_dir.glob("*/commit.json"):
            try:
                payload = _load_verified_json(commit_json)
            except Exception:
                continue
            if payload.get("commit_id") == commit_id:
                return commit_json
        return None

    def load_current_commit(self) -> Commit:
        """Load and verify the current commit referenced by the mounted head."""
        head = self._head()
        commit_json = self.commit_path_for_id(head.superblock.current_commit_id)
        if commit_json is None or not commit_json.exists():
            raise StoreCorruptionError("Current commit not found on disk")
        payload = _load_verified_json(commit_json)
        commit = Commit.from_dict(payload)
        if not commit.checksum_valid():
            raise ChecksumError("Commit checksum invalid")
        if commit.commit_id != head.superblock.current_commit_id:
            raise StoreCorruptionError(
                "Current commit id mismatch: "
                f"head={head.superblock.current_commit_id!r}, "
                f"commit={commit.commit_id!r}"
            )
        return commit

    def has_current_root(self, root_name: str) -> bool:
        commit = self.load_current_commit()
        return commit.has_root(_normalize_root_name(root_name))

    def current_archive_path(self) -> Path:
        """Return the current immutable archive payload path."""
        return self.current_root_path("archive")

    def current_root_entry(self, root_name: str) -> RootEntry:
        """Return the typed current commit entry for a named root."""
        commit = self.load_current_commit()
        normalized_root_name = _normalize_root_name(root_name)
        try:
            return commit.root_entry(normalized_root_name)
        except KeyError as exc:
            raise StoreCorruptionError(f"Commit missing roots.{normalized_root_name}") from exc

    def current_root_path(self, root_name: str) -> Path:
        """Return the current immutable payload path for a named commit root."""
        root_entry = self.current_root_entry(root_name)
        path = get_object_file(self.paths, root_entry.object_id, ext=root_entry.ext)
        if not path.exists():
            raise StoreCorruptionError(f"Commit root object missing: {path}")
        return path

    def commit_new_archive(
        self,
        archive_path: Path,
        *,
        reason: str,
        created_by: dict[str, Any] | None = None,
    ) -> str:
        """Create a new immutable commit from a staged archive file."""
        return self.commit_new_roots(
            {"archive": archive_path},
            reason=reason,
            created_by=created_by,
        )

    def commit_new_roots(
        self,
        root_paths: Mapping[str, Path],
        *,
        reason: str,
        created_by: dict[str, Any] | None = None,
    ) -> str:
        """Create a new immutable commit from staged named root payloads."""
        paths = self.paths

        with StoreLock(paths.root):
            head = self._head()
            sb = head.superblock
            new_gen = int(sb.generation) + 1
            base_commit_id = sb.current_commit_id

            txn_id = f"txn_{now_utc_iso().replace(':', '').replace('-', '')}_{secrets.token_hex(2)}"
            journal = Journal(
                txn_id=txn_id,
                state="staging",
                intent="commit.create",
                base_commit_id=base_commit_id,
                target_generation=new_gen,
                started_at=now_utc_iso(),
                updated_at=now_utc_iso(),
            ).with_checksum()
            write_journal(paths.active_journal, journal.to_dict())

            root_entries = _commit_root_entries(paths, root_paths)
            object_ids = [root_entry.object_id for root_entry in root_entries.values()]
            journal.new_object_id = object_ids[0] if object_ids else None
            journal.updated_at = now_utc_iso()
            journal.state = "validating"
            journal.with_checksum()
            write_journal(paths.active_journal, journal.to_dict())

            commit_id = f"c_{new_gen:012d}_{secrets.token_hex(4)}"
            commit = Commit(
                commit_id=commit_id,
                generation=new_gen,
                parent_commit_id=base_commit_id,
                created_at=now_utc_iso(),
                reason=reason,
                created_by=created_by or {},
                roots=root_entries,
            ).with_checksum()

            commit_path = paths.commit_json(new_gen)
            commit_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(commit_path, commit.to_dict(), fsync_file=True, fsync_dir=True)

            journal.state = "committing"
            journal.updated_at = now_utc_iso()
            journal.staged_commit_path = str(commit_path)
            journal.with_checksum()
            write_journal(paths.active_journal, journal.to_dict())

            new_sb = Superblock(
                format=_STORE_FORMAT,
                store_version=_SUPPORTED_STORE_VERSION,
                generation=new_gen,
                current_commit_id=commit_id,
                previous_commit_id=base_commit_id,
                last_clean_commit_id=commit_id,
                active_journal_txn_id=None,
                created_at=sb.created_at,
                updated_at=now_utc_iso(),
            ).with_checksum()

            slot = _inactive_slot(head.slot)
            if slot == "a":
                atomic_write_json(
                    paths.superblock_a,
                    new_sb.to_dict(),
                    fsync_file=True,
                    fsync_dir=True,
                )
            else:
                atomic_write_json(
                    paths.superblock_b,
                    new_sb.to_dict(),
                    fsync_file=True,
                    fsync_dir=True,
                )

            journal.state = "cleanup"
            journal.updated_at = now_utc_iso()
            journal.with_checksum()
            write_journal(paths.active_journal, journal.to_dict())

            clear_journal(paths.active_journal)
            return commit_id
