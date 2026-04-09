from __future__ import annotations

import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from xpkg.core.json_utils import load_json_dict
from xpkg.io.siesta_format.shared import CANONICAL_BUNDLE_SUFFIX, SUPPORTED_BUNDLE_SUFFIXES
from xpkg.io.siesta_store.errors import (
    ChecksumError,
    IncompatibleStoreVersionError,
    JournalStateError,
    StoreCorruptionError,
)
from xpkg.io.siesta_store.hashing import verify_checksum
from xpkg.io.siesta_store.journal import clear_journal, read_journal, write_journal
from xpkg.io.siesta_store.lock import StoreLock
from xpkg.io.siesta_store.object_store import get_object_file, put_object_file
from xpkg.io.siesta_store.paths import StorePaths
from xpkg.io.siesta_store.platform_io import atomic_write_json
from xpkg.io.siesta_store.schema import Commit, Journal, Superblock, now_utc_iso

_SUPPORTED_STORE_VERSION = 1
_STORE_FORMAT = "xpkg.siesta-store"


@dataclass(slots=True)
class MountedHead:
    slot: Literal["a", "b"]
    superblock: Superblock


def _object_ext_for_path(path: Path, *, default: str = ".sta") -> str:
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


def _commit_root_entry(object_id: str, *, ext: str) -> dict[str, str]:
    return {"object_id": object_id, "ext": ext}


def _normalize_object_ext(path: Path) -> str:
    suffix = _object_ext_for_path(path, default=CANONICAL_BUNDLE_SUFFIX).lower()
    if suffix in SUPPORTED_BUNDLE_SUFFIXES:
        return suffix
    return CANONICAL_BUNDLE_SUFFIX


class SiestaStore:
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
    ) -> SiestaStore:
        return cls.create_from_sta(
            store_root=store_root,
            initial_sta=initial_archive,
            created_by=created_by,
            reason=reason,
        )

    @classmethod
    def create_from_sta(
        cls,
        store_root: Path,
        initial_sta: Path,
        *,
        created_by: dict[str, Any] | None = None,
        reason: str = "init",
    ) -> SiestaStore:
        store = cls(store_root)
        paths = store.paths
        paths.root.mkdir(parents=True, exist_ok=True)
        paths.journal_dir.mkdir(parents=True, exist_ok=True)
        paths.commits_dir.mkdir(parents=True, exist_ok=True)
        paths.objects_dir.mkdir(parents=True, exist_ok=True)
        paths.workspace_dir.mkdir(parents=True, exist_ok=True)
        paths.snapshots_dir.mkdir(parents=True, exist_ok=True)

        initial_path = Path(initial_sta)
        object_ext = _normalize_object_ext(initial_path)

        with StoreLock(paths.root):
            obj_id = put_object_file(paths, initial_path, ext=object_ext)
            commit_id = f"c_{1:012d}_{secrets.token_hex(4)}"
            root_entry = _commit_root_entry(obj_id, ext=object_ext)
            commit = Commit(
                commit_id=commit_id,
                generation=1,
                parent_commit_id=None,
                created_at=now_utc_iso(),
                reason=reason,
                created_by=created_by or {},
                roots={"archive": root_entry},
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
    def open(cls, store_root: Path) -> SiestaStore:
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
        return commit

    def current_bundle_path(self) -> Path:
        """Legacy alias for `current_archive_path`."""
        commit = self.load_current_commit()
        root = commit.roots.get("archive") or commit.roots.get("bundle")
        if not isinstance(root, dict):
            raise StoreCorruptionError(
                "Commit missing roots.archive "
                "(legacy roots.bundle also accepted)"
            )
        object_id = str(root.get("object_id", ""))
        ext = str(root.get("ext", CANONICAL_BUNDLE_SUFFIX))
        path = get_object_file(self.paths, object_id, ext=ext)
        if not path.exists():
            raise StoreCorruptionError(f"Archive object missing: {path}")
        return path

    def current_archive_path(self) -> Path:
        """Alias for the current immutable payload path using archive terminology."""
        return self.current_bundle_path()

    def commit_new_bundle(
        self,
        bundle_path: Path,
        *,
        reason: str,
        created_by: dict[str, Any] | None = None,
    ) -> str:
        """Legacy alias for `commit_new_archive`."""
        return self.commit_new_archive(bundle_path, reason=reason, created_by=created_by)

    def commit_new_archive(
        self,
        archive_path: Path,
        *,
        reason: str,
        created_by: dict[str, Any] | None = None,
    ) -> str:
        """Create a new immutable commit from a staged archive file."""
        paths = self.paths
        candidate_path = Path(archive_path)
        object_ext = _normalize_object_ext(candidate_path)

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

            obj_id = put_object_file(paths, candidate_path, ext=object_ext)
            journal.new_object_id = obj_id
            journal.updated_at = now_utc_iso()
            journal.state = "validating"
            journal.with_checksum()
            write_journal(paths.active_journal, journal.to_dict())

            commit_id = f"c_{new_gen:012d}_{secrets.token_hex(4)}"
            root_entry = _commit_root_entry(obj_id, ext=object_ext)
            commit = Commit(
                commit_id=commit_id,
                generation=new_gen,
                parent_commit_id=base_commit_id,
                created_at=now_utc_iso(),
                reason=reason,
                created_by=created_by or {},
                roots={"archive": root_entry},
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
