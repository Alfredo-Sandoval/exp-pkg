from __future__ import annotations

from pathlib import Path

from xpkg.io.archive_store.journal import read_journal, write_journal
from xpkg.io.archive_store.paths import StorePaths
from xpkg.io.archive_store.platform_io import atomic_write_json
from xpkg.io.archive_store.schema import Journal, Superblock
from xpkg.io.archive_store.store import ArchiveStore


def test_recover_clears_staging_journal_and_keeps_last_clean_head(tmp_path: Path) -> None:
    archive = tmp_path / "initial.sta"
    archive.write_bytes(b"initial")

    store_root = tmp_path / "project.sta"
    store = ArchiveStore.create_from_sta(store_root, archive)
    initial_head = store.recover()

    journal = Journal(
        txn_id="txn_staging",
        state="staging",
        intent="commit.create",
        base_commit_id=initial_head.superblock.current_commit_id,
        target_generation=initial_head.superblock.generation + 1,
        started_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
    ).with_checksum()
    write_journal(StorePaths(store_root).active_journal, journal.to_dict())

    recovered = ArchiveStore.open(store_root).recover()
    assert recovered.superblock.current_commit_id == initial_head.superblock.last_clean_commit_id
    assert recovered.superblock.generation == initial_head.superblock.generation + 1
    assert read_journal(StorePaths(store_root).active_journal) is None


def test_recover_reverts_committing_state_when_commit_file_is_missing(tmp_path: Path) -> None:
    archive = tmp_path / "initial.sta"
    archive.write_bytes(b"initial")

    store_root = tmp_path / "project.sta"
    store = ArchiveStore.create_from_sta(store_root, archive)
    initial_head = store.recover()
    paths = StorePaths(store_root)
    original_commit_id = initial_head.superblock.current_commit_id

    broken_superblock = Superblock.from_dict(initial_head.superblock.to_dict())
    broken_superblock.generation = initial_head.superblock.generation + 1
    broken_superblock.previous_commit_id = original_commit_id
    broken_superblock.current_commit_id = "c_999999999999_missing"
    broken_superblock.updated_at = "2026-01-01T00:00:01Z"
    broken_superblock.with_checksum()
    atomic_write_json(paths.superblock_a, broken_superblock.to_dict())
    atomic_write_json(paths.superblock_b, broken_superblock.to_dict())

    journal = Journal(
        txn_id="txn_committing",
        state="committing",
        intent="commit.create",
        base_commit_id=original_commit_id,
        target_generation=broken_superblock.generation,
        started_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:01Z",
    ).with_checksum()
    write_journal(paths.active_journal, journal.to_dict())

    recovered = ArchiveStore.open(store_root).recover()
    assert recovered.superblock.current_commit_id == initial_head.superblock.last_clean_commit_id
    assert recovered.superblock.generation == broken_superblock.generation + 1
    assert read_journal(paths.active_journal) is None
