from __future__ import annotations

from pathlib import Path

from xpkg.io.archive_store.paths import StorePaths
from xpkg.io.archive_store.platform_io import atomic_write_json
from xpkg.io.archive_store.schema import Superblock
from xpkg.io.archive_store.store import ArchiveStore


def _superblock(
    *,
    generation: int,
    current_commit_id: str,
    updated_at: str,
) -> Superblock:
    return Superblock(
        format="xpkg.archive-store",
        store_version=1,
        generation=generation,
        current_commit_id=current_commit_id,
        previous_commit_id=None,
        last_clean_commit_id=current_commit_id,
        active_journal_txn_id=None,
        created_at="2026-01-01T00:00:00Z",
        updated_at=updated_at,
    ).with_checksum()


def test_recover_picks_highest_generation_superblock(tmp_path: Path) -> None:
    paths = StorePaths(root=tmp_path / "project.xpkg")
    paths.root.mkdir(parents=True)

    superblock_a = _superblock(
        generation=1,
        current_commit_id="c_000000000001_deadbeef",
        updated_at="2026-01-01T00:00:00Z",
    )
    superblock_b = _superblock(
        generation=2,
        current_commit_id="c_000000000002_deadbeef",
        updated_at="2026-01-01T00:00:01Z",
    )

    atomic_write_json(paths.superblock_a, superblock_a.to_dict())
    atomic_write_json(paths.superblock_b, superblock_b.to_dict())

    head = ArchiveStore(paths.root).recover()
    assert head.slot == "b"
    assert head.superblock.generation == 2
    assert head.superblock.current_commit_id == "c_000000000002_deadbeef"


def test_recover_breaks_generation_ties_with_updated_at(tmp_path: Path) -> None:
    paths = StorePaths(root=tmp_path / "project.xpkg")
    paths.root.mkdir(parents=True)

    superblock_a = _superblock(
        generation=3,
        current_commit_id="c_000000000003_alpha",
        updated_at="2026-01-01T00:00:00Z",
    )
    superblock_b = _superblock(
        generation=3,
        current_commit_id="c_000000000003_beta",
        updated_at="2026-01-01T00:00:01Z",
    )

    atomic_write_json(paths.superblock_a, superblock_a.to_dict())
    atomic_write_json(paths.superblock_b, superblock_b.to_dict())

    head = ArchiveStore(paths.root).recover()
    assert head.slot == "b"
    assert head.superblock.current_commit_id == "c_000000000003_beta"
