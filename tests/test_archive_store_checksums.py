from __future__ import annotations

from xpkg.io.workspace_durable_store import Commit, Journal, RootEntry, Superblock, now_utc_iso


def test_superblock_checksum_roundtrip() -> None:
    superblock = Superblock(
        format="xpkg.workspace-durable-store",
        store_version=1,
        generation=1,
        current_commit_id="c_000000000001_deadbeef",
        previous_commit_id=None,
        last_clean_commit_id="c_000000000001_deadbeef",
        active_journal_txn_id=None,
        created_at=now_utc_iso(),
        updated_at=now_utc_iso(),
    ).with_checksum()

    assert superblock.checksum is not None
    assert superblock.checksum_valid()


def test_commit_and_journal_checksums_fail_after_tampering() -> None:
    commit = Commit(
        commit_id="c_000000000001_deadbeef",
        generation=1,
        parent_commit_id=None,
        created_at=now_utc_iso(),
        reason="init",
        created_by={},
        roots={"snapshot": RootEntry(object_id="obj_deadbeef", ext=".json")},
    ).with_checksum()
    journal = Journal(
        txn_id="txn_deadbeef",
        state="staging",
        intent="commit.create",
        base_commit_id=commit.commit_id,
        target_generation=2,
        started_at=now_utc_iso(),
        updated_at=now_utc_iso(),
    ).with_checksum()

    assert commit.checksum_valid()
    assert journal.checksum_valid()

    commit.reason = "mutated"
    journal.state = "cleanup"

    assert not commit.checksum_valid()
    assert not journal.checksum_valid()
