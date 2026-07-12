"""Regression tests for narrow corruption handling in the durable store.

``_try_load_slot``, ``_choose_head``, and ``commit_path_for_id`` must treat
only corruption-shaped errors (``_CORRUPT_PAYLOAD_ERRORS``) as recoverable.
Programming errors raised while reading store files must propagate.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from xpkg.project.durable_store import (
    ChecksumError,
    Commit,
    Journal,
    JournalStateError,
    ProjectDurableStore,
    RootEntry,
    StoreCorruptionError,
    StorePaths,
    atomic_write_json,
    read_journal,
    write_journal,
)


def _make_store(tmp_path: Path) -> tuple[ProjectDurableStore, StorePaths, str]:
    state = tmp_path / "initial.json"
    state.write_text('{"version": 1}', encoding="utf-8")
    store_root = tmp_path / ".xpkg"
    store = ProjectDurableStore.create_from_roots(store_root, {"state": state})
    head = store.recover()
    return store, StorePaths(store_root), head.superblock.current_commit_id


def _tamper_checksum(path: Path) -> None:
    """Rewrite a store JSON file with an intentionally wrong checksum."""

    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["checksum"] = "sha256:" + "0" * 64
    atomic_write_json(path, payload)


def test_recover_uses_good_slot_when_other_slot_is_garbage_json(tmp_path: Path) -> None:
    _, paths, commit_id = _make_store(tmp_path)
    paths.superblock_a.write_text("{this is not json", encoding="utf-8")

    head = ProjectDurableStore.open(paths.root).recover()

    assert head.slot == "b"
    assert head.superblock.current_commit_id == commit_id


def test_recover_uses_good_slot_when_other_slot_fails_checksum(tmp_path: Path) -> None:
    _, paths, commit_id = _make_store(tmp_path)
    _tamper_checksum(paths.superblock_b)

    head = ProjectDurableStore.open(paths.root).recover()

    assert head.slot == "a"
    assert head.superblock.current_commit_id == commit_id


def test_recover_raises_checksum_error_when_both_slots_corrupt(tmp_path: Path) -> None:
    _, paths, _ = _make_store(tmp_path)
    _tamper_checksum(paths.superblock_a)
    _tamper_checksum(paths.superblock_b)

    with pytest.raises(ChecksumError, match="Checksum invalid"):
        ProjectDurableStore(paths.root).recover()


def test_recover_propagates_programming_error_from_superblock_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-corruption error in one slot must propagate, not fall back.

    Before the ``_CORRUPT_PAYLOAD_ERRORS`` narrowing, ``_try_load_slot``
    caught bare ``Exception`` and recover() silently mounted the healthy
    slot, masking the bug.
    """

    from xpkg.project import durable_store

    _, paths, _ = _make_store(tmp_path)
    real_load_json_dict = durable_store.load_json_dict

    def _bug_in_slot_a(path: Path) -> dict[str, object]:
        if Path(path).name == paths.superblock_a.name:
            raise AttributeError("simulated programming error")
        return real_load_json_dict(path)

    monkeypatch.setattr(durable_store, "load_json_dict", _bug_in_slot_a)

    with pytest.raises(AttributeError, match="simulated programming error"):
        ProjectDurableStore(paths.root).recover()


def test_recover_requires_existing_store_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Store root not found"):
        ProjectDurableStore(tmp_path / "missing" / ".xpkg").recover()


def test_recover_rejects_journal_with_invalid_checksum(tmp_path: Path) -> None:
    _, paths, commit_id = _make_store(tmp_path)
    journal = Journal(
        txn_id="txn_bad_checksum",
        state="staging",
        intent="commit.create",
        base_commit_id=commit_id,
        target_generation=2,
        started_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
    ).with_checksum()
    payload = journal.to_dict()
    payload["state"] = "committing"  # invalidates the checksum
    write_journal(paths.active_journal, payload)

    with pytest.raises(ChecksumError, match="Journal checksum invalid"):
        ProjectDurableStore(paths.root).recover()


def test_recover_rejects_unknown_journal_state(tmp_path: Path) -> None:
    _, paths, commit_id = _make_store(tmp_path)
    journal = Journal(
        txn_id="txn_unknown",
        state="weird",
        intent="commit.create",
        base_commit_id=commit_id,
        target_generation=2,
        started_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
    ).with_checksum()
    write_journal(paths.active_journal, journal.to_dict())

    with pytest.raises(JournalStateError, match="Unknown journal state: weird"):
        ProjectDurableStore(paths.root).recover()


def test_recover_keeps_head_and_clears_journal_when_commit_landed(tmp_path: Path) -> None:
    """A 'committing' journal whose commit exists on disk is a completed txn."""

    _, paths, commit_id = _make_store(tmp_path)
    journal = Journal(
        txn_id="txn_committing_done",
        state="committing",
        intent="commit.create",
        base_commit_id=commit_id,
        target_generation=1,
        started_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
    ).with_checksum()
    write_journal(paths.active_journal, journal.to_dict())

    head = ProjectDurableStore(paths.root).recover()

    assert head.superblock.current_commit_id == commit_id
    assert head.superblock.generation == 1
    assert read_journal(paths.active_journal) is None


def test_commit_path_for_id_returns_none_when_expected_commit_corrupt(tmp_path: Path) -> None:
    store, paths, commit_id = _make_store(tmp_path)
    paths.commit_json(1).write_text("{this is not json", encoding="utf-8")

    assert store.commit_path_for_id(commit_id) is None


def test_commit_path_for_id_scan_skips_corrupt_sibling_commits(tmp_path: Path) -> None:
    store, paths, _ = _make_store(tmp_path)
    # The searched id has no parseable generation, forcing the glob scan.
    target_commit = Commit(
        commit_id="release-tag-commit",
        generation=5,
        parent_commit_id=None,
        created_at="2026-01-01T00:00:00Z",
        reason="tag",
        created_by={},
        roots={"state": RootEntry(object_id="obj_" + "a" * 64, ext=".json")},
    ).with_checksum()
    commit_path = paths.commit_json(5)
    commit_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(commit_path, target_commit.to_dict())
    paths.commit_json(1).write_text("{this is not json", encoding="utf-8")

    assert store.commit_path_for_id("release-tag-commit") == commit_path


def test_commit_path_for_id_returns_none_for_unknown_commit(tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    assert store.commit_path_for_id("c_000000000099_deadbeef") is None


def test_commit_path_for_id_returns_none_without_commits_dir(tmp_path: Path) -> None:
    store, paths, commit_id = _make_store(tmp_path)
    shutil.rmtree(paths.commits_dir)
    assert store.commit_path_for_id(commit_id) is None


def test_commit_path_for_id_propagates_programming_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-corruption error during commit resolution must not become None."""

    store, _, commit_id = _make_store(tmp_path)

    def _programming_bug(_path: Path) -> dict[str, object]:
        raise AttributeError("simulated programming error")

    monkeypatch.setattr("xpkg.project.durable_store.load_json_dict", _programming_bug)

    with pytest.raises(AttributeError, match="simulated programming error"):
        store.commit_path_for_id(commit_id)


def test_load_current_commit_reports_missing_commit_payload(tmp_path: Path) -> None:
    store, paths, _ = _make_store(tmp_path)
    shutil.rmtree(paths.commits_dir)

    with pytest.raises(StoreCorruptionError, match="Current commit not found on disk"):
        store.load_current_commit()


def test_current_root_entry_reports_missing_root(tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)

    assert store.has_current_root("state")
    assert not store.has_current_root("predictions")
    with pytest.raises(StoreCorruptionError, match=r"Commit missing roots\.predictions"):
        store.current_root_entry("predictions")


def test_current_root_path_reports_missing_object_payload(tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    store.current_root_path("state").unlink()

    with pytest.raises(StoreCorruptionError, match="Commit root object missing"):
        store.current_root_path("state")
