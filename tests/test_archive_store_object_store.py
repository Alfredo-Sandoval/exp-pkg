from __future__ import annotations

from pathlib import Path

from xpkg.io.archive_store import ArchiveStore
from xpkg.io.archive_store.object_store import get_object_file, put_object_file
from xpkg.io.archive_store.paths import StorePaths


def test_put_object_file_is_content_addressed(tmp_path: Path) -> None:
    paths = StorePaths(root=tmp_path / "store")
    source = tmp_path / "payload.bin"
    source.write_bytes(b"payload-bytes")

    object_id_a = put_object_file(paths, source, ext=".bin")
    object_id_b = put_object_file(paths, source, ext=".bin")
    object_path = get_object_file(paths, object_id_a, ext=".bin")

    assert object_id_a == object_id_b
    assert object_path.exists()
    assert object_path.read_bytes() == b"payload-bytes"
    assert object_path.suffix == ".bin"


def test_store_roundtrip_preserves_payload_root_suffix(tmp_path: Path) -> None:
    initial_payload = tmp_path / "initial.bin"
    initial_payload.write_bytes(b"first")

    store = ArchiveStore.create_from_roots(
        tmp_path / "project.xpkg",
        {"payload": initial_payload},
    )
    current = store.current_root_path("payload")
    assert current.suffix == ".bin"
    assert current.read_bytes() == b"first"

    updated_payload = tmp_path / "updated.bin"
    updated_payload.write_bytes(b"second")
    commit_id = store.commit_new_roots({"payload": updated_payload}, reason="update")
    assert commit_id.startswith("c_")

    reopened = ArchiveStore.open(tmp_path / "project.xpkg")
    assert reopened.current_root_path("payload").suffix == ".bin"
    assert reopened.current_root_path("payload").read_bytes() == b"second"


def test_store_roundtrip_preserves_snapshot_root_suffix(tmp_path: Path) -> None:
    initial_snapshot = tmp_path / "initial.json"
    initial_snapshot.write_text('{"version": 1}', encoding="utf-8")

    store = ArchiveStore.create_from_roots(
        tmp_path / "project.xpkg",
        {"snapshot": initial_snapshot},
    )
    current_entry = store.current_root_entry("snapshot")
    current = store.current_root_path("snapshot")
    assert current_entry.ext == ".json"
    assert current_entry.object_id.startswith("obj_")
    assert current.suffix == ".json"
    assert current.read_text(encoding="utf-8") == '{"version": 1}'

    updated_snapshot = tmp_path / "updated.json"
    updated_snapshot.write_text('{"version": 2}', encoding="utf-8")
    commit_id = store.commit_new_roots({"snapshot": updated_snapshot}, reason="update")
    assert commit_id.startswith("c_")

    reopened = ArchiveStore.open(tmp_path / "project.xpkg")
    assert reopened.has_current_root("snapshot")
    assert not reopened.has_current_root("archive")
    assert reopened.current_root_path("snapshot").suffix == ".json"
    assert reopened.current_root_path("snapshot").read_text(encoding="utf-8") == '{"version": 2}'
