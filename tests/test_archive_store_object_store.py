from __future__ import annotations

from pathlib import Path

from xpkg.compat import create_store_from_archive, open_store
from xpkg.io.archive_store.object_store import get_object_file, put_object_file
from xpkg.io.archive_store.paths import StorePaths


def test_put_object_file_is_content_addressed(tmp_path: Path) -> None:
    paths = StorePaths(root=tmp_path / "store")
    source = tmp_path / "payload.sta"
    source.write_bytes(b"payload-bytes")

    object_id_a = put_object_file(paths, source, ext=".sta")
    object_id_b = put_object_file(paths, source, ext=".sta")
    object_path = get_object_file(paths, object_id_a, ext=".sta")

    assert object_id_a == object_id_b
    assert object_path.exists()
    assert object_path.read_bytes() == b"payload-bytes"
    assert object_path.suffix == ".sta"


def test_store_wrapper_roundtrip_preserves_archive_suffix(tmp_path: Path) -> None:
    initial_archive = tmp_path / "initial.sta"
    initial_archive.write_bytes(b"first")

    store = create_store_from_archive(tmp_path / "project.sta", initial_archive)
    current = store.current_archive_path()
    assert current.suffix == ".sta"
    assert current.read_bytes() == b"first"

    updated_archive = tmp_path / "updated.sta"
    updated_archive.write_bytes(b"second")
    commit_id = store.commit_new_archive(updated_archive, reason="update")
    assert commit_id.startswith("c_")

    reopened = open_store(tmp_path / "project.sta")
    assert reopened.current_archive_path().suffix == ".sta"
    assert reopened.current_archive_path().read_bytes() == b"second"
