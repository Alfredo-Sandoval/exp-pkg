from __future__ import annotations

from pathlib import Path

from xpkg.formats import create_store_from_sta, open_store
from xpkg.io.siesta_store.object_store import get_object_file, put_object_file
from xpkg.io.siesta_store.paths import StorePaths


def test_put_object_file_is_content_addressed(tmp_path: Path) -> None:
    paths = StorePaths(root=tmp_path / "store")
    source = tmp_path / "payload.siesta"
    source.write_bytes(b"payload-bytes")

    object_id_a = put_object_file(paths, source, ext=".siesta")
    object_id_b = put_object_file(paths, source, ext=".siesta")
    object_path = get_object_file(paths, object_id_a, ext=".siesta")

    assert object_id_a == object_id_b
    assert object_path.exists()
    assert object_path.read_bytes() == b"payload-bytes"
    assert object_path.suffix == ".siesta"


def test_store_wrapper_roundtrip_preserves_archive_suffix(tmp_path: Path) -> None:
    initial_archive = tmp_path / "initial.siesta"
    initial_archive.write_bytes(b"first")

    store = create_store_from_sta(tmp_path / "project.siesta", initial_archive)
    current = store.current_archive_path()
    assert current.suffix == ".siesta"
    assert current.read_bytes() == b"first"

    updated_archive = tmp_path / "updated.siesta"
    updated_archive.write_bytes(b"second")
    commit_id = store.commit_new_archive(updated_archive, reason="update")
    assert commit_id.startswith("c_")

    reopened = open_store(tmp_path / "project.siesta")
    assert reopened.current_archive_path().suffix == ".siesta"
    assert reopened.current_archive_path().read_bytes() == b"second"
