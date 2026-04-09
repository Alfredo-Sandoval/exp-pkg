from __future__ import annotations

from pathlib import Path

from xpkg.io.archive_store.oplog import OplogWriter, iter_oplog


def test_oplog_append_and_read(tmp_path: Path) -> None:
    oplog_path = tmp_path / "workspace" / "session.oplog.jsonl"
    with OplogWriter(oplog_path, fsync_every_n=1) as writer:
        writer.append({"op": "point.set", "x": 1.0, "y": 2.0})
        writer.append({"op": "point.set", "x": 3.0, "y": 4.0})

    rows = list(iter_oplog(oplog_path))
    assert len(rows) == 2
    assert rows[0]["seq"] == 1
    assert rows[0]["op"] == "point.set"
    assert rows[1]["seq"] == 2
    assert rows[1]["x"] == 3.0
    assert rows[1]["y"] == 4.0


def test_iter_oplog_missing_file_is_empty(tmp_path: Path) -> None:
    rows = list(iter_oplog(tmp_path / "missing.jsonl"))
    assert rows == []
