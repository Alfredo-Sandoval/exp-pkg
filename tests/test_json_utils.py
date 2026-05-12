from __future__ import annotations

import os
from pathlib import Path

import pytest

from xpkg import json_utils
from xpkg._core import json_utils as core_json_utils


def test_write_json_writes_valid_json_with_trailing_newline(tmp_path: Path) -> None:
    target = tmp_path / "payload.json"

    json_utils.write_json(target, {"b": 2, "a": 1}, sort_keys=True)

    assert target.read_text(encoding="utf-8") == '{\n  "a": 1,\n  "b": 2\n}\n'
    assert json_utils.load_json(target) == {"a": 1, "b": 2}


def test_write_json_compact_output_remains_one_line(tmp_path: Path) -> None:
    target = tmp_path / "payload.json"

    json_utils.write_json(target, {"items": [1, 2]}, compact=True, trailing_newline=False)

    assert target.read_text(encoding="utf-8") == '{"items":[1,2]}'


def test_load_json_and_load_json_dict_round_trip_through_write_json(tmp_path: Path) -> None:
    target = tmp_path / "payload.json"
    payload = {"name": "run", "meta": {"ok": True}}

    json_utils.write_json(target, payload)

    assert json_utils.load_json(target) == payload
    assert json_utils.load_json_dict(target) == payload


def test_write_json_does_not_use_path_write_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def _write_text_blocked(*args: object, **kwargs: object) -> int:
        raise AssertionError("Path.write_text must not be used by write_json")

    monkeypatch.setattr(Path, "write_text", _write_text_blocked)

    target = tmp_path / "payload.json"
    json_utils.write_json(target, {"ok": True})

    assert json_utils.load_json_dict(target) == {"ok": True}


def test_write_json_cleans_temp_file_when_replace_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "payload.json"
    seen_tmp: dict[str, Path] = {}

    def _failing_replace(
        src: str | os.PathLike[str],
        dst: str | os.PathLike[str],
    ) -> None:
        del dst
        seen_tmp["path"] = Path(src)
        raise OSError("boom")

    monkeypatch.setattr(core_json_utils.os, "replace", _failing_replace)

    with pytest.raises(OSError, match="boom"):
        json_utils.write_json(target, {"value": 5})

    assert not seen_tmp["path"].exists()
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []
    assert not target.exists()
