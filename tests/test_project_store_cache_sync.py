from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from tests.factories import make_labels
from xpkg.model import Labels
from xpkg.project import (
    current_project_state_path,
    init_project,
    save_project_labels,
)
from xpkg.project.durable_store import ProjectDurableStore
from xpkg.project.state_io import (
    project_state_cache_digest_matches,
    project_state_cache_digest_path,
    state_commit_id,
    write_project_state_cache_digest,
)
from xpkg.project.store import current_project_commit_id


def test_save_project_labels_creates_durable_head_and_state_commit_id(tmp_path: Path) -> None:
    project = tmp_path / "Project"
    init_project(project, title="Project")

    labels = make_labels(tmp_path, x=3.0, y=4.0)
    state_path = save_project_labels(project, labels)

    assert state_path == current_project_state_path(project)
    assert state_path.exists()

    commit_id = current_project_commit_id(project)
    assert commit_id is not None
    store = ProjectDurableStore.open(project / ".xpkg")
    assert store.has_current_root("state")
    assert not store.has_current_root("archive")

    state_payload = json.loads(state_path.read_text(encoding="utf-8"))["payload"]
    assert state_commit_id(state_payload) == commit_id
    assert project_state_cache_digest_path(state_path).is_file()
    assert project_state_cache_digest_matches(state_path, commit_id=commit_id)


def test_project_load_ignores_state_when_commit_id_mismatches_head(tmp_path: Path) -> None:
    project = tmp_path / "Project"
    init_project(project, title="Project")

    labels = make_labels(tmp_path, x=3.0, y=4.0)
    state_path = save_project_labels(project, labels)

    document = json.loads(state_path.read_text(encoding="utf-8"))
    document["payload"]["metadata"]["xpkg_commit_id"] = "c_stale_state"
    document["payload"]["data"]["keypoints"][0][0][0][0] = 99.0
    state_path.write_text(json.dumps(document, indent=2), encoding="utf-8")

    loaded = Labels.load_file(project.as_posix())
    pts = loaded.labeled_frames[0].instances[0].point_records(copy=False)

    assert float(pts["x"][0]) == 3.0
    assert float(pts["y"][0]) == 4.0


def _state_cache_with_digest(tmp_path: Path, *, commit_id: str) -> Path:
    state = tmp_path / "state.json"
    state.write_text('{"payload": {"data": 1}}', encoding="utf-8")
    write_project_state_cache_digest(state, commit_id=commit_id)
    return state


def test_state_cache_digest_matches_fresh_sidecar(tmp_path: Path) -> None:
    state = _state_cache_with_digest(tmp_path, commit_id="c_000000000001_aa")

    assert project_state_cache_digest_matches(state, commit_id="c_000000000001_aa")


def test_state_cache_digest_does_not_match_without_sidecar(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    state.write_text('{"payload": {}}', encoding="utf-8")

    assert not project_state_cache_digest_matches(state, commit_id="c_000000000001_aa")


def test_state_cache_digest_warns_and_rejects_corrupt_sidecar(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Corrupt digest sidecars are warn-logged misses, not silent ones."""

    state = _state_cache_with_digest(tmp_path, commit_id="c_000000000001_aa")
    digest_path = project_state_cache_digest_path(state)
    digest_path.write_text("{this is not json", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="xpkg.project.state_io"):
        assert not project_state_cache_digest_matches(state, commit_id="c_000000000001_aa")

    assert "Unreadable state cache digest" in caplog.text
    assert digest_path.as_posix() in caplog.text


def test_state_cache_digest_warns_and_rejects_non_object_sidecar(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    state = _state_cache_with_digest(tmp_path, commit_id="c_000000000001_aa")
    project_state_cache_digest_path(state).write_text("[1, 2]", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="xpkg.project.state_io"):
        assert not project_state_cache_digest_matches(state, commit_id="c_000000000001_aa")

    assert "Unreadable state cache digest" in caplog.text


def test_state_cache_digest_commit_mismatch_is_silent_miss(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    state = _state_cache_with_digest(tmp_path, commit_id="c_000000000001_aa")

    with caplog.at_level(logging.WARNING, logger="xpkg.project.state_io"):
        assert not project_state_cache_digest_matches(state, commit_id="c_000000000002_bb")

    assert caplog.records == []


def test_state_cache_digest_rejects_modified_state_payload(tmp_path: Path) -> None:
    state = _state_cache_with_digest(tmp_path, commit_id="c_000000000001_aa")
    state.write_text('{"payload": {"data": 2}}', encoding="utf-8")

    assert not project_state_cache_digest_matches(state, commit_id="c_000000000001_aa")


def test_state_cache_digest_rejects_sidecar_missing_digest_value(tmp_path: Path) -> None:
    state = _state_cache_with_digest(tmp_path, commit_id="c_000000000001_aa")
    project_state_cache_digest_path(state).write_text(
        '{"xpkg_commit_id": "c_000000000001_aa", "sha256": ""}',
        encoding="utf-8",
    )

    assert not project_state_cache_digest_matches(state, commit_id="c_000000000001_aa")
