from __future__ import annotations

import json
from pathlib import Path

from tests.factories import make_labels
from xpkg.model import Labels
from xpkg.project import (
    current_project_state_path,
    init_project,
    save_project_labels,
)
from xpkg.project.durable_store import ProjectDurableStore
from xpkg.project.state import project_state_commit_id_from_document
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

    state_document = json.loads(state_path.read_text(encoding="utf-8"))
    assert project_state_commit_id_from_document(state_document) == commit_id


def test_project_load_ignores_state_when_commit_id_mismatches_head(tmp_path: Path) -> None:
    project = tmp_path / "Project"
    init_project(project, title="Project")

    labels = make_labels(tmp_path, x=3.0, y=4.0)
    state_path = save_project_labels(project, labels)

    document = json.loads(state_path.read_text(encoding="utf-8"))
    document["payload"]["metadata"]["xpkg_commit_id"] = "c_stale_state"
    document["payload"]["experiment"]["sessions"][0]["session"]["payload"][
        "session"
    ]["poses"][0]["data"]["data"]["keypoints"][0][0][0][0] = 99.0
    state_path.write_text(json.dumps(document, indent=2), encoding="utf-8")

    loaded = Labels.load_file(project.as_posix())
    pts = loaded.labeled_frames[0].instances[0].point_records(copy=False)

    assert float(pts["x"][0]) == 3.0
    assert float(pts["y"][0]) == 4.0
