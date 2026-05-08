"""Round-trip tests for DatasetDatasheet across project, service, CLI, and pack."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from xpkg.model import DatasetDatasheet, DatasheetMotivation
from xpkg.project import (
    init_project,
    load_project_datasheet,
    pack_project,
    project_datasheet_path,
    save_project_datasheet,
    unpack_project,
)
from xpkg.services.project import ProjectService


def _datasheet_payload() -> dict:
    return {
        "title": "FAIR demo dataset",
        "dataset_id": "fair-demo-2026",
        "version": "1.0",
        "summary": "Smoke-test datasheet for CLI round-trip.",
        "motivation": {
            "purpose": "Verify Datasheet CLI plumbing.",
            "creators": ["Sandoval, A."],
            "funders": ["NIH MH128177"],
        },
        "composition": {
            "instances": "One session per mouse.",
            "instance_count": 12,
            "splits": {"train": "8", "test": "4"},
        },
        "distribution": {
            "license": "CC-BY-4.0",
            "doi": "10.5281/zenodo.42",
        },
        "maintenance": {
            "maintainer": "Sandoval Lab",
            "contact": "sandoval@example.org",
        },
    }


def _capture_json(capsys: pytest.CaptureFixture[str]) -> dict:
    out = capsys.readouterr().out.strip()
    assert out, "expected JSON envelope on stdout"
    envelope = json.loads(out)
    assert envelope["ok"] is True
    return envelope["data"]


def test_save_and_load_project_datasheet(tmp_path: Path) -> None:
    project = tmp_path / "Datasheet Project"
    init_project(project, title="Datasheet Project")

    datasheet = DatasetDatasheet(
        title="Round-trip datasheet",
        motivation=DatasheetMotivation(purpose="Persist and reload"),
    )

    saved = save_project_datasheet(project, datasheet)
    assert saved == project_datasheet_path(project)
    assert saved.is_file()

    loaded = load_project_datasheet(project)
    assert loaded == datasheet


def test_load_project_datasheet_returns_none_when_unset(tmp_path: Path) -> None:
    project = tmp_path / "No Datasheet Project"
    init_project(project, title="No Datasheet Project")
    assert load_project_datasheet(project) is None


def test_project_service_save_and_load_datasheet(tmp_path: Path) -> None:
    project = ProjectService.create(tmp_path / "Service Datasheet Project")
    project.metadata.update(datasheet=DatasetDatasheet(title="Service round-trip"))
    record = project.metadata.datasheet
    assert record is not None
    assert record.title == "Service round-trip"


def test_cli_set_and_show_datasheet_round_trip(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from xpkg.cli import main

    project = tmp_path / "Datasheet CLI Project"
    init_project(project, title="Datasheet CLI Project")
    payload_path = tmp_path / "datasheet.json"
    payload_path.write_text(json.dumps(_datasheet_payload()), encoding="utf-8")

    code = main(
        [
            "project",
            "metadata",
            "set",
            "datasheet",
            str(project),
            "--from",
            str(payload_path),
            "--json",
        ]
    )
    assert code == 0
    saved = _capture_json(capsys)
    assert saved["status"] == "saved"
    assert saved["metadata"] == "datasheet"
    assert saved["datasheet"]["distribution"]["doi"] == "10.5281/zenodo.42"

    code = main(["project", "metadata", "show", "datasheet", str(project), "--json"])
    assert code == 0
    shown = _capture_json(capsys)
    assert shown["status"] == "loaded"
    assert shown["datasheet"]["title"] == "FAIR demo dataset"
    assert shown["datasheet"]["composition"]["instance_count"] == 12
    assert shown["datasheet"]["composition"]["splits"] == {"train": "8", "test": "4"}


def test_cli_show_datasheet_reports_missing_when_unset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from xpkg.cli import main

    project = tmp_path / "No Datasheet CLI Project"
    init_project(project, title="No Datasheet CLI Project")

    code = main(["project", "metadata", "show", "datasheet", str(project), "--json"])
    assert code == 0
    shown = _capture_json(capsys)
    assert shown["status"] == "missing"
    assert shown["datasheet"] is None


def test_datasheet_round_trips_through_pack_unpack(tmp_path: Path) -> None:
    project = tmp_path / "Pack Datasheet Project"
    init_project(project, title="Pack Datasheet Project")
    save_project_datasheet(project, DatasetDatasheet.from_dict(_datasheet_payload()))

    expkg = pack_project(project)
    with zipfile.ZipFile(expkg) as zf:
        manifest = json.loads(zf.read("EXPKG.json"))
    assert manifest["datasheet"]["title"] == "FAIR demo dataset"
    assert manifest["datasheet"]["distribution"]["license"] == "CC-BY-4.0"

    restored = unpack_project(expkg, tmp_path / "Restored")
    record = load_project_datasheet(restored)
    assert record is not None
    assert record.title == "FAIR demo dataset"
    assert record.distribution.doi == "10.5281/zenodo.42"
