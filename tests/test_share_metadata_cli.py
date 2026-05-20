"""CLI contract tests for ``xpkg project metadata set/show`` slots."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xpkg.project import init_project


def _write_share_payload(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "title": "FAIR demo dataset",
                "creators": ["Sandoval, A.", "Doe, J."],
                "doi": "10.5281/zenodo.42",
                "license": "CC-BY-4.0",
                "version": "1.0",
                "repository_url": "https://github.com/example/demo",
                "keywords": ["mouse", "reach"],
                "related_publications": ["https://doi.org/10.1234/demo.2026"],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_acquisition_payload(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "acquisition_id": "session-001",
                "recorded_at": "2026-05-07T10:00:00Z",
                "site": "Lab Rig 2",
                "cameras": [
                    {
                        "camera_id": "cam_top",
                        "model": "Basler acA1920",
                        "frame_rate_hz": 120.0,
                        "resolution_px": [1920, 1080],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _capture_json(capsys: pytest.CaptureFixture[str]) -> dict:
    out = capsys.readouterr().out.strip()
    assert out, "expected JSON envelope on stdout"
    envelope = json.loads(out)
    assert envelope["ok"] is True
    return envelope["data"]


def test_cli_set_and_show_dataset_share_round_trip(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from xpkg.cli import main

    project = tmp_path / "Share CLI Project"
    init_project(project, title="Share CLI Project")
    payload = _write_share_payload(tmp_path / "share.json")

    code = main(
        [
            "project",
            "metadata",
            "set",
            "dataset-share",
            str(project),
            "--from",
            str(payload),
            "--json",
        ]
    )
    assert code == 0
    saved = _capture_json(capsys)
    assert saved["status"] == "saved"
    assert saved["metadata"] == "dataset_share"
    assert saved["dataset_share"]["doi"] == "10.5281/zenodo.42"

    code = main(["project", "metadata", "show", "dataset-share", str(project), "--json"])
    assert code == 0
    shown = _capture_json(capsys)
    assert shown["status"] == "loaded"
    assert shown["dataset_share"]["title"] == "FAIR demo dataset"
    assert shown["dataset_share"]["creators"] == ["Sandoval, A.", "Doe, J."]
    assert shown["dataset_share"]["keywords"] == ["mouse", "reach"]


def test_cli_show_dataset_share_reports_missing_when_unset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from xpkg.cli import main

    project = tmp_path / "No Share Project"
    init_project(project, title="No Share Project")

    code = main(["project", "metadata", "show", "dataset-share", str(project), "--json"])
    assert code == 0
    shown = _capture_json(capsys)
    assert shown["status"] == "missing"
    assert shown["dataset_share"] is None


def test_cli_set_and_show_acquisition_round_trip(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from xpkg.cli import main

    project = tmp_path / "Acquisition CLI Project"
    init_project(project, title="Acquisition CLI Project")
    payload = _write_acquisition_payload(tmp_path / "acquisition.json")

    code = main(
        [
            "project",
            "metadata",
            "set",
            "acquisition",
            str(project),
            "--from",
            str(payload),
            "--json",
        ]
    )
    assert code == 0
    saved = _capture_json(capsys)
    assert saved["status"] == "saved"
    assert saved["acquisition"]["site"] == "Lab Rig 2"
    assert saved["acquisition"]["cameras"][0]["camera_id"] == "cam_top"

    code = main(["project", "metadata", "show", "acquisition", str(project), "--json"])
    assert code == 0
    shown = _capture_json(capsys)
    assert shown["status"] == "loaded"
    assert shown["acquisition"]["acquisition_id"] == "session-001"
    assert shown["acquisition"]["cameras"][0]["frame_rate_hz"] == 120.0
