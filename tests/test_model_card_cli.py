"""Round-trip tests for ModelCard across project, service, CLI, and pack."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from xpkg.model import ModelCard, ModelCardDetails, ModelCardIntendedUse
from xpkg.project import (
    init_project,
    load_project_model_card,
    pack_project,
    project_model_card_path,
    save_project_model_card,
    unpack_project,
)
from xpkg.services.project import ProjectService


def _model_card_payload() -> dict:
    return {
        "details": {
            "name": "dlc-mouse-reach",
            "version": "2.3.4",
            "type": "Pose estimation",
            "architecture": "ResNet-50",
            "license": "CC-BY-4.0",
            "developers": ["A. Sandoval"],
        },
        "intended_use": {
            "primary_uses": "Mouse keypoint inference on open-field reach video.",
            "primary_users": ["Behavior labs"],
            "out_of_scope_uses": "Not validated for multi-animal scenes.",
        },
        "metrics": {
            "measures": ["RMSE px", "PCK@0.05"],
            "decision_thresholds": {"min_confidence": "0.4"},
        },
        "training_data": {
            "description": "200 manually labeled frames.",
            "source": "In-lab acquisition.",
        },
        "evaluation_data": {
            "description": "10 held-out sessions.",
        },
        "ethical_considerations": "IACUC-approved animal use.",
        "caveats": ["Single-animal only."],
    }


def _capture_json(capsys: pytest.CaptureFixture[str]) -> dict:
    out = capsys.readouterr().out.strip()
    assert out, "expected JSON envelope on stdout"
    return json.loads(out)


def test_save_and_load_project_model_card(tmp_path: Path) -> None:
    project = tmp_path / "Model Card Project"
    init_project(project, title="Model Card Project")

    card = ModelCard(
        details=ModelCardDetails(name="round-trip"),
        intended_use=ModelCardIntendedUse(primary_uses="Smoke test"),
    )

    saved = save_project_model_card(project, card)
    assert saved == project_model_card_path(project)
    assert saved.is_file()

    loaded = load_project_model_card(project)
    assert loaded == card


def test_load_project_model_card_returns_none_when_unset(tmp_path: Path) -> None:
    project = tmp_path / "No Card Project"
    init_project(project, title="No Card Project")
    assert load_project_model_card(project) is None


def test_project_service_save_and_load_model_card(tmp_path: Path) -> None:
    project = ProjectService.create(tmp_path / "Service Card Project")
    project.save_model_card(ModelCard(details=ModelCardDetails(name="service-card")))
    record = project.load_model_card()
    assert record is not None
    assert record.details.name == "service-card"


def test_cli_set_and_show_model_card_round_trip(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from xpkg.cli import main

    project = tmp_path / "Card CLI Project"
    init_project(project, title="Card CLI Project")
    payload_path = tmp_path / "model_card.json"
    payload_path.write_text(json.dumps(_model_card_payload()), encoding="utf-8")

    code = main(
        [
            "project",
            "set-model-card",
            str(project),
            "--from",
            str(payload_path),
            "--json",
        ]
    )
    assert code == 0
    saved = _capture_json(capsys)
    assert saved["status"] == "saved"
    assert saved["metadata"] == "model_card"
    assert saved["model_card"]["details"]["name"] == "dlc-mouse-reach"

    code = main(["project", "show-model-card", str(project), "--json"])
    assert code == 0
    shown = _capture_json(capsys)
    assert shown["status"] == "loaded"
    assert shown["model_card"]["intended_use"]["primary_users"] == ["Behavior labs"]
    assert shown["model_card"]["metrics"]["decision_thresholds"] == {"min_confidence": "0.4"}
    assert shown["model_card"]["caveats"] == ["Single-animal only."]


def test_cli_show_model_card_reports_missing_when_unset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from xpkg.cli import main

    project = tmp_path / "No Card CLI Project"
    init_project(project, title="No Card CLI Project")

    code = main(["project", "show-model-card", str(project), "--json"])
    assert code == 0
    shown = _capture_json(capsys)
    assert shown["status"] == "missing"
    assert shown["model_card"] is None


def test_model_card_round_trips_through_pack_unpack(tmp_path: Path) -> None:
    project = tmp_path / "Pack Card Project"
    init_project(project, title="Pack Card Project")
    save_project_model_card(project, ModelCard.from_dict(_model_card_payload()))

    expkg = pack_project(project)
    with zipfile.ZipFile(expkg) as zf:
        manifest = json.loads(zf.read("EXPKG.json"))
    assert manifest["model_card"]["details"]["name"] == "dlc-mouse-reach"
    assert manifest["model_card"]["details"]["version"] == "2.3.4"

    restored = unpack_project(expkg, tmp_path / "Restored")
    record = load_project_model_card(restored)
    assert record is not None
    assert record.details.name == "dlc-mouse-reach"
    assert record.intended_use.primary_uses is not None
