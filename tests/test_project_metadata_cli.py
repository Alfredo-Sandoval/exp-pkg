from __future__ import annotations

import json
from pathlib import Path

from xpkg.project import init_project


def test_cli_sets_and_shows_acquisition_metadata_json_mode(
    tmp_path: Path,
    capsys,
) -> None:
    from xpkg.cli import main

    project = tmp_path / "Acquisition CLI Project"
    source = tmp_path / "acquisition.json"
    init_project(project, title="Acquisition CLI Project")
    source.write_text(
        json.dumps(
            {
                "acquisition_id": "acq-cli",
                "system": "open field rig",
                "arena_size": "40 x 40 cm",
                "arena_material": "matte acrylic",
                "arena_color": "gray",
                "lighting": "IR",
                "ir_lighting": True,
                "cameras": [
                    {
                        "camera_id": "cam-top",
                        "frame_rate_hz": 120.0,
                        "resolution_px": [1920, 1080],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    set_code = main(
        [
            "project",
            "metadata",
            "set",
            "acquisition",
            str(project),
            "--from",
            str(source),
            "--json",
        ]
    )
    set_envelope = json.loads(capsys.readouterr().out)
    assert set_envelope["ok"] is True
    set_payload = set_envelope["data"]
    show_code = main(["project", "metadata", "show", "acquisition", str(project), "--json"])
    show_envelope = json.loads(capsys.readouterr().out)
    assert show_envelope["ok"] is True
    show_payload = show_envelope["data"]

    assert set_code == 0
    assert show_code == 0
    assert set_payload["status"] == "saved"
    assert set_payload["path"].endswith("/.xpkg/state/current.json")
    assert show_payload["acquisition"]["acquisition_id"] == "acq-cli"
    assert show_payload["acquisition"]["ir_lighting"] is True


def test_cli_sets_and_shows_dataset_share_metadata_json_mode(
    tmp_path: Path,
    capsys,
) -> None:
    from xpkg.cli import main

    project = tmp_path / "Dataset Share CLI Project"
    source = tmp_path / "dataset_share.json"
    init_project(project, title="Dataset Share CLI Project")
    source.write_text(
        json.dumps(
            {
                "title": "Dataset share CLI fixture",
                "creators": ["Sandoval Lab"],
                "license": "BSD-3-Clause",
                "doi": "10.0000/cli",
                "funders": ["NIH"],
                "related_publications": ["Example et al. 2024"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    set_code = main(
        [
            "project",
            "metadata",
            "set",
            "dataset-share",
            str(project),
            "--from",
            str(source),
            "--json",
        ]
    )
    set_envelope = json.loads(capsys.readouterr().out)
    assert set_envelope["ok"] is True
    set_payload = set_envelope["data"]
    show_code = main(["project", "metadata", "show", "dataset-share", str(project), "--json"])
    show_envelope = json.loads(capsys.readouterr().out)
    assert show_envelope["ok"] is True
    show_payload = show_envelope["data"]

    assert set_code == 0
    assert show_code == 0
    assert set_payload["status"] == "saved"
    assert set_payload["path"].endswith("/.xpkg/state/current.json")
    assert show_payload["dataset_share"]["doi"] == "10.0000/cli"
    assert show_payload["dataset_share"]["funders"] == ["NIH"]
