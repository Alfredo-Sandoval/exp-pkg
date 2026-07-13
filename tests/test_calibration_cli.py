from __future__ import annotations

import json
from pathlib import Path

from tests.calibration_helpers import write_anipose_toml, write_opencv_stereo_yaml
from xpkg.project import init_project, load_project_calibration


def test_cli_imports_anipose_calibration_json_mode(tmp_path: Path, capsys) -> None:
    from xpkg.cli import main

    project = tmp_path / "Calibration CLI Project"
    init_project(project, title="Calibration CLI Project")
    source = write_anipose_toml(tmp_path / "calibration.toml")

    code = main(
        [
            "import",
            "calibration",
            "anipose",
            "--path",
            str(source),
            "--out",
            str(project),
            "--calibration-id",
            "rig",
            "--name",
            "rig-2024-03-15",
            "--units",
            "mm",
            "--json",
        ]
    )

    assert code == 0
    captured_streams = capsys.readouterr()
    assert captured_streams.err == ""
    envelope = json.loads(captured_streams.out)
    assert envelope["ok"] is True
    payload = envelope["data"]
    assert payload["status"] == "imported"
    assert payload["source"] == "anipose"
    assert payload["project"] == str(project)
    assert payload["state_path"].endswith("/.xpkg/state/current.json")
    assert (project / "Media" / "calibrations" / "rig" / "calibration.toml").is_file()
    assert load_project_calibration(project, "rig").name == "rig-2024-03-15"


def test_cli_imports_opencv_stereo_calibration_json_mode(
    tmp_path: Path,
    capsys,
) -> None:
    from xpkg.cli import main

    project = tmp_path / "OpenCV Calibration CLI Project"
    init_project(project, title="OpenCV Calibration CLI Project")
    source = write_opencv_stereo_yaml(tmp_path / "stereo.yml")

    code = main(
        [
            "import",
            "calibration",
            "opencv-stereo-yaml",
            "--path",
            str(source),
            "--out",
            str(project),
            "--calibration-id",
            "stereo-rig",
            "--name",
            "arena-stereo",
            "--units",
            "mm",
            "--json",
        ]
    )

    assert code == 0
    captured_streams = capsys.readouterr()
    assert captured_streams.err == ""
    envelope = json.loads(captured_streams.out)
    assert envelope["ok"] is True
    payload = envelope["data"]
    assert payload["status"] == "imported"
    assert payload["source"] == "opencv-stereo-yaml"
    assert payload["project"] == str(project)
    assert payload["state_path"].endswith("/.xpkg/state/current.json")
    assert (project / "Media" / "calibrations" / "stereo-rig" / "stereo.yml").is_file()
    loaded = load_project_calibration(project, "stereo-rig")
    assert loaded.name == "arena-stereo"
    assert loaded.source is not None
    assert loaded.source.tool == "opencv"
