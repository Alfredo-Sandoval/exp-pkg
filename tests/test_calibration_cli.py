from __future__ import annotations

import json
from pathlib import Path

from xpkg.project import init_project, load_project_calibration


def _write_anipose_toml(path: Path) -> Path:
    path.write_text(
        """
[cam_top]
name = "cam_top"
size = [1920, 1080]
matrix = [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]]
distortions = [0.1, -0.01, 0.001, 0.002, 0.0]
rotation = [0.0, 0.0, 0.0]
translation = [1.0, 2.0, 3.0]
fisheye = false
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def test_cli_imports_anipose_calibration_json_mode(tmp_path: Path, capsys) -> None:
    from xpkg.cli import main

    project = tmp_path / "Calibration CLI Project"
    init_project(project, title="Calibration CLI Project")
    source = _write_anipose_toml(tmp_path / "calibration.toml")

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
    assert payload["calibration_path"].endswith("/.xpkg/calibrations/rig/calibration.json")
    assert (project / ".xpkg" / "calibrations" / "rig" / "source" / "calibration.toml").is_file()
    assert load_project_calibration(project, "rig").name == "rig-2024-03-15"
