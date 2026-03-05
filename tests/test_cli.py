from __future__ import annotations

from pathlib import Path


def test_cli_routes_dlc_csv(monkeypatch, capsys) -> None:
    from posetta.cli import main
    from posetta.io.converters.converter_helpers import ConversionResult

    captured: dict[str, object] = {}

    def fake_convert_dlc_csv(
        csv_path: str,
        video_path: str,
        out_path: str,
        *,
        skeleton_name: str,
        likelihood_threshold: float,
        progress_callback,
    ) -> ConversionResult:
        captured["csv_path"] = csv_path
        captured["video_path"] = video_path
        captured["out_path"] = out_path
        captured["skeleton_name"] = skeleton_name
        captured["likelihood_threshold"] = likelihood_threshold
        progress_callback("csv-progress")
        return ConversionResult(
            source_dir=Path("input"),
            project_root=Path("out"),
            videos=[Path(video_path)],
            siesta_path=Path(out_path),
        )

    monkeypatch.setattr("posetta.cli.convert_dlc_csv", fake_convert_dlc_csv)

    code = main(
        [
            "convert",
            "dlc",
            "csv",
            "--csv",
            "tracking.csv",
            "--video",
            "clip.mp4",
            "--out",
            "tracking.siesta",
            "--skeleton-name",
            "mouse",
            "--threshold",
            "0.25",
        ]
    )

    assert code == 0
    assert captured == {
        "csv_path": "tracking.csv",
        "video_path": "clip.mp4",
        "out_path": "tracking.siesta",
        "skeleton_name": "mouse",
        "likelihood_threshold": 0.25,
    }

    stdout = capsys.readouterr().out
    assert "csv-progress" in stdout
    assert "tracking.siesta" in stdout


def test_cli_routes_dlc_project(monkeypatch, capsys) -> None:
    from posetta.cli import main
    from posetta.io.converters.converter_helpers import ConversionResult

    captured: dict[str, object] = {}

    def fake_convert_dlc_project(
        project_dir: str,
        out_dir: str,
        *,
        likelihood_threshold: float,
        progress_callback,
    ) -> list[ConversionResult]:
        captured["project_dir"] = project_dir
        captured["out_dir"] = out_dir
        captured["likelihood_threshold"] = likelihood_threshold
        progress_callback("project-progress")
        return [
            ConversionResult(
                source_dir=Path(project_dir),
                project_root=Path(out_dir),
                videos=[Path("video1.mp4")],
                siesta_path=Path(out_dir) / "video1.siesta",
            ),
            ConversionResult(
                source_dir=Path(project_dir),
                project_root=Path(out_dir),
                videos=[Path("video2.mp4")],
                siesta_path=Path(out_dir) / "video2.siesta",
            ),
        ]

    monkeypatch.setattr("posetta.cli.convert_dlc_project", fake_convert_dlc_project)

    code = main(
        [
            "convert",
            "dlc",
            "project",
            "--project",
            "dlc-project",
            "--out",
            "exports",
            "--threshold",
            "0.5",
        ]
    )

    assert code == 0
    assert captured == {
        "project_dir": "dlc-project",
        "out_dir": "exports",
        "likelihood_threshold": 0.5,
    }

    stdout = capsys.readouterr().out
    assert "Converted 2 project item(s)" in stdout
    assert "exports/video1.siesta" in stdout
    assert "exports/video2.siesta" in stdout


def test_cli_routes_sleap(monkeypatch, capsys) -> None:
    from posetta.cli import main
    from posetta.io.converters.converter_helpers import ConversionResult

    captured: dict[str, object] = {}

    def fake_convert_sleap_package(
        slp: str,
        out_dir: str,
        *,
        fps: int,
        encode_videos: bool | None,
        progress_callback,
    ) -> ConversionResult:
        captured["slp"] = slp
        captured["out_dir"] = out_dir
        captured["fps"] = fps
        captured["encode_videos"] = encode_videos
        progress_callback("sleap-progress")
        return ConversionResult(
            source_dir=Path(slp),
            project_root=Path(out_dir),
            videos=[],
            siesta_path=Path(out_dir) / "project.siesta",
        )

    monkeypatch.setattr("posetta.cli.convert_sleap_package", fake_convert_sleap_package)

    code = main(
        [
            "convert",
            "sleap",
            "--slp",
            "labels.pkg.slp",
            "--out",
            "sleap-export",
            "--fps",
            "24",
            "--no-videos",
        ]
    )

    assert code == 0
    assert captured == {
        "slp": "labels.pkg.slp",
        "out_dir": "sleap-export",
        "fps": 24,
        "encode_videos": False,
    }

    stdout = capsys.readouterr().out
    assert "sleap-progress" in stdout
    assert "sleap-export/project.siesta" in stdout
