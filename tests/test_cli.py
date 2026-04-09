from __future__ import annotations

from pathlib import Path


def test_cli_routes_dlc_csv(monkeypatch, capsys) -> None:
    from xpkg.adapters import ConversionResult
    from xpkg.cli import main

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
            bundle_path=Path(out_path),
        )

    monkeypatch.setattr("xpkg.cli.convert_dlc_csv", fake_convert_dlc_csv)

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
            "tracking.sta",
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
        "out_path": "tracking.sta",
        "skeleton_name": "mouse",
        "likelihood_threshold": 0.25,
    }

    stdout = capsys.readouterr().out
    assert "csv-progress" in stdout
    assert "tracking.sta" in stdout


def test_cli_routes_dlc_project(monkeypatch, capsys) -> None:
    from xpkg.adapters import ConversionResult
    from xpkg.cli import main

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
                bundle_path=Path(out_dir) / "video1.sta",
            ),
            ConversionResult(
                source_dir=Path(project_dir),
                project_root=Path(out_dir),
                videos=[Path("video2.mp4")],
                bundle_path=Path(out_dir) / "video2.sta",
            ),
        ]

    monkeypatch.setattr("xpkg.cli.convert_dlc_project", fake_convert_dlc_project)

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
    assert "exports/video1.sta" in stdout
    assert "exports/video2.sta" in stdout


def test_cli_routes_sleap(monkeypatch, capsys) -> None:
    from xpkg.adapters import ConversionResult
    from xpkg.cli import main

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
            bundle_path=Path(out_dir) / "project.sta",
        )

    monkeypatch.setattr("xpkg.cli.convert_sleap_package", fake_convert_sleap_package)

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
    assert "sleap-export/project.sta" in stdout


def test_cli_routes_init_workspace(monkeypatch, capsys) -> None:
    from xpkg.cli import main

    captured: dict[str, object] = {}

    def fake_init_project(
        workspace: str,
        *,
        title: str | None,
        project_id: str | None,
        default_pack_mode: str,
        force: bool,
    ) -> object:
        captured["workspace"] = workspace
        captured["title"] = title
        captured["project_id"] = project_id
        captured["default_pack_mode"] = default_pack_mode
        captured["force"] = force
        return object()

    monkeypatch.setattr("xpkg.cli.init_project", fake_init_project)

    code = main(
        [
            "init",
            "My Project",
            "--title",
            "My Project",
            "--id",
            "project-123",
            "--pack-mode",
            "snapshot",
            "--force",
        ]
    )

    assert code == 0
    assert captured == {
        "workspace": "My Project",
        "title": "My Project",
        "project_id": "project-123",
        "default_pack_mode": "snapshot",
        "force": True,
    }
    assert "Initialized workspace My Project" in capsys.readouterr().out


def test_cli_routes_import_legacy(monkeypatch, capsys) -> None:
    from xpkg.cli import main

    captured: dict[str, object] = {}

    def fake_import_legacy_archive(
        legacy_archive: str,
        workspace: str,
        *,
        title: str | None,
        default_pack_mode: str = "portable",
        force: bool,
    ) -> Path:
        captured["legacy_archive"] = legacy_archive
        captured["workspace"] = workspace
        captured["title"] = title
        captured["default_pack_mode"] = default_pack_mode
        captured["force"] = force
        return Path(workspace) / ".posetta" / "state" / "current.sta"

    monkeypatch.setattr("xpkg.cli.import_legacy_archive", fake_import_legacy_archive)

    code = main(
        [
            "import",
            "legacy",
            "--file",
            "tracking.siesta",
            "--out",
            "My Project",
            "--title",
            "Imported",
            "--force",
        ]
    )

    assert code == 0
    assert captured == {
        "legacy_archive": "tracking.siesta",
        "workspace": "My Project",
        "title": "Imported",
        "default_pack_mode": "portable",
        "force": True,
    }
    stdout = capsys.readouterr().out
    assert "Imported legacy archive tracking.siesta -> My Project" in stdout
    assert ".posetta/state/current.sta" in stdout


def test_cli_routes_import_dlc_csv_workspace(monkeypatch, capsys) -> None:
    from xpkg.cli import main

    captured: dict[str, object] = {}

    def fake_import_dlc_csv_workspace(
        csv_path: str,
        video_path: str,
        workspace: str,
        *,
        skeleton_name: str,
        likelihood_threshold: float,
        default_pack_mode: str = "portable",
        force: bool = False,
        progress_callback,
    ) -> Path:
        captured["csv_path"] = csv_path
        captured["video_path"] = video_path
        captured["workspace"] = workspace
        captured["skeleton_name"] = skeleton_name
        captured["likelihood_threshold"] = likelihood_threshold
        captured["default_pack_mode"] = default_pack_mode
        captured["force"] = force
        progress_callback("import-progress")
        return Path(workspace) / ".posetta" / "state" / "current.sta"

    monkeypatch.setattr("xpkg.cli.import_dlc_csv_workspace", fake_import_dlc_csv_workspace)

    code = main(
        [
            "import",
            "dlc",
            "csv",
            "--csv",
            "tracking.csv",
            "--video",
            "clip.mp4",
            "--out",
            "My Project",
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
        "workspace": "My Project",
        "skeleton_name": "mouse",
        "likelihood_threshold": 0.25,
        "default_pack_mode": "portable",
        "force": False,
    }
    stdout = capsys.readouterr().out
    assert "import-progress" in stdout
    assert "Imported DLC CSV into My Project" in stdout
    assert ".posetta/state/current.sta" in stdout


def test_cli_routes_pack_unpack_and_validate(monkeypatch, capsys) -> None:
    from xpkg.cli import main

    captured: dict[str, object] = {}

    def fake_pack_project(
        workspace: str,
        *,
        out: str | None,
        mode: str | None,
        overwrite: bool,
    ) -> Path:
        captured["pack_workspace"] = workspace
        captured["pack_out"] = out
        captured["pack_mode"] = mode
        captured["pack_overwrite"] = overwrite
        return Path("My Project.expkg")

    def fake_unpack_project(
        artifact: str,
        out: str,
        *,
        force: bool,
        rename_title: str | None,
    ) -> Path:
        captured["unpack_artifact"] = artifact
        captured["unpack_out"] = out
        captured["unpack_force"] = force
        captured["unpack_rename_title"] = rename_title
        return Path(out)

    def fake_validate_artifact(path: str) -> None:
        captured.setdefault("validated", []).append(path)

    monkeypatch.setattr("xpkg.cli.pack_project", fake_pack_project)
    monkeypatch.setattr("xpkg.cli.unpack_project", fake_unpack_project)
    monkeypatch.setattr("xpkg.cli.validate_artifact", fake_validate_artifact)

    pack_code = main(["pack", "My Project", "--mode", "snapshot", "--overwrite"])
    unpack_code = main(
        [
            "unpack",
            "My Project.expkg",
            "--out",
            "Unpacked Project",
            "--force",
            "--rename",
            "Renamed Project",
        ]
    )
    validate_code = main(["validate", "My Project.expkg"])

    assert pack_code == 0
    assert unpack_code == 0
    assert validate_code == 0
    assert captured == {
        "pack_workspace": "My Project",
        "pack_out": None,
        "pack_mode": "snapshot",
        "pack_overwrite": True,
        "unpack_artifact": "My Project.expkg",
        "unpack_out": "Unpacked Project",
        "unpack_force": True,
        "unpack_rename_title": "Renamed Project",
        "validated": ["My Project.expkg"],
    }
    stdout = capsys.readouterr().out
    assert "Packed My Project" in stdout
    assert "Unpacked My Project.expkg" in stdout
    assert "Valid My Project.expkg" in stdout
