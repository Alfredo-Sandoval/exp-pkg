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
            archive_path=Path(out_path),
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
            "tracking.xpkg",
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
        "out_path": "tracking.xpkg",
        "skeleton_name": "mouse",
        "likelihood_threshold": 0.25,
    }

    stdout = capsys.readouterr().out
    assert "csv-progress" in stdout
    assert "tracking.xpkg" in stdout


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
                archive_path=Path(out_dir) / "video1.xpkg",
            ),
            ConversionResult(
                source_dir=Path(project_dir),
                project_root=Path(out_dir),
                videos=[Path("video2.mp4")],
                archive_path=Path(out_dir) / "video2.xpkg",
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
    assert "exports/video1.xpkg" in stdout
    assert "exports/video2.xpkg" in stdout


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
            archive_path=Path(out_dir) / "project.xpkg",
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
    assert "sleap-export/project.xpkg" in stdout


def test_cli_routes_sleap_h5(monkeypatch, capsys) -> None:
    from xpkg.adapters import ConversionResult
    from xpkg.cli import main

    captured: dict[str, object] = {}

    def fake_convert_sleap_h5(
        h5_path: str,
        video_path: str,
        out_path: str,
        *,
        skeleton_name: str,
        likelihood_threshold: float,
        progress_callback,
    ) -> ConversionResult:
        captured["h5_path"] = h5_path
        captured["video_path"] = video_path
        captured["out_path"] = out_path
        captured["skeleton_name"] = skeleton_name
        captured["likelihood_threshold"] = likelihood_threshold
        progress_callback("sleap-h5-progress")
        return ConversionResult(
            source_dir=Path(h5_path),
            project_root=Path(out_path).parent,
            videos=[Path(video_path)],
            archive_path=Path(out_path),
        )

    monkeypatch.setattr("xpkg.cli.convert_sleap_h5", fake_convert_sleap_h5)

    code = main(
        [
            "convert",
            "sleap",
            "--h5",
            "analysis.h5",
            "--video",
            "clip.mp4",
            "--out",
            "analysis.xpkg",
            "--skeleton-name",
            "mouse",
            "--threshold",
            "0.25",
        ]
    )

    assert code == 0
    assert captured == {
        "h5_path": "analysis.h5",
        "video_path": "clip.mp4",
        "out_path": "analysis.xpkg",
        "skeleton_name": "mouse",
        "likelihood_threshold": 0.25,
    }

    stdout = capsys.readouterr().out
    assert "sleap-h5-progress" in stdout
    assert "analysis.xpkg" in stdout


def test_cli_routes_mmpose(monkeypatch, capsys) -> None:
    from xpkg.adapters import ConversionResult
    from xpkg.cli import main

    captured: dict[str, object] = {}

    def fake_convert_mmpose_topdown_json(
        json_path: str,
        video_path: str,
        out_path: str,
        *,
        skeleton_name: str,
        instance_index: int,
        likelihood_threshold: float,
        progress_callback,
    ) -> ConversionResult:
        captured["json_path"] = json_path
        captured["video_path"] = video_path
        captured["out_path"] = out_path
        captured["skeleton_name"] = skeleton_name
        captured["instance_index"] = instance_index
        captured["likelihood_threshold"] = likelihood_threshold
        progress_callback("mmpose-progress")
        return ConversionResult(
            source_dir=Path(json_path),
            project_root=Path(out_path).parent,
            videos=[Path(video_path)],
            archive_path=Path(out_path),
        )

    monkeypatch.setattr("xpkg.cli.convert_mmpose_topdown_json", fake_convert_mmpose_topdown_json)

    code = main(
        [
            "convert",
            "mmpose",
            "--json",
            "results.json",
            "--video",
            "clip.mp4",
            "--out",
            "mmpose.xpkg",
            "--skeleton-name",
            "mouse",
            "--instance-index",
            "1",
            "--threshold",
            "0.4",
        ]
    )

    assert code == 0
    assert captured == {
        "json_path": "results.json",
        "video_path": "clip.mp4",
        "out_path": "mmpose.xpkg",
        "skeleton_name": "mouse",
        "instance_index": 1,
        "likelihood_threshold": 0.4,
    }

    stdout = capsys.readouterr().out
    assert "mmpose-progress" in stdout
    assert "mmpose.xpkg" in stdout


def test_cli_routes_mediapipe(monkeypatch, capsys) -> None:
    from xpkg.adapters import ConversionResult
    from xpkg.cli import main

    captured: dict[str, object] = {}

    def fake_convert_mediapipe_pose_landmarks_json(
        json_path: str,
        video_path: str,
        out_path: str,
        *,
        skeleton_name: str,
        likelihood_threshold: float,
        progress_callback,
    ) -> ConversionResult:
        captured["json_path"] = json_path
        captured["video_path"] = video_path
        captured["out_path"] = out_path
        captured["skeleton_name"] = skeleton_name
        captured["likelihood_threshold"] = likelihood_threshold
        progress_callback("mediapipe-progress")
        return ConversionResult(
            source_dir=Path(json_path),
            project_root=Path(out_path).parent,
            videos=[Path(video_path)],
            archive_path=Path(out_path),
        )

    monkeypatch.setattr(
        "xpkg.cli.convert_mediapipe_pose_landmarks_json",
        fake_convert_mediapipe_pose_landmarks_json,
    )

    code = main(
        [
            "convert",
            "mediapipe",
            "--json",
            "pose_landmarks.json",
            "--video",
            "clip.mp4",
            "--out",
            "mediapipe.xpkg",
            "--threshold",
            "0.3",
        ]
    )

    assert code == 0
    assert captured == {
        "json_path": "pose_landmarks.json",
        "video_path": "clip.mp4",
        "out_path": "mediapipe.xpkg",
        "skeleton_name": "mediapipe_pose",
        "likelihood_threshold": 0.3,
    }

    stdout = capsys.readouterr().out
    assert "mediapipe-progress" in stdout
    assert "mediapipe.xpkg" in stdout


def test_cli_routes_openpose(monkeypatch, capsys) -> None:
    from xpkg.adapters import ConversionResult
    from xpkg.cli import main

    captured: dict[str, object] = {}

    def fake_convert_openpose_json(
        json_dir: str,
        video_path: str,
        out_path: str,
        *,
        skeleton_name: str,
        likelihood_threshold: float,
        progress_callback,
    ) -> ConversionResult:
        captured["json_dir"] = json_dir
        captured["video_path"] = video_path
        captured["out_path"] = out_path
        captured["skeleton_name"] = skeleton_name
        captured["likelihood_threshold"] = likelihood_threshold
        progress_callback("openpose-progress")
        return ConversionResult(
            source_dir=Path(json_dir),
            project_root=Path(out_path).parent,
            videos=[Path(video_path)],
            archive_path=Path(out_path),
        )

    monkeypatch.setattr("xpkg.cli.convert_openpose_json", fake_convert_openpose_json)

    code = main(
        [
            "convert",
            "openpose",
            "--json",
            "openpose_json",
            "--video",
            "clip.mp4",
            "--out",
            "openpose.xpkg",
            "--threshold",
            "0.2",
        ]
    )

    assert code == 0
    assert captured == {
        "json_dir": "openpose_json",
        "video_path": "clip.mp4",
        "out_path": "openpose.xpkg",
        "skeleton_name": "imported",
        "likelihood_threshold": 0.2,
    }

    stdout = capsys.readouterr().out
    assert "openpose-progress" in stdout
    assert "openpose.xpkg" in stdout


def test_cli_routes_detectron2(monkeypatch, capsys) -> None:
    from xpkg.adapters import ConversionResult
    from xpkg.cli import main

    captured: dict[str, object] = {}

    def fake_convert_detectron2_coco(
        predictions_path: str,
        dataset_json_path: str,
        image_root: str,
        out_path: str,
        *,
        category_id: int | None,
        skeleton_name: str | None,
        likelihood_threshold: float,
        progress_callback,
    ) -> ConversionResult:
        captured["predictions_path"] = predictions_path
        captured["dataset_json_path"] = dataset_json_path
        captured["image_root"] = image_root
        captured["out_path"] = out_path
        captured["category_id"] = category_id
        captured["skeleton_name"] = skeleton_name
        captured["likelihood_threshold"] = likelihood_threshold
        progress_callback("detectron2-progress")
        return ConversionResult(
            source_dir=Path(predictions_path),
            project_root=Path(out_path).parent,
            videos=[Path(image_root)],
            archive_path=Path(out_path),
        )

    monkeypatch.setattr("xpkg.cli.convert_detectron2_coco", fake_convert_detectron2_coco)

    code = main(
        [
            "convert",
            "detectron2",
            "--predictions",
            "coco_instances_results.json",
            "--dataset-json",
            "dataset.json",
            "--image-root",
            "images",
            "--out",
            "detectron2.xpkg",
            "--category-id",
            "7",
            "--skeleton-name",
            "mouse",
            "--threshold",
            "0.6",
        ]
    )

    assert code == 0
    assert captured == {
        "predictions_path": "coco_instances_results.json",
        "dataset_json_path": "dataset.json",
        "image_root": "images",
        "out_path": "detectron2.xpkg",
        "category_id": 7,
        "skeleton_name": "mouse",
        "likelihood_threshold": 0.6,
    }

    stdout = capsys.readouterr().out
    assert "detectron2-progress" in stdout
    assert "detectron2.xpkg" in stdout


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
        return Path(workspace) / ".xpkg" / "state" / "current.json"

    monkeypatch.setattr("xpkg.cli.import_legacy_archive", fake_import_legacy_archive)

    code = main(
        [
            "import",
            "legacy",
            "--file",
            "tracking.xpkg",
            "--out",
            "My Project",
            "--title",
            "Imported",
            "--force",
        ]
    )

    assert code == 0
    assert captured == {
        "legacy_archive": "tracking.xpkg",
        "workspace": "My Project",
        "title": "Imported",
        "default_pack_mode": "portable",
        "force": True,
    }
    stdout = capsys.readouterr().out
    assert "Imported archive tracking.xpkg -> My Project" in stdout
    assert ".xpkg/state/current.json" in stdout


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
        return Path(workspace) / ".xpkg" / "state" / "current.json"

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
    assert ".xpkg/state/current.json" in stdout


def test_cli_routes_import_dlc_project_workspace(monkeypatch, capsys) -> None:
    from xpkg.cli import main

    captured: dict[str, object] = {}

    def fake_import_dlc_project_workspace(
        project_dir: str,
        workspace: str,
        *,
        skeleton_name: str | None = None,
        likelihood_threshold: float,
        default_pack_mode: str = "portable",
        force: bool = False,
        progress_callback,
    ) -> Path:
        captured["project_dir"] = project_dir
        captured["workspace"] = workspace
        captured["skeleton_name"] = skeleton_name
        captured["likelihood_threshold"] = likelihood_threshold
        captured["default_pack_mode"] = default_pack_mode
        captured["force"] = force
        progress_callback("project-import-progress")
        return Path(workspace) / ".xpkg" / "state" / "current.json"

    monkeypatch.setattr(
        "xpkg.cli.import_dlc_project_workspace",
        fake_import_dlc_project_workspace,
    )

    code = main(
        [
            "import",
            "dlc",
            "project",
            "--project",
            "dlc-project",
            "--out",
            "My Project",
            "--threshold",
            "0.5",
        ]
    )

    assert code == 0
    assert captured == {
        "project_dir": "dlc-project",
        "workspace": "My Project",
        "skeleton_name": None,
        "likelihood_threshold": 0.5,
        "default_pack_mode": "portable",
        "force": False,
    }
    stdout = capsys.readouterr().out
    assert "project-import-progress" in stdout
    assert "Imported DLC project into My Project" in stdout
    assert ".xpkg/state/current.json" in stdout


def test_cli_routes_import_sleap_h5_workspace(monkeypatch, capsys) -> None:
    from xpkg.cli import main

    captured: dict[str, object] = {}

    def fake_import_sleap_h5_workspace(
        h5_path: str,
        video_path: str,
        workspace: str,
        *,
        skeleton_name: str,
        likelihood_threshold: float,
        default_pack_mode: str = "portable",
        force: bool = False,
        progress_callback,
    ) -> Path:
        captured["h5_path"] = h5_path
        captured["video_path"] = video_path
        captured["workspace"] = workspace
        captured["skeleton_name"] = skeleton_name
        captured["likelihood_threshold"] = likelihood_threshold
        captured["default_pack_mode"] = default_pack_mode
        captured["force"] = force
        progress_callback("sleap-h5-import-progress")
        return Path(workspace) / ".xpkg" / "state" / "current.json"

    monkeypatch.setattr("xpkg.cli.import_sleap_h5_workspace", fake_import_sleap_h5_workspace)

    code = main(
        [
            "import",
            "sleap",
            "--h5",
            "analysis.h5",
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
        "h5_path": "analysis.h5",
        "video_path": "clip.mp4",
        "workspace": "My Project",
        "skeleton_name": "mouse",
        "likelihood_threshold": 0.25,
        "default_pack_mode": "portable",
        "force": False,
    }
    stdout = capsys.readouterr().out
    assert "sleap-h5-import-progress" in stdout
    assert "Imported SLEAP H5 into My Project" in stdout
    assert ".xpkg/state/current.json" in stdout


def test_cli_routes_import_mmpose_workspace(monkeypatch, capsys) -> None:
    from xpkg.cli import main

    captured: dict[str, object] = {}

    def fake_import_mmpose_topdown_json_workspace(
        json_path: str,
        video_path: str,
        workspace: str,
        *,
        skeleton_name: str,
        instance_index: int,
        likelihood_threshold: float,
        default_pack_mode: str = "portable",
        force: bool = False,
        progress_callback,
    ) -> Path:
        captured["json_path"] = json_path
        captured["video_path"] = video_path
        captured["workspace"] = workspace
        captured["skeleton_name"] = skeleton_name
        captured["instance_index"] = instance_index
        captured["likelihood_threshold"] = likelihood_threshold
        captured["default_pack_mode"] = default_pack_mode
        captured["force"] = force
        progress_callback("mmpose-import-progress")
        return Path(workspace) / ".xpkg" / "state" / "current.json"

    monkeypatch.setattr(
        "xpkg.cli.import_mmpose_topdown_json_workspace",
        fake_import_mmpose_topdown_json_workspace,
    )

    code = main(
        [
            "import",
            "mmpose",
            "--json",
            "results.json",
            "--video",
            "clip.mp4",
            "--out",
            "My Project",
            "--instance-index",
            "1",
            "--threshold",
            "0.4",
        ]
    )

    assert code == 0
    assert captured == {
        "json_path": "results.json",
        "video_path": "clip.mp4",
        "workspace": "My Project",
        "skeleton_name": "imported",
        "instance_index": 1,
        "likelihood_threshold": 0.4,
        "default_pack_mode": "portable",
        "force": False,
    }
    stdout = capsys.readouterr().out
    assert "mmpose-import-progress" in stdout
    assert "Imported MMPose JSON into My Project" in stdout
    assert ".xpkg/state/current.json" in stdout


def test_cli_routes_import_mediapipe_workspace(monkeypatch, capsys) -> None:
    from xpkg.cli import main

    captured: dict[str, object] = {}

    def fake_import_mediapipe_pose_landmarks_json_workspace(
        json_path: str,
        video_path: str,
        workspace: str,
        *,
        skeleton_name: str,
        likelihood_threshold: float,
        default_pack_mode: str = "portable",
        force: bool = False,
        progress_callback,
    ) -> Path:
        captured["json_path"] = json_path
        captured["video_path"] = video_path
        captured["workspace"] = workspace
        captured["skeleton_name"] = skeleton_name
        captured["likelihood_threshold"] = likelihood_threshold
        captured["default_pack_mode"] = default_pack_mode
        captured["force"] = force
        progress_callback("mediapipe-import-progress")
        return Path(workspace) / ".xpkg" / "state" / "current.json"

    monkeypatch.setattr(
        "xpkg.cli.import_mediapipe_pose_landmarks_json_workspace",
        fake_import_mediapipe_pose_landmarks_json_workspace,
    )

    code = main(
        [
            "import",
            "mediapipe",
            "--json",
            "pose_landmarks.json",
            "--video",
            "clip.mp4",
            "--out",
            "My Project",
            "--threshold",
            "0.3",
        ]
    )

    assert code == 0
    assert captured == {
        "json_path": "pose_landmarks.json",
        "video_path": "clip.mp4",
        "workspace": "My Project",
        "skeleton_name": "mediapipe_pose",
        "likelihood_threshold": 0.3,
        "default_pack_mode": "portable",
        "force": False,
    }
    stdout = capsys.readouterr().out
    assert "mediapipe-import-progress" in stdout
    assert "Imported MediaPipe JSON into My Project" in stdout
    assert ".xpkg/state/current.json" in stdout


def test_cli_routes_import_openpose_workspace(monkeypatch, capsys) -> None:
    from xpkg.cli import main

    captured: dict[str, object] = {}

    def fake_import_openpose_json_workspace(
        json_dir: str,
        video_path: str,
        workspace: str,
        *,
        skeleton_name: str,
        likelihood_threshold: float,
        default_pack_mode: str = "portable",
        force: bool = False,
        progress_callback,
    ) -> Path:
        captured["json_dir"] = json_dir
        captured["video_path"] = video_path
        captured["workspace"] = workspace
        captured["skeleton_name"] = skeleton_name
        captured["likelihood_threshold"] = likelihood_threshold
        captured["default_pack_mode"] = default_pack_mode
        captured["force"] = force
        progress_callback("openpose-import-progress")
        return Path(workspace) / ".xpkg" / "state" / "current.json"

    monkeypatch.setattr(
        "xpkg.cli.import_openpose_json_workspace",
        fake_import_openpose_json_workspace,
    )

    code = main(
        [
            "import",
            "openpose",
            "--json",
            "openpose_json",
            "--video",
            "clip.mp4",
            "--out",
            "My Project",
            "--threshold",
            "0.2",
        ]
    )

    assert code == 0
    assert captured == {
        "json_dir": "openpose_json",
        "video_path": "clip.mp4",
        "workspace": "My Project",
        "skeleton_name": "imported",
        "likelihood_threshold": 0.2,
        "default_pack_mode": "portable",
        "force": False,
    }
    stdout = capsys.readouterr().out
    assert "openpose-import-progress" in stdout
    assert "Imported OpenPose JSON into My Project" in stdout
    assert ".xpkg/state/current.json" in stdout


def test_cli_routes_import_detectron2_workspace(monkeypatch, capsys) -> None:
    from xpkg.cli import main

    captured: dict[str, object] = {}

    def fake_import_detectron2_coco_workspace(
        predictions_path: str,
        dataset_json_path: str,
        image_root: str,
        workspace: str,
        *,
        category_id: int | None,
        skeleton_name: str | None,
        likelihood_threshold: float,
        default_pack_mode: str = "portable",
        force: bool = False,
        progress_callback,
    ) -> Path:
        captured["predictions_path"] = predictions_path
        captured["dataset_json_path"] = dataset_json_path
        captured["image_root"] = image_root
        captured["workspace"] = workspace
        captured["category_id"] = category_id
        captured["skeleton_name"] = skeleton_name
        captured["likelihood_threshold"] = likelihood_threshold
        captured["default_pack_mode"] = default_pack_mode
        captured["force"] = force
        progress_callback("detectron2-import-progress")
        return Path(workspace) / ".xpkg" / "state" / "current.json"

    monkeypatch.setattr(
        "xpkg.cli.import_detectron2_coco_workspace",
        fake_import_detectron2_coco_workspace,
    )

    code = main(
        [
            "import",
            "detectron2",
            "--predictions",
            "coco_instances_results.json",
            "--dataset-json",
            "dataset.json",
            "--image-root",
            "images",
            "--out",
            "My Project",
            "--category-id",
            "7",
            "--skeleton-name",
            "mouse",
            "--threshold",
            "0.6",
        ]
    )

    assert code == 0
    assert captured == {
        "predictions_path": "coco_instances_results.json",
        "dataset_json_path": "dataset.json",
        "image_root": "images",
        "workspace": "My Project",
        "category_id": 7,
        "skeleton_name": "mouse",
        "likelihood_threshold": 0.6,
        "default_pack_mode": "portable",
        "force": False,
    }
    stdout = capsys.readouterr().out
    assert "detectron2-import-progress" in stdout
    assert "Imported Detectron2 COCO into My Project" in stdout
    assert ".xpkg/state/current.json" in stdout


def test_cli_routes_pack_unpack_and_validate(monkeypatch, capsys) -> None:
    from xpkg.cli import main

    captured: dict[str, object] = {}
    validated: list[str] = []

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
        validated.append(path)

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
    }
    assert validated == ["My Project.expkg"]
    stdout = capsys.readouterr().out
    assert "Packed My Project" in stdout
    assert "Unpacked My Project.expkg" in stdout
    assert "Valid My Project.expkg" in stdout
