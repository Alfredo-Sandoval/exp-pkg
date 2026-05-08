from __future__ import annotations

import json
from pathlib import Path

import pytest


def _project_state_path(project: str) -> Path:
    return Path(project) / ".xpkg" / "state" / "current.json"


def test_cli_rejects_removed_compat_commands() -> None:
    from xpkg.cli import main

    with pytest.raises(SystemExit):
        main(["init", "My Project"])

    with pytest.raises(SystemExit):
        main(["convert", "dlc", "csv"])

    with pytest.raises(SystemExit):
        main(["import", "detectron2", "--predictions", "coco_instances_results.json"])

    with pytest.raises(SystemExit):
        main(["import", "openpose", "--json", "openpose_json", "--out", "My Project"])

    with pytest.raises(SystemExit):
        main(["import", "sleap", "--slp", "labels.pkg.slp", "--out", "My Project"])

    with pytest.raises(SystemExit):
        main(["import", "vicon", "--csv", "trial.csv", "--out", "My Project"])


def test_cli_routes_init_project(monkeypatch, capsys) -> None:
    from xpkg.cli import main

    captured: dict[str, object] = {}

    def fake_init_project(
        project: str,
        *,
        title: str | None,
        project_id: str | None,
        force: bool,
    ) -> object:
        captured["project"] = project
        captured["title"] = title
        captured["project_id"] = project_id
        captured["force"] = force
        return object()

    monkeypatch.setattr("xpkg.cli.commands.project.init_project", fake_init_project)

    code = main(
        [
            "project",
            "init",
            "My Project",
            "--title",
            "My Project",
            "--id",
            "project-123",
            "--force",
        ]
    )

    assert code == 0
    assert captured == {
        "project": "My Project",
        "title": "My Project",
        "project_id": "project-123",
        "force": True,
    }
    assert "Initialized project My Project" in capsys.readouterr().out


def test_cli_init_json_mode(monkeypatch, capsys) -> None:
    from xpkg.cli import main

    captured: dict[str, object] = {}

    def fake_init_project(
        project: str,
        *,
        title: str | None,
        project_id: str | None,
        force: bool,
    ) -> object:
        captured["project"] = project
        captured["title"] = title
        captured["project_id"] = project_id
        captured["force"] = force
        return object()

    monkeypatch.setattr("xpkg.cli.commands.project.init_project", fake_init_project)

    code = main(["project", "init", "My Project", "--title", "My Project", "--json"])

    assert code == 0
    assert captured == {
        "project": "My Project",
        "title": "My Project",
        "project_id": None,
        "force": False,
    }
    captured_streams = capsys.readouterr()
    assert captured_streams.err == ""
    payload = json.loads(captured_streams.out)
    assert payload == {
        "ok": True,
        "data": {
            "status": "initialized",
            "project": "My Project",
            "title": "My Project",
            "project_id": None,
        },
    }


def test_cli_routes_import_dlc_csv_project(monkeypatch, capsys) -> None:
    from xpkg.cli import main

    captured: dict[str, object] = {}

    def fake_import_dlc_csv_project(
        csv_path: str,
        video_path: str,
        project: str,
        *,
        skeleton_name: str,
        likelihood_threshold: float,
        prediction_provenance=None,
        provenance=None,
        force: bool = False,
        progress_callback,
    ) -> Path:
        captured["csv_path"] = csv_path
        captured["video_path"] = video_path
        captured["project"] = project
        captured["skeleton_name"] = skeleton_name
        captured["likelihood_threshold"] = likelihood_threshold
        captured["force"] = force
        progress_callback("import-progress")
        return _project_state_path(project)

    monkeypatch.setattr(
        "xpkg.cli.commands.imports.import_dlc_csv_project",
        fake_import_dlc_csv_project,
    )

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
            "subject",
            "--threshold",
            "0.25",
        ]
    )

    assert code == 0
    assert captured == {
        "csv_path": "tracking.csv",
        "video_path": "clip.mp4",
        "project": "My Project",
        "skeleton_name": "subject",
        "likelihood_threshold": 0.25,
        "force": False,
    }
    stdout = capsys.readouterr().out
    assert "import-progress" in stdout
    assert "Imported DLC CSV into My Project" in stdout
    assert ".xpkg/state/current.json" in stdout


def test_cli_routes_pose_prediction_provenance_json(monkeypatch, tmp_path: Path) -> None:
    from xpkg.cli import main

    captured: dict[str, object] = {}
    provenance_path = tmp_path / "prediction_provenance.json"
    provenance_path.write_text(
        json.dumps(
            {
                "model_name": "dlc-model",
                "model_version": "2.0",
                "training_set": "open-field-training-v1",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_import_dlc_csv_project(
        csv_path: str,
        video_path: str,
        project: str,
        *,
        skeleton_name: str,
        likelihood_threshold: float,
        prediction_provenance,
        provenance=None,
        force: bool = False,
        progress_callback,
    ) -> Path:
        del csv_path, video_path, skeleton_name, likelihood_threshold, force, provenance
        captured["project"] = project
        captured["prediction_provenance"] = prediction_provenance
        progress_callback("import-progress")
        return _project_state_path(project)

    monkeypatch.setattr(
        "xpkg.cli.commands.imports.import_dlc_csv_project",
        fake_import_dlc_csv_project,
    )

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
            "--provenance-json",
            str(provenance_path),
        ]
    )

    assert code == 0
    assert captured["project"] == "My Project"
    assert captured["prediction_provenance"] == {
        "model_name": "dlc-model",
        "model_version": "2.0",
        "training_set": "open-field-training-v1",
    }


def test_cli_import_json_mode_suppresses_progress(monkeypatch, capsys) -> None:
    from xpkg.cli import main

    def fake_import_dlc_csv_project(
        csv_path: str,
        video_path: str,
        project: str,
        *,
        skeleton_name: str,
        likelihood_threshold: float,
        prediction_provenance=None,
        provenance=None,
        force: bool = False,
        progress_callback,
    ) -> Path:
        progress_callback("import-progress")
        return _project_state_path(project)

    monkeypatch.setattr(
        "xpkg.cli.commands.imports.import_dlc_csv_project",
        fake_import_dlc_csv_project,
    )

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
            "--json",
        ]
    )

    assert code == 0
    captured_streams = capsys.readouterr()
    assert captured_streams.err == ""
    assert "import-progress" not in captured_streams.out
    payload = json.loads(captured_streams.out)
    assert payload == {
        "ok": True,
        "data": {
            "status": "imported",
            "source": "dlc_csv",
            "project": "My Project",
            "state_path": "My Project/.xpkg/state/current.json",
        },
    }


def test_cli_routes_import_vicon_csv_project(monkeypatch, capsys) -> None:
    from xpkg.cli import main

    captured: dict[str, object] = {}

    def fake_import_vicon_csv_project(
        csv_path: str,
        project: str,
        *,
        force: bool = False,
        progress_callback,
    ) -> Path:
        captured["csv_path"] = csv_path
        captured["project"] = project
        captured["force"] = force
        progress_callback("vicon-csv-progress")
        return _project_state_path(project)

    monkeypatch.setattr(
        "xpkg.cli.commands.imports.import_vicon_csv_project",
        fake_import_vicon_csv_project,
    )

    code = main(
        [
            "import",
            "vicon",
            "csv",
            "--csv",
            "trial.csv",
            "--out",
            "My Project",
        ]
    )

    assert code == 0
    assert captured == {
        "csv_path": "trial.csv",
        "project": "My Project",
        "force": False,
    }
    stdout = capsys.readouterr().out
    assert "vicon-csv-progress" in stdout
    assert "Imported Vicon CSV into My Project" in stdout
    assert ".xpkg/state/current.json" in stdout


def test_cli_routes_import_vicon_c3d_project(monkeypatch, capsys) -> None:
    from xpkg.cli import main

    captured: dict[str, object] = {}

    def fake_import_vicon_c3d_project(
        c3d_path: str,
        project: str,
        *,
        force: bool = False,
        progress_callback,
    ) -> Path:
        captured["c3d_path"] = c3d_path
        captured["project"] = project
        captured["force"] = force
        progress_callback("vicon-c3d-progress")
        return _project_state_path(project)

    monkeypatch.setattr(
        "xpkg.cli.commands.imports.import_vicon_c3d_project",
        fake_import_vicon_c3d_project,
    )

    code = main(
        [
            "import",
            "vicon",
            "c3d",
            "--c3d",
            "trial.c3d",
            "--out",
            "My Project",
        ]
    )

    assert code == 0
    assert captured == {
        "c3d_path": "trial.c3d",
        "project": "My Project",
        "force": False,
    }
    stdout = capsys.readouterr().out
    assert "vicon-c3d-progress" in stdout
    assert "Imported Vicon C3D into My Project" in stdout
    assert ".xpkg/state/current.json" in stdout


def test_cli_routes_import_vicon_recording_project(monkeypatch, capsys) -> None:
    from xpkg.cli import main

    captured: dict[str, object] = {}

    def fake_import_vicon_project(
        recording_path: str,
        project: str,
        *,
        force: bool = False,
        progress_callback,
    ) -> Path:
        captured["recording_path"] = recording_path
        captured["project"] = project
        captured["force"] = force
        progress_callback("vicon-auto-progress")
        return _project_state_path(project)

    monkeypatch.setattr(
        "xpkg.cli.commands.imports.import_vicon_project",
        fake_import_vicon_project,
    )

    code = main(
        [
            "import",
            "vicon",
            "recording",
            "--recording",
            "trial.csv",
            "--out",
            "My Project",
        ]
    )

    assert code == 0
    assert captured == {
        "recording_path": "trial.csv",
        "project": "My Project",
        "force": False,
    }
    stdout = capsys.readouterr().out
    assert "vicon-auto-progress" in stdout
    assert "Imported Vicon recording into My Project" in stdout
    assert ".xpkg/state/current.json" in stdout


def test_cli_routes_import_dlc_project_directory(monkeypatch, capsys) -> None:
    from xpkg.cli import main

    captured: dict[str, object] = {}

    def fake_import_dlc_project_directory(
        project_dir: str,
        project: str,
        *,
        skeleton_name: str | None = None,
        likelihood_threshold: float,
        prediction_provenance=None,
        provenance=None,
        force: bool = False,
        progress_callback,
    ) -> Path:
        captured["project_dir"] = project_dir
        captured["project"] = project
        captured["skeleton_name"] = skeleton_name
        captured["likelihood_threshold"] = likelihood_threshold
        captured["force"] = force
        progress_callback("project-import-progress")
        return _project_state_path(project)

    monkeypatch.setattr(
        "xpkg.cli.commands.imports.import_dlc_project_directory",
        fake_import_dlc_project_directory,
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
        "project": "My Project",
        "skeleton_name": None,
        "likelihood_threshold": 0.5,
        "force": False,
    }
    stdout = capsys.readouterr().out
    assert "project-import-progress" in stdout
    assert "Imported DLC project into My Project" in stdout
    assert ".xpkg/state/current.json" in stdout


def test_cli_routes_import_sleap_package_project(monkeypatch, capsys) -> None:
    from xpkg.cli import main

    captured: dict[str, object] = {}

    def fake_import_sleap_package_project(
        slp: str,
        project: str,
        *,
        fps: int,
        encode_videos: bool | None,
        prediction_provenance=None,
        provenance=None,
        force: bool = False,
        progress_callback,
    ) -> Path:
        captured["slp"] = slp
        captured["project"] = project
        captured["fps"] = fps
        captured["encode_videos"] = encode_videos
        captured["force"] = force
        progress_callback("sleap-progress")
        return _project_state_path(project)

    monkeypatch.setattr(
        "xpkg.cli.commands.imports.import_sleap_package_project",
        fake_import_sleap_package_project,
    )

    code = main(
        [
            "import",
            "sleap",
            "package",
            "--slp",
            "labels.pkg.slp",
            "--out",
            "My Project",
            "--fps",
            "24",
            "--no-videos",
        ]
    )

    assert code == 0
    assert captured == {
        "slp": "labels.pkg.slp",
        "project": "My Project",
        "fps": 24,
        "encode_videos": False,
        "force": False,
    }
    stdout = capsys.readouterr().out
    assert "sleap-progress" in stdout
    assert "Imported SLEAP package into My Project" in stdout
    assert ".xpkg/state/current.json" in stdout


def test_cli_routes_import_sleap_h5_project(monkeypatch, capsys) -> None:
    from xpkg.cli import main

    captured: dict[str, object] = {}

    def fake_import_sleap_h5_project(
        h5_path: str,
        video_path: str,
        project: str,
        *,
        skeleton_name: str,
        likelihood_threshold: float,
        prediction_provenance=None,
        provenance=None,
        force: bool = False,
        progress_callback,
    ) -> Path:
        captured["h5_path"] = h5_path
        captured["video_path"] = video_path
        captured["project"] = project
        captured["skeleton_name"] = skeleton_name
        captured["likelihood_threshold"] = likelihood_threshold
        captured["force"] = force
        progress_callback("sleap-h5-import-progress")
        return _project_state_path(project)

    monkeypatch.setattr(
        "xpkg.cli.commands.imports.import_sleap_h5_project",
        fake_import_sleap_h5_project,
    )

    code = main(
        [
            "import",
            "sleap",
            "h5",
            "--h5",
            "analysis.h5",
            "--video",
            "clip.mp4",
            "--out",
            "My Project",
            "--skeleton-name",
            "subject",
            "--threshold",
            "0.25",
        ]
    )

    assert code == 0
    assert captured == {
        "h5_path": "analysis.h5",
        "video_path": "clip.mp4",
        "project": "My Project",
        "skeleton_name": "subject",
        "likelihood_threshold": 0.25,
        "force": False,
    }
    stdout = capsys.readouterr().out
    assert "sleap-h5-import-progress" in stdout
    assert "Imported SLEAP H5 into My Project" in stdout
    assert ".xpkg/state/current.json" in stdout


def test_cli_routes_import_mmpose_project(monkeypatch, capsys) -> None:
    from xpkg.cli import main

    captured: dict[str, object] = {}

    def fake_import_mmpose_topdown_json_project(
        json_path: str,
        video_path: str,
        project: str,
        *,
        skeleton_name: str,
        instance_index: int,
        likelihood_threshold: float,
        prediction_provenance=None,
        provenance=None,
        force: bool = False,
        progress_callback,
    ) -> Path:
        captured["json_path"] = json_path
        captured["video_path"] = video_path
        captured["project"] = project
        captured["skeleton_name"] = skeleton_name
        captured["instance_index"] = instance_index
        captured["likelihood_threshold"] = likelihood_threshold
        captured["force"] = force
        progress_callback("mmpose-import-progress")
        return _project_state_path(project)

    monkeypatch.setattr(
        "xpkg.cli.commands.imports.import_mmpose_topdown_json_project",
        fake_import_mmpose_topdown_json_project,
    )

    code = main(
        [
            "import",
            "mmpose",
            "--input-json",
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
        "project": "My Project",
        "skeleton_name": "imported",
        "instance_index": 1,
        "likelihood_threshold": 0.4,
        "force": False,
    }
    stdout = capsys.readouterr().out
    assert "mmpose-import-progress" in stdout
    assert "Imported MMPose JSON into My Project" in stdout
    assert ".xpkg/state/current.json" in stdout


def test_cli_routes_import_mediapipe_project(monkeypatch, capsys) -> None:
    from xpkg.cli import main

    captured: dict[str, object] = {}

    def fake_import_mediapipe_pose_landmarks_json_project(
        json_path: str,
        video_path: str,
        project: str,
        *,
        skeleton_name: str,
        likelihood_threshold: float,
        prediction_provenance=None,
        provenance=None,
        force: bool = False,
        progress_callback,
    ) -> Path:
        captured["json_path"] = json_path
        captured["video_path"] = video_path
        captured["project"] = project
        captured["skeleton_name"] = skeleton_name
        captured["likelihood_threshold"] = likelihood_threshold
        captured["force"] = force
        progress_callback("mediapipe-import-progress")
        return _project_state_path(project)

    monkeypatch.setattr(
        "xpkg.cli.commands.imports.import_mediapipe_pose_landmarks_json_project",
        fake_import_mediapipe_pose_landmarks_json_project,
    )

    code = main(
        [
            "import",
            "mediapipe",
            "--input-json",
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
        "project": "My Project",
        "skeleton_name": "mediapipe_pose",
        "likelihood_threshold": 0.3,
        "force": False,
    }
    stdout = capsys.readouterr().out
    assert "mediapipe-import-progress" in stdout
    assert "Imported MediaPipe JSON into My Project" in stdout
    assert ".xpkg/state/current.json" in stdout


def test_cli_routes_import_lightning_pose_project(monkeypatch, capsys) -> None:
    from xpkg.cli import main

    captured: dict[str, object] = {}

    def fake_import_lightning_pose_csv_project(
        csv_path: str,
        video_path: str,
        project: str,
        *,
        skeleton_name: str,
        likelihood_threshold: float,
        prediction_provenance=None,
        provenance=None,
        force: bool = False,
        progress_callback,
    ) -> Path:
        captured["csv_path"] = csv_path
        captured["video_path"] = video_path
        captured["project"] = project
        captured["skeleton_name"] = skeleton_name
        captured["likelihood_threshold"] = likelihood_threshold
        captured["force"] = force
        progress_callback("lightning-pose-import-progress")
        return _project_state_path(project)

    monkeypatch.setattr(
        "xpkg.cli.commands.imports.import_lightning_pose_csv_project",
        fake_import_lightning_pose_csv_project,
    )

    code = main(
        [
            "import",
            "lightning-pose",
            "--csv",
            "video_preds/session0.csv",
            "--video",
            "clip.mp4",
            "--out",
            "My Project",
            "--threshold",
            "0.4",
        ]
    )

    assert code == 0
    assert captured == {
        "csv_path": "video_preds/session0.csv",
        "video_path": "clip.mp4",
        "project": "My Project",
        "skeleton_name": "imported",
        "likelihood_threshold": 0.4,
        "force": False,
    }
    stdout = capsys.readouterr().out
    assert "lightning-pose-import-progress" in stdout
    assert "Imported Lightning Pose CSV into My Project" in stdout
    assert ".xpkg/state/current.json" in stdout


def test_cli_json_errors_use_stderr(capsys) -> None:
    from xpkg.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main(["import", "sleap", "h5", "--h5", "analysis.h5", "--out", "My Project", "--json"])

    assert exc_info.value.code == 1
    captured_streams = capsys.readouterr()
    assert captured_streams.out == ""
    payload = json.loads(captured_streams.err)
    assert payload == {
        "ok": False,
        "error": {
            "code": "usage_error",
            "message": "Missing option '--video'.",
            "hint": "Run `xpkg --help` or `xpkg describe --json` for the command contract.",
        },
    }


def test_cli_routes_pack_unpack_and_validate(monkeypatch, capsys) -> None:
    from xpkg.cli import main

    captured: dict[str, object] = {}
    validated: list[str] = []

    def fake_pack_project(
        project: str,
        *,
        out: str | None,
        media: str,
        overwrite: bool,
    ) -> Path:
        captured["pack_project"] = project
        captured["pack_out"] = out
        captured["pack_media"] = media
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

    monkeypatch.setattr("xpkg.cli.commands.project.pack_project", fake_pack_project)
    monkeypatch.setattr("xpkg.cli.commands.project.unpack_project", fake_unpack_project)
    monkeypatch.setattr(
        "xpkg.cli.commands.project.validate_artifact_target",
        fake_validate_artifact,
    )

    pack_code = main(
        [
            "project",
            "pack",
            "My Project",
            "--media",
            "package",
            "--overwrite",
        ]
    )
    unpack_code = main(
        [
            "project",
            "unpack",
            "My Project.expkg",
            "--out",
            "Unpacked Project",
            "--force",
            "--rename",
            "Renamed Project",
        ]
    )
    validate_code = main(["project", "validate", "My Project.expkg"])

    assert pack_code == 0
    assert unpack_code == 0
    assert validate_code == 0
    assert captured == {
        "pack_project": "My Project",
        "pack_out": None,
        "pack_media": "package",
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


def test_cli_project_describe_json_mode(tmp_path: Path, capsys) -> None:
    from xpkg.cli import main
    from xpkg.services import ProjectService

    project = ProjectService.create(tmp_path / "Described Project", title="Described Project")

    code = main(["project", "describe", str(project.project_root), "--json"])

    assert code == 0
    captured_streams = capsys.readouterr()
    assert captured_streams.err == ""
    envelope = json.loads(captured_streams.out)
    assert envelope["ok"] is True
    payload = envelope["data"]
    assert payload["status"] == "described"
    assert payload["project"] == str(project.project_root)
    assert payload["descriptor"]["title"] == "Described Project"
    assert payload["paths"]["descriptor"].endswith("/PROJECT.json")
    assert payload["paths"]["store"].endswith("/.xpkg")
    assert payload["paths"]["current_state"].endswith("/.xpkg/state/current.json")
    assert payload["has_current_state"] is False


def test_cli_describe_lists_top_level_inspect(capsys) -> None:
    from xpkg.cli import main

    code = main(["describe", "--json"])

    assert code == 0
    captured_streams = capsys.readouterr()
    assert captured_streams.err == ""
    envelope = json.loads(captured_streams.out)
    assert envelope["ok"] is True
    payload = envelope["data"]
    assert payload["resources"]["inspect"] == ["path"]
    assert "inspect" in payload["commands"]


def test_cli_inspect_json_mode(monkeypatch, capsys) -> None:
    from xpkg.cli import main
    from xpkg.inspection import InspectionKind, InspectionReport

    captured: dict[str, object] = {}

    def fake_inspect_path(target: str, *, confidence_threshold: float) -> InspectionReport:
        captured["target"] = target
        captured["confidence_threshold"] = confidence_threshold
        return InspectionReport(
            path=target,
            name="tracking.csv",
            suffix=".csv",
            exists=True,
            is_dir=False,
            size_bytes=42,
            kind=InspectionKind.POSE_PREDICTIONS,
            likely_importers=("dlc_csv",),
            summary={"frames": 2, "keypoints": 3},
            warnings=(),
        )

    monkeypatch.setattr("xpkg.cli.commands.inspect.inspect_path", fake_inspect_path)

    code = main(["inspect", "tracking.csv", "--threshold", "0.7", "--json"])

    assert code == 0
    assert captured == {"target": "tracking.csv", "confidence_threshold": 0.7}
    captured_streams = capsys.readouterr()
    assert captured_streams.err == ""
    envelope = json.loads(captured_streams.out)
    assert envelope["ok"] is True
    payload = envelope["data"]
    assert payload["status"] == "inspected"
    assert payload["kind"] == "pose_predictions"
    assert payload["likely_importers"] == ["dlc_csv"]


def test_cli_inspect_human_mode(monkeypatch, capsys) -> None:
    from xpkg.cli import main
    from xpkg.inspection import InspectionKind, InspectionReport

    def fake_inspect_path(target: str, *, confidence_threshold: float) -> InspectionReport:
        return InspectionReport(
            path=target,
            name="clip.mp4",
            suffix=".mp4",
            exists=True,
            is_dir=False,
            size_bytes=1024,
            kind=InspectionKind.VIDEO,
            likely_importers=(),
            summary={"frames": 12, "fps": 30.0, "width": 640, "height": 480},
            warnings=("Frame count is approximate.",),
        )

    monkeypatch.setattr("xpkg.cli.commands.inspect.inspect_path", fake_inspect_path)

    code = main(["inspect", "clip.mp4"])

    assert code == 0
    stdout = capsys.readouterr().out
    assert "Kind: video" in stdout
    assert "frames: 12" in stdout
    assert "Warning: Frame count is approximate." in stdout


def test_cli_artifacts_list_inspect_validate_and_rebuild(tmp_path: Path, capsys) -> None:
    from xpkg.cli import main
    from xpkg.services import ProjectService

    project = ProjectService.create(tmp_path / "Artifact CLI")
    source = tmp_path / "summary.csv"
    source.write_text("x\n1\n", encoding="utf-8")
    project.artifacts.register(
        artifact_id="summary",
        artifact_type="table",
        outputs=[source],
    )

    assert main(["artifacts", "list", str(project.project_root), "--kind", "table"]) == 0
    stdout = capsys.readouterr().out
    assert "table\t-\tsummary\t.xpkg/artifacts/tables/summary/manifest.json" in stdout

    assert (
        main(
            [
                "artifacts",
                "inspect",
                str(project.project_root),
                "summary",
                "--kind",
                "table",
            ]
        )
        == 0
    )
    stdout = capsys.readouterr().out
    assert '"artifact_type": "table"' in stdout
    assert '"artifact_id": "summary"' in stdout

    assert (
        main(
            [
                "artifacts",
                "validate",
                str(project.project_root),
                "summary",
                "--kind",
                "table",
            ]
        )
        == 0
    )
    assert "Valid artifact summary" in capsys.readouterr().out

    assert main(["artifacts", "rebuild-index", str(project.project_root)]) == 0
    assert "Indexed artifacts 1" in capsys.readouterr().out
