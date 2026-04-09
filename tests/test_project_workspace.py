from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest


def _write_test_image(path: Path, value: int = 128) -> None:
    image = np.full((12, 16, 3), value, dtype=np.uint8)
    ok = cv2.imwrite(path.as_posix(), image)
    assert ok


def _write_test_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        path.as_posix(),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (16, 12),
    )
    assert writer.isOpened()
    for value in (32, 64, 96):
        frame = np.full((12, 16, 3), value, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    assert path.exists()


def _make_single_frame_video(tmp_path: Path):
    from xpkg.model import Video

    frame_path = tmp_path / "frame.png"
    _write_test_image(frame_path)
    video = Video.from_image_filenames([frame_path.as_posix()])
    video.filename = frame_path.as_posix()
    return frame_path, video


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _make_labels(tmp_path: Path, *, x: float, y: float):
    from xpkg.core.annotations import Instance, LabeledFrame, Point
    from xpkg.model import Labels, build_keypoint_skeleton

    _, video = _make_single_frame_video(tmp_path)
    skeleton = build_keypoint_skeleton(["nose"], name="mouse")
    frame = LabeledFrame(
        video=video,
        frame_idx=0,
        instances=[
            Instance(
                skeleton=skeleton,
                init_points={"nose": Point(x, y, visible=True, complete=True)},
            )
        ],
    )
    labels = Labels(labeled_frames=[frame], videos=[video], skeletons=[skeleton])
    labels.validate()
    return labels


def _make_media_labels(video_path: Path, *, x: float, y: float):
    from xpkg.core.annotations import Instance, LabeledFrame, Point
    from xpkg.model import Labels, Video, build_keypoint_skeleton

    video = Video.from_filename(video_path.as_posix())
    skeleton = build_keypoint_skeleton(["nose"], name="mouse")
    frame = LabeledFrame(
        video=video,
        frame_idx=0,
        instances=[
            Instance(
                skeleton=skeleton,
                init_points={"nose": Point(x, y, visible=True, complete=True)},
            )
        ],
    )
    labels = Labels(labeled_frames=[frame], videos=[video], skeletons=[skeleton])
    labels.validate()
    return labels


def test_init_project_writes_workspace_contract(tmp_path: Path) -> None:
    from xpkg.formats import (
        current_project_state_path,
        init_project,
        is_workspace_root,
        load_project_descriptor,
    )
    from xpkg.model import Labels

    workspace = tmp_path / "My Project"
    descriptor = init_project(workspace, title="My Project")

    assert is_workspace_root(workspace)
    assert (workspace / "PROJECT.json").is_file()
    assert (workspace / ".posetta").is_dir()
    assert (workspace / "Media").is_dir()
    assert (workspace / "Exports").is_dir()
    assert not current_project_state_path(workspace).exists()
    assert load_project_descriptor(workspace).title == "My Project"
    assert descriptor.default_pack_mode == "portable"

    loaded = Labels.load_file(workspace.as_posix())
    assert loaded.labeled_frames == []


def test_migrate_legacy_archive_creates_workspace_and_workspace_loads(tmp_path: Path) -> None:
    from xpkg.formats import (
        current_project_archive_path,
        current_project_snapshot_path,
        migrate_legacy_archive,
        workspace_media_root,
        workspace_state_root,
        workspace_store_root,
        write_siesta,
    )
    from xpkg.model import Labels

    source_root = tmp_path / "source"
    source_root.mkdir()
    labels = _make_labels(source_root, x=3.0, y=4.0)
    legacy_path = tmp_path / "tracking.siesta"
    workspace = tmp_path / "Migrated Project"
    write_siesta(legacy_path, labels)

    migrated_archive = migrate_legacy_archive(legacy_path, workspace)

    assert migrated_archive == current_project_snapshot_path(workspace)
    assert migrated_archive.exists()
    assert not current_project_archive_path(workspace).exists()
    assert not (workspace_store_root(workspace) / "superblock.a.json").exists()
    assert not (workspace_state_root(workspace) / "current.sta").exists()
    assert (workspace_state_root(workspace) / "current.json").is_file()

    loaded = Labels.load_file(workspace.as_posix())
    pts = loaded.labeled_frames[0].instances[0].get_points_array(copy=False, full=True)
    assert float(pts["x"][0]) == 3.0
    assert float(pts["y"][0]) == 4.0
    media_root = workspace_media_root(workspace)
    managed_files = sorted(path for path in media_root.rglob("*") if path.is_file())
    assert managed_files
    for video in loaded.videos:
        assert _is_within(Path(str(video.filename)), media_root)
        for frame_path in video.image_filenames or []:
            resolved = Path(str(frame_path))
            assert _is_within(resolved, media_root)
            assert resolved.exists()


def test_migrate_legacy_archive_rewrites_stale_project_metadata_paths(tmp_path: Path) -> None:
    from xpkg.formats import (
        current_project_snapshot_path,
        init_project,
        migrate_legacy_archive,
        write_siesta,
    )
    from xpkg.io.workspace_snapshot_backend import read_workspace_snapshot_payload

    legacy_root = tmp_path / "bootstrap_2026-01-13"
    legacy_root.mkdir()
    downloads_root = tmp_path / "Downloads"
    downloads_root.mkdir()
    source_video = downloads_root / "session_video.avi"
    _write_test_video(source_video)
    labels = _make_media_labels(source_video, x=2.0, y=3.0)
    legacy_output_dir = legacy_root / "models" / "pose" / "run-1"
    legacy_output_dir.mkdir(parents=True)
    legacy_path = legacy_root / "tracking.siesta"
    training_state = {
        "schema_version": 1,
        "latest": {
            "run_id": "latest",
            "created_ns": 2,
            "source_bundle": legacy_root.as_posix(),
            "output_dir": legacy_output_dir.as_posix(),
        },
        "runs": [
            {
                "run_id": "rebased",
                "created_ns": 1,
                "source_bundle": legacy_root.as_posix(),
                "output_dir": legacy_output_dir.as_posix(),
            },
            {
                "run_id": "cleared",
                "created_ns": 0,
                "source_bundle": (legacy_root / "missing-bundle").as_posix(),
                "output_dir": (legacy_root / "missing-output").as_posix(),
            },
        ],
    }
    session_state = {
        "active_video_path": source_video.as_posix(),
        "active_frame_idx": 2,
    }
    write_siesta(
        legacy_path,
        labels,
        metadata={
            "training_state_json": training_state,
            "session_json": session_state,
        },
    )

    workspace = tmp_path / "Migrated Project"
    init_project(workspace, title="Migrated Project", force=True)
    workspace_output_dir = workspace / "models" / "pose" / "run-1"
    workspace_output_dir.mkdir(parents=True)

    migrate_legacy_archive(legacy_path, workspace)

    snapshot_payload = read_workspace_snapshot_payload(current_project_snapshot_path(workspace))
    migrated_training = snapshot_payload["metadata"]["training_state_json"]
    migrated_session = snapshot_payload["session"]

    assert migrated_training["latest"]["source_bundle"] == workspace.resolve().as_posix()
    assert migrated_training["latest"]["output_dir"] == workspace_output_dir.resolve().as_posix()
    assert migrated_training["runs"][0]["source_bundle"] == workspace.resolve().as_posix()
    assert migrated_training["runs"][0]["output_dir"] == workspace_output_dir.resolve().as_posix()
    assert migrated_training["runs"][1]["source_bundle"] == ""
    assert migrated_training["runs"][1]["output_dir"] == ""
    assert migrated_session["active_video_path"] == f"Media/{source_video.name}"
    assert migrated_session["active_frame_idx"] == 2


def test_pack_snapshot_and_unpack_roundtrip_workspace(tmp_path: Path) -> None:
    from xpkg.formats import pack_project, unpack_project, validate_artifact, write_siesta
    from xpkg.model import Labels

    labels = _make_labels(tmp_path, x=5.0, y=6.0)
    legacy_path = tmp_path / "tracking.siesta"
    write_siesta(legacy_path, labels)

    workspace = tmp_path / "Roundtrip Project"
    from xpkg.formats import migrate_legacy_archive

    migrate_legacy_archive(legacy_path, workspace)

    artifact = pack_project(workspace, mode="snapshot")
    validate_artifact(artifact)

    unpacked = tmp_path / "Unpacked Project"
    unpack_project(artifact, unpacked)
    validate_artifact(unpacked)

    loaded = Labels.load_file(unpacked.as_posix())
    pts = loaded.labeled_frames[0].instances[0].get_points_array(copy=False, full=True)
    assert float(pts["x"][0]) == 5.0
    assert float(pts["y"][0]) == 6.0


def test_pack_portable_and_unpack_uses_managed_media_after_source_removal(tmp_path: Path) -> None:
    from xpkg.formats import (
        migrate_legacy_archive,
        pack_project,
        unpack_project,
        validate_artifact,
        workspace_media_root,
        write_siesta,
    )
    from xpkg.model import Labels

    source_root = tmp_path / "source"
    source_root.mkdir()
    labels = _make_labels(source_root, x=7.0, y=8.0)
    legacy_path = tmp_path / "external.siesta"
    write_siesta(legacy_path, labels)

    workspace = tmp_path / "Portable Project"
    migrate_legacy_archive(legacy_path, workspace)
    managed_files = sorted(
        path for path in workspace_media_root(workspace).rglob("*") if path.is_file()
    )
    assert managed_files

    artifact = tmp_path / "Portable Project.expkg"
    pack_project(workspace, mode="portable", out=artifact)
    validate_artifact(artifact)

    shutil.rmtree(source_root)
    shutil.rmtree(workspace)

    unpacked = tmp_path / "Unpacked Portable Project"
    unpack_project(artifact, unpacked)
    validate_artifact(unpacked)

    loaded = Labels.load_file(unpacked.as_posix())
    pts = loaded.labeled_frames[0].instances[0].get_points_array(copy=False, full=True)
    assert float(pts["x"][0]) == 7.0
    assert float(pts["y"][0]) == 8.0

    unpacked_media_root = workspace_media_root(unpacked)
    for video in loaded.videos:
        assert _is_within(Path(str(video.filename)), unpacked_media_root)
        for frame_path in video.image_filenames or []:
            resolved = Path(str(frame_path))
            assert _is_within(resolved, unpacked_media_root)
            assert resolved.exists()


def test_workspace_load_auto_adopts_legacy_state_archive(tmp_path: Path) -> None:
    from xpkg.formats import (
        current_project_archive_path,
        init_project,
        workspace_state_root,
        workspace_store_root,
        write_siesta,
    )
    from xpkg.model import Labels

    source_root = tmp_path / "source"
    source_root.mkdir()
    labels = _make_labels(source_root, x=9.0, y=10.0)

    workspace = tmp_path / "Legacy Workspace"
    init_project(workspace, title="Legacy Workspace")
    legacy_state_path = workspace_state_root(workspace) / "current.siesta"
    legacy_state_path.parent.mkdir(parents=True, exist_ok=True)
    write_siesta(legacy_state_path, labels)

    loaded = Labels.load_file(workspace.as_posix())
    pts = loaded.labeled_frames[0].instances[0].get_points_array(copy=False, full=True)
    assert float(pts["x"][0]) == 9.0
    assert float(pts["y"][0]) == 10.0

    current_archive = current_project_archive_path(workspace)
    assert current_archive.exists()
    assert current_archive != legacy_state_path
    assert (workspace_store_root(workspace) / "superblock.a.json").is_file()
    assert not legacy_state_path.exists()


def test_labels_save_file_to_workspace_creates_first_committed_state(tmp_path: Path) -> None:
    from xpkg.formats import (
        current_project_archive_path,
        current_project_snapshot_path,
        init_project,
    )
    from xpkg.model import Labels

    source_root = tmp_path / "source"
    source_root.mkdir()
    labels = _make_labels(source_root, x=11.0, y=12.0)

    workspace = tmp_path / "Saved Workspace"
    init_project(workspace, title="Saved Workspace")

    saved_target = Labels.save_file(labels, workspace.as_posix())
    current_snapshot = current_project_snapshot_path(workspace)

    assert saved_target == workspace.as_posix()
    assert current_snapshot.exists()
    assert not current_project_archive_path(workspace).exists()
    assert labels.path == workspace

    loaded = Labels.load_file(workspace.as_posix())
    pts = loaded.labeled_frames[0].instances[0].get_points_array(copy=False, full=True)
    assert float(pts["x"][0]) == 11.0
    assert float(pts["y"][0]) == 12.0


def test_labels_save_file_to_workspace_preserves_predictions(tmp_path: Path) -> None:
    from xpkg.formats import (
        PredictionAppendItem,
        SerializerPredictedInstance,
        current_project_snapshot_path,
        migrate_legacy_archive,
        write_siesta,
    )
    from xpkg.io.workspace_snapshot_backend import read_workspace_snapshot_payload
    from xpkg.model import Labels

    source_root = tmp_path / "source"
    source_root.mkdir()
    initial_labels = _make_labels(source_root, x=1.0, y=2.0)
    updated_labels = _make_labels(source_root, x=21.0, y=22.0)
    legacy_path = tmp_path / "with_predictions.siesta"
    workspace = tmp_path / "Workspace Save"

    predictions = [
        PredictionAppendItem(
            video_index=0,
            frame_index=0,
            instances=[
                SerializerPredictedInstance(
                    keypoints=[(13.0, 14.0, 0.9)],
                    keypoint_scores=[0.9],
                    score=0.97,
                    track_id=4,
                )
            ],
        )
    ]

    write_siesta(legacy_path, initial_labels, predictions=predictions)
    migrate_legacy_archive(legacy_path, workspace)

    saved_target = Labels.save_file(updated_labels, workspace.as_posix())
    payload = read_workspace_snapshot_payload(current_project_snapshot_path(workspace))

    assert saved_target == workspace.as_posix()
    label_keypoints = np.asarray(payload["data"]["keypoints"], dtype=np.float32)
    prediction_scores = np.asarray(
        payload["predictions"]["data"]["keypoint_score"],
        dtype=np.float32,
    )
    prediction_track_ids = np.asarray(payload["predictions"]["data"]["track_id"], dtype=np.int32)

    assert float(label_keypoints[0, 0, 0, 0]) == 21.0
    assert float(label_keypoints[0, 0, 0, 1]) == 22.0
    assert float(prediction_scores[0, 0, 0]) == pytest.approx(0.9)
    assert int(prediction_track_ids[0, 0]) == 4


def test_workspace_load_prefers_current_snapshot(tmp_path: Path) -> None:
    from xpkg.formats import init_project, workspace_state_root
    from xpkg.model import Labels

    source_root = tmp_path / "source"
    source_root.mkdir()
    labels = _make_labels(source_root, x=11.0, y=12.0)
    workspace = tmp_path / "Snapshot Preferred"
    init_project(workspace, title="Snapshot Preferred")

    Labels.save_file(labels, workspace.as_posix())
    snapshot_path = workspace_state_root(workspace) / "current.json"
    snapshot_doc = json.loads(snapshot_path.read_text())
    keypoints = snapshot_doc["payload"]["data"]["keypoints"]
    keypoints[0][0][0][0] = 101.0
    keypoints[0][0][0][1] = 102.0
    snapshot_path.write_text(json.dumps(snapshot_doc, indent=2) + "\n", encoding="utf-8")

    loaded = Labels.load_file(workspace.as_posix())
    pts = loaded.labeled_frames[0].instances[0].get_points_array(copy=False, full=True)
    assert float(pts["x"][0]) == 101.0
    assert float(pts["y"][0]) == 102.0


def test_summarize_project_and_validate_project_read_labels_video_group(tmp_path: Path) -> None:
    from xpkg.formats import (
        current_project_archive_path,
        current_project_snapshot_path,
        init_project,
        summarize_project,
        validate_project,
    )
    from xpkg.model import Labels

    source_video = tmp_path / "source.avi"
    _write_test_video(source_video)
    labels = _make_media_labels(source_video, x=4.0, y=5.0)
    workspace = tmp_path / "Summary Project"
    init_project(workspace, title="Summary Project")

    saved_target = Labels.save_file(labels, workspace.as_posix())
    archive = current_project_archive_path(workspace)
    labels.save_file(labels, archive.as_posix())
    summary = summarize_project(archive)

    assert saved_target == workspace.as_posix()
    assert current_project_snapshot_path(workspace).exists()
    assert archive.exists()
    validate_project(archive)
    assert summary.n_videos == 1
    assert len(summary.video_filenames) == 1
    assert Path(summary.video_filenames[0]).name == source_video.name
    assert summary.video_shapes == (1, 4)
    assert summary.label_frames == 1
    assert summary.prediction_frames == 0
