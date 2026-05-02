from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest


class _ForeignTrack:
    def __init__(self, track_id: int) -> None:
        self.spawned_on = int(track_id)

    @property
    def id(self) -> int:
        return int(self.spawned_on)


class _ForeignPredictedInstance:
    def __init__(
        self,
        *,
        x: float,
        y: float,
        point_score: float,
        instance_score: float,
        track_id: int,
    ) -> None:
        self.score = float(instance_score)
        self.track = _ForeignTrack(track_id)
        self._points = np.array(
            [(x, y, True, True, point_score, 0)],
            dtype=[
                ("x", "f4"),
                ("y", "f4"),
                ("visible", "?"),
                ("complete", "?"),
                ("score", "f4"),
                ("flags", "u1"),
            ],
        )

    def get_points_array(self, *, copy: bool, full: bool) -> np.ndarray:
        assert full
        return self._points.copy() if copy else self._points


class _ForeignFrame:
    def __init__(
        self,
        *,
        video: object,
        frame_idx: int,
        predicted_instances: list[_ForeignPredictedInstance],
        heatmaps: np.ndarray | None = None,
    ) -> None:
        self.video = video
        self.frame_idx = int(frame_idx)
        self.instances: list[object] = []
        self.predicted_instances = list(predicted_instances)
        self.heatmaps = heatmaps


class _ForeignSkeleton:
    def __init__(self) -> None:
        self.keypoints = ["nose"]


class _ForeignLabels:
    def __init__(self, *, videos: list[object], labeled_frames: list[_ForeignFrame]) -> None:
        self.videos = list(videos)
        self.labeled_frames = list(labeled_frames)
        self.skeletons = [_ForeignSkeleton()]
        self.skeleton = self.skeletons[0]


def _video_writer_fourcc(code: str) -> int:
    fourcc_fn = getattr(cv2, "VideoWriter_fourcc", None)
    if not callable(fourcc_fn):
        raise RuntimeError("OpenCV build does not expose VideoWriter_fourcc")
    return int(fourcc_fn(*code))


def _write_test_image(path: Path, value: int = 128) -> None:
    image = np.full((12, 16, 3), value, dtype=np.uint8)
    ok = cv2.imwrite(path.as_posix(), image)
    assert ok


def _write_test_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        path.as_posix(),
        _video_writer_fourcc("MJPG"),
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
    skeleton = build_keypoint_skeleton(["nose"], name="subject")
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
    skeleton = build_keypoint_skeleton(["nose"], name="subject")
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


def _make_predicted_labels(
    tmp_path: Path,
    *,
    x: float,
    y: float,
    point_score: float = 0.9,
    instance_score: float = 0.97,
    track_id: int = 7,
    heatmaps: np.ndarray | None = None,
):
    from xpkg.core.annotations import LabeledFrame, PredictedInstance, PredictedPoint, Track
    from xpkg.model import Labels, build_keypoint_skeleton

    _, video = _make_single_frame_video(tmp_path)
    skeleton = build_keypoint_skeleton(["nose"], name="subject")
    frame = LabeledFrame(video=video, frame_idx=0)
    frame.heatmaps = heatmaps
    frame.instances = [
        PredictedInstance(
            skeleton=skeleton,
            frame=frame,
            track=Track(spawned_on=track_id, name=f"track-{track_id}"),
            init_points={
                "nose": PredictedPoint(
                    x,
                    y,
                    visible=True,
                    complete=True,
                    score=point_score,
                )
            },
            score=instance_score,
        )
    ]
    labels = Labels(labeled_frames=[frame], videos=[video], skeletons=[skeleton])
    labels.validate()
    return labels


def _make_unconfigured_labels():
    from xpkg.core.skeleton import Skeleton
    from xpkg.model import Labels

    return Labels(
        skeletons=[Skeleton(name="unconfigured", keypoints=[], links_ids=[])],
        keypoints=[],
    )


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
    assert (workspace / ".xpkg").is_dir()
    assert (workspace / "Media").is_dir()
    assert (workspace / "Exports").is_dir()
    assert not current_project_state_path(workspace).exists()
    assert load_project_descriptor(workspace).title == "My Project"
    assert descriptor.default_pack_mode == "portable"

    loaded = Labels.load_file(workspace.as_posix())
    assert loaded.labeled_frames == []


def test_migrate_legacy_archive_creates_workspace_and_workspace_loads(tmp_path: Path) -> None:
    from xpkg.formats import (
        current_project_snapshot_path,
        migrate_legacy_archive,
        workspace_media_root,
        workspace_state_root,
        workspace_store_root,
    )
    from xpkg.io.archive_format import write_archive
    from xpkg.io.archive_store import ArchiveStore
    from xpkg.model import Labels

    source_root = tmp_path / "source"
    source_root.mkdir()
    labels = _make_labels(source_root, x=3.0, y=4.0)
    source_archive_path = tmp_path / "tracking.xpkg"
    workspace = tmp_path / "Migrated Project"
    write_archive(source_archive_path, labels)

    migrated_archive = migrate_legacy_archive(source_archive_path, workspace)

    assert migrated_archive == current_project_snapshot_path(workspace)
    assert migrated_archive.exists()
    assert (workspace_store_root(workspace) / "superblock.a.json").is_file()
    assert not (workspace_state_root(workspace) / "current.xpkg").exists()
    assert (workspace_state_root(workspace) / "current.json").is_file()
    store = ArchiveStore.open(workspace_store_root(workspace))
    assert store.has_current_root("snapshot")
    assert not store.has_current_root("archive")

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


def test_workspace_metadata_helpers_roundtrip_without_existing_labels(tmp_path: Path) -> None:
    from xpkg.formats import (
        init_project,
        load_workspace_metadata,
        load_workspace_payload,
        save_workspace_labels,
        save_workspace_metadata,
    )

    workspace = tmp_path / "Metadata Project"
    init_project(workspace, title="Metadata Project")

    assert load_workspace_payload(workspace) == {"metadata": {}}
    assert load_workspace_metadata(workspace) == {}

    labels = _make_labels(tmp_path, x=1.5, y=2.5)
    save_workspace_labels(workspace, labels)
    manifest_payload = {
        "entries": [
            {
                "id": "archive",
                "label": "Metadata Project.xpkg",
                "path": "Metadata Project.xpkg",
                "asset_type": "predictions",
                "metadata": {"role": "archive"},
            }
        ]
    }

    saved_path = save_workspace_metadata(
        workspace,
        {
            "manifest_json": manifest_payload,
            "session_json": {"active_frame_idx": 7},
        },
        reason="test.metadata",
    )

    assert saved_path.is_file()
    assert load_workspace_metadata(workspace) == {
        "manifest_json": manifest_payload,
        "preferences": {},
        "session_json": {"active_frame_idx": 7},
    }
    payload = load_workspace_payload(workspace)
    assert "labels" in payload
    assert payload["labels"]["frames"]["frame_index"] == [0]
    assert payload["metadata"]["manifest_json"] == manifest_payload
    assert payload["metadata"]["session_json"]["active_frame_idx"] == 7
    assert payload["labels"]["metadata"]["preferences"] == {}
    assert payload["predictions"]["attrs"]["committed_length"] == 0
    resolved_path = Path(payload["labels"]["videos"]["resolved_paths"][0])
    assert resolved_path.is_absolute()
    assert resolved_path.exists()


def test_save_workspace_metadata_commits_snapshot_without_labels_recommit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import xpkg.io.project_workspace as project_workspace
    from xpkg.formats import (
        init_project,
        load_workspace_metadata,
        save_workspace_labels,
        save_workspace_metadata,
    )

    workspace = tmp_path / "Metadata Fast Path"
    init_project(workspace, title="Metadata Fast Path")
    save_workspace_labels(workspace, _make_labels(tmp_path, x=1.0, y=2.0))

    def fail_commit(*_args, **_kwargs):
        raise AssertionError("metadata-only save should not recommit Labels")

    monkeypatch.setattr(project_workspace, "_commit_labels_to_workspace", fail_commit)

    saved_path = save_workspace_metadata(
        workspace,
        {"session_json": {"active_frame_idx": 12}},
        reason="test.metadata.fast_path",
    )

    assert saved_path.is_file()
    assert load_workspace_metadata(workspace)["session_json"] == {"active_frame_idx": 12}


def test_empty_placeholder_workspace_loads(tmp_path: Path) -> None:
    from xpkg.formats import (
        init_project,
        load_workspace_payload,
        save_workspace_labels,
    )
    from xpkg.model import Labels

    workspace = tmp_path / "Placeholder Project"
    init_project(workspace, title="Placeholder Project")
    save_workspace_labels(
        workspace,
        _make_unconfigured_labels(),
        metadata={"project_name": "Placeholder Project"},
    )

    loaded = Labels.load_file(workspace.as_posix())
    assert loaded.labeled_frames == []
    assert len(loaded.skeletons) == 1
    assert loaded.skeletons[0].name == "unconfigured"

    payload = load_workspace_payload(workspace)
    assert "labels" in payload
    assert np.asarray(payload["labels"]["data"]["keypoints"], dtype=np.float32).size == 0
    assert payload["predictions"]["attrs"]["committed_length"] == 0


def test_load_workspace_payload_keeps_predictions_out_of_labels_bundle(tmp_path: Path) -> None:
    from xpkg.formats import init_project, load_workspace_payload
    from xpkg.model import Labels

    workspace = tmp_path / "Prediction Project"
    init_project(workspace, title="Prediction Project")
    labels = _make_predicted_labels(
        tmp_path,
        x=10.0,
        y=20.0,
        track_id=7,
        heatmaps=np.ones((1, 2, 2), dtype=np.float32),
    )
    Labels.save_file(labels, workspace.as_posix())

    payload = load_workspace_payload(workspace)
    label_frames = np.asarray(payload["labels"]["frames"]["frame_index"], dtype=np.int32)
    label_track_ids = np.asarray(payload["labels"]["data"]["track_id"], dtype=np.int32)
    prediction_track_ids = np.asarray(payload["predictions"]["data"]["track_id"], dtype=np.int32)

    assert label_frames.size == 0
    assert label_track_ids.size == 0
    assert int(payload["predictions"]["attrs"]["committed_length"]) == 1
    assert int(prediction_track_ids[0, 0]) == 7


def test_load_workspace_payload_uses_snapshot_without_archive_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import xpkg.io.project_workspace as project_workspace
    from xpkg.formats import init_project, load_workspace_payload
    from xpkg.model import Labels

    workspace = tmp_path / "Direct Payload Project"
    init_project(workspace, title="Direct Payload Project")
    Labels.save_file(_make_labels(tmp_path, x=4.0, y=5.0), workspace.as_posix())

    def fail_export(*_args, **_kwargs):
        raise AssertionError("load_workspace_payload should not export an archive")

    monkeypatch.setattr(project_workspace, "export_project_archive", fail_export)

    payload = load_workspace_payload(workspace)

    assert payload["labels"]["frames"]["frame_index"].tolist() == [0]
    assert payload["labels"]["data"]["track_id"].shape == (1, 1)
    assert payload["predictions"]["attrs"]["committed_length"] == 0


def test_migrate_legacy_archive_rewrites_stale_project_metadata_paths(tmp_path: Path) -> None:
    from xpkg.formats import (
        current_project_snapshot_path,
        init_project,
        migrate_legacy_archive,
    )
    from xpkg.io.archive_format import write_archive
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
    source_archive_path = legacy_root / "tracking.xpkg"
    training_state = {
        "schema_version": 1,
        "latest": {
            "run_id": "latest",
            "created_ns": 2,
            "source_archive": legacy_root.as_posix(),
            "output_dir": legacy_output_dir.as_posix(),
        },
        "runs": [
            {
                "run_id": "rebased",
                "created_ns": 1,
                "source_archive": legacy_root.as_posix(),
                "output_dir": legacy_output_dir.as_posix(),
            },
            {
                "run_id": "cleared",
                "created_ns": 0,
                "source_archive": (legacy_root / "missing-archive").as_posix(),
                "output_dir": (legacy_root / "missing-output").as_posix(),
            },
        ],
    }
    session_state = {
        "active_video_path": source_video.as_posix(),
        "active_frame_idx": 2,
    }
    write_archive(
        source_archive_path,
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

    migrate_legacy_archive(source_archive_path, workspace)

    snapshot_payload = read_workspace_snapshot_payload(current_project_snapshot_path(workspace))
    migrated_training = snapshot_payload["metadata"]["training_state_json"]
    migrated_session = snapshot_payload["session"]

    assert migrated_training["latest"]["source_archive"] == workspace.resolve().as_posix()
    assert migrated_training["latest"]["output_dir"] == workspace_output_dir.resolve().as_posix()
    assert migrated_training["runs"][0]["source_archive"] == workspace.resolve().as_posix()
    assert migrated_training["runs"][0]["output_dir"] == workspace_output_dir.resolve().as_posix()
    assert migrated_training["runs"][1]["source_archive"] == ""
    assert migrated_training["runs"][1]["output_dir"] == ""
    assert migrated_session["active_video_path"] == f"Media/{source_video.name}"
    assert migrated_session["active_frame_idx"] == 2


def test_pack_snapshot_and_unpack_roundtrip_workspace(tmp_path: Path) -> None:
    from xpkg.formats import (
        current_project_snapshot_path,
        pack_project,
        unpack_project,
        validate_artifact,
    )
    from xpkg.io.archive_format import write_archive
    from xpkg.model import Labels

    labels = _make_labels(tmp_path, x=5.0, y=6.0)
    source_archive_path = tmp_path / "tracking.xpkg"
    write_archive(source_archive_path, labels)

    workspace = tmp_path / "Roundtrip Project"
    from xpkg.formats import migrate_legacy_archive

    migrate_legacy_archive(source_archive_path, workspace)
    snapshot_path = current_project_snapshot_path(workspace)
    snapshot_doc = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot_doc["payload"]["data"]["keypoints"][0][0][0][0] = 51.0
    snapshot_doc["payload"]["data"]["keypoints"][0][0][0][1] = 61.0
    snapshot_path.write_text(json.dumps(snapshot_doc, indent=2) + "\n", encoding="utf-8")

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
    )
    from xpkg.io.archive_format import write_archive
    from xpkg.model import Labels

    source_root = tmp_path / "source"
    source_root.mkdir()
    labels = _make_labels(source_root, x=7.0, y=8.0)
    source_archive_path = tmp_path / "external.xpkg"
    write_archive(source_archive_path, labels)

    workspace = tmp_path / "Portable Project"
    migrate_legacy_archive(source_archive_path, workspace)
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


def test_legacy_workspace_current_state_archive_requires_explicit_cutover(tmp_path: Path) -> None:
    from xpkg.formats import (
        current_project_state_path,
        init_project,
        workspace_state_root,
        workspace_store_root,
    )
    from xpkg.io.archive_format import write_archive
    from xpkg.io.project_workspace import (
        LegacyWorkspaceMigrationRequiredError,
        export_project_archive,
    )
    from xpkg.model import Labels

    source_root = tmp_path / "source"
    source_root.mkdir()
    legacy_labels = _make_labels(source_root, x=9.0, y=10.0)
    updated_labels = _make_labels(source_root, x=11.0, y=12.0)

    workspace = tmp_path / "Legacy Workspace"
    init_project(workspace, title="Legacy Workspace")
    current_state_archive = workspace_state_root(workspace) / "current.xpkg"
    current_state_archive.parent.mkdir(parents=True, exist_ok=True)
    write_archive(current_state_archive, legacy_labels)

    with pytest.raises(
        LegacyWorkspaceMigrationRequiredError,
        match="migrate_legacy_archive",
    ):
        Labels.load_file(workspace.as_posix())

    with pytest.raises(
        LegacyWorkspaceMigrationRequiredError,
        match="migrate_legacy_archive",
    ):
        Labels.save_file(updated_labels, workspace.as_posix())

    with pytest.raises(
        LegacyWorkspaceMigrationRequiredError,
        match="migrate_legacy_archive",
    ):
        export_project_archive(workspace)

    assert current_project_state_path(workspace) == workspace_state_root(workspace) / "current.json"
    assert not current_project_state_path(workspace).exists()
    assert current_state_archive.exists()
    assert not (workspace_store_root(workspace) / "superblock.a.json").exists()


def test_labels_save_file_to_workspace_creates_first_committed_state(tmp_path: Path) -> None:
    from xpkg.formats import (
        current_project_snapshot_path,
        init_project,
        workspace_store_root,
    )
    from xpkg.io.archive_store import ArchiveStore
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
    assert not (workspace / ".xpkg" / "state" / "current.xpkg").exists()
    store = ArchiveStore.open(workspace_store_root(workspace))
    assert store.has_current_root("snapshot")
    assert not store.has_current_root("archive")
    assert labels.path == workspace

    loaded = Labels.load_file(workspace.as_posix())
    pts = loaded.labeled_frames[0].instances[0].get_points_array(copy=False, full=True)
    assert float(pts["x"][0]) == 11.0
    assert float(pts["y"][0]) == 12.0


def test_labels_save_file_to_workspace_preserves_predictions(tmp_path: Path) -> None:
    from xpkg.formats import (
        current_project_snapshot_path,
        migrate_legacy_archive,
    )
    from xpkg.io.archive_format import (
        PredictionAppendItem,
        SerializerPredictedInstance,
        write_archive,
    )
    from xpkg.io.workspace_snapshot_backend import read_workspace_snapshot_payload
    from xpkg.model import Labels

    source_root = tmp_path / "source"
    source_root.mkdir()
    initial_labels = _make_labels(source_root, x=1.0, y=2.0)
    updated_labels = _make_labels(source_root, x=21.0, y=22.0)
    source_archive_path = tmp_path / "with_predictions.xpkg"
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

    write_archive(source_archive_path, initial_labels, predictions=predictions)
    migrate_legacy_archive(source_archive_path, workspace)

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


def test_labels_save_file_to_workspace_seeds_predictions_from_labels_when_current_is_empty(
    tmp_path: Path,
) -> None:
    from xpkg.formats import current_project_snapshot_path, init_project
    from xpkg.io.workspace_snapshot_backend import read_workspace_snapshot_payload
    from xpkg.model import Labels

    source_root = tmp_path / "source"
    source_root.mkdir()
    labels_without_predictions = _make_labels(source_root, x=1.0, y=2.0)
    labels_with_predictions = _make_predicted_labels(
        source_root,
        x=13.0,
        y=14.0,
        track_id=7,
    )
    workspace = tmp_path / "Workspace Save"

    init_project(workspace, title="Workspace Save")
    Labels.save_file(labels_without_predictions, workspace.as_posix())
    Labels.save_file(labels_with_predictions, workspace.as_posix())
    payload = read_workspace_snapshot_payload(current_project_snapshot_path(workspace))

    assert int(payload["predictions"]["attrs"]["committed_length"]) == 1
    prediction_track_ids = np.asarray(payload["predictions"]["data"]["track_id"], dtype=np.int32)
    assert int(prediction_track_ids[0, 0]) == 7


def test_predictions_payload_from_labels_uses_frame_predicted_instances_view() -> None:
    from xpkg.io.workspace_snapshot_backend import predictions_payload_from_labels

    video = object()
    predicted = _ForeignPredictedInstance(
        x=10.0,
        y=20.0,
        point_score=0.9,
        instance_score=0.97,
        track_id=9,
    )
    labels = _ForeignLabels(
        videos=[video],
        labeled_frames=[
            _ForeignFrame(
                video=video,
                frame_idx=3,
                predicted_instances=[predicted],
                heatmaps=np.ones((1, 2, 2), dtype=np.float32),
            )
        ],
    )

    payload = predictions_payload_from_labels(labels)
    frame_index = np.asarray(payload["frames"]["frame_index"], dtype=np.int32)
    track_ids = np.asarray(payload["data"]["track_id"], dtype=np.int32)
    keypoints = np.asarray(payload["data"]["keypoints"], dtype=np.float32)
    heatmaps = np.asarray(payload["data"]["heatmaps"], dtype=np.float16)

    assert int(payload["attrs"]["committed_length"]) == 1
    assert int(frame_index[0]) == 3
    assert int(track_ids[0, 0]) == 9
    assert float(keypoints[0, 0, 0, 0]) == pytest.approx(10.0)
    assert float(keypoints[0, 0, 0, 1]) == pytest.approx(20.0)
    assert heatmaps.shape == (1, 1, 2, 2)


def test_workspace_load_rebuilds_tampered_snapshot_cache_when_commit_id_matches_head(
    tmp_path: Path,
) -> None:
    from xpkg.formats import current_project_snapshot_path, init_project
    from xpkg.model import Labels

    source_root = tmp_path / "source"
    source_root.mkdir()
    labels = _make_labels(source_root, x=11.0, y=12.0)
    workspace = tmp_path / "Snapshot Preferred"
    init_project(workspace, title="Snapshot Preferred")

    Labels.save_file(labels, workspace.as_posix())
    snapshot_path = current_project_snapshot_path(workspace)
    snapshot_doc = json.loads(snapshot_path.read_text(encoding="utf-8"))
    keypoints = snapshot_doc["payload"]["data"]["keypoints"]
    keypoints[0][0][0][0] = 101.0
    keypoints[0][0][0][1] = 102.0
    snapshot_path.write_text(json.dumps(snapshot_doc, indent=2) + "\n", encoding="utf-8")

    loaded = Labels.load_file(workspace.as_posix())
    pts = loaded.labeled_frames[0].instances[0].get_points_array(copy=False, full=True)
    repaired_snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))

    assert float(pts["x"][0]) == 11.0
    assert float(pts["y"][0]) == 12.0
    assert float(repaired_snapshot["payload"]["data"]["keypoints"][0][0][0][0]) == 11.0
    assert float(repaired_snapshot["payload"]["data"]["keypoints"][0][0][0][1]) == 12.0


def test_workspace_load_rebuilds_missing_snapshot_from_committed_state(tmp_path: Path) -> None:
    from xpkg.formats import current_project_snapshot_path, init_project
    from xpkg.model import Labels

    source_root = tmp_path / "source"
    source_root.mkdir()
    labels = _make_labels(source_root, x=21.0, y=22.0)
    workspace = tmp_path / "Rebuild Missing Snapshot"
    init_project(workspace, title="Rebuild Missing Snapshot")

    Labels.save_file(labels, workspace.as_posix())
    snapshot_path = current_project_snapshot_path(workspace)
    snapshot_path.unlink()

    loaded = Labels.load_file(workspace.as_posix())
    pts = loaded.labeled_frames[0].instances[0].get_points_array(copy=False, full=True)

    assert float(pts["x"][0]) == 21.0
    assert float(pts["y"][0]) == 22.0
    assert snapshot_path.exists()


def test_workspace_load_ignores_stale_snapshot_when_commit_id_mismatches(tmp_path: Path) -> None:
    from xpkg.formats import current_project_snapshot_path, init_project
    from xpkg.model import Labels

    source_root = tmp_path / "source"
    source_root.mkdir()
    initial_labels = _make_labels(source_root, x=11.0, y=12.0)
    updated_labels = _make_labels(source_root, x=31.0, y=32.0)
    workspace = tmp_path / "Stale Snapshot"
    init_project(workspace, title="Stale Snapshot")

    Labels.save_file(initial_labels, workspace.as_posix())
    snapshot_path = current_project_snapshot_path(workspace)
    stale_snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))

    Labels.save_file(updated_labels, workspace.as_posix())
    snapshot_path.write_text(json.dumps(stale_snapshot, indent=2) + "\n", encoding="utf-8")

    loaded = Labels.load_file(workspace.as_posix())
    pts = loaded.labeled_frames[0].instances[0].get_points_array(copy=False, full=True)
    assert float(pts["x"][0]) == 31.0
    assert float(pts["y"][0]) == 32.0


def test_summarize_project_and_validate_project_read_labels_video_group(tmp_path: Path) -> None:
    from xpkg.formats import current_project_snapshot_path, init_project
    from xpkg.io.archive_format import summarize_project, validate_project
    from xpkg.io.project_workspace import export_project_archive
    from xpkg.model import Labels

    source_video = tmp_path / "source.avi"
    _write_test_video(source_video)
    labels = _make_media_labels(source_video, x=4.0, y=5.0)
    workspace = tmp_path / "Summary Project"
    init_project(workspace, title="Summary Project")

    saved_target = Labels.save_file(labels, workspace.as_posix())
    archive = export_project_archive(workspace)
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


def test_workspace_metadata_field_helpers_roundtrip_current_head(tmp_path: Path) -> None:
    from xpkg.formats import (
        init_project,
        load_workspace_metadata,
        load_workspace_metadata_field,
        save_workspace_labels,
        save_workspace_metadata_field,
    )

    workspace = tmp_path / "Field Metadata Project"
    init_project(workspace, title="Field Metadata Project")
    save_workspace_labels(workspace, _make_labels(tmp_path, x=1.0, y=2.0))

    saved_path = save_workspace_metadata_field(
        workspace,
        "session_json",
        {"active_frame_idx": 7},
        reason="test.workspace_metadata_field",
    )

    assert saved_path.is_file()
    assert load_workspace_metadata_field(workspace, "session_json") == {"active_frame_idx": 7}
    assert load_workspace_metadata(workspace)["session_json"] == {"active_frame_idx": 7}


def test_inspect_workspace_reports_current_head_summary_and_metadata(tmp_path: Path) -> None:
    from xpkg.formats import (
        WorkspaceInspection,
        init_project,
        inspect_workspace,
        save_workspace_labels,
    )
    from xpkg.services import WorkspaceService

    workspace = tmp_path / "Inspection Project"
    training_state = {
        "schema_version": 1,
        "latest": {
            "run_id": "run_1",
            "created_ns": 1,
        },
        "runs": [],
    }

    init_project(workspace, title="Inspection Project")
    save_workspace_labels(
        workspace,
        _make_labels(tmp_path, x=3.0, y=4.0),
        metadata={
            "project_name": "Inspection Project",
            "training_state_json": training_state,
        },
    )

    inspection = inspect_workspace(workspace)
    service_inspection = WorkspaceService.open(workspace).inspect()

    assert isinstance(inspection, WorkspaceInspection)
    assert inspection.current_state_path.exists()
    assert inspection.state_kind == "labels"
    assert inspection.summary is not None
    assert inspection.summary.label_frames == 1
    assert inspection.summary.prediction_frames == 0
    assert inspection.metadata["project_name"] == "Inspection Project"
    assert inspection.metadata["training_state_json"] == training_state
    assert inspection.is_valid is True
    assert service_inspection == inspection
