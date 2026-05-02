from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

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


def _read_expkg_manifest(artifact: Path) -> dict[str, Any]:
    from xpkg.workspace import EXPKG_MANIFEST_FILENAME

    with zipfile.ZipFile(artifact) as archive:
        raw = archive.read(EXPKG_MANIFEST_FILENAME).decode("utf-8")
    return json.loads(raw)


def _archive_member_sha256(archive: zipfile.ZipFile, member: str) -> str:
    digest = hashlib.sha256()
    with archive.open(member, mode="r") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _make_labels(tmp_path: Path, *, x: float, y: float):
    from xpkg.model import Labels, build_keypoint_skeleton
    from xpkg.pose.annotations import Instance, LabeledFrame, Point

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
    from xpkg.model import Labels, Video, build_keypoint_skeleton
    from xpkg.pose.annotations import Instance, LabeledFrame, Point

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
    from xpkg.model import Labels, build_keypoint_skeleton
    from xpkg.pose.annotations import LabeledFrame, PredictedInstance, PredictedPoint, Track

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
    from xpkg.model import Labels
    from xpkg.pose.skeleton import Skeleton

    return Labels(
        skeletons=[Skeleton(name="unconfigured", keypoints=[], links_ids=[])],
        keypoints=[],
    )


def test_init_project_writes_workspace_contract(tmp_path: Path) -> None:
    from xpkg.model import Labels
    from xpkg.workspace import (
        current_project_state_path,
        init_project,
        is_workspace_root,
        load_project_descriptor,
    )

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


def test_workspace_metadata_helpers_roundtrip_without_existing_labels(tmp_path: Path) -> None:
    from xpkg.workspace import (
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
                "id": "predictions",
                "label": "predictions.json",
                "path": "predictions/predictions.json",
                "asset_type": "predictions",
                "metadata": {"role": "predictions"},
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
    from xpkg.workspace import (
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
    from xpkg.model import Labels
    from xpkg.workspace import (
        init_project,
        load_workspace_payload,
        save_workspace_labels,
    )

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
    from xpkg.model import Labels
    from xpkg.workspace import init_project, load_workspace_payload

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


def test_load_workspace_payload_uses_snapshot_payload(tmp_path: Path) -> None:
    from xpkg.model import Labels
    from xpkg.workspace import init_project, load_workspace_payload

    workspace = tmp_path / "Direct Payload Project"
    init_project(workspace, title="Direct Payload Project")
    Labels.save_file(_make_labels(tmp_path, x=4.0, y=5.0), workspace.as_posix())

    payload = load_workspace_payload(workspace)

    assert payload["labels"]["frames"]["frame_index"].tolist() == [0]
    assert payload["labels"]["data"]["track_id"].shape == (1, 1)
    assert payload["predictions"]["attrs"]["committed_length"] == 0


def test_pack_snapshot_and_unpack_roundtrip_workspace(tmp_path: Path) -> None:
    from xpkg.model import Labels
    from xpkg.workspace import (
        current_project_snapshot_path,
        init_project,
        pack_project,
        unpack_project,
        validate_artifact,
    )

    labels = _make_labels(tmp_path, x=5.0, y=6.0)

    workspace = tmp_path / "Roundtrip Project"
    init_project(workspace, title="Roundtrip Project")
    Labels.save_file(labels, workspace.as_posix())
    snapshot_path = current_project_snapshot_path(workspace)
    snapshot_doc = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot_doc["payload"]["data"]["keypoints"][0][0][0][0] = 51.0
    snapshot_doc["payload"]["data"]["keypoints"][0][0][0][1] = 61.0
    snapshot_path.write_text(json.dumps(snapshot_doc, indent=2) + "\n", encoding="utf-8")

    artifact = pack_project(workspace, mode="snapshot", media_policy="include")
    validate_artifact(artifact)
    manifest = _read_expkg_manifest(artifact)
    with zipfile.ZipFile(artifact) as archive:
        archive_names = set(archive.namelist())

    assert manifest["pack_mode"] == "snapshot"
    assert manifest["media_policy"] == "include"
    assert manifest["media"]["included_files"] == 1
    assert any(name.startswith("Media/") for name in archive_names)

    unpacked = tmp_path / "Unpacked Project"
    unpack_project(artifact, unpacked)

    loaded = Labels.load_file(unpacked.as_posix())
    pts = loaded.labeled_frames[0].instances[0].get_points_array(copy=False, full=True)
    assert float(pts["x"][0]) == 5.0
    assert float(pts["y"][0]) == 6.0


def test_pack_snapshot_defaults_to_manifest_media_without_bundling(tmp_path: Path) -> None:
    from xpkg.model import Labels
    from xpkg.workspace import init_project, pack_project, validate_artifact

    labels = _make_labels(tmp_path, x=6.0, y=7.0)
    workspace = tmp_path / "Snapshot Manifest Project"
    init_project(workspace, title="Snapshot Manifest Project")
    Labels.save_file(labels, workspace.as_posix())

    artifact = pack_project(workspace, mode="snapshot")
    validate_artifact(artifact)
    manifest = _read_expkg_manifest(artifact)
    with zipfile.ZipFile(artifact) as archive:
        archive_names = set(archive.namelist())

    assert manifest["pack_mode"] == "snapshot"
    assert manifest["media_policy"] == "manifest"
    assert manifest["media"]["external_files"] == 1
    assert manifest["media"]["external_bytes"] > 0
    assert not any(name.startswith("Media/") for name in archive_names)


def test_pack_portable_and_unpack_uses_managed_media_after_source_removal(tmp_path: Path) -> None:
    from xpkg.model import Labels
    from xpkg.workspace import (
        init_project,
        pack_project,
        unpack_project,
        validate_artifact,
        workspace_media_root,
    )

    source_root = tmp_path / "source"
    source_root.mkdir()
    labels = _make_labels(source_root, x=7.0, y=8.0)

    workspace = tmp_path / "Portable Project"
    init_project(workspace, title="Portable Project")
    Labels.save_file(labels, workspace.as_posix())
    managed_files = sorted(
        path for path in workspace_media_root(workspace).rglob("*") if path.is_file()
    )
    assert managed_files

    artifact = tmp_path / "Portable Project.expkg"
    pack_project(workspace, mode="portable", out=artifact)
    validate_artifact(artifact)
    manifest = _read_expkg_manifest(artifact)
    with zipfile.ZipFile(artifact) as archive:
        archive_names = set(archive.namelist())
        for entry in manifest["members"]:
            assert _archive_member_sha256(archive, entry["path"]) == entry["sha256"]

    assert manifest["format"] == "xpkg-packed-project"
    assert manifest["container"] == "zip"
    assert manifest["pack_mode"] == "portable"
    assert manifest["media_policy"] == "include"
    assert manifest["media"]["included_files"] == len(managed_files)
    assert all(name.startswith("Media/") for name in archive_names if name.startswith("Media/"))
    assert any(name.startswith("Media/") for name in archive_names)

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


def test_pack_portable_rejects_nonportable_media_policy(tmp_path: Path) -> None:
    from xpkg.model import Labels
    from xpkg.workspace import init_project, pack_project

    labels = _make_labels(tmp_path, x=9.0, y=10.0)
    workspace = tmp_path / "Portable Policy Project"
    init_project(workspace, title="Portable Policy Project")
    Labels.save_file(labels, workspace.as_posix())

    with pytest.raises(ValueError, match="Portable pack requires media_policy='include'"):
        pack_project(
            workspace,
            mode="portable",
            media_policy="manifest",
            out=tmp_path / "bad.expkg",
        )


def test_validate_expkg_rejects_tampered_member_payload(tmp_path: Path) -> None:
    from xpkg.model import Labels
    from xpkg.workspace import init_project, pack_project, validate_expkg

    labels = _make_labels(tmp_path, x=11.0, y=12.0)
    workspace = tmp_path / "Tamper Project"
    init_project(workspace, title="Tamper Project")
    Labels.save_file(labels, workspace.as_posix())

    artifact = pack_project(workspace, mode="portable", out=tmp_path / "good.expkg")
    tampered = tmp_path / "tampered.expkg"
    with zipfile.ZipFile(artifact) as source, zipfile.ZipFile(tampered, mode="w") as dest:
        for info in source.infolist():
            data = source.read(info.filename)
            if info.filename == "PROJECT.json":
                data = data.replace(b"Tamper Project", b"Mutant Project")
            dest.writestr(info, data)

    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_expkg(tampered)


def test_labels_save_file_to_workspace_creates_first_committed_state(tmp_path: Path) -> None:
    from xpkg.io.archive_store import ArchiveStore
    from xpkg.model import Labels
    from xpkg.workspace import (
        current_project_snapshot_path,
        init_project,
        workspace_store_root,
    )

    source_root = tmp_path / "source"
    source_root.mkdir()
    labels = _make_labels(source_root, x=11.0, y=12.0)

    workspace = tmp_path / "Saved Workspace"
    init_project(workspace, title="Saved Workspace")

    saved_target = Labels.save_file(labels, workspace.as_posix())
    current_snapshot = current_project_snapshot_path(workspace)

    assert saved_target == workspace.as_posix()
    assert current_snapshot.exists()
    store = ArchiveStore.open(workspace_store_root(workspace))
    assert store.has_current_root("snapshot")
    assert not store.has_current_root("archive")
    assert labels.path == workspace

    loaded = Labels.load_file(workspace.as_posix())
    pts = loaded.labeled_frames[0].instances[0].get_points_array(copy=False, full=True)
    assert float(pts["x"][0]) == 11.0
    assert float(pts["y"][0]) == 12.0


def test_labels_save_file_to_workspace_preserves_predictions(tmp_path: Path) -> None:
    from xpkg.io.workspace_snapshot_backend import read_workspace_snapshot_payload
    from xpkg.model import Labels
    from xpkg.workspace import (
        current_project_snapshot_path,
        init_project,
    )

    source_root = tmp_path / "source"
    source_root.mkdir()
    initial_labels = _make_predicted_labels(
        source_root,
        x=13.0,
        y=14.0,
        point_score=0.9,
        instance_score=0.97,
        track_id=4,
    )
    updated_labels = _make_labels(source_root, x=21.0, y=22.0)
    workspace = tmp_path / "Workspace Save"

    init_project(workspace, title="Workspace Save")
    Labels.save_file(initial_labels, workspace.as_posix())

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
    from xpkg.io.workspace_snapshot_backend import read_workspace_snapshot_payload
    from xpkg.model import Labels
    from xpkg.workspace import current_project_snapshot_path, init_project

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
    from xpkg.model import Labels
    from xpkg.workspace import current_project_snapshot_path, init_project

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
    from xpkg.model import Labels
    from xpkg.workspace import current_project_snapshot_path, init_project

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
    from xpkg.model import Labels
    from xpkg.workspace import current_project_snapshot_path, init_project

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


def test_summarize_loaded_project_and_validate_loaded_project_read_labels_video_group(
    tmp_path: Path,
) -> None:
    from xpkg.io.project_validation import summarize_loaded_project, validate_loaded_project
    from xpkg.model import Labels
    from xpkg.workspace import current_project_snapshot_path, init_project, load_workspace_payload

    source_video = tmp_path / "source.avi"
    _write_test_video(source_video)
    labels = _make_media_labels(source_video, x=4.0, y=5.0)
    workspace = tmp_path / "Summary Project"
    init_project(workspace, title="Summary Project")

    saved_target = Labels.save_file(labels, workspace.as_posix())
    snapshot = current_project_snapshot_path(workspace)
    payload = load_workspace_payload(workspace)
    summary = summarize_loaded_project(payload, path=snapshot)

    assert saved_target == workspace.as_posix()
    assert snapshot.exists()
    validate_loaded_project(payload)
    assert summary.n_videos == 1
    assert len(summary.video_filenames) == 1
    assert Path(summary.video_filenames[0]).name == source_video.name
    assert summary.video_shapes == (1, 4)
    assert summary.label_frames == 1
    assert summary.prediction_frames == 0


def test_workspace_metadata_field_helpers_roundtrip_current_head(tmp_path: Path) -> None:
    from xpkg.workspace import (
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
    from xpkg.services import WorkspaceService
    from xpkg.workspace import (
        WorkspaceInspection,
        init_project,
        inspect_workspace,
        save_workspace_labels,
    )

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
