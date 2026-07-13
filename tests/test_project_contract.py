from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.factories import (
    make_labels,
    make_single_frame_video,
)


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

    def point_records(self, *, copy: bool) -> np.ndarray:
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


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _read_expkg_manifest(artifact: Path) -> dict[str, Any]:
    from xpkg.project.artifact import EXPKG_MANIFEST_FILENAME

    with zipfile.ZipFile(artifact) as archive:
        raw = archive.read(EXPKG_MANIFEST_FILENAME).decode("utf-8")
    return json.loads(raw)


def _archive_member_sha256(archive: zipfile.ZipFile, member: str) -> str:
    digest = hashlib.sha256()
    with archive.open(member, mode="r") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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

    _, video = make_single_frame_video(tmp_path)
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


def test_init_project_writes_project_contract(tmp_path: Path) -> None:
    from xpkg.model import Labels
    from xpkg.project import (
        current_project_state_path,
        init_project,
        is_project_root,
        load_project_descriptor,
    )

    project = tmp_path / "My Project"
    descriptor = init_project(project, title="My Project")

    assert is_project_root(project)
    assert (project / "PROJECT.json").is_file()
    assert (project / ".xpkg").is_dir()
    assert (project / "Media").is_dir()
    assert (project / "Exports").is_dir()
    assert not current_project_state_path(project).exists()
    assert load_project_descriptor(project).title == "My Project"
    assert descriptor.to_dict()["exports_root"] == "Exports"

    loaded = Labels.load_file(project.as_posix())
    assert loaded.labeled_frames == []


def test_ensure_project_adopts_nonempty_directory(tmp_path: Path) -> None:
    from xpkg.project import ensure_project, is_project_root, load_project_descriptor

    project = tmp_path / "Existing Experiment"
    project.mkdir()
    source = project / "recording.csv"
    source.write_text("time,signal\n0.0,1.0\n", encoding="utf-8")

    descriptor = ensure_project(project, title="Existing Experiment")

    assert is_project_root(project)
    assert source.read_text(encoding="utf-8") == "time,signal\n0.0,1.0\n"
    assert (project / "PROJECT.json").is_file()
    assert (project / ".xpkg").is_dir()
    assert (project / "Media").is_dir()
    assert (project / "Exports").is_dir()
    assert descriptor.title == "Existing Experiment"
    assert load_project_descriptor(project).project_id == descriptor.project_id


def test_ensure_project_preserves_existing_descriptor(tmp_path: Path) -> None:
    from xpkg.project import ensure_project, init_project, load_project_descriptor

    project = tmp_path / "Existing Project"
    init_project(project, title="Original Title")
    original = load_project_descriptor(project)

    descriptor = ensure_project(project, title="Replacement Title")

    assert descriptor.to_dict() == original.to_dict()
    assert load_project_descriptor(project).to_dict() == original.to_dict()


def test_project_descriptor_rejects_unsupported_fields(tmp_path: Path) -> None:
    from xpkg.project import init_project, load_project_descriptor

    project = tmp_path / "Strict Descriptor"
    init_project(project, title="Strict Descriptor")
    descriptor_path = project / "PROJECT.json"
    payload = json.loads(descriptor_path.read_text(encoding="utf-8"))
    payload["default_pack_mode"] = "portable"
    descriptor_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported field.*default_pack_mode"):
        load_project_descriptor(project)


def test_project_descriptor_rejects_invalid_project_id() -> None:
    from xpkg.project import ProjectDescriptor

    payload = ProjectDescriptor.new(title="Strict Descriptor").to_dict()
    payload["project_id"] = "project-123"

    with pytest.raises(ValueError, match="project_id must be a UUID or ULID"):
        ProjectDescriptor.from_dict(payload)


def test_project_metadata_helpers_roundtrip_without_existing_labels(tmp_path: Path) -> None:
    from xpkg.project import (
        init_project,
        load_project_metadata,
        save_project_labels,
        save_project_metadata,
    )

    project = tmp_path / "Metadata Project"
    init_project(project, title="Metadata Project")

    assert load_project_metadata(project) == {}

    labels = make_labels(tmp_path, x=1.5, y=2.5)
    save_project_labels(project, labels)
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

    saved_path = save_project_metadata(
        project,
        {
            "manifest_json": manifest_payload,
            "session_json": {"active_frame_idx": 7},
        },
        reason="test.metadata",
    )

    assert saved_path.is_file()
    assert load_project_metadata(project) == {
        "manifest_json": manifest_payload,
        "session_json": {"active_frame_idx": 7},
    }
    from xpkg.project import load_project_experiment

    experiment = load_project_experiment(project)
    assert experiment.metadata["manifest_json"] == manifest_payload
    assert experiment.metadata["session_json"]["active_frame_idx"] == 7
    resolved_path = Path(experiment.sessions[0].pose().videos[0].filename)
    assert resolved_path.is_absolute()
    assert resolved_path.exists()


def test_save_project_metadata_commits_state_without_labels_recommit(
    tmp_path: Path,
) -> None:
    from xpkg.project import (
        init_project,
        load_project_labels,
        load_project_metadata,
        save_project_labels,
        save_project_metadata,
    )

    project = tmp_path / "Metadata Fast Path"
    init_project(project, title="Metadata Fast Path")
    save_project_labels(project, make_labels(tmp_path, x=1.0, y=2.0))

    saved_path = save_project_metadata(
        project,
        {"session_json": {"active_frame_idx": 12}},
        reason="test.metadata.fast_path",
    )

    assert saved_path.is_file()
    assert load_project_metadata(project)["session_json"] == {"active_frame_idx": 12}
    assert load_project_labels(project).user_instances[0]["nose"].x == pytest.approx(1.0)


def test_empty_placeholder_project_loads(tmp_path: Path) -> None:
    from xpkg.model import Labels
    from xpkg.project import (
        init_project,
        save_project_labels,
    )

    project = tmp_path / "Placeholder Project"
    init_project(project, title="Placeholder Project")
    save_project_labels(
        project,
        _make_unconfigured_labels(),
        metadata={"project_name": "Placeholder Project"},
    )

    loaded = Labels.load_file(project.as_posix())
    assert loaded.labeled_frames == []
    assert len(loaded.skeletons) == 1
    assert loaded.skeletons[0].name == "unconfigured"



def test_session_pose_roundtrip_preserves_predicted_instances(tmp_path: Path) -> None:
    from xpkg.model import Labels
    from xpkg.project import init_project

    project = tmp_path / "Prediction Project"
    init_project(project, title="Prediction Project")
    labels = _make_predicted_labels(
        tmp_path,
        x=10.0,
        y=20.0,
        track_id=7,
        heatmaps=np.ones((1, 2, 2), dtype=np.float32),
    )
    Labels.save_file(labels, project.as_posix())

    restored = Labels.load_file(project.as_posix())

    assert restored.user_instances == []
    assert len(restored.predicted_instances) == 1
    assert restored.predicted_instances[0].track is not None
    assert restored.predicted_instances[0].track.id == 7
    assert restored.predicted_instances[0]["nose"].score == pytest.approx(0.9)


def test_pose_replacement_is_exact_and_does_not_preserve_removed_predictions(
    tmp_path: Path,
) -> None:
    from xpkg.model import Labels
    from xpkg.project import init_project

    project = tmp_path / "Prediction Label Frame Project"
    init_project(project, title="Prediction Label Frame Project")
    Labels.save_file(
        _make_predicted_labels(tmp_path, x=10.0, y=20.0, track_id=-1),
        project.as_posix(),
    )
    Labels.save_file(make_labels(tmp_path, x=11.0, y=22.0), project.as_posix())

    restored = Labels.load_file(project.as_posix())

    assert len(restored.user_instances) == 1
    assert restored.user_instances[0]["nose"].x == pytest.approx(11.0)
    assert restored.user_instances[0]["nose"].y == pytest.approx(22.0)
    assert restored.predicted_instances == []


def test_project_session_exposes_typed_pose_state(tmp_path: Path) -> None:
    from xpkg.model import Labels
    from xpkg.project import init_project, load_project_session

    project = tmp_path / "Direct Payload Project"
    init_project(project, title="Direct Payload Project")
    Labels.save_file(make_labels(tmp_path, x=4.0, y=5.0), project.as_posix())

    session = load_project_session(project)
    pose = session.pose()

    assert pose.labeled_frames[0].frame_idx == 0
    assert pose.user_instances[0]["nose"].x == pytest.approx(4.0)
    assert session.modality_names == ("videos", "pose")


def test_pack_and_unpack_roundtrip_project(tmp_path: Path) -> None:
    from xpkg.model import Labels
    from xpkg.project import (
        current_project_state_path,
        init_project,
        pack_project,
        unpack_project,
        validate_artifact,
    )

    labels = make_labels(tmp_path, x=5.0, y=6.0)

    project = tmp_path / "Roundtrip Project"
    init_project(project, title="Roundtrip Project")
    Labels.save_file(labels, project.as_posix())
    state_path = current_project_state_path(project)
    state_doc = json.loads(state_path.read_text(encoding="utf-8"))
    pose_payload = state_doc["payload"]["experiment"]["sessions"][0]["session"][
        "payload"
        ]["session"]["poses"][0]["data"]
    pose_payload["data"]["keypoints"][0][0][0][0] = 51.0
    pose_payload["data"]["keypoints"][0][0][0][1] = 61.0
    state_path.write_text(json.dumps(state_doc, indent=2) + "\n", encoding="utf-8")

    artifact = pack_project(project)
    validate_artifact(artifact)
    manifest = _read_expkg_manifest(artifact)
    with zipfile.ZipFile(artifact) as archive:
        archive_names = set(archive.namelist())

    assert manifest["media"]["mode"] == "full"
    assert manifest["media"]["included_files"] == 1
    assert manifest["media"]["external_files"] == 0
    assert manifest["media"]["files"][0]["included"] is True
    assert any(name.startswith("Media/") for name in archive_names)

    unpacked = tmp_path / "Unpacked Project"
    unpack_project(artifact, unpacked)

    loaded = Labels.load_file(unpacked.as_posix())
    pts = loaded.labeled_frames[0].instances[0].point_records(copy=False)
    assert float(pts["x"][0]) == 5.0
    assert float(pts["y"][0]) == 6.0


def test_pack_portable_and_unpack_uses_managed_media_after_source_removal(tmp_path: Path) -> None:
    from xpkg.model import Labels
    from xpkg.project import (
        init_project,
        pack_project,
        unpack_project,
        validate_artifact,
    )
    from xpkg.project.layout import project_media_root

    source_root = tmp_path / "source"
    source_root.mkdir()
    labels = make_labels(source_root, x=7.0, y=8.0)

    project = tmp_path / "Portable Project"
    init_project(project, title="Portable Project")
    Labels.save_file(labels, project.as_posix())
    managed_files = sorted(
        path for path in project_media_root(project).rglob("*") if path.is_file()
    )
    assert managed_files

    artifact = tmp_path / "Portable Project.expkg"
    pack_project(project, out=artifact)
    validate_artifact(artifact)
    manifest = _read_expkg_manifest(artifact)
    with zipfile.ZipFile(artifact) as archive:
        archive_names = set(archive.namelist())
        for entry in manifest["members"]:
            assert _archive_member_sha256(archive, entry["path"]) == entry["sha256"]

    assert manifest["format"] == "xpkg-packed-project"
    assert manifest["container"] == "zip"
    assert manifest["media"]["mode"] == "full"
    assert manifest["media"]["included_files"] == len(managed_files)
    assert manifest["media"]["external_files"] == 0
    assert all(entry["included"] is True for entry in manifest["media"]["files"])
    assert all(name.startswith("Media/") for name in archive_names if name.startswith("Media/"))
    assert any(name.startswith("Media/") for name in archive_names)

    shutil.rmtree(source_root)
    shutil.rmtree(project)

    unpacked = tmp_path / "Unpacked Portable Project"
    unpack_project(artifact, unpacked)
    validate_artifact(unpacked)

    loaded = Labels.load_file(unpacked.as_posix())
    pts = loaded.labeled_frames[0].instances[0].point_records(copy=False)
    assert float(pts["x"][0]) == 7.0
    assert float(pts["y"][0]) == 8.0

    unpacked_media_root = project_media_root(unpacked)
    for video in loaded.videos:
        assert _is_within(Path(str(video.filename)), unpacked_media_root)
        for frame_path in video.image_filenames or []:
            resolved = Path(str(frame_path))
            assert _is_within(resolved, unpacked_media_root)
            assert resolved.exists()


def test_pack_manifest_media_mode_records_media_without_storing_bytes(tmp_path: Path) -> None:
    from xpkg.model import Labels
    from xpkg.project import init_project, pack_project, unpack_project
    from xpkg.project.layout import project_media_root

    source_root = tmp_path / "source"
    source_root.mkdir()
    labels = make_labels(source_root, x=9.0, y=10.0)
    project = tmp_path / "Manifest Media Project"
    init_project(project, title="Manifest Media Project")
    Labels.save_file(labels, project.as_posix())

    artifact = pack_project(project, out=tmp_path / "manifest.expkg", media="manifest")
    manifest = _read_expkg_manifest(artifact)
    with zipfile.ZipFile(artifact) as archive:
        archive_names = set(archive.namelist())

    media_files = manifest["media"]["files"]
    assert manifest["media"]["mode"] == "manifest"
    assert manifest["media"]["included_files"] == 0
    assert manifest["media"]["external_files"] == len(media_files)
    assert all(entry["included"] is False for entry in media_files)
    assert all(entry["compression"] == "none" for entry in media_files)
    assert not any(name.startswith("Media/") for name in archive_names)

    unpacked = tmp_path / "Unpacked Manifest Media Project"
    unpack_project(artifact, unpacked)
    assert list(project_media_root(unpacked).rglob("*")) == []


def test_pack_package_media_mode_includes_images_and_manifests_videos(
    tmp_path: Path,
) -> None:
    from xpkg.project import init_project, pack_project, validate_expkg
    from xpkg.project.layout import project_media_root

    project = tmp_path / "Package Media Project"
    init_project(project, title="Package Media Project")
    media_root = project_media_root(project)
    image_path = media_root / "frames" / "frame_000.png"
    video_path = media_root / "raw" / "trial.mp4"
    image_path.parent.mkdir(parents=True)
    video_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"image-bytes")
    video_path.write_bytes(b"video-bytes")

    artifact = pack_project(project, out=tmp_path / "package.expkg", media="package")
    validate_expkg(artifact)
    manifest = _read_expkg_manifest(artifact)
    with zipfile.ZipFile(artifact) as archive:
        archive_names = set(archive.namelist())

    entries = {entry["path"]: entry for entry in manifest["media"]["files"]}
    assert manifest["media"]["mode"] == "package"
    assert manifest["media"]["included_files"] == 1
    assert manifest["media"]["external_files"] == 1
    assert entries["Media/frames/frame_000.png"]["included"] is True
    assert entries["Media/raw/trial.mp4"]["included"] is False
    assert "Media/frames/frame_000.png" in archive_names
    assert "Media/raw/trial.mp4" not in archive_names


def test_validate_expkg_rejects_tampered_member_payload(tmp_path: Path) -> None:
    from xpkg.model import Labels
    from xpkg.project import init_project, pack_project, validate_expkg

    labels = make_labels(tmp_path, x=11.0, y=12.0)
    project = tmp_path / "Tamper Project"
    init_project(project, title="Tamper Project")
    Labels.save_file(labels, project.as_posix())

    artifact = pack_project(project, out=tmp_path / "good.expkg")
    tampered = tmp_path / "tampered.expkg"
    with zipfile.ZipFile(artifact) as source, zipfile.ZipFile(tampered, mode="w") as dest:
        for info in source.infolist():
            data = source.read(info.filename)
            if info.filename == "PROJECT.json":
                data = data.replace(b"Tamper Project", b"Mutant Project")
            dest.writestr(info, data)

    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_expkg(tampered)


def test_labels_save_file_to_project_creates_first_committed_state(tmp_path: Path) -> None:
    from xpkg.model import Labels
    from xpkg.project import current_project_state_path, init_project
    from xpkg.project.durable_store import ProjectDurableStore
    from xpkg.project.layout import project_store_root

    source_root = tmp_path / "source"
    source_root.mkdir()
    labels = make_labels(source_root, x=11.0, y=12.0)

    project = tmp_path / "Saved Project"
    init_project(project, title="Saved Project")

    saved_target = Labels.save_file(labels, project.as_posix())
    current_state = current_project_state_path(project)

    assert saved_target == project.as_posix()
    assert current_state.exists()
    store = ProjectDurableStore.open(project_store_root(project))
    assert store.has_current_root("state")
    assert not store.has_current_root("archive")
    assert labels.path == project

    loaded = Labels.load_file(project.as_posix())
    pts = loaded.labeled_frames[0].instances[0].point_records(copy=False)
    assert float(pts["x"][0]) == 11.0
    assert float(pts["y"][0]) == 12.0


def test_predictions_payload_from_labels_uses_frame_predicted_instances_view() -> None:
    from xpkg.io.pose_json import predictions_payload_from_labels

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


def test_project_load_rebuilds_tampered_state_cache_when_commit_id_matches_head(
    tmp_path: Path,
) -> None:
    from xpkg.model import Labels
    from xpkg.project import current_project_state_path, init_project

    source_root = tmp_path / "source"
    source_root.mkdir()
    labels = make_labels(source_root, x=11.0, y=12.0)
    project = tmp_path / "State Preferred"
    init_project(project, title="State Preferred")

    Labels.save_file(labels, project.as_posix())
    state_path = current_project_state_path(project)
    state_doc = json.loads(state_path.read_text(encoding="utf-8"))
    pose_payload = state_doc["payload"]["experiment"]["sessions"][0]["session"][
        "payload"
        ]["session"]["poses"][0]["data"]
    keypoints = pose_payload["data"]["keypoints"]
    keypoints[0][0][0][0] = 101.0
    keypoints[0][0][0][1] = 102.0
    state_path.write_text(json.dumps(state_doc, indent=2) + "\n", encoding="utf-8")

    loaded = Labels.load_file(project.as_posix())
    pts = loaded.labeled_frames[0].instances[0].point_records(copy=False)
    repaired_state = json.loads(state_path.read_text(encoding="utf-8"))

    assert float(pts["x"][0]) == 11.0
    assert float(pts["y"][0]) == 12.0
    repaired_pose = repaired_state["payload"]["experiment"]["sessions"][0]["session"][
        "payload"
    ]["session"]["poses"][0]["data"]["data"]
    assert float(repaired_pose["keypoints"][0][0][0][0]) == 11.0
    assert float(repaired_pose["keypoints"][0][0][0][1]) == 12.0


def test_project_load_rebuilds_missing_state_from_committed_state(tmp_path: Path) -> None:
    from xpkg.model import Labels
    from xpkg.project import current_project_state_path, init_project

    source_root = tmp_path / "source"
    source_root.mkdir()
    labels = make_labels(source_root, x=21.0, y=22.0)
    project = tmp_path / "Rebuild Missing State"
    init_project(project, title="Rebuild Missing State")

    Labels.save_file(labels, project.as_posix())
    state_path = current_project_state_path(project)
    state_path.unlink()

    loaded = Labels.load_file(project.as_posix())
    pts = loaded.labeled_frames[0].instances[0].point_records(copy=False)

    assert float(pts["x"][0]) == 21.0
    assert float(pts["y"][0]) == 22.0
    assert state_path.exists()


def test_project_load_ignores_stale_state_when_commit_id_mismatches(tmp_path: Path) -> None:
    from xpkg.model import Labels
    from xpkg.project import current_project_state_path, init_project

    source_root = tmp_path / "source"
    source_root.mkdir()
    initial_labels = make_labels(source_root, x=11.0, y=12.0)
    updated_labels = make_labels(source_root, x=31.0, y=32.0)
    project = tmp_path / "Stale State"
    init_project(project, title="Stale State")

    Labels.save_file(initial_labels, project.as_posix())
    state_path = current_project_state_path(project)
    stale_state = json.loads(state_path.read_text(encoding="utf-8"))

    Labels.save_file(updated_labels, project.as_posix())
    state_path.write_text(json.dumps(stale_state, indent=2) + "\n", encoding="utf-8")

    loaded = Labels.load_file(project.as_posix())
    pts = loaded.labeled_frames[0].instances[0].point_records(copy=False)
    assert float(pts["x"][0]) == 31.0
    assert float(pts["y"][0]) == 32.0


def test_project_metadata_field_helpers_roundtrip_current_head(tmp_path: Path) -> None:
    from xpkg.project import (
        init_project,
        load_project_metadata,
        load_project_metadata_field,
        save_project_labels,
        save_project_metadata_field,
    )

    project = tmp_path / "Field Metadata Project"
    init_project(project, title="Field Metadata Project")
    save_project_labels(project, make_labels(tmp_path, x=1.0, y=2.0))

    saved_path = save_project_metadata_field(
        project,
        "session_json",
        {"active_frame_idx": 7},
        reason="test.project_metadata_field",
    )

    assert saved_path.is_file()
    assert load_project_metadata_field(project, "session_json") == {"active_frame_idx": 7}
    assert load_project_metadata(project)["session_json"] == {"active_frame_idx": 7}


def test_inspect_project_reports_shallow_current_head_summary(tmp_path: Path) -> None:
    from xpkg.project import (
        ProjectInspection,
        init_project,
        inspect_project,
        save_project_labels,
    )
    from xpkg.services import ProjectService

    project = tmp_path / "Inspection Project"
    training_state = {
        "schema_version": 1,
        "latest": {
            "run_id": "run_1",
            "created_ns": 1,
        },
        "runs": [],
    }

    init_project(project, title="Inspection Project")
    save_project_labels(
        project,
        make_labels(tmp_path, x=3.0, y=4.0),
        metadata={
            "project_name": "Inspection Project",
            "training_state_json": training_state,
        },
    )

    inspection = inspect_project(project)
    service_inspection = ProjectService.open(project).inspect()

    assert isinstance(inspection, ProjectInspection)
    assert inspection.current_state_path.exists()
    assert inspection.state_kind == "experiment"
    assert inspection.summary.state_summary["label_frame_count"] == 1
    assert inspection.summary.state_summary["prediction_frame_count"] == 0
    assert inspection.summary.modalities == ("videos", "pose")
    assert inspection.is_valid is True
    assert service_inspection == inspection
