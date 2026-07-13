from __future__ import annotations

import json
import zipfile
from pathlib import Path

import cv2
import numpy as np

from xpkg.model import AcquisitionMetadata, CameraMetadata, DatasetShareMetadata
from xpkg.project import (
    current_project_state_path,
    init_project,
    load_project_acquisition,
    load_project_dataset_share,
    load_project_metadata,
    load_project_session,
    pack_project,
    save_project_acquisition,
    save_project_dataset_share,
    save_project_labels,
    save_project_metadata,
    unpack_project,
)


def _make_labels(tmp_path: Path):
    from xpkg.model import Labels, Video, build_keypoint_skeleton
    from xpkg.pose.annotations import Instance, LabeledFrame, Point

    frame_path = tmp_path / "frame.png"
    ok = cv2.imwrite(frame_path.as_posix(), np.full((12, 16, 3), 128, dtype=np.uint8))
    assert ok
    video = Video.from_image_filenames([frame_path.as_posix()])
    video.filename = frame_path.as_posix()
    skeleton = build_keypoint_skeleton(["nose"], name="subject")
    frame = LabeledFrame(
        video=video,
        frame_idx=0,
        instances=[
            Instance(
                skeleton=skeleton,
                init_points={"nose": Point(1.0, 2.0, visible=True, complete=True)},
            )
        ],
    )
    labels = Labels(labeled_frames=[frame], videos=[video], skeletons=[skeleton])
    labels.validate()
    return labels


def test_project_metadata_roundtrips_on_project_head(tmp_path: Path) -> None:
    project = tmp_path / "Metadata Project"
    init_project(project, title="Metadata Project")
    labels = _make_labels(tmp_path)
    save_project_labels(project, labels, metadata={"project_name": "Metadata Project"})

    metadata = {
        "session_json": {"active_frame_idx": 3},
        "training_state_json": {
            "schema_version": 1,
            "latest": {
                "run_id": "run-1",
                "created_ns": 1,
                "output_dir": str(project / "models" / "pose" / "run-1"),
                "summary": {"status": "completed"},
            },
            "runs": [],
        },
        "manifest_json": {"version": 1, "entries": []},
    }

    state_path = save_project_metadata(project, metadata)

    assert state_path == current_project_state_path(project)
    loaded_metadata = load_project_metadata(project)
    assert loaded_metadata is not None
    assert loaded_metadata["session_json"] == metadata["session_json"]
    assert loaded_metadata["training_state_json"] == metadata["training_state_json"]
    assert loaded_metadata["manifest_json"] == metadata["manifest_json"]
    from xpkg.project import load_project_experiment

    experiment = load_project_experiment(project)
    assert experiment.metadata["session_json"] == metadata["session_json"]
    assert experiment.metadata["training_state_json"] == metadata["training_state_json"]
    assert experiment.metadata["manifest_json"] == metadata["manifest_json"]


def test_project_metadata_load_returns_empty_before_first_commit(tmp_path: Path) -> None:
    project = tmp_path / "Empty Project"
    init_project(project, title="Empty Project")

    assert load_project_metadata(project) == {}


def test_project_metadata_load_does_not_materialize_predictions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "Metadata Project"
    init_project(project, title="Metadata Project")
    labels = _make_labels(tmp_path)
    save_project_labels(project, labels, metadata={"project_name": "Metadata Project"})
    save_project_metadata(
        project,
        {"session_json": {"active_frame_idx": 11}},
        reason="test.metadata",
    )

    def _fail_session_hydration(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("metadata load must not hydrate the full session")

    monkeypatch.setattr(
        "xpkg.project.recording.read_experiment_json",
        _fail_session_hydration,
    )

    assert load_project_metadata(project) == {
        "session_json": {"active_frame_idx": 11},
    }


def test_session_acquisition_and_experiment_dataset_share_roundtrip(tmp_path: Path) -> None:
    project = tmp_path / "Scoped Metadata Project"
    init_project(project, title="Scoped Metadata Project")
    acquisition = AcquisitionMetadata(
        acquisition_id="acq-001",
        system="open field rig",
        arena_size="40 x 40 cm",
        arena_material="matte acrylic",
        arena_color="gray",
        lighting="overhead visible plus IR",
        ir_lighting=True,
        cameras=(
            CameraMetadata(
                camera_id="cam-top",
                model="FastCam",
                lens="12 mm",
                distance_to_arena_mm=350.0,
                frame_rate_hz=120.0,
                resolution_px=(1920, 1080),
            ),
        ),
    )
    dataset_share = DatasetShareMetadata(
        title="Scoped behavior dataset",
        creators=("Sandoval Lab",),
        license="BSD-3-Clause",
        doi="10.0000/scoped",
        funders=("NIH",),
        related_publications=("Example et al. 2024",),
    )

    acquisition_path = save_project_acquisition(project, acquisition)
    share_path = save_project_dataset_share(project, dataset_share)

    assert acquisition_path == current_project_state_path(project)
    assert share_path == current_project_state_path(project)
    assert load_project_acquisition(project) == acquisition
    assert load_project_dataset_share(project) == dataset_share


def test_semantic_metadata_packs_only_in_canonical_experiment_state(tmp_path: Path) -> None:
    project = tmp_path / "Scoped Metadata Pack Project"
    init_project(project, title="Scoped Metadata Pack Project")
    save_project_acquisition(
        project,
        {
            "acquisition_id": "acq-pack",
            "system": "multi-camera rig",
            "cameras": [{"camera_id": "cam-top", "frame_rate_hz": 120.0}],
        },
    )
    save_project_dataset_share(
        project,
        {
            "title": "Packable behavior dataset",
            "creators": ["Sandoval Lab"],
            "license": "BSD-3-Clause",
            "doi": "10.0000/packable",
            "funders": ["NIH"],
            "related_publications": ["Example et al. 2024"],
        },
    )

    artifact = pack_project(project, out=tmp_path / "scoped-metadata.expkg")

    with zipfile.ZipFile(artifact, mode="r") as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("EXPKG.json").decode("utf-8"))

    assert ".xpkg/metadata/acquisition.json" not in names
    assert ".xpkg/metadata/dataset_share.json" not in names
    assert "acquisition" not in manifest
    assert "dataset_share" not in manifest
    assert manifest["artifact_schema_version"] == 2

    restored = unpack_project(artifact, tmp_path / "Restored Scoped Metadata Project")
    restored_acquisition = load_project_acquisition(restored)
    restored_dataset_share = load_project_dataset_share(restored)

    assert restored_acquisition is not None
    assert restored_acquisition.acquisition_id == "acq-pack"
    assert restored_dataset_share is not None
    assert restored_dataset_share.doi == "10.0000/packable"


def test_project_session_loads_rebased_pose_head(tmp_path: Path) -> None:
    project = tmp_path / "Payload Project"
    init_project(project, title="Payload Project")
    labels = _make_labels(tmp_path)
    save_project_labels(project, labels, metadata={"project_name": "Payload Project"})

    from xpkg.project import load_project_experiment

    experiment = load_project_experiment(project)
    session = load_project_session(project)
    restored = session.pose()

    assert experiment.metadata["project_name"] == "Payload Project"
    assert restored.labeled_frames[0].frame_idx == 0
    assert restored.videos[0].filename == labels.videos[0].filename
