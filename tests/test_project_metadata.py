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
    load_project_acquisition_metadata,
    load_project_dataset_share_metadata,
    load_project_metadata,
    load_project_payload,
    pack_project,
    save_project_acquisition_metadata,
    save_project_dataset_share_metadata,
    save_project_labels,
    save_project_metadata,
    unpack_project,
)
from xpkg.project.metadata import (
    project_acquisition_metadata_path,
    project_dataset_share_metadata_path,
)
from xpkg.project.state_io import read_project_state_payload


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
    assert loaded_metadata["preferences"] == {}

    state_payload = read_project_state_payload(state_path)
    assert state_payload["metadata"]["session_json"] == metadata["session_json"]
    assert state_payload["metadata"]["training_state_json"] == metadata["training_state_json"]
    assert state_payload["metadata"]["manifest_json"] == metadata["manifest_json"]


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

    import xpkg.project.store.cache as cache

    def _fail_prediction_materialization(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("metadata load must not materialize predictions")

    def _fail_full_state_load(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("metadata load must not read the full labels state")

    monkeypatch.setattr(cache, "read_project_state", _fail_full_state_load)
    monkeypatch.setattr(
        cache,
        "_predictions_payload_from_state_payload",
        _fail_prediction_materialization,
    )

    assert load_project_metadata(project) == {
        "preferences": {},
        "session_json": {"active_frame_idx": 11},
    }


def test_project_scoped_metadata_slots_roundtrip_without_project_head(tmp_path: Path) -> None:
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

    acquisition_path = save_project_acquisition_metadata(project, acquisition)
    share_path = save_project_dataset_share_metadata(project, dataset_share)

    assert acquisition_path == project_acquisition_metadata_path(project)
    assert share_path == project_dataset_share_metadata_path(project)
    assert load_project_acquisition_metadata(project) == acquisition
    assert load_project_dataset_share_metadata(project) == dataset_share


def test_project_scoped_metadata_slots_pack_into_expkg_manifest(tmp_path: Path) -> None:
    project = tmp_path / "Scoped Metadata Pack Project"
    init_project(project, title="Scoped Metadata Pack Project")
    save_project_acquisition_metadata(
        project,
        {
            "acquisition_id": "acq-pack",
            "system": "multi-camera rig",
            "cameras": [{"camera_id": "cam-top", "frame_rate_hz": 120.0}],
        },
    )
    save_project_dataset_share_metadata(
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

    assert ".xpkg/metadata/acquisition.json" in names
    assert ".xpkg/metadata/dataset_share.json" in names
    assert manifest["acquisition"]["acquisition_id"] == "acq-pack"
    assert manifest["dataset_share"]["doi"] == "10.0000/packable"
    assert manifest["dataset_share"]["license"] == "BSD-3-Clause"
    assert manifest["dataset_share"]["funders"] == ["NIH"]
    assert manifest["dataset_share"]["related_publications"] == ["Example et al. 2024"]

    restored = unpack_project(artifact, tmp_path / "Restored Scoped Metadata Project")
    restored_acquisition = load_project_acquisition_metadata(restored)
    restored_dataset_share = load_project_dataset_share_metadata(restored)

    assert restored_acquisition is not None
    assert restored_acquisition.acquisition_id == "acq-pack"
    assert restored_dataset_share is not None
    assert restored_dataset_share.doi == "10.0000/packable"


def test_project_payload_loads_rebased_project_head(tmp_path: Path) -> None:
    project = tmp_path / "Payload Project"
    init_project(project, title="Payload Project")
    labels = _make_labels(tmp_path)
    save_project_labels(project, labels, metadata={"project_name": "Payload Project"})

    payload = load_project_payload(project)

    assert payload["metadata"]["project_name"] == "Payload Project"
    assert payload["labels"]["frames"]["frame_index"] == [0]
    assert payload["labels"]["videos"]["resolved_paths"][0] == labels.videos[0].filename
