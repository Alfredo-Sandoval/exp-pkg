from __future__ import annotations

import pytest

import xpkg.model as model
from xpkg.model import AcquisitionMetadata, CameraMetadata, DatasetShareMetadata


def test_camera_metadata_round_trips_json_friendly_payload() -> None:
    camera = CameraMetadata(
        camera_id="cam-1",
        name="overhead",
        manufacturer="Example Imaging",
        model="FastCam",
        lens="12 mm",
        distance_to_arena_mm=350.0,
        frame_rate_hz=120.0,
        resolution_px=(1920, 1080),
        exposure_ms=2.5,
        metadata={"mount": "top"},
    )

    payload = camera.to_dict()

    assert payload == {
        "camera_id": "cam-1",
        "name": "overhead",
        "manufacturer": "Example Imaging",
        "model": "FastCam",
        "lens": "12 mm",
        "distance_to_arena_mm": 350.0,
        "frame_rate_hz": 120.0,
        "exposure_ms": 2.5,
        "resolution_px": [1920, 1080],
        "metadata": {"mount": "top"},
    }
    assert CameraMetadata.from_dict(payload) == camera


def test_camera_metadata_validates_capture_settings() -> None:
    with pytest.raises(ValueError, match="camera_id"):
        CameraMetadata(camera_id=" ")

    with pytest.raises(ValueError, match="frame_rate_hz"):
        CameraMetadata(camera_id="cam-1", frame_rate_hz=0.0)

    with pytest.raises(ValueError, match="resolution_px"):
        CameraMetadata(camera_id="cam-1", resolution_px=(1920, 0))

    with pytest.raises(ValueError, match="distance_to_arena_mm"):
        CameraMetadata(camera_id="cam-1", distance_to_arena_mm=-1.0)


def test_acquisition_metadata_collects_cameras_and_setup_payload() -> None:
    payload = {
        "acquisition_id": "acq-001",
        "recorded_at": "2023-10-05T14:30:00Z",
        "experimenter": "A. Researcher",
        "site": "behavior room 2",
        "system": "open field rig",
        "arena_size": "40 x 40 cm",
        "arena_material": "matte acrylic",
        "arena_color": "gray",
        "lighting": "overhead visible plus IR",
        "ir_lighting": True,
        "cameras": [
            {
                "camera_id": "cam-1",
                "name": "side",
                "distance_to_arena_mm": 250.0,
                "frame_rate_hz": 60.0,
                "resolution_px": [1280, 720],
            }
        ],
        "software": {"bonsai": "2.8"},
        "hardware": {"controller": "ttl-box"},
        "notes": "ambient light controlled",
    }

    acquisition = AcquisitionMetadata.from_dict(payload)

    assert acquisition.cameras == (
        CameraMetadata(
            camera_id="cam-1",
            name="side",
            distance_to_arena_mm=250.0,
            frame_rate_hz=60.0,
            resolution_px=(1280, 720),
        ),
    )
    assert acquisition.to_dict() == payload


def test_acquisition_metadata_requires_unique_camera_ids() -> None:
    camera = CameraMetadata(camera_id="cam-1")

    with pytest.raises(ValueError, match="camera_id values must be unique"):
        AcquisitionMetadata(cameras=(camera, camera))


def test_dataset_share_metadata_round_trips_citation_and_access_fields() -> None:
    share = DatasetShareMetadata(
        title="Open behavioral video analysis dataset",
        creators=("Luxem Lab", "Sandoval Lab"),
        dataset_id="dataset-001",
        description="Example multimodal behavior package.",
        license="BSD-3-Clause",
        doi="10.0000/example",
        repository_url="https://example.org/datasets/behavior",
        version="1.0.0",
        funders=("NIH",),
        keywords=("behavior", "video", "pose"),
        access="open",
        related_publications=("Luxem et al. 2023",),
    )

    payload = share.to_dict()

    assert payload["creators"] == ["Luxem Lab", "Sandoval Lab"]
    assert payload["funders"] == ["NIH"]
    assert payload["keywords"] == ["behavior", "video", "pose"]
    assert payload["related_publications"] == ["Luxem et al. 2023"]
    assert DatasetShareMetadata.from_dict(payload) == share
    legacy_payload = dict(payload)
    legacy_payload.pop("funders")
    legacy_payload["funder"] = "NSF"
    assert DatasetShareMetadata.from_dict(legacy_payload).funders == ("NSF",)


def test_dataset_share_metadata_requires_title_and_creator() -> None:
    with pytest.raises(ValueError, match="title"):
        DatasetShareMetadata(title="", creators=("A. Researcher",))

    with pytest.raises(ValueError, match="creators"):
        DatasetShareMetadata(title="Dataset", creators=())

    with pytest.raises(TypeError, match="creators"):
        DatasetShareMetadata.from_dict({"title": "Dataset", "creators": "A. Researcher"})


def test_metadata_models_are_available_from_public_surfaces() -> None:
    assert model.AcquisitionMetadata is AcquisitionMetadata
    assert model.CameraMetadata is CameraMetadata
    assert model.DatasetShareMetadata is DatasetShareMetadata
