from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from xpkg.formats import (
    current_project_snapshot_path,
    init_project,
    load_workspace_metadata,
    load_workspace_payload,
    save_workspace_labels,
    save_workspace_metadata,
)
from xpkg.io.workspace_snapshot_backend import read_workspace_snapshot_payload


def _make_labels(tmp_path: Path):
    from xpkg.core.annotations import Instance, LabeledFrame, Point
    from xpkg.model import Labels, Video, build_keypoint_skeleton

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


def test_workspace_metadata_roundtrips_on_workspace_head(tmp_path: Path) -> None:
    workspace = tmp_path / "Metadata Project"
    init_project(workspace, title="Metadata Project")
    labels = _make_labels(tmp_path)
    save_workspace_labels(workspace, labels, metadata={"project_name": "Metadata Project"})

    metadata = {
        "session_json": {"active_frame_idx": 3},
        "training_state_json": {
            "schema_version": 1,
            "latest": {
                "run_id": "run-1",
                "created_ns": 1,
                "output_dir": str(workspace / "models" / "pose" / "run-1"),
                "summary": {"status": "completed"},
            },
            "runs": [],
        },
        "manifest_json": {"version": 1, "entries": []},
    }

    state_path = save_workspace_metadata(workspace, metadata)

    assert state_path == current_project_snapshot_path(workspace)
    loaded_metadata = load_workspace_metadata(workspace)
    assert loaded_metadata is not None
    assert loaded_metadata["session_json"] == metadata["session_json"]
    assert loaded_metadata["training_state_json"] == metadata["training_state_json"]
    assert loaded_metadata["manifest_json"] == metadata["manifest_json"]
    assert loaded_metadata["preferences"] == {}

    snapshot_payload = read_workspace_snapshot_payload(state_path)
    assert snapshot_payload["metadata"]["session_json"] == metadata["session_json"]
    assert snapshot_payload["metadata"]["training_state_json"] == metadata["training_state_json"]
    assert snapshot_payload["metadata"]["manifest_json"] == metadata["manifest_json"]


def test_workspace_metadata_load_returns_empty_before_first_commit(tmp_path: Path) -> None:
    workspace = tmp_path / "Empty Project"
    init_project(workspace, title="Empty Project")

    assert load_workspace_metadata(workspace) == {}


def test_workspace_payload_loads_rebased_workspace_head(tmp_path: Path) -> None:
    workspace = tmp_path / "Payload Project"
    init_project(workspace, title="Payload Project")
    labels = _make_labels(tmp_path)
    save_workspace_labels(workspace, labels, metadata={"project_name": "Payload Project"})

    payload = load_workspace_payload(workspace)

    assert payload["metadata"]["project_name"] == "Payload Project"
    assert payload["labels"]["frames"]["frame_index"] == [0]
    assert payload["labels"]["videos"]["resolved_paths"][0] == labels.videos[0].filename
