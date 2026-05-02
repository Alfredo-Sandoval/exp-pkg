from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from xpkg.model import Labels, Video, build_keypoint_skeleton
from xpkg.pose.annotations import Instance, LabeledFrame, Point
from xpkg.project import (
    current_project_snapshot_path,
    init_project,
    save_project_labels,
)
from xpkg.project.durable_store import ProjectDurableStore
from xpkg.project.snapshot_backend import (
    project_snapshot_cache_digest_matches,
    project_snapshot_cache_digest_path,
    snapshot_commit_id,
)
from xpkg.project.store import current_project_commit_id


def _write_test_image(path: Path, value: int = 128) -> None:
    image = np.full((12, 16, 3), value, dtype=np.uint8)
    ok = cv2.imwrite(path.as_posix(), image)
    assert ok


def _make_labels(tmp_path: Path, *, x: float, y: float) -> Labels:
    frame_path = tmp_path / "frame.png"
    _write_test_image(frame_path)
    video = Video.from_image_filenames([frame_path.as_posix()])
    video.filename = frame_path.as_posix()
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


def test_save_project_labels_creates_durable_head_and_snapshot_commit_id(tmp_path: Path) -> None:
    project = tmp_path / "Project"
    init_project(project, title="Project")

    labels = _make_labels(tmp_path, x=3.0, y=4.0)
    snapshot_path = save_project_labels(project, labels)

    assert snapshot_path == current_project_snapshot_path(project)
    assert snapshot_path.exists()

    commit_id = current_project_commit_id(project)
    assert commit_id is not None
    store = ProjectDurableStore.open(project / ".xpkg")
    assert store.has_current_root("snapshot")
    assert not store.has_current_root("archive")

    snapshot_payload = json.loads(snapshot_path.read_text(encoding="utf-8"))["payload"]
    assert snapshot_commit_id(snapshot_payload) == commit_id
    assert project_snapshot_cache_digest_path(snapshot_path).is_file()
    assert project_snapshot_cache_digest_matches(snapshot_path, commit_id=commit_id)


def test_project_load_ignores_snapshot_when_commit_id_mismatches_head(tmp_path: Path) -> None:
    project = tmp_path / "Project"
    init_project(project, title="Project")

    labels = _make_labels(tmp_path, x=3.0, y=4.0)
    snapshot_path = save_project_labels(project, labels)

    document = json.loads(snapshot_path.read_text(encoding="utf-8"))
    document["payload"]["metadata"]["xpkg_commit_id"] = "c_stale_snapshot"
    document["payload"]["data"]["keypoints"][0][0][0][0] = 99.0
    snapshot_path.write_text(json.dumps(document, indent=2), encoding="utf-8")

    loaded = Labels.load_file(project.as_posix())
    pts = loaded.labeled_frames[0].instances[0].get_points_array(copy=False, full=True)

    assert float(pts["x"][0]) == 3.0
    assert float(pts["y"][0]) == 4.0
