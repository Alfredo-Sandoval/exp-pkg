from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest


def _video_writer_fourcc(code: str) -> int:
    fourcc_fn = getattr(cv2, "VideoWriter_fourcc", None)
    if not callable(fourcc_fn):
        raise RuntimeError("OpenCV build does not expose VideoWriter_fourcc")
    return int(fourcc_fn(*code))


def _sample_dlc_dataframe(*, x_offset: float = 0.0) -> pd.DataFrame:
    columns = pd.MultiIndex.from_product(
        [["demo"], ["nose", "tail"], ["x", "y", "likelihood"]],
        names=["scorer", "bodyparts", "coords"],
    )
    return pd.DataFrame(
        [
            [10.0 + x_offset, 20.0, 0.95, 30.0 + x_offset, 40.0, 0.90],
            [11.0 + x_offset, 21.0, 0.85, 31.0 + x_offset, 41.0, 0.80],
        ],
        columns=columns,
    )


def _write_sample_dlc_h5(path: Path, *, x_offset: float = 0.0) -> pd.DataFrame:
    df = _sample_dlc_dataframe(x_offset=x_offset)
    df.to_hdf(path, key="df")
    return df


def _write_sample_dlc_csv(path: Path, *, x_offset: float = 0.0) -> pd.DataFrame:
    df = _sample_dlc_dataframe(x_offset=x_offset)
    df.to_csv(path)
    return df


def _write_dummy_video(path: Path) -> None:
    fourcc = _video_writer_fourcc("MJPG")
    writer = cv2.VideoWriter(path.as_posix(), fourcc, 5.0, (16, 12))
    assert writer.isOpened()
    try:
        for idx in range(2):
            frame = np.full((12, 16, 3), (idx + 1) * 20, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def _make_dlc_project_fixture(tmp_path: Path) -> Path:
    project_root = tmp_path / "dlc-project"
    labeled_data = project_root / "labeled-data"
    videos_dir = project_root / "videos"
    labeled_data.mkdir(parents=True)
    videos_dir.mkdir()

    csv_dir = labeled_data / "session-csv"
    h5_dir = labeled_data / "session-h5"
    missing_video_dir = labeled_data / "session-missing-video"
    no_data_dir = labeled_data / "session-no-data"
    csv_dir.mkdir()
    h5_dir.mkdir()
    missing_video_dir.mkdir()
    no_data_dir.mkdir()

    _write_sample_dlc_csv(csv_dir / "CollectedData_demo.csv", x_offset=0.0)
    _write_sample_dlc_h5(h5_dir / "CollectedData_demo.h5", x_offset=100.0)
    _write_sample_dlc_csv(missing_video_dir / "CollectedData_demo.csv", x_offset=200.0)

    _write_dummy_video(videos_dir / "session-csv.avi")
    _write_dummy_video(videos_dir / "session-h5.avi")

    return project_root


def test_convert_dlc_h5_project_builds_multi_video_labels(tmp_path: Path) -> None:
    from xpkg.io.converters.dlc_import import convert_dlc_h5_project

    recording_dir = tmp_path / "session-0"
    (recording_dir / "tracking").mkdir(parents=True)
    (recording_dir / "alpha_view").mkdir()
    (recording_dir / "beta_view").mkdir()
    tracking_path = recording_dir / "tracking" / "session-0-tracking.h5"
    df = _write_sample_dlc_h5(tracking_path)
    video_a = recording_dir / "alpha_view" / "session-0-leftCam.avi"
    video_b = recording_dir / "beta_view" / "session-0-underGlass.avi"
    _write_dummy_video(video_a)
    _write_dummy_video(video_b)

    result = convert_dlc_h5_project(
        tracking_path,
        [video_a, video_b],
        recording_dir,
    )

    assert result.project_root == recording_dir
    assert result.metadata["source"] == "dlc_h5_import"
    assert result.metadata["source_h5"] == "tracking/session-0-tracking.h5"
    assert result.metadata["source_videos"] == [
        "alpha_view/session-0-leftCam.avi",
        "beta_view/session-0-underGlass.avi",
    ]

    labels = result.labels
    assert [video.filename for video in labels.videos] == [
        "alpha_view/session-0-leftCam.avi",
        "beta_view/session-0-underGlass.avi",
    ]
    assert len(labels.videos) == 2
    assert len(labels.labeled_frames) == len(df) * 2
    counts = Counter(Path(frame.video.filename or "").name for frame in labels.labeled_frames)
    assert counts == {
        "session-0-leftCam.avi": len(df),
        "session-0-underGlass.avi": len(df),
    }


def test_convert_dlc_project_skips_incomplete_entries(tmp_path: Path) -> None:
    from xpkg.io.converters.dlc_import import convert_dlc_project

    project_root = _make_dlc_project_fixture(tmp_path)
    out_dir = tmp_path / "converted"
    progress: list[str] = []

    results = convert_dlc_project(
        project_root,
        out_dir,
        progress_callback=progress.append,
    )

    assert [result.project_root.name for result in results] == [
        "session-csv",
        "session-h5",
    ]
    assert [result.metadata["source"] for result in results] == [
        "dlc_csv_import",
        "dlc_h5_import",
    ]
    assert "IMPORT: Skipping session-missing-video (no video found)" in progress
    assert "IMPORT: Skipping session-no-data (no data file)" in progress


def test_import_dlc_project_directory_imports_supported_items_into_one_project(
    tmp_path: Path,
) -> None:
    from xpkg.model import Labels
    from xpkg.project import (
        current_project_state_path,
        import_dlc_project_directory,
        project_media_root,
    )
    from xpkg.project.state_io import read_project_state_payload

    project_root = _make_dlc_project_fixture(tmp_path)
    project = tmp_path / "Imported DLC Project"
    progress: list[str] = []

    state_path = import_dlc_project_directory(
        project_root,
        project,
        progress_callback=progress.append,
    )

    assert state_path == current_project_state_path(project)
    assert "IMPORT: Skipping session-missing-video (no video found)" in progress
    assert "IMPORT: Skipping session-no-data (no data file)" in progress

    payload = read_project_state_payload(state_path)
    assert payload["metadata"]["source"] == "dlc_project_import"
    assert payload["metadata"]["project_name"] == "dlc-project"
    assert payload["metadata"]["source_items"] == [
        {
            "name": "session-csv",
            "source": "dlc_csv_import",
            "source_data": "labeled-data/session-csv/CollectedData_demo.csv",
            "source_video": "videos/session-csv.avi",
        },
        {
            "name": "session-h5",
            "source": "dlc_h5_import",
            "source_data": "labeled-data/session-h5/CollectedData_demo.h5",
            "source_video": "videos/session-h5.avi",
        },
    ]
    assert payload["metadata"]["skipped_items"] == [
        {"name": "session-missing-video", "reason": "no video found"},
        {"name": "session-no-data", "reason": "no data file"},
    ]

    loaded = Labels.load_file(project.as_posix())
    assert len(loaded.videos) == 2
    assert len(loaded.skeletons) == 1
    assert len(loaded.labeled_frames) == 4
    counts = Counter(Path(str(frame.video.filename or "")).name for frame in loaded.labeled_frames)
    assert counts == {
        "session-csv.avi": 2,
        "session-h5.avi": 2,
    }

    media_root = project_media_root(project).resolve()
    for video in loaded.videos:
        assert Path(str(video.filename)).resolve().parent == media_root


def test_import_lightning_pose_csv_project_uses_dlc_style_predictions(
    tmp_path: Path,
) -> None:
    from xpkg.model import Labels
    from xpkg.project import current_project_state_path, import_lightning_pose_csv_project
    from xpkg.project.state_io import read_project_state_payload

    csv_path = tmp_path / "video_preds" / "session0.csv"
    csv_path.parent.mkdir()
    _write_sample_dlc_csv(csv_path)
    video_path = tmp_path / "session0.avi"
    _write_dummy_video(video_path)
    project = tmp_path / "Imported Lightning Pose"
    progress: list[str] = []

    state_path = import_lightning_pose_csv_project(
        csv_path,
        video_path,
        project,
        skeleton_name="lp",
        likelihood_threshold=0.5,
        progress_callback=progress.append,
    )

    assert state_path == current_project_state_path(project)
    assert "IMPORT: Reading Lightning Pose CSV session0.csv" in progress
    payload = read_project_state_payload(state_path)
    assert payload["metadata"]["source"] == "lightning_pose_csv_import"
    assert payload["metadata"]["source_csv"] == csv_path.as_posix()
    loaded = Labels.load_file(project.as_posix())
    assert len(loaded.skeletons) == 1
    assert loaded.skeletons[0].keypoint_names == ["nose", "tail"]
    assert len(loaded.labeled_frames) == 2


def test_import_dlc_project_directory_requires_supported_items(tmp_path: Path) -> None:
    from xpkg.project import import_dlc_project_directory

    project_root = tmp_path / "empty-project"
    (project_root / "labeled-data" / "session-no-data").mkdir(parents=True)
    (project_root / "videos").mkdir()

    with pytest.raises(ValueError, match="No supported DLC project items found"):
        import_dlc_project_directory(project_root, tmp_path / "project")
