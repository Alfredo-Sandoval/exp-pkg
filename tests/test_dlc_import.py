from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from tests.factories import (
    write_dummy_video,
    write_sample_dlc_csv,
    write_sample_dlc_h5,
)


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

    write_sample_dlc_csv(csv_dir / "CollectedData_demo.csv", x_offset=0.0)
    write_sample_dlc_h5(h5_dir / "CollectedData_demo.h5", x_offset=100.0)
    write_sample_dlc_csv(missing_video_dir / "CollectedData_demo.csv", x_offset=200.0)

    write_dummy_video(videos_dir / "session-csv.avi")
    write_dummy_video(videos_dir / "session-h5.avi")

    return project_root


def test_convert_dlc_h5_project_builds_multi_video_labels(tmp_path: Path) -> None:
    from xpkg.io.converters.dlc_import import convert_dlc_h5_project

    recording_dir = tmp_path / "session-0"
    (recording_dir / "tracking").mkdir(parents=True)
    (recording_dir / "alpha_view").mkdir()
    (recording_dir / "beta_view").mkdir()
    tracking_path = recording_dir / "tracking" / "session-0-tracking.h5"
    df = write_sample_dlc_h5(tracking_path)
    video_a = recording_dir / "alpha_view" / "session-0-leftCam.avi"
    video_b = recording_dir / "beta_view" / "session-0-underGlass.avi"
    write_dummy_video(video_a)
    write_dummy_video(video_b)

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
    from xpkg.project import current_project_state_path
    from xpkg.project.layout import project_media_root
    from xpkg.project.state_io import read_project_state
    from xpkg.project.store.imports import import_dlc_project_directory

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

    payload = read_project_state(state_path)
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
    from xpkg._core.hashing import sha256_file
    from xpkg.model import Labels
    from xpkg.project import current_project_state_path
    from xpkg.project.state_io import read_project_state
    from xpkg.project.store.imports import import_lightning_pose_csv_project

    csv_path = tmp_path / "video_preds" / "session0.csv"
    csv_path.parent.mkdir()
    write_sample_dlc_csv(csv_path)
    video_path = tmp_path / "session0.avi"
    write_dummy_video(video_path)
    config_path = tmp_path / "lightning_pose_config.yaml"
    config_path.write_text("model: lp-demo\n", encoding="utf-8")
    project = tmp_path / "Imported Lightning Pose"
    progress: list[str] = []

    state_path = import_lightning_pose_csv_project(
        csv_path,
        video_path,
        project,
        skeleton_name="lp",
        likelihood_threshold=0.5,
        prediction_provenance={
            "model_name": "lp-demo-model",
            "model_version": "1.2.3",
            "training_set": "open-field-training-v1",
            "config_snapshot_path": config_path,
        },
        progress_callback=progress.append,
    )

    assert state_path == current_project_state_path(project)
    assert "IMPORT: Reading Lightning Pose CSV session0.csv" in progress
    payload = read_project_state(state_path)
    assert payload["metadata"]["source"] == "lightning_pose_csv_import"
    assert payload["metadata"]["source_csv"] == csv_path.name
    provenance = payload["provenance"]["pose_prediction"]
    assert provenance["tool"]["name"] == "Lightning Pose"
    assert provenance["source_format"] == "csv"
    assert provenance["inputs"]["source_csv"] == csv_path.name
    assert provenance["model"] == {
        "name": "lp-demo-model",
        "version": "1.2.3",
        "training_set": "open-field-training-v1",
    }
    assert provenance["config_snapshot"] == {
        "path": config_path.as_posix(),
        "sha256": sha256_file(config_path),
    }
    assert payload["metadata"]["prediction_provenance"] == provenance
    loaded = Labels.load_file(project.as_posix())
    assert loaded.provenance["pose_prediction"] == provenance
    assert len(loaded.skeletons) == 1
    assert loaded.skeletons[0].keypoint_names == ["nose", "tail"]
    assert len(loaded.labeled_frames) == 2


def test_import_dlc_project_directory_requires_supported_items(tmp_path: Path) -> None:
    from xpkg.project.store.imports import import_dlc_project_directory

    project_root = tmp_path / "empty-project"
    (project_root / "labeled-data" / "session-no-data").mkdir(parents=True)
    (project_root / "videos").mkdir()

    with pytest.raises(ValueError, match="No supported DLC project items found"):
        import_dlc_project_directory(project_root, tmp_path / "project")


def test_dlc_csv_import_preserves_per_keypoint_likelihood_through_project_state(
    tmp_path: Path,
) -> None:
    from xpkg.model import Labels
    from xpkg.pose.annotations import PredictedInstance
    from xpkg.project.store.imports import import_dlc_csv_project

    csv_path = tmp_path / "session.csv"
    write_sample_dlc_csv(csv_path)
    video_path = tmp_path / "session.avi"
    write_dummy_video(video_path)
    project = tmp_path / "DLC Project"

    import_dlc_csv_project(csv_path, video_path, project)

    loaded = Labels.load_file(project.as_posix())
    assert len(loaded.labeled_frames) == 2
    expected_scores_by_frame = {
        0: {"nose": 0.95, "tail": 0.90},
        1: {"nose": 0.85, "tail": 0.80},
    }
    for frame in loaded.labeled_frames:
        assert frame.frame_idx in expected_scores_by_frame
        expected = expected_scores_by_frame[frame.frame_idx]
        instances = frame.predicted_instances
        assert instances, f"Frame {frame.frame_idx} should expose predicted instances"
        instance = instances[0]
        assert isinstance(instance, PredictedInstance)
        # Instance-level prediction score should be the mean of the per-keypoint likelihoods.
        assert instance.score == pytest.approx(
            float(np.mean(list(expected.values()))), rel=1e-5
        )
        observed = {
            keypoint.name: float(point["score"]) for keypoint, point in instance.keypoints_points
        }
        for name, score in expected.items():
            assert observed[name] == pytest.approx(score, rel=1e-5)


def test_dlc_csv_import_preserves_scores_through_pack_unpack_roundtrip(
    tmp_path: Path,
) -> None:
    from xpkg.model import Labels
    from xpkg.pose.annotations import PredictedInstance
    from xpkg.project import pack_project, unpack_project
    from xpkg.project.store.imports import import_dlc_csv_project

    csv_path = tmp_path / "session.csv"
    write_sample_dlc_csv(csv_path)
    video_path = tmp_path / "session.avi"
    write_dummy_video(video_path)
    project = tmp_path / "DLC Project"

    import_dlc_csv_project(csv_path, video_path, project)
    artifact = pack_project(project, out=tmp_path / "session.expkg")
    restored = tmp_path / "Restored"
    unpack_project(artifact, restored)

    loaded = Labels.load_file(restored.as_posix())
    assert len(loaded.labeled_frames) == 2
    expected_scores_by_frame = {
        0: {"nose": 0.95, "tail": 0.90},
        1: {"nose": 0.85, "tail": 0.80},
    }
    for frame in loaded.labeled_frames:
        instance = frame.predicted_instances[0]
        assert isinstance(instance, PredictedInstance)
        observed = {
            keypoint.name: float(point["score"]) for keypoint, point in instance.keypoints_points
        }
        expected = expected_scores_by_frame[frame.frame_idx]
        for name, score in expected.items():
            assert observed[name] == pytest.approx(score, rel=1e-5)
