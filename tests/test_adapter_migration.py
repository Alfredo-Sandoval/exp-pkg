from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd

from xpkg.io.converters.converter_helpers import ConversionResult, remap_labels_to_videos


def _video_writer_fourcc(code: str) -> int:
    fourcc_fn = getattr(cv2, "VideoWriter_fourcc", None)
    if not callable(fourcc_fn):
        raise RuntimeError("OpenCV build does not expose VideoWriter_fourcc")
    return int(fourcc_fn(*code))


def test_remap_labels_to_videos_preserves_multi_suffix_directory_names(
    monkeypatch,
    tmp_path: Path,
) -> None:
    frames_dir = tmp_path / "labeled-data" / "sleap_labels.pkg"
    frames_dir.mkdir(parents=True)
    frame_paths = []
    for idx in range(2):
        frame_path = frames_dir / f"img{idx:08d}.png"
        frame_path.write_bytes(b"png")
        frame_paths.append(frame_path.as_posix())

    image_sequence_video = SimpleNamespace(
        filename=None,
        image_filenames=frame_paths,
    )
    mp4_path = tmp_path / "videos" / "sleap_labels.pkg.mp4"
    mp4_path.parent.mkdir(parents=True)
    mp4_path.write_bytes(b"mp4")
    encoded_video = SimpleNamespace(filename=mp4_path.as_posix(), image_filenames=[])
    monkeypatch.setattr(
        "xpkg.io.converters.converter_helpers.Video.from_filename",
        lambda path: encoded_video,
    )

    class _LabelsStub:
        def __init__(self) -> None:
            self.videos = [image_sequence_video]
            self.labeled_frames = [SimpleNamespace(video=image_sequence_video)]
            self.merge_calls = 0
            self.cache_updates = 0

        def merge_matching_frames(self) -> None:
            self.merge_calls += 1

        def update_cache(self) -> None:
            self.cache_updates += 1

    labels = _LabelsStub()
    remap_labels_to_videos(labels, [mp4_path], tmp_path)

    assert labels.videos == [encoded_video]
    assert labels.labeled_frames[0].video is encoded_video
    assert labels.merge_calls == 1
    assert labels.cache_updates == 1


def test_dlc_adapter_bridges_progress_and_uses_xpkg_extension(monkeypatch) -> None:
    from xpkg.adapters.dlc import convert_dlc_h5

    captured: dict[str, object] = {}
    progress_events: list[tuple[int, str]] = []

    def fake_convert_dlc_h5_project(
        h5_path: str,
        video_paths: list[str],
        project_root: str,
        *,
        likelihood_threshold: float,
        progress_callback,
        archive_extension: str,
    ) -> ConversionResult:
        captured["h5_path"] = h5_path
        captured["video_paths"] = video_paths
        captured["project_root"] = project_root
        captured["likelihood_threshold"] = likelihood_threshold
        captured["archive_extension"] = archive_extension
        assert progress_callback is not None
        progress_callback("DLC_IMPORT STEP: read_h5")
        progress_callback("IMPORT: still reading")
        progress_callback("DLC_IMPORT STEP: build_labels")
        progress_callback("DLC_IMPORT DONE")
        return ConversionResult(
            source_dir=Path("input"),
            project_root=Path(project_root),
            videos=[Path(video_paths[0])],
            archive_path=Path(project_root) / "project.xpkg",
        )

    monkeypatch.setattr(
        "xpkg.io.converters.dlc_import.convert_dlc_h5_project",
        fake_convert_dlc_h5_project,
    )

    result = convert_dlc_h5(
        "tracking.h5",
        "clip.mp4",
        "project",
        likelihood_threshold=0.25,
        progress_callback=lambda progress, message: progress_events.append((progress, message)),
    )

    assert captured == {
        "h5_path": "tracking.h5",
        "video_paths": ["clip.mp4"],
        "project_root": "project",
        "likelihood_threshold": 0.25,
        "archive_extension": ".xpkg",
    }
    assert progress_events == [
        (10, "DLC_IMPORT STEP: read_h5"),
        (10, "IMPORT: still reading"),
        (55, "DLC_IMPORT STEP: build_labels"),
        (100, "DLC_IMPORT DONE"),
    ]
    assert result.archive_path == Path("project") / "project.xpkg"


def test_dlc_adapter_main_routes_cli_arguments(monkeypatch, tmp_path: Path) -> None:
    from xpkg.adapters.dlc import main

    captured: dict[str, object] = {}

    def fake_convert_dlc_h5(
        h5_path: str,
        video_path: str,
        project_root: str,
        *,
        likelihood_threshold: float,
        progress_callback,
    ) -> ConversionResult:
        captured["h5_path"] = h5_path
        captured["video_path"] = video_path
        captured["project_root"] = project_root
        captured["likelihood_threshold"] = likelihood_threshold
        captured["progress_callback"] = progress_callback
        return ConversionResult(
            source_dir=Path(h5_path),
            project_root=Path(project_root),
            videos=[Path(video_path)],
            archive_path=Path(project_root) / f"{Path(project_root).name}.xpkg",
        )

    monkeypatch.setattr("xpkg.adapters.dlc.convert_dlc_h5", fake_convert_dlc_h5)

    rc = main(
        [
            "--h5",
            str(tmp_path / "sample-tracking.h5"),
            "--video",
            str(tmp_path / "video-a.avi"),
            "--out",
            str(tmp_path / "project-root"),
            "--likelihood-threshold",
            "0.2",
        ]
    )

    assert rc == 0
    assert captured["h5_path"] == str(tmp_path / "sample-tracking.h5")
    assert captured["video_path"] == str(tmp_path / "video-a.avi")
    assert captured["project_root"] == str(tmp_path / "project-root")
    assert captured["likelihood_threshold"] == 0.2
    assert captured["progress_callback"] is None


def test_dlc_adapter_stores_project_relative_video_filename(tmp_path: Path) -> None:
    from xpkg.adapters.dlc import convert_dlc_h5
    from xpkg.io.archive_format import read_archive

    recording_dir = tmp_path / "session-0"
    tracking_dir = recording_dir / "tracking"
    video_dir = recording_dir / "alpha_view"
    tracking_dir.mkdir(parents=True)
    video_dir.mkdir()

    tracking_path = tracking_dir / "session-0-tracking.h5"
    columns = pd.MultiIndex.from_product(
        [["demo"], ["nose"], ["x", "y", "likelihood"]],
        names=["scorer", "bodyparts", "coords"],
    )
    df = pd.DataFrame([[10.0, 20.0, 0.95], [11.0, 21.0, 0.85]], columns=columns)
    df.to_hdf(tracking_path, key="df")

    video_path = video_dir / "session-0-leftCam.avi"
    writer = cv2.VideoWriter(
        video_path.as_posix(),
        _video_writer_fourcc("MJPG"),
        5.0,
        (16, 12),
    )
    assert writer.isOpened()
    for _ in range(2):
        frame = np.zeros((12, 16, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()

    result = convert_dlc_h5(tracking_path, video_path, recording_dir)

    payload = read_archive(result.archive_path, lazy=False)
    assert payload["labels"]["videos"]["filenames"] == ["alpha_view/session-0-leftCam.avi"]


def test_sleap_adapter_bridges_progress_and_uses_xpkg_extension(monkeypatch) -> None:
    from xpkg.adapters.sleap import convert_sleap_package

    captured: dict[str, object] = {}
    progress_events: list[tuple[int, str]] = []

    def fake_convert_sleap_package(
        slp: str,
        out_dir: str,
        *,
        fps: int,
        encode_videos: bool | None,
        archive_extension: str,
        progress_callback,
    ) -> ConversionResult:
        captured["slp"] = slp
        captured["out_dir"] = out_dir
        captured["fps"] = fps
        captured["encode_videos"] = encode_videos
        captured["archive_extension"] = archive_extension
        assert progress_callback is not None
        progress_callback("XPKG_IMPORT START: extracting_frames")
        progress_callback("XPKG_IMPORT OK: label_table_ready")
        progress_callback("XPKG_IMPORT DONE")
        return ConversionResult(
            source_dir=Path(slp),
            project_root=Path(out_dir),
            videos=[],
            archive_path=Path(out_dir) / "project.xpkg",
        )

    monkeypatch.setattr(
        "xpkg.io.converters.sleap_import.convert_sleap_package",
        fake_convert_sleap_package,
    )

    result = convert_sleap_package(
        "labels.pkg.slp",
        "project",
        fps=24,
        encode_videos=False,
        progress_callback=lambda progress, message: progress_events.append((progress, message)),
    )

    assert captured == {
        "slp": "labels.pkg.slp",
        "out_dir": "project",
        "fps": 24,
        "encode_videos": False,
        "archive_extension": ".xpkg",
    }
    assert progress_events == [
        (10, "XPKG_IMPORT START: extracting_frames"),
        (45, "XPKG_IMPORT OK: label_table_ready"),
        (100, "XPKG_IMPORT DONE"),
    ]
    assert result.archive_path == Path("project") / "project.xpkg"


def test_sleap_h5_adapter_bridges_progress_and_uses_xpkg_extension(monkeypatch) -> None:
    from xpkg.adapters.sleap import convert_sleap_h5

    captured: dict[str, object] = {}
    progress_events: list[tuple[int, str]] = []

    def fake_convert_sleap_h5(
        h5_path: str,
        video_path: str,
        out_path: str,
        *,
        skeleton_name: str,
        likelihood_threshold: float,
        archive_extension: str,
        progress_callback,
    ) -> ConversionResult:
        captured["h5_path"] = h5_path
        captured["video_path"] = video_path
        captured["out_path"] = out_path
        captured["skeleton_name"] = skeleton_name
        captured["likelihood_threshold"] = likelihood_threshold
        captured["archive_extension"] = archive_extension
        assert progress_callback is not None
        progress_callback("SLEAP_H5_IMPORT STEP: read_h5")
        progress_callback("SLEAP_H5_IMPORT STEP: build_labels")
        progress_callback("SLEAP_H5_IMPORT DONE")
        return ConversionResult(
            source_dir=Path(h5_path),
            project_root=Path(out_path).parent,
            videos=[Path(video_path)],
            archive_path=Path(out_path),
        )

    monkeypatch.setattr(
        "xpkg.io.converters.sleap_import.convert_sleap_h5",
        fake_convert_sleap_h5,
    )

    result = convert_sleap_h5(
        "analysis.h5",
        "clip.mp4",
        "project.xpkg",
        skeleton_name="mouse",
        likelihood_threshold=0.25,
        progress_callback=lambda progress, message: progress_events.append((progress, message)),
    )

    assert captured == {
        "h5_path": "analysis.h5",
        "video_path": "clip.mp4",
        "out_path": "project.xpkg",
        "skeleton_name": "mouse",
        "likelihood_threshold": 0.25,
        "archive_extension": ".xpkg",
    }
    assert progress_events == [
        (10, "SLEAP_H5_IMPORT STEP: read_h5"),
        (60, "SLEAP_H5_IMPORT STEP: build_labels"),
        (100, "SLEAP_H5_IMPORT DONE"),
    ]
    assert result.archive_path == Path("project.xpkg")


def test_sleap_adapter_main_routes_cli_arguments(monkeypatch, tmp_path: Path) -> None:
    from xpkg.adapters.sleap import main

    captured: dict[str, object] = {}

    def fake_convert_sleap_package(
        slp: str,
        out_dir: str,
        *,
        fps: int,
        encode_videos: bool | None,
        progress_callback,
    ) -> ConversionResult:
        captured["slp"] = slp
        captured["out_dir"] = out_dir
        captured["fps"] = fps
        captured["encode_videos"] = encode_videos
        captured["progress_callback"] = progress_callback
        return ConversionResult(
            source_dir=Path(slp),
            project_root=Path(out_dir),
            videos=[],
            archive_path=Path(out_dir) / f"{Path(out_dir).name}.xpkg",
        )

    monkeypatch.setattr("xpkg.adapters.sleap.convert_sleap_package", fake_convert_sleap_package)

    rc = main(
        [
            "--slp",
            str(tmp_path / "input.pkg.slp"),
            "--out",
            str(tmp_path / "project"),
            "--fps",
            "24",
            "--no-videos",
        ]
    )

    assert rc == 0
    assert captured["slp"] == str(tmp_path / "input.pkg.slp")
    assert captured["out_dir"] == str(tmp_path / "project")
    assert captured["fps"] == 24
    assert captured["encode_videos"] is False
    assert captured["progress_callback"] is None


def test_sleap_h5_adapter_main_routes_cli_arguments(monkeypatch, tmp_path: Path) -> None:
    from xpkg.adapters.sleap import main

    captured: dict[str, object] = {}

    def fake_convert_sleap_h5(
        h5_path: str,
        video_path: str,
        out_path: str,
        *,
        skeleton_name: str,
        likelihood_threshold: float,
        progress_callback,
    ) -> ConversionResult:
        captured["h5_path"] = h5_path
        captured["video_path"] = video_path
        captured["out_path"] = out_path
        captured["skeleton_name"] = skeleton_name
        captured["likelihood_threshold"] = likelihood_threshold
        captured["progress_callback"] = progress_callback
        return ConversionResult(
            source_dir=Path(h5_path),
            project_root=Path(out_path).parent,
            videos=[Path(video_path)],
            archive_path=Path(out_path),
        )

    monkeypatch.setattr("xpkg.adapters.sleap.convert_sleap_h5", fake_convert_sleap_h5)

    rc = main(
        [
            "--h5",
            str(tmp_path / "input.analysis.h5"),
            "--video",
            str(tmp_path / "clip.mp4"),
            "--out",
            str(tmp_path / "project.xpkg"),
            "--skeleton-name",
            "mouse",
            "--likelihood-threshold",
            "0.2",
        ]
    )

    assert rc == 0
    assert captured["h5_path"] == str(tmp_path / "input.analysis.h5")
    assert captured["video_path"] == str(tmp_path / "clip.mp4")
    assert captured["out_path"] == str(tmp_path / "project.xpkg")
    assert captured["skeleton_name"] == "mouse"
    assert captured["likelihood_threshold"] == 0.2
    assert captured["progress_callback"] is None


def test_mmpose_adapter_bridges_progress_and_uses_xpkg_extension(monkeypatch) -> None:
    from xpkg.adapters.mmpose import convert_mmpose_topdown_json

    captured: dict[str, object] = {}
    progress_events: list[tuple[int, str]] = []

    def fake_convert_mmpose_topdown_json(
        json_path: str,
        video_path: str,
        out_path: str,
        *,
        skeleton_name: str,
        instance_index: int,
        likelihood_threshold: float,
        archive_extension: str,
        progress_callback,
    ) -> ConversionResult:
        captured["json_path"] = json_path
        captured["video_path"] = video_path
        captured["out_path"] = out_path
        captured["skeleton_name"] = skeleton_name
        captured["instance_index"] = instance_index
        captured["likelihood_threshold"] = likelihood_threshold
        captured["archive_extension"] = archive_extension
        assert progress_callback is not None
        progress_callback("MMPOSE_JSON_IMPORT STEP: read_json")
        progress_callback("MMPOSE_JSON_IMPORT STEP: build_labels")
        progress_callback("MMPOSE_JSON_IMPORT DONE")
        return ConversionResult(
            source_dir=Path(json_path),
            project_root=Path(out_path).parent,
            videos=[Path(video_path)],
            archive_path=Path(out_path),
        )

    monkeypatch.setattr(
        "xpkg.io.converters.mmpose_import.convert_mmpose_topdown_json",
        fake_convert_mmpose_topdown_json,
    )

    result = convert_mmpose_topdown_json(
        "results.json",
        "clip.mp4",
        "mmpose.xpkg",
        skeleton_name="mouse",
        instance_index=1,
        likelihood_threshold=0.4,
        progress_callback=lambda progress, message: progress_events.append((progress, message)),
    )

    assert captured == {
        "json_path": "results.json",
        "video_path": "clip.mp4",
        "out_path": "mmpose.xpkg",
        "skeleton_name": "mouse",
        "instance_index": 1,
        "likelihood_threshold": 0.4,
        "archive_extension": ".xpkg",
    }
    assert progress_events == [
        (10, "MMPOSE_JSON_IMPORT STEP: read_json"),
        (60, "MMPOSE_JSON_IMPORT STEP: build_labels"),
        (100, "MMPOSE_JSON_IMPORT DONE"),
    ]
    assert result.archive_path == Path("mmpose.xpkg")


def test_mmpose_adapter_main_routes_cli_arguments(monkeypatch, tmp_path: Path) -> None:
    from xpkg.adapters.mmpose import main

    captured: dict[str, object] = {}

    def fake_convert_mmpose_topdown_json(
        json_path: str,
        video_path: str,
        out_path: str,
        *,
        skeleton_name: str,
        instance_index: int,
        likelihood_threshold: float,
        progress_callback,
    ) -> ConversionResult:
        captured["json_path"] = json_path
        captured["video_path"] = video_path
        captured["out_path"] = out_path
        captured["skeleton_name"] = skeleton_name
        captured["instance_index"] = instance_index
        captured["likelihood_threshold"] = likelihood_threshold
        captured["progress_callback"] = progress_callback
        return ConversionResult(
            source_dir=Path(json_path),
            project_root=Path(out_path).parent,
            videos=[Path(video_path)],
            archive_path=Path(out_path),
        )

    monkeypatch.setattr(
        "xpkg.adapters.mmpose.convert_mmpose_topdown_json",
        fake_convert_mmpose_topdown_json,
    )

    rc = main(
        [
            "--json",
            str(tmp_path / "results.json"),
            "--video",
            str(tmp_path / "clip.mp4"),
            "--out",
            str(tmp_path / "mmpose.xpkg"),
            "--instance-index",
            "1",
            "--skeleton-name",
            "mouse",
            "--likelihood-threshold",
            "0.4",
        ]
    )

    assert rc == 0
    assert captured["json_path"] == str(tmp_path / "results.json")
    assert captured["video_path"] == str(tmp_path / "clip.mp4")
    assert captured["out_path"] == str(tmp_path / "mmpose.xpkg")
    assert captured["skeleton_name"] == "mouse"
    assert captured["instance_index"] == 1
    assert captured["likelihood_threshold"] == 0.4
    assert captured["progress_callback"] is None


def test_mediapipe_adapter_bridges_progress_and_uses_xpkg_extension(monkeypatch) -> None:
    from xpkg.adapters.mediapipe import convert_mediapipe_pose_landmarks_json

    captured: dict[str, object] = {}
    progress_events: list[tuple[int, str]] = []

    def fake_convert_mediapipe_pose_landmarks_json(
        json_path: str,
        video_path: str,
        out_path: str,
        *,
        skeleton_name: str,
        likelihood_threshold: float,
        archive_extension: str,
        progress_callback,
    ) -> ConversionResult:
        captured["json_path"] = json_path
        captured["video_path"] = video_path
        captured["out_path"] = out_path
        captured["skeleton_name"] = skeleton_name
        captured["likelihood_threshold"] = likelihood_threshold
        captured["archive_extension"] = archive_extension
        assert progress_callback is not None
        progress_callback("MEDIAPIPE_IMPORT STEP: read_json")
        progress_callback("MEDIAPIPE_IMPORT STEP: build_labels")
        progress_callback("MEDIAPIPE_IMPORT DONE")
        return ConversionResult(
            source_dir=Path(json_path),
            project_root=Path(out_path).parent,
            videos=[Path(video_path)],
            archive_path=Path(out_path),
        )

    monkeypatch.setattr(
        "xpkg.io.converters.mediapipe_import.convert_mediapipe_pose_landmarks_json",
        fake_convert_mediapipe_pose_landmarks_json,
    )

    result = convert_mediapipe_pose_landmarks_json(
        "pose_landmarks.json",
        "clip.mp4",
        "mediapipe.xpkg",
        likelihood_threshold=0.3,
        progress_callback=lambda progress, message: progress_events.append((progress, message)),
    )

    assert captured == {
        "json_path": "pose_landmarks.json",
        "video_path": "clip.mp4",
        "out_path": "mediapipe.xpkg",
        "skeleton_name": "mediapipe_pose",
        "likelihood_threshold": 0.3,
        "archive_extension": ".xpkg",
    }
    assert progress_events == [
        (10, "MEDIAPIPE_IMPORT STEP: read_json"),
        (60, "MEDIAPIPE_IMPORT STEP: build_labels"),
        (100, "MEDIAPIPE_IMPORT DONE"),
    ]
    assert result.archive_path == Path("mediapipe.xpkg")


def test_mediapipe_adapter_main_routes_cli_arguments(monkeypatch, tmp_path: Path) -> None:
    from xpkg.adapters.mediapipe import main

    captured: dict[str, object] = {}

    def fake_convert_mediapipe_pose_landmarks_json(
        json_path: str,
        video_path: str,
        out_path: str,
        *,
        skeleton_name: str,
        likelihood_threshold: float,
        progress_callback,
    ) -> ConversionResult:
        captured["json_path"] = json_path
        captured["video_path"] = video_path
        captured["out_path"] = out_path
        captured["skeleton_name"] = skeleton_name
        captured["likelihood_threshold"] = likelihood_threshold
        captured["progress_callback"] = progress_callback
        return ConversionResult(
            source_dir=Path(json_path),
            project_root=Path(out_path).parent,
            videos=[Path(video_path)],
            archive_path=Path(out_path),
        )

    monkeypatch.setattr(
        "xpkg.adapters.mediapipe.convert_mediapipe_pose_landmarks_json",
        fake_convert_mediapipe_pose_landmarks_json,
    )

    rc = main(
        [
            "--json",
            str(tmp_path / "pose_landmarks.json"),
            "--video",
            str(tmp_path / "clip.mp4"),
            "--out",
            str(tmp_path / "mediapipe.xpkg"),
            "--likelihood-threshold",
            "0.3",
        ]
    )

    assert rc == 0
    assert captured["json_path"] == str(tmp_path / "pose_landmarks.json")
    assert captured["video_path"] == str(tmp_path / "clip.mp4")
    assert captured["out_path"] == str(tmp_path / "mediapipe.xpkg")
    assert captured["skeleton_name"] == "mediapipe_pose"
    assert captured["likelihood_threshold"] == 0.3
    assert captured["progress_callback"] is None


def test_openpose_adapter_bridges_progress_and_uses_xpkg_extension(monkeypatch) -> None:
    from xpkg.adapters.openpose import convert_openpose_json

    captured: dict[str, object] = {}
    progress_events: list[tuple[int, str]] = []

    def fake_convert_openpose_json(
        json_dir: str,
        video_path: str,
        out_path: str,
        *,
        skeleton_name: str,
        likelihood_threshold: float,
        progress_callback,
    ) -> ConversionResult:
        captured["json_dir"] = json_dir
        captured["video_path"] = video_path
        captured["out_path"] = out_path
        captured["skeleton_name"] = skeleton_name
        captured["likelihood_threshold"] = likelihood_threshold
        assert progress_callback is not None
        progress_callback("OPENPOSE_IMPORT STEP: read_json")
        progress_callback("OPENPOSE_IMPORT STEP: build_labels")
        progress_callback("OPENPOSE_IMPORT DONE")
        return ConversionResult(
            source_dir=Path(json_dir),
            project_root=Path(out_path).parent,
            videos=[Path(video_path)],
            archive_path=Path(out_path),
        )

    monkeypatch.setattr(
        "xpkg.io.converters.openpose_import.convert_openpose_json",
        fake_convert_openpose_json,
    )

    result = convert_openpose_json(
        "openpose_json",
        "clip.mp4",
        "openpose.xpkg",
        likelihood_threshold=0.2,
        progress_callback=lambda progress, message: progress_events.append((progress, message)),
    )

    assert captured == {
        "json_dir": "openpose_json",
        "video_path": "clip.mp4",
        "out_path": "openpose.xpkg",
        "skeleton_name": "imported",
        "likelihood_threshold": 0.2,
    }
    assert progress_events == [
        (10, "OPENPOSE_IMPORT STEP: read_json"),
        (60, "OPENPOSE_IMPORT STEP: build_labels"),
        (100, "OPENPOSE_IMPORT DONE"),
    ]
    assert result.archive_path == Path("openpose.xpkg")


def test_openpose_adapter_main_routes_cli_arguments(monkeypatch, tmp_path: Path) -> None:
    from xpkg.adapters.openpose import main

    captured: dict[str, object] = {}

    def fake_convert_openpose_json(
        json_dir: str,
        video_path: str,
        out_path: str,
        *,
        skeleton_name: str,
        likelihood_threshold: float,
        progress_callback,
    ) -> ConversionResult:
        captured["json_dir"] = json_dir
        captured["video_path"] = video_path
        captured["out_path"] = out_path
        captured["skeleton_name"] = skeleton_name
        captured["likelihood_threshold"] = likelihood_threshold
        captured["progress_callback"] = progress_callback
        return ConversionResult(
            source_dir=Path(json_dir),
            project_root=Path(out_path).parent,
            videos=[Path(video_path)],
            archive_path=Path(out_path),
        )

    monkeypatch.setattr("xpkg.adapters.openpose.convert_openpose_json", fake_convert_openpose_json)

    rc = main(
        [
            "--json",
            str(tmp_path / "openpose_json"),
            "--video",
            str(tmp_path / "clip.mp4"),
            "--out",
            str(tmp_path / "openpose.xpkg"),
            "--likelihood-threshold",
            "0.2",
        ]
    )

    assert rc == 0
    assert captured["json_dir"] == str(tmp_path / "openpose_json")
    assert captured["video_path"] == str(tmp_path / "clip.mp4")
    assert captured["out_path"] == str(tmp_path / "openpose.xpkg")
    assert captured["skeleton_name"] == "imported"
    assert captured["likelihood_threshold"] == 0.2
    assert captured["progress_callback"] is None


def test_detectron2_adapter_bridges_progress(monkeypatch) -> None:
    from xpkg.adapters.detectron2 import convert_detectron2_coco

    captured: dict[str, object] = {}
    progress_events: list[tuple[int, str]] = []

    def fake_convert_detectron2_coco(
        predictions_path: str,
        dataset_json_path: str,
        image_root: str,
        out_path: str,
        *,
        category_id: int | None,
        skeleton_name: str | None,
        likelihood_threshold: float,
        progress_callback,
    ) -> ConversionResult:
        captured["predictions_path"] = predictions_path
        captured["dataset_json_path"] = dataset_json_path
        captured["image_root"] = image_root
        captured["out_path"] = out_path
        captured["category_id"] = category_id
        captured["skeleton_name"] = skeleton_name
        captured["likelihood_threshold"] = likelihood_threshold
        assert progress_callback is not None
        progress_callback("DETECTRON2_IMPORT STEP: read_json")
        progress_callback("IMPORT: still reading")
        progress_callback("DETECTRON2_IMPORT STEP: build_labels")
        progress_callback("DETECTRON2_IMPORT DONE")
        return ConversionResult(
            source_dir=Path("input"),
            project_root=Path("out"),
            videos=[Path(image_root)],
            archive_path=Path(out_path),
        )

    monkeypatch.setattr(
        "xpkg.io.converters.detectron2_import.convert_detectron2_coco",
        fake_convert_detectron2_coco,
    )

    result = convert_detectron2_coco(
        "coco_instances_results.json",
        "dataset.json",
        "images",
        "detectron2.xpkg",
        category_id=7,
        skeleton_name="mouse",
        likelihood_threshold=0.5,
        progress_callback=lambda progress, message: progress_events.append((progress, message)),
    )

    assert captured == {
        "predictions_path": "coco_instances_results.json",
        "dataset_json_path": "dataset.json",
        "image_root": "images",
        "out_path": "detectron2.xpkg",
        "category_id": 7,
        "skeleton_name": "mouse",
        "likelihood_threshold": 0.5,
    }
    assert progress_events == [
        (10, "DETECTRON2_IMPORT STEP: read_json"),
        (10, "IMPORT: still reading"),
        (60, "DETECTRON2_IMPORT STEP: build_labels"),
        (100, "DETECTRON2_IMPORT DONE"),
    ]
    assert result.archive_path == Path("detectron2.xpkg")


def test_detectron2_adapter_main_routes_cli_arguments(monkeypatch, tmp_path: Path) -> None:
    from xpkg.adapters.detectron2 import main

    captured: dict[str, object] = {}

    def fake_convert_detectron2_coco(
        predictions_path: str,
        dataset_json_path: str,
        image_root: str,
        out_path: str,
        *,
        category_id: int | None,
        skeleton_name: str | None,
        likelihood_threshold: float,
        progress_callback,
    ) -> ConversionResult:
        captured["predictions_path"] = predictions_path
        captured["dataset_json_path"] = dataset_json_path
        captured["image_root"] = image_root
        captured["out_path"] = out_path
        captured["category_id"] = category_id
        captured["skeleton_name"] = skeleton_name
        captured["likelihood_threshold"] = likelihood_threshold
        captured["progress_callback"] = progress_callback
        return ConversionResult(
            source_dir=Path(predictions_path),
            project_root=Path(out_path).parent,
            videos=[Path(image_root)],
            archive_path=Path(out_path),
        )

    monkeypatch.setattr(
        "xpkg.adapters.detectron2.convert_detectron2_coco",
        fake_convert_detectron2_coco,
    )

    rc = main(
        [
            "--predictions",
            str(tmp_path / "coco_instances_results.json"),
            "--dataset-json",
            str(tmp_path / "dataset.json"),
            "--image-root",
            str(tmp_path / "images"),
            "--out",
            str(tmp_path / "detectron2.xpkg"),
            "--category-id",
            "3",
            "--skeleton-name",
            "mouse",
            "--likelihood-threshold",
            "0.4",
        ]
    )

    assert rc == 0
    assert captured["predictions_path"] == str(tmp_path / "coco_instances_results.json")
    assert captured["dataset_json_path"] == str(tmp_path / "dataset.json")
    assert captured["image_root"] == str(tmp_path / "images")
    assert captured["out_path"] == str(tmp_path / "detectron2.xpkg")
    assert captured["category_id"] == 3
    assert captured["skeleton_name"] == "mouse"
    assert captured["likelihood_threshold"] == 0.4
    assert captured["progress_callback"] is None
