"""Per-format project importer implementations used by services and CLI."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from xpkg._core.path_registry import resolve_path
from xpkg.project.store.conversion import (
    _import_pose_project,
    _merge_labels_for_import,
)

if TYPE_CHECKING:
    from xpkg.model import PoseModelProvenance


def import_dlc_csv_project(
    csv_path: str | Path,
    video_path: str | Path,
    project: str | Path,
    *,
    skeleton_name: str = "imported",
    likelihood_threshold: float = 0.0,
    prediction_provenance: Mapping[str, Any] | None = None,
    provenance: PoseModelProvenance | Mapping[str, Any] | None = None,
    force: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> Path:
    """Import a DeepLabCut CSV plus video into a project."""
    from xpkg.io.converters.dlc_import import convert_dlc_csv

    return _import_pose_project(
        project,
        force=force,
        reason="project.import.dlc_csv",
        prediction_provenance=prediction_provenance,
        provenance=provenance,
        default_tool="deeplabcut",
        source_path=csv_path,
        convert=lambda _tmp_dir: convert_dlc_csv(
            csv_path,
            video_path,
            skeleton_name=skeleton_name,
            likelihood_threshold=likelihood_threshold,
            progress_callback=progress_callback,
        ),
    )


def import_lightning_pose_csv_project(
    csv_path: str | Path,
    video_path: str | Path,
    project: str | Path,
    *,
    skeleton_name: str = "imported",
    likelihood_threshold: float = 0.0,
    prediction_provenance: Mapping[str, Any] | None = None,
    provenance: PoseModelProvenance | Mapping[str, Any] | None = None,
    force: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> Path:
    """Import a Lightning Pose prediction CSV plus video into a project."""
    from xpkg.io.converters.dlc_import import convert_lightning_pose_csv

    return _import_pose_project(
        project,
        force=force,
        reason="project.import.lightning_pose_csv",
        prediction_provenance=prediction_provenance,
        provenance=provenance,
        default_tool="lightning-pose",
        source_path=csv_path,
        convert=lambda _tmp_dir: convert_lightning_pose_csv(
            csv_path,
            video_path,
            skeleton_name=skeleton_name,
            likelihood_threshold=likelihood_threshold,
            progress_callback=progress_callback,
        ),
    )


def import_dlc_h5_project(
    h5_path: str | Path,
    video_path: str | Path,
    project: str | Path,
    *,
    skeleton_name: str = "imported",
    likelihood_threshold: float = 0.0,
    prediction_provenance: Mapping[str, Any] | None = None,
    provenance: PoseModelProvenance | Mapping[str, Any] | None = None,
    force: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> Path:
    """Import a DeepLabCut H5 export plus video into a project."""
    from xpkg.io.converters.dlc_import import convert_dlc_h5

    return _import_pose_project(
        project,
        force=force,
        reason="project.import.dlc_h5",
        prediction_provenance=prediction_provenance,
        provenance=provenance,
        default_tool="deeplabcut",
        source_path=h5_path,
        convert=lambda _tmp_dir: convert_dlc_h5(
            h5_path,
            video_path,
            skeleton_name=skeleton_name,
            likelihood_threshold=likelihood_threshold,
            progress_callback=progress_callback,
        ),
    )


def import_dlc_project_directory(
    project_dir: str | Path,
    project: str | Path,
    *,
    skeleton_name: str | None = None,
    likelihood_threshold: float = 0.0,
    prediction_provenance: Mapping[str, Any] | None = None,
    provenance: PoseModelProvenance | Mapping[str, Any] | None = None,
    force: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> Path:
    """Import a supported DeepLabCut project into one project."""
    from xpkg.io.converters.dlc_import import (
        _discover_dlc_project_items,
        _stored_project_path,
        convert_dlc_csv,
        convert_dlc_h5,
    )
    from xpkg.io.converters.result import ConversionResult

    resolved_project_dir = resolve_path(project_dir)
    resolved_skeleton_name = skeleton_name or resolved_project_dir.name or "dlc"

    def _convert_project(_tmp_dir: Path) -> ConversionResult:
        project_items, skipped_items = _discover_dlc_project_items(
            resolved_project_dir,
            progress_callback=progress_callback,
        )
        if not project_items:
            raise ValueError(f"No supported DLC project items found in {resolved_project_dir}")

        merged_labels = None
        videos: list[Path] = []
        source_items: list[dict[str, str]] = []
        for project_item in project_items:
            if project_item.source_type == "h5":
                result = convert_dlc_h5(
                    project_item.data_path,
                    project_item.video_path,
                    skeleton_name=resolved_skeleton_name,
                    likelihood_threshold=likelihood_threshold,
                    progress_callback=progress_callback,
                )
            else:
                result = convert_dlc_csv(
                    project_item.data_path,
                    project_item.video_path,
                    skeleton_name=resolved_skeleton_name,
                    likelihood_threshold=likelihood_threshold,
                    progress_callback=progress_callback,
                )

            merged_labels = _merge_labels_for_import(merged_labels, result.labels)
            videos.extend(result.videos)
            source_items.append(
                {
                    "name": project_item.name,
                    "source": f"dlc_{project_item.source_type}_import",
                    "source_data": _stored_project_path(
                        project_item.data_path,
                        project_root=resolved_project_dir,
                    ),
                    "source_video": _stored_project_path(
                        project_item.video_path,
                        project_root=resolved_project_dir,
                    ),
                }
            )

        assert merged_labels is not None
        merged_labels.validate()
        metadata = {
            "project_name": resolved_project_dir.name,
            "source": "dlc_project_import",
            "source_project": resolved_project_dir.name,
            "source_items": source_items,
            "skipped_items": [
                {"name": skipped_item.name, "reason": skipped_item.reason}
                for skipped_item in skipped_items
            ],
        }
        return ConversionResult(
            source_dir=resolved_project_dir,
            project_root=resolved_project_dir,
            videos=videos,
            labels=merged_labels,
            metadata=metadata,
        )

    return _import_pose_project(
        project,
        force=force,
        reason="project.import.dlc_project",
        prediction_provenance=prediction_provenance,
        provenance=provenance,
        default_tool="deeplabcut",
        source_path=resolved_project_dir,
        convert=_convert_project,
    )


def import_sleap_package_project(
    slp: str | Path,
    project: str | Path,
    *,
    fps: int = 30,
    encode_videos: bool | None = None,
    prediction_provenance: Mapping[str, Any] | None = None,
    provenance: PoseModelProvenance | Mapping[str, Any] | None = None,
    force: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> Path:
    """Import a SLEAP package into a project."""
    from xpkg.io.converters.sleap_import import convert_sleap_package

    return _import_pose_project(
        project,
        force=force,
        reason="project.import.sleap",
        prediction_provenance=prediction_provenance,
        provenance=provenance,
        default_tool="sleap",
        source_path=slp,
        convert=lambda tmp_dir: convert_sleap_package(
            slp,
            tmp_dir,
            fps=int(fps),
            encode_videos=encode_videos,
            progress_callback=progress_callback,
        ),
    )


def import_sleap_h5_project(
    h5_path: str | Path,
    video_path: str | Path,
    project: str | Path,
    *,
    skeleton_name: str = "imported",
    likelihood_threshold: float = 0.0,
    prediction_provenance: Mapping[str, Any] | None = None,
    provenance: PoseModelProvenance | Mapping[str, Any] | None = None,
    force: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> Path:
    """Import a SLEAP analysis H5 export plus video into a project."""
    from xpkg.io.converters.sleap_import import convert_sleap_h5

    return _import_pose_project(
        project,
        force=force,
        reason="project.import.sleap_h5",
        prediction_provenance=prediction_provenance,
        provenance=provenance,
        default_tool="sleap",
        source_path=h5_path,
        convert=lambda _tmp_dir: convert_sleap_h5(
            h5_path,
            video_path,
            skeleton_name=skeleton_name,
            likelihood_threshold=likelihood_threshold,
            progress_callback=progress_callback,
        ),
    )


def import_mmpose_topdown_json_project(
    json_path: str | Path,
    video_path: str | Path,
    project: str | Path,
    *,
    skeleton_name: str = "imported",
    instance_index: int = 0,
    likelihood_threshold: float = 0.0,
    prediction_provenance: Mapping[str, Any] | None = None,
    provenance: PoseModelProvenance | Mapping[str, Any] | None = None,
    force: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> Path:
    """Import an MMPose top-down JSON export plus video into a project."""
    from xpkg.io.converters.mmpose_import import convert_mmpose_topdown_json

    return _import_pose_project(
        project,
        force=force,
        reason="project.import.mmpose_topdown_json",
        prediction_provenance=prediction_provenance,
        provenance=provenance,
        default_tool="mmpose",
        source_path=json_path,
        convert=lambda _tmp_dir: convert_mmpose_topdown_json(
            json_path,
            video_path,
            skeleton_name=skeleton_name,
            instance_index=int(instance_index),
            likelihood_threshold=likelihood_threshold,
            progress_callback=progress_callback,
        ),
    )


def import_mediapipe_pose_landmarks_json_project(
    json_path: str | Path,
    video_path: str | Path,
    project: str | Path,
    *,
    skeleton_name: str = "mediapipe_pose",
    likelihood_threshold: float = 0.0,
    prediction_provenance: Mapping[str, Any] | None = None,
    provenance: PoseModelProvenance | Mapping[str, Any] | None = None,
    force: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> Path:
    """Import MediaPipe pose-landmarks JSON plus video into a project."""
    from xpkg.io.converters.mediapipe_import import convert_mediapipe_pose_landmarks_json

    return _import_pose_project(
        project,
        force=force,
        reason="project.import.mediapipe_pose_landmarks_json",
        prediction_provenance=prediction_provenance,
        provenance=provenance,
        default_tool="mediapipe",
        source_path=json_path,
        convert=lambda _tmp_dir: convert_mediapipe_pose_landmarks_json(
            json_path,
            video_path,
            skeleton_name=skeleton_name,
            likelihood_threshold=likelihood_threshold,
            progress_callback=progress_callback,
        ),
    )
