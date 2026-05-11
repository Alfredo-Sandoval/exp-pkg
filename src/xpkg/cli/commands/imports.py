"""CLI commands that import external data into project-first xpkg projects.

The CLI mirrors the ``ProjectService`` Python dispatch surface. The user picks
one of three families (``pose``, ``calibration``, ``motion``) and a kebab-case
``format`` positional argument that matches ``ProjectService.import_pose``,
``ProjectService.import_calibration``, or ``ProjectService.import_motion``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, Any, get_args

import typer

from xpkg.cli.shared import (
    JsonOption,
    progress_callback,
    require_likelihood_threshold,
    require_nonnegative_int,
    require_option_value,
    require_positive_int,
    run_command,
    write_path,
)
from xpkg.project import (
    import_anipose_calibration_project,
    import_dlc_csv_project,
    import_dlc_h5_project,
    import_dlc_project_directory,
    import_lightning_pose_csv_project,
    import_mediapipe_pose_landmarks_json_project,
    import_mmpose_topdown_json_project,
    import_sleap_h5_project,
    import_sleap_package_project,
    import_vicon_c3d_project,
    import_vicon_csv_project,
    import_vicon_project,
)
from xpkg.services.project import CalibrationFormat, MotionFormat, PoseFormat

from ..._core.json_utils import load_json_dict

app = typer.Typer(
    add_completion=False,
    help="Import external tracking, calibration, and motion data into a project.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


_POSE_FORMATS: tuple[str, ...] = get_args(PoseFormat)
_CALIBRATION_FORMATS: tuple[str, ...] = get_args(CalibrationFormat)
_MOTION_FORMATS: tuple[str, ...] = get_args(MotionFormat)

_PER_CLIP_POSE_FORMATS: frozenset[str] = frozenset(
    {
        "dlc-csv",
        "dlc-h5",
        "lightning-pose-csv",
        "mediapipe-pose-landmarks-json",
        "mmpose-topdown-json",
        "sleap-h5",
    }
)


_POSE_FORMAT_LABELS: dict[str, str] = {
    "dlc-csv": "DLC CSV",
    "dlc-h5": "DLC H5",
    "dlc-project": "DLC project",
    "lightning-pose-csv": "Lightning Pose CSV",
    "mediapipe-pose-landmarks-json": "MediaPipe JSON",
    "mmpose-topdown-json": "MMPose JSON",
    "sleap-h5": "SLEAP H5",
    "sleap-package": "SLEAP package",
}


_MOTION_FORMAT_LABELS: dict[str, str] = {
    "vicon": "Vicon recording",
    "vicon-csv": "Vicon CSV",
    "vicon-c3d": "Vicon C3D",
}


_MODEL_PROVENANCE_HELP = (
    "Optional JSON block with PoseModelProvenance fields "
    "(tool_version, model_name, checkpoint_id, training_set_reference, ...)."
)


def _validate_format(format_value: str, allowed: tuple[str, ...], family: str) -> str:
    if format_value not in allowed:
        choices = ", ".join(allowed)
        raise typer.BadParameter(
            f"unknown {family} format {format_value!r}. Choose from: {choices}"
        )
    return format_value


def _import_payload(source: str, project: str, state_path: Path) -> dict[str, str]:
    return {
        "status": "imported",
        "source": source,
        "project": project,
        "state_path": str(state_path),
    }


def _calibration_payload(source: str, project: str, calibration_path: Path) -> dict[str, str]:
    return {
        "status": "imported",
        "source": source,
        "project": project,
        "calibration_path": str(calibration_path),
    }


def _emit_import_result(payload: dict[str, Any], label: str) -> None:
    sys.stdout.write(f"Imported {label} into {payload['project']}\n")
    write_path(Path(str(payload["state_path"])))


def _emit_calibration_import_result(payload: dict[str, Any], label: str) -> None:
    sys.stdout.write(f"Imported {label} into {payload['project']}\n")
    write_path(Path(str(payload["calibration_path"])))


def _load_provenance_json(path: str | None, option_name: str) -> dict[str, Any] | None:
    if path is None:
        return None
    return load_json_dict(require_option_value(path, option_name))


@app.command("pose")
def import_pose(
    format: Annotated[
        str,
        typer.Argument(
            help="Pose format to import (e.g. dlc-csv, sleap-h5, sleap-package).",
        ),
    ],
    out: Annotated[str, typer.Option("--out", help="Output project directory.")],
    path: Annotated[
        str | None,
        typer.Option(
            "--path",
            "--input-json",
            help=(
                "Path to the input file or directory (CSV/H5/JSON/.pkg.slp/DLC project)."
            ),
        ),
    ] = None,
    video: Annotated[
        str | None,
        typer.Option("--video", help="Path to the matching video file (per-clip formats)."),
    ] = None,
    skeleton_name: Annotated[
        str | None,
        typer.Option("--skeleton-name", help="Skeleton name to store in the project."),
    ] = None,
    likelihood_threshold: Annotated[
        float,
        typer.Option(
            "--likelihood-threshold",
            "--threshold",
            callback=require_likelihood_threshold,
            help="Drop keypoints below this likelihood (0.0-1.0).",
        ),
    ] = 0.0,
    instance_index: Annotated[
        int,
        typer.Option(
            "--instance-index",
            callback=require_nonnegative_int,
            help="Instance index for multi-person formats (mmpose-topdown-json only).",
        ),
    ] = 0,
    fps: Annotated[
        int,
        typer.Option(
            "--fps",
            callback=require_positive_int,
            help="Frames per second for sleap-package video re-encoding.",
        ),
    ] = 30,
    encode_videos: Annotated[
        bool,
        typer.Option(
            "--encode-videos/--no-encode-videos",
            help="Encode MP4 videos into the internal store (sleap-package only).",
        ),
    ] = True,
    prediction_provenance: Annotated[
        str | None,
        typer.Option(
            "--prediction-provenance",
            "--provenance",
            help="Path to a JSON block with upstream prediction provenance.",
        ),
    ] = None,
    provenance_json: Annotated[
        str | None,
        typer.Option(
            "--provenance-json",
            help=_MODEL_PROVENANCE_HELP,
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite an existing project state."),
    ] = False,
    json_output: JsonOption = False,
) -> None:
    """Import a pose track using the kebab-case dispatch format string."""
    format = _validate_format(format, _POSE_FORMATS, "pose")
    out = require_option_value(out, "--out")
    path = require_option_value(path, "--path")

    if format in _PER_CLIP_POSE_FORMATS:
        video_path: str | None = require_option_value(video, "--video")
    else:
        video_path = video

    def action() -> dict[str, str]:
        prediction = _load_provenance_json(prediction_provenance, "--prediction-provenance")
        model_provenance = _load_provenance_json(provenance_json, "--provenance-json")
        progress = progress_callback(json_output)

        common: dict[str, Any] = {
            "prediction_provenance": prediction,
            "provenance": model_provenance,
            "force": force,
            "progress_callback": progress,
        }

        # No-video formats dispatch first so the per-clip branches can rely
        # on `video_path` being non-None.
        if format == "dlc-project":
            state_path = import_dlc_project_directory(
                path,
                out,
                skeleton_name=skeleton_name,
                likelihood_threshold=likelihood_threshold,
                **common,
            )
        elif format == "sleap-package":
            state_path = import_sleap_package_project(
                path,
                out,
                fps=fps,
                encode_videos=encode_videos,
                **common,
            )
        else:
            assert video_path is not None  # narrowed via _PER_CLIP_POSE_FORMATS above
            if format == "dlc-csv":
                state_path = import_dlc_csv_project(
                    path,
                    video_path,
                    out,
                    skeleton_name=skeleton_name or "imported",
                    likelihood_threshold=likelihood_threshold,
                    **common,
                )
            elif format == "dlc-h5":
                state_path = import_dlc_h5_project(
                    path,
                    video_path,
                    out,
                    skeleton_name=skeleton_name or "imported",
                    likelihood_threshold=likelihood_threshold,
                    **common,
                )
            elif format == "lightning-pose-csv":
                state_path = import_lightning_pose_csv_project(
                    path,
                    video_path,
                    out,
                    skeleton_name=skeleton_name or "imported",
                    likelihood_threshold=likelihood_threshold,
                    **common,
                )
            elif format == "mediapipe-pose-landmarks-json":
                state_path = import_mediapipe_pose_landmarks_json_project(
                    path,
                    video_path,
                    out,
                    skeleton_name=skeleton_name or "mediapipe_pose",
                    likelihood_threshold=likelihood_threshold,
                    **common,
                )
            elif format == "mmpose-topdown-json":
                state_path = import_mmpose_topdown_json_project(
                    path,
                    video_path,
                    out,
                    skeleton_name=skeleton_name or "imported",
                    instance_index=instance_index,
                    likelihood_threshold=likelihood_threshold,
                    **common,
                )
            elif format == "sleap-h5":
                state_path = import_sleap_h5_project(
                    path,
                    video_path,
                    out,
                    skeleton_name=skeleton_name or "imported",
                    likelihood_threshold=likelihood_threshold,
                    **common,
                )
            else:  # pragma: no cover - defended above
                raise ValueError(f"Unknown pose format: {format!r}")

        return _import_payload(format, out, state_path)

    label = _POSE_FORMAT_LABELS[format]
    run_command(
        json_output=json_output,
        action=action,
        human_output=lambda payload: _emit_import_result(payload, label),
    )


@app.command("calibration")
def import_calibration(
    format: Annotated[
        str,
        typer.Argument(help="Calibration format to import (e.g. anipose)."),
    ],
    out: Annotated[str, typer.Option("--out", help="Output project directory.")],
    path: Annotated[
        str | None,
        typer.Option("--path", help="Path to the calibration source file."),
    ] = None,
    calibration_id: Annotated[
        str | None,
        typer.Option("--calibration-id", help="Managed calibration ID to write."),
    ] = None,
    name: Annotated[
        str | None,
        typer.Option("--name", help="Human-readable calibration name."),
    ] = None,
    units: Annotated[
        str,
        typer.Option("--units", help="World-coordinate units for translations."),
    ] = "unknown",
    captured_at: Annotated[
        str | None,
        typer.Option("--captured-at", help="Calibration capture timestamp."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite an existing calibration JSON file."),
    ] = False,
    json_output: JsonOption = False,
) -> None:
    """Import a camera calibration using the kebab-case dispatch format string."""
    format = _validate_format(format, _CALIBRATION_FORMATS, "calibration")
    out = require_option_value(out, "--out")
    path = require_option_value(path, "--path")

    def action() -> dict[str, str]:
        if format == "anipose":
            calibration_path = import_anipose_calibration_project(
                path,
                out,
                calibration_id=calibration_id,
                name=name,
                units=units,
                captured_at=captured_at,
                force=force,
            )
        else:  # pragma: no cover - defended above
            raise ValueError(f"Unknown calibration format: {format!r}")
        return _calibration_payload(format, out, calibration_path)

    label = "Anipose calibration" if format == "anipose" else format
    run_command(
        json_output=json_output,
        action=action,
        human_output=lambda payload: _emit_calibration_import_result(payload, label),
    )


@app.command("motion")
def import_motion(
    format: Annotated[
        str,
        typer.Argument(help="Motion format to import (vicon, vicon-csv, vicon-c3d)."),
    ],
    out: Annotated[str, typer.Option("--out", help="Output project directory.")],
    path: Annotated[
        str | None,
        typer.Option("--path", help="Path to the motion-capture recording."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite an existing project state."),
    ] = False,
    json_output: JsonOption = False,
) -> None:
    """Import a motion-capture recording using the kebab-case dispatch format string."""
    format = _validate_format(format, _MOTION_FORMATS, "motion")
    out = require_option_value(out, "--out")
    path = require_option_value(path, "--path")

    def action() -> dict[str, str]:
        progress = progress_callback(json_output)
        if format == "vicon":
            state_path = import_vicon_project(
                path, out, force=force, progress_callback=progress
            )
        elif format == "vicon-csv":
            state_path = import_vicon_csv_project(
                path, out, force=force, progress_callback=progress
            )
        elif format == "vicon-c3d":
            state_path = import_vicon_c3d_project(
                path, out, force=force, progress_callback=progress
            )
        else:  # pragma: no cover - defended above
            raise ValueError(f"Unknown motion format: {format!r}")
        return _import_payload(format, out, state_path)

    label = _MOTION_FORMAT_LABELS[format]
    run_command(
        json_output=json_output,
        action=action,
        human_output=lambda payload: _emit_import_result(payload, label),
    )
