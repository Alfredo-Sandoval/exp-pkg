"""CLI commands that import external data into project-first xpkg projects."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, Any

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

app = typer.Typer(
    add_completion=False,
    help="Import external tracking data into a project-first exp-pkg project.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
dlc_app = typer.Typer(
    add_completion=False,
    help="Import DeepLabCut tracking data into a project.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
sleap_app = typer.Typer(
    add_completion=False,
    help="Import SLEAP data into a project.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
vicon_app = typer.Typer(
    add_completion=False,
    help="Import Vicon recordings into a project.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


def _import_payload(source: str, project: str, state_path: Path) -> dict[str, str]:
    return {
        "status": "imported",
        "source": source,
        "project": project,
        "state_path": str(state_path),
    }


def _emit_import_result(payload: dict[str, Any], label: str) -> None:
    sys.stdout.write(f"Imported {label} into {payload['project']}\n")
    write_path(Path(str(payload["state_path"])))


@dlc_app.command("csv")
def import_dlc_csv(
    csv: Annotated[str, typer.Option("--csv", help="Path to a DLC CSV tracking file.")],
    video: Annotated[str, typer.Option("--video", help="Path to the matching video file.")],
    out: Annotated[str, typer.Option("--out", help="Output project directory.")],
    skeleton_name: Annotated[
        str,
        typer.Option("--skeleton-name", help="Skeleton name to store in the project."),
    ] = "imported",
    threshold: Annotated[
        float,
        typer.Option("--threshold", callback=require_likelihood_threshold),
    ] = 0.0,
    json_output: JsonOption = False,
) -> None:
    """Import a DLC CSV tracking file."""
    csv = require_option_value(csv, "--csv")
    video = require_option_value(video, "--video")
    out = require_option_value(out, "--out")

    def action() -> dict[str, str]:
        state_path = import_dlc_csv_project(
            csv,
            video,
            out,
            skeleton_name=skeleton_name,
            likelihood_threshold=threshold,
            progress_callback=progress_callback(json_output),
        )
        return _import_payload("dlc_csv", out, state_path)

    run_command(
        json_output=json_output,
        action=action,
        human_output=lambda payload: _emit_import_result(payload, "DLC CSV"),
    )


@dlc_app.command("h5")
def import_dlc_h5(
    h5: Annotated[str, typer.Option("--h5", help="Path to a DLC H5 tracking file.")],
    video: Annotated[str, typer.Option("--video", help="Path to the matching video file.")],
    out: Annotated[str, typer.Option("--out", help="Output project directory.")],
    skeleton_name: Annotated[
        str,
        typer.Option("--skeleton-name", help="Skeleton name to store in the project."),
    ] = "imported",
    threshold: Annotated[
        float,
        typer.Option("--threshold", callback=require_likelihood_threshold),
    ] = 0.0,
    json_output: JsonOption = False,
) -> None:
    """Import a DLC H5 tracking file."""
    h5 = require_option_value(h5, "--h5")
    video = require_option_value(video, "--video")
    out = require_option_value(out, "--out")

    def action() -> dict[str, str]:
        state_path = import_dlc_h5_project(
            h5,
            video,
            out,
            skeleton_name=skeleton_name,
            likelihood_threshold=threshold,
            progress_callback=progress_callback(json_output),
        )
        return _import_payload("dlc_h5", out, state_path)

    run_command(
        json_output=json_output,
        action=action,
        human_output=lambda payload: _emit_import_result(payload, "DLC H5"),
    )


@dlc_app.command("project")
def import_dlc_project(
    project: Annotated[str, typer.Option("--project", help="Path to the DLC project root.")],
    out: Annotated[str, typer.Option("--out", help="Output project directory.")],
    threshold: Annotated[
        float,
        typer.Option("--threshold", callback=require_likelihood_threshold),
    ] = 0.0,
    json_output: JsonOption = False,
) -> None:
    """Import an entire DLC project into one project."""
    project = require_option_value(project, "--project")
    out = require_option_value(out, "--out")

    def action() -> dict[str, str]:
        state_path = import_dlc_project_directory(
            project,
            out,
            likelihood_threshold=threshold,
            progress_callback=progress_callback(json_output),
        )
        return _import_payload("dlc_project", out, state_path)

    run_command(
        json_output=json_output,
        action=action,
        human_output=lambda payload: _emit_import_result(payload, "DLC project"),
    )


@app.command("lightning-pose")
def import_lightning_pose(
    csv: Annotated[
        str,
        typer.Option("--csv", help="Path to a Lightning Pose prediction CSV."),
    ],
    video: Annotated[str, typer.Option("--video", help="Path to the matching video file.")],
    out: Annotated[str, typer.Option("--out", help="Output project directory.")],
    skeleton_name: Annotated[
        str,
        typer.Option("--skeleton-name", help="Skeleton name to store in the project."),
    ] = "imported",
    threshold: Annotated[
        float,
        typer.Option("--threshold", callback=require_likelihood_threshold),
    ] = 0.0,
    json_output: JsonOption = False,
) -> None:
    """Import Lightning Pose CSV predictions into a project."""
    csv = require_option_value(csv, "--csv")
    video = require_option_value(video, "--video")
    out = require_option_value(out, "--out")

    def action() -> dict[str, str]:
        state_path = import_lightning_pose_csv_project(
            csv,
            video,
            out,
            skeleton_name=skeleton_name,
            likelihood_threshold=threshold,
            progress_callback=progress_callback(json_output),
        )
        return _import_payload("lightning_pose_csv", out, state_path)

    run_command(
        json_output=json_output,
        action=action,
        human_output=lambda payload: _emit_import_result(payload, "Lightning Pose CSV"),
    )


@app.command("mediapipe")
def import_mediapipe(
    json_path: Annotated[
        str,
        typer.Option("--input-json", help="Path to MediaPipe pose-landmarks JSON."),
    ],
    video: Annotated[str, typer.Option("--video", help="Path to the matching video file.")],
    out: Annotated[str, typer.Option("--out", help="Output project directory.")],
    skeleton_name: Annotated[
        str,
        typer.Option("--skeleton-name", help="Skeleton name to store in the project."),
    ] = "mediapipe_pose",
    threshold: Annotated[
        float,
        typer.Option("--threshold", callback=require_likelihood_threshold),
    ] = 0.0,
    json_output: JsonOption = False,
) -> None:
    """Import MediaPipe pose-landmarks JSON into a project."""
    json_path = require_option_value(json_path, "--input-json")
    video = require_option_value(video, "--video")
    out = require_option_value(out, "--out")

    def action() -> dict[str, str]:
        state_path = import_mediapipe_pose_landmarks_json_project(
            json_path,
            video,
            out,
            skeleton_name=skeleton_name,
            likelihood_threshold=threshold,
            progress_callback=progress_callback(json_output),
        )
        return _import_payload("mediapipe_pose_landmarks_json", out, state_path)

    run_command(
        json_output=json_output,
        action=action,
        human_output=lambda payload: _emit_import_result(payload, "MediaPipe JSON"),
    )


@app.command("mmpose")
def import_mmpose(
    json_path: Annotated[
        str,
        typer.Option("--input-json", help="Path to an MMPose JSON export."),
    ],
    video: Annotated[str, typer.Option("--video", help="Path to the matching video file.")],
    out: Annotated[str, typer.Option("--out", help="Output project directory.")],
    skeleton_name: Annotated[
        str,
        typer.Option("--skeleton-name", help="Skeleton name to store in the project."),
    ] = "imported",
    instance_index: Annotated[
        int,
        typer.Option("--instance-index", callback=require_nonnegative_int),
    ] = 0,
    threshold: Annotated[
        float,
        typer.Option("--threshold", callback=require_likelihood_threshold),
    ] = 0.0,
    json_output: JsonOption = False,
) -> None:
    """Import MMPose top-down demo JSON predictions into a project."""
    json_path = require_option_value(json_path, "--input-json")
    video = require_option_value(video, "--video")
    out = require_option_value(out, "--out")

    def action() -> dict[str, str]:
        state_path = import_mmpose_topdown_json_project(
            json_path,
            video,
            out,
            skeleton_name=skeleton_name,
            instance_index=instance_index,
            likelihood_threshold=threshold,
            progress_callback=progress_callback(json_output),
        )
        return _import_payload("mmpose_topdown_json", out, state_path)

    run_command(
        json_output=json_output,
        action=action,
        human_output=lambda payload: _emit_import_result(payload, "MMPose JSON"),
    )


@sleap_app.command("h5")
def import_sleap_h5(
    out: Annotated[str, typer.Option("--out", help="Output project directory.")],
    h5: Annotated[
        str,
        typer.Option("--h5", help="Path to the SLEAP analysis H5 export."),
    ],
    video: Annotated[
        str,
        typer.Option("--video", help="Path to the matching video file."),
    ],
    skeleton_name: Annotated[
        str,
        typer.Option("--skeleton-name", help="Skeleton name to store in the project."),
    ] = "imported",
    threshold: Annotated[
        float,
        typer.Option("--threshold", callback=require_likelihood_threshold),
    ] = 0.0,
    json_output: JsonOption = False,
) -> None:
    """Import a SLEAP analysis H5 export into a project."""
    out = require_option_value(out, "--out")
    h5 = require_option_value(h5, "--h5")
    video = require_option_value(video, "--video")

    def action() -> dict[str, str]:
        state_path = import_sleap_h5_project(
            h5,
            video,
            out,
            skeleton_name=skeleton_name,
            likelihood_threshold=threshold,
            progress_callback=progress_callback(json_output),
        )
        return _import_payload("sleap_h5", out, state_path)

    run_command(
        json_output=json_output,
        action=action,
        human_output=lambda payload: _emit_import_result(payload, "SLEAP H5"),
    )


@sleap_app.command("package")
def import_sleap_package(
    out: Annotated[str, typer.Option("--out", help="Output project directory.")],
    slp: Annotated[
        str,
        typer.Option("--slp", help="Path to the input .pkg.slp archive."),
    ],
    fps: Annotated[int, typer.Option("--fps", callback=require_positive_int)] = 30,
    encode_videos: Annotated[
        bool,
        typer.Option("--videos/--no-videos", help="Encode MP4 videos into the internal store."),
    ] = True,
    json_output: JsonOption = False,
) -> None:
    """Import a SLEAP package into a project."""
    out = require_option_value(out, "--out")
    slp = require_option_value(slp, "--slp")

    def action() -> dict[str, str]:
        state_path = import_sleap_package_project(
            slp,
            out,
            fps=fps,
            encode_videos=encode_videos,
            progress_callback=progress_callback(json_output),
        )
        return _import_payload("sleap_package", out, state_path)

    run_command(
        json_output=json_output,
        action=action,
        human_output=lambda payload: _emit_import_result(payload, "SLEAP package"),
    )


@vicon_app.command("c3d")
def import_vicon_c3d(
    out: Annotated[str, typer.Option("--out", help="Output project directory.")],
    c3d: Annotated[
        str,
        typer.Option("--c3d", help="Path to a Vicon C3D recording."),
    ],
    json_output: JsonOption = False,
) -> None:
    """Import a Vicon C3D recording into a project."""
    out = require_option_value(out, "--out")
    c3d = require_option_value(c3d, "--c3d")

    def action() -> dict[str, str]:
        state_path = import_vicon_c3d_project(
            c3d,
            out,
            progress_callback=progress_callback(json_output),
        )
        return _import_payload("vicon_c3d", out, state_path)

    run_command(
        json_output=json_output,
        action=action,
        human_output=lambda payload: _emit_import_result(payload, "Vicon C3D"),
    )


@vicon_app.command("csv")
def import_vicon_csv(
    out: Annotated[str, typer.Option("--out", help="Output project directory.")],
    csv: Annotated[
        str,
        typer.Option("--csv", help="Path to a Vicon Nexus CSV export."),
    ],
    json_output: JsonOption = False,
) -> None:
    """Import a Vicon Nexus CSV recording into a project."""
    out = require_option_value(out, "--out")
    csv = require_option_value(csv, "--csv")

    def action() -> dict[str, str]:
        state_path = import_vicon_csv_project(
            csv,
            out,
            progress_callback=progress_callback(json_output),
        )
        return _import_payload("vicon_csv", out, state_path)

    run_command(
        json_output=json_output,
        action=action,
        human_output=lambda payload: _emit_import_result(payload, "Vicon CSV"),
    )


@vicon_app.command("recording")
def import_vicon_recording(
    out: Annotated[str, typer.Option("--out", help="Output project directory.")],
    recording: Annotated[
        str,
        typer.Option("--recording", help="Path to a Vicon recording (.csv or .c3d)."),
    ],
    json_output: JsonOption = False,
) -> None:
    """Import an auto-detected Vicon recording into a project."""
    out = require_option_value(out, "--out")
    recording = require_option_value(recording, "--recording")

    def action() -> dict[str, str]:
        state_path = import_vicon_project(
            recording,
            out,
            progress_callback=progress_callback(json_output),
        )
        return _import_payload("vicon_recording", out, state_path)

    run_command(
        json_output=json_output,
        action=action,
        human_output=lambda payload: _emit_import_result(payload, "Vicon recording"),
    )


app.add_typer(dlc_app, name="dlc")
app.add_typer(sleap_app, name="sleap")
app.add_typer(vicon_app, name="vicon")
