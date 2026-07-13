"""Pose import command."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, cast, get_args

import typer

from xpkg._core.json_utils import load_json_dict
from xpkg.cli.commands.imports import app
from xpkg.cli.commands.imports._shared import (
    emit_import_result,
    import_payload,
    open_import_project,
    validate_format,
)
from xpkg.cli.shared import (
    JsonOption,
    progress_callback,
    require_likelihood_threshold,
    require_nonnegative_int,
    require_option_value,
    require_positive_int,
    run_command,
)

if TYPE_CHECKING:
    from xpkg.services.project import PoseFormat

_PER_CLIP_FORMATS = frozenset(
    {
        "dlc-csv",
        "dlc-h5",
        "lightning-pose-csv",
        "mediapipe-pose-landmarks-json",
        "mmpose-topdown-json",
        "sleap-h5",
    }
)
_LABELS = {
    "dlc-csv": "DLC CSV",
    "dlc-h5": "DLC H5",
    "dlc-project": "DLC project",
    "lightning-pose-csv": "Lightning Pose CSV",
    "mediapipe-pose-landmarks-json": "MediaPipe JSON",
    "mmpose-topdown-json": "MMPose JSON",
    "sleap-h5": "SLEAP H5",
    "sleap-package": "SLEAP package",
}


def _pose_formats() -> tuple[str, ...]:
    from xpkg.services.project import PoseFormat

    return tuple(str(value) for value in get_args(PoseFormat))


def _provenance(path: str | None, option: str) -> dict[str, Any] | None:
    return None if path is None else load_json_dict(require_option_value(path, option))


@app.command("pose")
def import_pose(
    format: Annotated[str, typer.Argument(help="Pose format.")],
    out: Annotated[str, typer.Option("--out", help="Output project directory.")],
    path: Annotated[str | None, typer.Option("--path", "--input-json")] = None,
    video: Annotated[str | None, typer.Option("--video")] = None,
    skeleton_name: Annotated[str | None, typer.Option("--skeleton-name")] = None,
    likelihood_threshold: Annotated[
        float,
        typer.Option(
            "--likelihood-threshold", "--threshold", callback=require_likelihood_threshold
        ),
    ] = 0.0,
    instance_index: Annotated[
        int, typer.Option("--instance-index", callback=require_nonnegative_int)
    ] = 0,
    fps: Annotated[int, typer.Option("--fps", callback=require_positive_int)] = 30,
    encode_videos: Annotated[bool, typer.Option("--encode-videos/--no-encode-videos")] = True,
    prediction_provenance: Annotated[
        str | None, typer.Option("--prediction-provenance", "--provenance")
    ] = None,
    provenance_json: Annotated[str | None, typer.Option("--provenance-json")] = None,
    session_id: Annotated[str | None, typer.Option("--session-id")] = None,
    force: Annotated[bool, typer.Option("--force")] = False,
    json_output: JsonOption = False,
) -> None:
    """Import pose data through ``ProjectService``."""
    selected = cast("PoseFormat", validate_format(format, _pose_formats(), "pose"))
    project_path = require_option_value(out, "--out")
    source_path = require_option_value(path, "--path")
    video_path = require_option_value(video, "--video") if selected in _PER_CLIP_FORMATS else video

    def action() -> dict[str, str]:
        service = open_import_project(project_path, force=force)
        state_path = service.import_pose(
            selected,
            path=source_path,
            video=video_path,
            skeleton_name=skeleton_name,
            likelihood_threshold=None if selected == "sleap-package" else likelihood_threshold,
            instance_index=instance_index if selected == "mmpose-topdown-json" else None,
            fps=fps if selected == "sleap-package" else None,
            encode_videos=encode_videos if selected == "sleap-package" else None,
            prediction_provenance=_provenance(prediction_provenance, "--prediction-provenance"),
            provenance=_provenance(provenance_json, "--provenance-json"),
            session_id=session_id,
            force=force,
            progress_callback=progress_callback(json_output),
        )
        return import_payload(selected, project_path, state_path)

    run_command(
        json_output=json_output,
        action=action,
        human_output=lambda payload: emit_import_result(payload, _LABELS[selected]),
    )
