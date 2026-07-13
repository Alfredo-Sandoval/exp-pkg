"""Camera-calibration import command."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, cast, get_args

import typer

from xpkg.cli.commands.imports import app
from xpkg.cli.commands.imports._shared import (
    emit_import_result,
    import_payload,
    open_import_project,
    validate_format,
)
from xpkg.cli.shared import JsonOption, require_option_value, run_command

if TYPE_CHECKING:
    from xpkg.services.project import CalibrationFormat

_LABELS = {
    "anipose": "Anipose calibration",
    "opencv-stereo-yaml": "OpenCV stereo calibration",
}


def _calibration_formats() -> tuple[str, ...]:
    from xpkg.services.project import CalibrationFormat

    return tuple(str(value) for value in get_args(CalibrationFormat))


@app.command("calibration")
def import_calibration(
    format: Annotated[str, typer.Argument(help="Calibration format.")],
    out: Annotated[str, typer.Option("--out", help="Output project directory.")],
    path: Annotated[str | None, typer.Option("--path", help="Calibration source file.")] = None,
    calibration_id: Annotated[
        str | None, typer.Option("--calibration-id", help="Managed calibration ID.")
    ] = None,
    session_id: Annotated[
        str | None, typer.Option("--session-id", help="Target recording-session ID.")
    ] = None,
    name: Annotated[str | None, typer.Option("--name", help="Calibration name.")] = None,
    units: Annotated[str, typer.Option("--units", help="World-coordinate units.")] = "unknown",
    captured_at: Annotated[
        str | None, typer.Option("--captured-at", help="Capture timestamp.")
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite an existing calibration file.")
    ] = False,
    json_output: JsonOption = False,
) -> None:
    """Import one camera calibration through ``ProjectService``."""
    selected = cast(
        "CalibrationFormat",
        validate_format(format, _calibration_formats(), "calibration"),
    )
    project_path = require_option_value(out, "--out")
    source_path = require_option_value(path, "--path")

    def action() -> dict[str, str]:
        service = open_import_project(project_path, force=force)
        result = service.import_calibration(
            selected,
            path=source_path,
            calibration_id=calibration_id,
            session_id=session_id,
            name=name,
            units=units,
            captured_at=captured_at,
            force=force,
        )
        return import_payload(selected, project_path, result)

    run_command(
        json_output=json_output,
        action=action,
        human_output=lambda payload: emit_import_result(payload, _LABELS[selected]),
    )
