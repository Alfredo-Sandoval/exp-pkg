"""Behavior-label import command."""

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
    from xpkg.services.project import BehaviorFormat


def _behavior_formats() -> tuple[str, ...]:
    from xpkg.services.project import BehaviorFormat

    return tuple(str(value) for value in get_args(BehaviorFormat))


@app.command("behavior")
def import_behavior(
    format: Annotated[str, typer.Argument(help="Behavior-label format.")],
    out: Annotated[str, typer.Option("--out", help="Output project directory.")],
    path: Annotated[str | None, typer.Option("--path", help="Behavior source file.")] = None,
    behavior_name: Annotated[
        str, typer.Option("--behavior-name", help="Session behavior link name.")
    ] = "behavior",
    session_id: Annotated[
        str | None, typer.Option("--session-id", help="Canonical recording session ID.")
    ] = None,
    video_role: Annotated[
        str | None, typer.Option("--video-role", help="Existing session video role.")
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Replace the named behavior link.")
    ] = False,
    json_output: JsonOption = False,
) -> None:
    """Import behavior labels into canonical recording-session state."""
    selected = cast(
        "BehaviorFormat",
        validate_format(format, _behavior_formats(), "behavior"),
    )
    project_path = require_option_value(out, "--out")
    source_path = require_option_value(path, "--path")

    def action() -> dict[str, str]:
        service = open_import_project(project_path, title=session_id, force=force)
        state_path = service.import_behavior(
            selected,
            path=source_path,
            behavior_name=behavior_name,
            session_id=session_id,
            video_role=video_role,
            force=force,
        )
        return import_payload(selected, project_path, state_path)

    run_command(
        json_output=json_output,
        action=action,
        human_output=lambda payload: emit_import_result(payload, "behavior labels"),
    )
