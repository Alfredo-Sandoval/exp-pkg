"""Event-table import command."""

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
    from xpkg.services.project import EventFormat


def _event_formats() -> tuple[str, ...]:
    from xpkg.services.project import EventFormat

    return tuple(str(value) for value in get_args(EventFormat))


@app.command("events")
def import_events(
    format: Annotated[str, typer.Argument(help="Event format (events-csv).")],
    out: Annotated[str, typer.Option("--out", help="Output project directory.")],
    path: Annotated[str | None, typer.Option("--path", help="Event source file.")] = None,
    session_id: Annotated[
        str | None, typer.Option("--session-id", help="Canonical recording session ID.")
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Replace the existing session event table.")
    ] = False,
    json_output: JsonOption = False,
) -> None:
    """Import event records into canonical recording-session state."""
    selected = cast(
        "EventFormat",
        validate_format(format, _event_formats(), "event"),
    )
    project_path = require_option_value(out, "--out")
    source_path = require_option_value(path, "--path")

    def action() -> dict[str, str]:
        service = open_import_project(project_path, title=session_id, force=force)
        state_path = service.import_events(
            selected,
            path=source_path,
            session_id=session_id,
            force=force,
        )
        return import_payload(selected, project_path, state_path)

    run_command(
        json_output=json_output,
        action=action,
        human_output=lambda payload: emit_import_result(payload, "event CSV"),
    )
