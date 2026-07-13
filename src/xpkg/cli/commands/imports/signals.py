"""Sampled-signal import command."""

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
    from xpkg.services.project import SignalFormat


def _signal_formats() -> tuple[str, ...]:
    from xpkg.services.project import SignalFormat

    return tuple(str(value) for value in get_args(SignalFormat))


@app.command("signals")
def import_signals(
    format: Annotated[str, typer.Argument(help="Signal format (e.g. photometry-csv).")],
    out: Annotated[str, typer.Option("--out", help="Output project directory.")],
    path: Annotated[str | None, typer.Option("--path", help="Signal source file.")] = None,
    session_id: Annotated[
        str | None, typer.Option("--session-id", help="Canonical recording session ID.")
    ] = None,
    signal_name: Annotated[
        str, typer.Option("--signal-name", help="Session signal link name.")
    ] = "photometry",
    force: Annotated[
        bool, typer.Option("--force", help="Replace existing recording-session state.")
    ] = False,
    json_output: JsonOption = False,
) -> None:
    """Import sampled signals into canonical recording-session state."""
    selected = cast(
        "SignalFormat",
        validate_format(format, _signal_formats(), "signal"),
    )
    project_path = require_option_value(out, "--out")
    source_path = require_option_value(path, "--path")

    def action() -> dict[str, str]:
        service = open_import_project(project_path, title=session_id, force=force)
        state_path = service.import_signals(
            selected,
            path=source_path,
            session_id=session_id,
            signal_name=signal_name,
            force=force,
        )
        return import_payload(selected, project_path, state_path)

    run_command(
        json_output=json_output,
        action=action,
        human_output=lambda payload: emit_import_result(payload, "photometry CSV"),
    )
