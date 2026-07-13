"""Timebase-synchronization import command."""

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
    from xpkg.services.project import SynchronizationFormat


def _synchronization_formats() -> tuple[str, ...]:
    from xpkg.services.project import SynchronizationFormat

    return tuple(str(value) for value in get_args(SynchronizationFormat))


@app.command("synchronization")
def import_synchronization(
    format: Annotated[str, typer.Argument(help="Synchronization format.")],
    out: Annotated[str, typer.Option("--out", help="Output project directory.")],
    path: Annotated[
        str | None, typer.Option("--path", help="Paired synchronization source file.")
    ] = None,
    source_timebase: Annotated[
        str | None, typer.Option("--source-timebase", help="Source clock name.")
    ] = None,
    target_timebase: Annotated[
        str | None, typer.Option("--target-timebase", help="Target clock name.")
    ] = None,
    model: Annotated[
        str, typer.Option("--model", help="Transform model: offset or affine.")
    ] = "affine",
    method: Annotated[
        str, typer.Option("--method", help="Evidence method: pulses or timestamps.")
    ] = "pulses",
    alignment_name: Annotated[
        str | None, typer.Option("--alignment-name", help="Session alignment link name.")
    ] = None,
    session_id: Annotated[
        str | None, typer.Option("--session-id", help="Canonical recording session ID.")
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Replace the named alignment link.")
    ] = False,
    json_output: JsonOption = False,
) -> None:
    """Import paired clock observations into canonical session state."""
    selected = cast(
        "SynchronizationFormat",
        validate_format(format, _synchronization_formats(), "synchronization"),
    )
    project_path = require_option_value(out, "--out")
    source_path = require_option_value(path, "--path")
    source_name = require_option_value(source_timebase, "--source-timebase")
    target_name = require_option_value(target_timebase, "--target-timebase")

    def action() -> dict[str, str]:
        from xpkg.model import AlignmentModel, SynchronizationMethod, Timebase

        service = open_import_project(project_path, title=session_id, force=force)
        state_path = service.import_synchronization(
            selected,
            path=source_path,
            source_timebase=Timebase(name=source_name),
            target_timebase=Timebase(name=target_name),
            model=AlignmentModel(model),
            method=SynchronizationMethod(method),
            alignment_name=alignment_name,
            session_id=session_id,
            force=force,
        )
        return import_payload(selected, project_path, state_path)

    run_command(
        json_output=json_output,
        action=action,
        human_output=lambda payload: emit_import_result(payload, "timebase alignment"),
    )
