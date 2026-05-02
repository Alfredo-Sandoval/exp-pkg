"""Workspace lifecycle commands."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer

from xpkg.cli.shared import JsonOption, PackMode, run_command, write_path
from xpkg.formats import init_project, pack_project, unpack_project
from xpkg.formats import validate_artifact as validate_artifact_target


def register(app: typer.Typer) -> None:
    """Register workspace lifecycle commands on the root app."""

    @app.command("init")
    def init(
        workspace: Annotated[str, typer.Argument(help="Workspace directory to create.")],
        title: Annotated[
            str | None,
            typer.Option("--title", help="Optional project title."),
        ] = None,
        project_id: Annotated[
            str | None,
            typer.Option("--id", help="Optional project identifier."),
        ] = None,
        pack_mode: Annotated[
            PackMode,
            typer.Option("--pack-mode", help="Default pack mode recorded in PROJECT.json."),
        ] = PackMode.portable,
        force: Annotated[
            bool,
            typer.Option("--force", help="Allow initialization into an existing empty directory."),
        ] = False,
        json_output: JsonOption = False,
    ) -> None:
        """Create a new empty exp-pkg workspace."""

        def action() -> dict[str, object]:
            init_project(
                workspace,
                title=title,
                project_id=project_id,
                default_pack_mode=pack_mode.value,
                force=force,
            )
            return {
                "status": "initialized",
                "workspace": workspace,
                "title": title,
                "project_id": project_id,
                "pack_mode": pack_mode.value,
            }

        def human_output(payload: dict[str, object]) -> None:
            sys.stdout.write(f"Initialized workspace {Path(str(payload['workspace']))}\n")

        run_command(json_output=json_output, action=action, human_output=human_output)

    @app.command("pack")
    def pack(
        workspace: Annotated[str, typer.Argument(help="Workspace directory to pack.")],
        out: Annotated[
            str | None,
            typer.Option("--out", help="Explicit output .expkg path."),
        ] = None,
        mode: Annotated[
            PackMode | None,
            typer.Option("--mode", help="Pack mode. Defaults to the workspace default."),
        ] = None,
        overwrite: Annotated[
            bool,
            typer.Option("--overwrite", help="Replace an existing output artifact."),
        ] = False,
        json_output: JsonOption = False,
    ) -> None:
        """Pack a workspace into a .expkg artifact."""

        def action() -> dict[str, object]:
            artifact_path = pack_project(
                workspace,
                out=out,
                mode=mode.value if mode is not None else None,
                overwrite=overwrite,
            )
            return {
                "status": "packed",
                "workspace": workspace,
                "artifact": str(artifact_path),
                "mode": mode.value if mode is not None else None,
            }

        def human_output(payload: dict[str, object]) -> None:
            sys.stdout.write(f"Packed {payload['workspace']}\n")
            write_path(Path(str(payload["artifact"])))

        run_command(json_output=json_output, action=action, human_output=human_output)

    @app.command("unpack")
    def unpack(
        artifact: Annotated[str, typer.Argument(help="Path to the .expkg artifact.")],
        out: Annotated[str, typer.Option("--out", help="Destination workspace directory.")],
        force: Annotated[
            bool,
            typer.Option("--force", help="Allow unpacking into an existing empty directory."),
        ] = False,
        rename: Annotated[
            str | None,
            typer.Option("--rename", help="Optional new project title."),
        ] = None,
        json_output: JsonOption = False,
    ) -> None:
        """Unpack a .expkg artifact into a workspace."""

        def action() -> dict[str, object]:
            workspace_path = unpack_project(
                artifact,
                out,
                force=force,
                rename_title=rename,
            )
            return {
                "status": "unpacked",
                "artifact": artifact,
                "workspace": str(workspace_path),
                "title": rename,
            }

        def human_output(payload: dict[str, object]) -> None:
            sys.stdout.write(f"Unpacked {payload['artifact']}\n")
            write_path(Path(str(payload["workspace"])))

        run_command(json_output=json_output, action=action, human_output=human_output)

    @app.command("validate")
    def validate(
        path: Annotated[str, typer.Argument(help="Workspace or .expkg artifact to validate.")],
        json_output: JsonOption = False,
    ) -> None:
        """Validate a workspace or packed .expkg artifact."""

        def action() -> dict[str, object]:
            validate_artifact_target(path)
            return {"status": "valid", "path": path}

        def human_output(payload: dict[str, object]) -> None:
            sys.stdout.write(f"Valid {payload['path']}\n")

        run_command(json_output=json_output, action=action, human_output=human_output)
