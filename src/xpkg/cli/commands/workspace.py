"""Workspace lifecycle commands."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, Any

import typer

from xpkg.cli.shared import JsonOption, PackMedia, run_command, write_path
from xpkg.workspace import (
    current_project_state_path,
    init_project,
    load_project_descriptor,
    pack_project,
    project_descriptor_path,
    resolve_workspace_root,
    unpack_project,
    workspace_artifacts_root,
    workspace_exports_root,
    workspace_media_root,
    workspace_state_root,
    workspace_store_root,
)
from xpkg.workspace import (
    validate_artifact as validate_artifact_target,
)

app = typer.Typer(
    add_completion=False,
    help="Create, inspect, validate, pack, and unpack workspace-first projects.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


def _workspace_describe_payload(path: str) -> dict[str, Any]:
    root = resolve_workspace_root(path)
    if root is None:
        raise FileNotFoundError(f"Not an xpkg workspace: {path}")
    descriptor = load_project_descriptor(root)
    current_state = current_project_state_path(root)
    return {
        "status": "described",
        "workspace": str(root),
        "descriptor": descriptor.to_dict(),
        "paths": {
            "descriptor": str(project_descriptor_path(root)),
            "store": str(workspace_store_root(root)),
            "artifacts": str(workspace_artifacts_root(root)),
            "state": str(workspace_state_root(root)),
            "media": str(workspace_media_root(root)),
            "exports": str(workspace_exports_root(root)),
            "current_state": str(current_state),
        },
        "has_current_state": current_state.exists(),
    }


def _emit_workspace_description(payload: dict[str, Any]) -> None:
    paths = payload["paths"]
    sys.stdout.write(f"Workspace {payload['workspace']}\n")
    sys.stdout.write(f"Descriptor {paths['descriptor']}\n")
    sys.stdout.write(f"Store {paths['store']}\n")
    sys.stdout.write(f"Artifacts {paths['artifacts']}\n")
    sys.stdout.write(f"State {paths['state']}\n")
    sys.stdout.write(f"Media {paths['media']}\n")
    sys.stdout.write(f"Exports {paths['exports']}\n")
    sys.stdout.write(f"Current state present: {payload['has_current_state']}\n")


@app.command("describe")
def describe(
    workspace: Annotated[str, typer.Argument(help="Workspace directory to inspect.")],
    json_output: JsonOption = False,
) -> None:
    """Describe the normalized workspace layout and descriptor."""

    run_command(
        json_output=json_output,
        action=lambda: _workspace_describe_payload(workspace),
        human_output=_emit_workspace_description,
    )


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
            force=force,
        )
        return {
            "status": "initialized",
            "workspace": workspace,
            "title": title,
            "project_id": project_id,
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
    media: Annotated[
        PackMedia,
        typer.Option(
            "--media",
            help=(
                "Media scope: full includes all managed media, package omits video "
                "containers, manifest records media without storing bytes."
            ),
        ),
    ] = PackMedia.full,
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
            media=media.value,
            overwrite=overwrite,
        )
        return {
            "status": "packed",
            "workspace": workspace,
            "artifact": str(artifact_path),
            "media": media.value,
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
