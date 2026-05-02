"""Project lifecycle commands."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, Any

import typer

from xpkg.cli.shared import JsonOption, PackMedia, require_option_value, run_command, write_path
from xpkg.project import (
    current_project_state_path,
    init_project,
    load_project_descriptor,
    pack_project,
    project_artifacts_root,
    project_descriptor_path,
    project_exports_root,
    project_media_root,
    project_state_root,
    project_store_root,
    resolve_project_root,
    unpack_project,
)
from xpkg.project import (
    validate_artifact as validate_artifact_target,
)

app = typer.Typer(
    add_completion=False,
    help="Create, inspect, validate, pack, and unpack project-first projects.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


def _project_describe_payload(path: str) -> dict[str, Any]:
    root = resolve_project_root(path)
    if root is None:
        raise FileNotFoundError(f"Not an xpkg project: {path}")
    descriptor = load_project_descriptor(root)
    current_state = current_project_state_path(root)
    return {
        "status": "described",
        "project": str(root),
        "descriptor": descriptor.to_dict(),
        "paths": {
            "descriptor": str(project_descriptor_path(root)),
            "store": str(project_store_root(root)),
            "artifacts": str(project_artifacts_root(root)),
            "state": str(project_state_root(root)),
            "media": str(project_media_root(root)),
            "exports": str(project_exports_root(root)),
            "current_state": str(current_state),
        },
        "has_current_state": current_state.exists(),
    }


def _emit_project_description(payload: dict[str, Any]) -> None:
    paths = payload["paths"]
    sys.stdout.write(f"Project {payload['project']}\n")
    sys.stdout.write(f"Descriptor {paths['descriptor']}\n")
    sys.stdout.write(f"Store {paths['store']}\n")
    sys.stdout.write(f"Artifacts {paths['artifacts']}\n")
    sys.stdout.write(f"State {paths['state']}\n")
    sys.stdout.write(f"Media {paths['media']}\n")
    sys.stdout.write(f"Exports {paths['exports']}\n")
    sys.stdout.write(f"Current state present: {payload['has_current_state']}\n")


@app.command("describe")
def describe(
    project: Annotated[str, typer.Argument(help="Project directory to inspect.")],
    json_output: JsonOption = False,
) -> None:
    """Describe the normalized project layout and descriptor."""

    run_command(
        json_output=json_output,
        action=lambda: _project_describe_payload(project),
        human_output=_emit_project_description,
    )


@app.command("init")
def init(
    project: Annotated[str, typer.Argument(help="Project directory to create.")],
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
    """Create a new empty exp-pkg project."""

    def action() -> dict[str, object]:
        init_project(
            project,
            title=title,
            project_id=project_id,
            force=force,
        )
        return {
            "status": "initialized",
            "project": project,
            "title": title,
            "project_id": project_id,
        }

    def human_output(payload: dict[str, object]) -> None:
        sys.stdout.write(f"Initialized project {Path(str(payload['project']))}\n")

    run_command(json_output=json_output, action=action, human_output=human_output)


@app.command("pack")
def pack(
    project: Annotated[str, typer.Argument(help="Project directory to pack.")],
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
    """Pack a project into a .expkg artifact."""

    def action() -> dict[str, object]:
        artifact_path = pack_project(
            project,
            out=out,
            media=media.value,
            overwrite=overwrite,
        )
        return {
            "status": "packed",
            "project": project,
            "artifact": str(artifact_path),
            "media": media.value,
        }

    def human_output(payload: dict[str, object]) -> None:
        sys.stdout.write(f"Packed {payload['project']}\n")
        write_path(Path(str(payload["artifact"])))

    run_command(json_output=json_output, action=action, human_output=human_output)


@app.command("unpack")
def unpack(
    artifact: Annotated[str, typer.Argument(help="Path to the .expkg artifact.")],
    out: Annotated[str, typer.Option("--out", help="Destination project directory.")],
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
    """Unpack a .expkg artifact into a project."""
    out = require_option_value(out, "--out")

    def action() -> dict[str, object]:
        project_path = unpack_project(
            artifact,
            out,
            force=force,
            rename_title=rename,
        )
        return {
            "status": "unpacked",
            "artifact": artifact,
            "project": str(project_path),
            "title": rename,
        }

    def human_output(payload: dict[str, object]) -> None:
        sys.stdout.write(f"Unpacked {payload['artifact']}\n")
        write_path(Path(str(payload["project"])))

    run_command(json_output=json_output, action=action, human_output=human_output)


@app.command("validate")
def validate(
    path: Annotated[str, typer.Argument(help="Project or .expkg artifact to validate.")],
    json_output: JsonOption = False,
) -> None:
    """Validate a project or packed .expkg artifact."""

    def action() -> dict[str, object]:
        validate_artifact_target(path)
        return {"status": "valid", "path": path}

    def human_output(payload: dict[str, object]) -> None:
        sys.stdout.write(f"Valid {payload['path']}\n")

    run_command(json_output=json_output, action=action, human_output=human_output)
