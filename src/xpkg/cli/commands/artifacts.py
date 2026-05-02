"""Workspace artifact commands."""

from __future__ import annotations

import sys
from typing import Annotated, Any

import typer

from xpkg._core.json_utils import dump_json
from xpkg.cli.shared import JsonOption, run_command
from xpkg.workspace import (
    list_workspace_artifact_index,
    load_workspace_artifact,
    rebuild_workspace_artifact_index,
    validate_workspace_artifact,
    validate_workspace_artifacts,
)

app = typer.Typer(
    add_completion=False,
    help="Inspect and validate workspace artifact manifests.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


@app.command("list")
def artifacts_list(
    workspace: Annotated[str, typer.Argument(help="Workspace directory to inspect.")],
    kind: Annotated[str | None, typer.Option("--kind", help="Optional artifact kind.")] = None,
    namespace: Annotated[
        str | None,
        typer.Option("--namespace", help="Optional caller-owned namespace filter."),
    ] = None,
    json_output: JsonOption = False,
) -> None:
    """List registered artifacts."""

    def action() -> dict[str, Any]:
        entries = list_workspace_artifact_index(
            workspace,
            artifact_type=kind,
            namespace=namespace,
        )
        return {
            "status": "listed",
            "workspace": workspace,
            "count": len(entries),
            "artifacts": [entry.to_dict() for entry in entries],
        }

    def human_output(payload: dict[str, Any]) -> None:
        entries = payload["artifacts"]
        if not entries:
            sys.stdout.write("No artifacts\n")
            return
        for entry in entries:
            namespace_value = entry["namespace"] or "-"
            sys.stdout.write(
                f"{entry['artifact_type']}\t{namespace_value}\t"
                f"{entry['artifact_id']}\t{entry['manifest_path']}\n"
            )

    run_command(json_output=json_output, action=action, human_output=human_output)


@app.command("inspect")
def artifacts_inspect(
    workspace: Annotated[str, typer.Argument(help="Workspace directory to inspect.")],
    artifact_id: Annotated[str, typer.Argument(help="Artifact id to inspect.")],
    kind: Annotated[str | None, typer.Option("--kind", help="Optional artifact kind.")] = None,
    namespace: Annotated[
        str | None,
        typer.Option("--namespace", help="Optional caller-owned namespace."),
    ] = None,
    json_output: JsonOption = False,
) -> None:
    """Print one artifact manifest."""

    def action() -> dict[str, Any]:
        artifact = load_workspace_artifact(
            workspace,
            artifact_id,
            artifact_type=kind,
            namespace=namespace,
        )
        return artifact.to_dict()

    def human_output(payload: dict[str, Any]) -> None:
        sys.stdout.write(dump_json(payload, indent=2, sort_keys=False) + "\n")

    run_command(json_output=json_output, action=action, human_output=human_output)


@app.command("validate")
def artifacts_validate(
    workspace: Annotated[str, typer.Argument(help="Workspace directory to validate.")],
    artifact_id: Annotated[str | None, typer.Argument(help="Optional artifact id.")] = None,
    kind: Annotated[str | None, typer.Option("--kind", help="Optional artifact kind.")] = None,
    namespace: Annotated[
        str | None,
        typer.Option("--namespace", help="Optional caller-owned namespace filter."),
    ] = None,
    json_output: JsonOption = False,
) -> None:
    """Validate one artifact or every matching artifact."""

    def action() -> dict[str, Any]:
        if artifact_id:
            artifact = validate_workspace_artifact(
                workspace,
                artifact_id,
                artifact_type=kind,
                namespace=namespace,
            )
            return {
                "status": "valid",
                "workspace": workspace,
                "artifact_id": artifact.artifact_id,
                "artifact_type": artifact.artifact_type,
                "namespace": artifact.namespace,
                "count": 1,
            }
        artifacts = validate_workspace_artifacts(
            workspace,
            artifact_type=kind,
            namespace=namespace,
        )
        return {
            "status": "valid",
            "workspace": workspace,
            "count": len(artifacts),
            "artifacts": [artifact.to_dict() for artifact in artifacts],
        }

    def human_output(payload: dict[str, Any]) -> None:
        if payload.get("artifact_id"):
            sys.stdout.write(f"Valid artifact {payload['artifact_id']}\n")
            return
        sys.stdout.write(f"Valid artifacts {payload['count']}\n")

    run_command(json_output=json_output, action=action, human_output=human_output)


@app.command("rebuild-index")
def artifacts_rebuild_index(
    workspace: Annotated[str, typer.Argument(help="Workspace directory to index.")],
    json_output: JsonOption = False,
) -> None:
    """Rebuild the workspace artifact index from manifests."""

    def action() -> dict[str, Any]:
        entries = rebuild_workspace_artifact_index(workspace)
        return {
            "status": "indexed",
            "workspace": workspace,
            "count": len(entries),
            "artifacts": [entry.to_dict() for entry in entries],
        }

    def human_output(payload: dict[str, Any]) -> None:
        sys.stdout.write(f"Indexed artifacts {payload['count']}\n")

    run_command(json_output=json_output, action=action, human_output=human_output)
