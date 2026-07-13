"""Shared formatting and validation for import commands."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

from xpkg.cli.shared import write_path

if TYPE_CHECKING:
    from xpkg.services import ProjectService


def open_import_project(
    project: str,
    *,
    force: bool,
    title: str | None = None,
) -> ProjectService:
    """Open an existing project or create a new import target."""
    from xpkg.services import ProjectService

    try:
        return ProjectService.open(project)
    except FileNotFoundError:
        return ProjectService.create(project, title=title, force=force)


def validate_format(format_value: str, allowed: tuple[str, ...], family: str) -> str:
    if format_value not in allowed:
        choices = ", ".join(allowed)
        raise typer.BadParameter(
            f"unknown {family} format {format_value!r}. Choose from: {choices}"
        )
    return format_value


def import_payload(source: str, project: str, state_path: Path) -> dict[str, str]:
    return {
        "status": "imported",
        "source": source,
        "project": project,
        "state_path": str(state_path),
    }


def emit_import_result(payload: dict[str, Any], label: str) -> None:
    sys.stdout.write(f"Imported {label} into {payload['project']}\n")
    write_path(Path(str(payload["state_path"])))


__all__ = [
    "emit_import_result",
    "import_payload",
    "open_import_project",
    "validate_format",
]
