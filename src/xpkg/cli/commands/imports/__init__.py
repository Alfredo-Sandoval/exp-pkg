"""Project import commands grouped by experiment domain."""

from __future__ import annotations

import typer

app = typer.Typer(
    add_completion=False,
    help="Import external experiment data into a project.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

from xpkg.cli.commands.imports import behavior as behavior  # noqa: E402
from xpkg.cli.commands.imports import calibration as calibration  # noqa: E402
from xpkg.cli.commands.imports import events as events  # noqa: E402
from xpkg.cli.commands.imports import pose as pose  # noqa: E402
from xpkg.cli.commands.imports import signals as signals  # noqa: E402
from xpkg.cli.commands.imports import synchronization as synchronization  # noqa: E402

__all__ = ["app"]
