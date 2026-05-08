"""Main Typer entrypoint for the xpkg CLI."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated

import typer

from xpkg.cli.commands import (
    artifacts,
    completion,
    imports,
    project,
)
from xpkg.cli.commands.inspect import inspect_target
from xpkg.cli.shared import (
    JsonOption,
    run_command,
    run_typer_app,
)
from xpkg.version import __version__

app = typer.Typer(
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Project-first IO and artifact tools for experiment projects.",
    no_args_is_help=False,
    rich_markup_mode="rich",
)


def _describe_payload() -> dict[str, object]:
    return {
        "name": "xpkg",
        "version": __version__,
        "distribution": "exp-pkg",
        "entrypoint": "xpkg",
        "import_name": "xpkg",
        "profile": "built-for-agents",
        "json_contract": {
            "success_stream": "stdout",
            "error_stream": "stderr",
            "success": {
                "shape": {
                    "ok": True,
                    "data": "<command-specific JSON object>",
                }
            },
            "error": {
                "shape": {
                    "ok": False,
                    "error": {
                        "code": "string",
                        "message": "string",
                        "hint": "string",
                    },
                }
            },
            "progress": "Progress messages are suppressed in --json mode.",
        },
        "exit_codes": {
            "0": "success",
            "1": "usage or runtime error",
            "2": "reserved for auth or config failure",
            "3": "not found",
        },
        "resources": {
            "artifacts": ["inspect", "list", "rebuild-index", "validate"],
            "completion": ["bash", "fish", "zsh"],
            "inspect": ["path"],
            "import": [
                "anipose calibration",
                "dlc csv",
                "dlc h5",
                "dlc project",
                "lightning-pose",
                "mediapipe",
                "mmpose",
                "sleap h5",
                "sleap package",
                "vicon c3d",
                "vicon csv",
                "vicon recording",
            ],
            "project": ["describe", "init", "pack", "unpack", "validate"],
        },
        "commands": [
            "artifacts inspect",
            "artifacts list",
            "artifacts rebuild-index",
            "artifacts validate",
            "completion bash",
            "completion fish",
            "completion zsh",
            "describe",
            "inspect",
            "import dlc csv",
            "import anipose calibration",
            "import dlc h5",
            "import dlc project",
            "import lightning-pose",
            "import mediapipe",
            "import mmpose",
            "import sleap h5",
            "import sleap package",
            "import vicon c3d",
            "import vicon csv",
            "import vicon recording",
            "project describe",
            "project init",
            "project pack",
            "project unpack",
            "project validate",
        ],
    }


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Show the xpkg version and exit.",
        ),
    ] = False,
) -> None:
    """Run xpkg commands."""
    if version:
        typer.echo(f"xpkg {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


@app.command("describe")
def describe(json_output: JsonOption = False) -> None:
    """Describe the machine-readable command contract."""

    def action() -> dict[str, object]:
        return _describe_payload()

    def human_output(payload: dict[str, object]) -> None:
        typer.echo(f"{payload['name']} {payload['version']}")
        typer.echo("Machine contract: success JSON on stdout, errors on stderr.")
        typer.echo("Run `xpkg describe --json` for the full contract.")

    run_command(json_output=json_output, action=action, human_output=human_output)


app.command("inspect")(inspect_target)
app.add_typer(artifacts.app, name="artifacts")
app.add_typer(completion.build_app(app), name="completion")
app.add_typer(imports.app, name="import")
app.add_typer(project.app, name="project")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the xpkg CLI and return a process-style exit code."""
    return run_typer_app(app, argv=argv, prog_name="xpkg")
