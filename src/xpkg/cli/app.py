"""Top-level Typer app assembly for the xpkg CLI."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Annotated

import click
import typer

from xpkg.cli.commands import artifacts, completion, imports, workspace
from xpkg.cli.shared import (
    JsonOption,
    argv_requests_json,
    run_command,
    usage_error_payload,
    write_json,
)
from xpkg.version import __version__

app = typer.Typer(
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Workspace-first IO and artifact tools for experiment projects.",
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
            "success": "Each command emits one command-specific JSON object in --json mode.",
            "error": {
                "shape": {
                    "error": {
                        "code": "string",
                        "message": "string",
                        "hint": "string",
                    }
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
        "commands": [
            "artifacts inspect",
            "artifacts list",
            "artifacts rebuild-index",
            "artifacts validate",
            "completion bash",
            "completion fish",
            "completion zsh",
            "describe",
            "import dlc csv",
            "import dlc h5",
            "import dlc project",
            "import lightning-pose",
            "import mediapipe",
            "import mmpose",
            "import sleap",
            "import vicon",
            "init",
            "migrate",
            "pack",
            "unpack",
            "validate",
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


workspace.register(app)
app.add_typer(artifacts.app, name="artifacts")
app.add_typer(completion.build_app(app), name="completion")
app.add_typer(imports.app, name="import")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the xpkg CLI and return a process-style exit code."""
    args = list(argv) if argv is not None else None
    try:
        result = app(args=args, prog_name="xpkg", standalone_mode=False)
    except typer.Exit as exc:
        return int(exc.exit_code or 0)
    except click.ClickException as exc:
        payload, exit_code = usage_error_payload(exc)
        if argv_requests_json(args):
            write_json(payload, stderr=True)
            raise SystemExit(exit_code) from exc
        exc.show(file=sys.stderr)
        raise SystemExit(exit_code) from exc
    if isinstance(result, int):
        return result
    return 0
