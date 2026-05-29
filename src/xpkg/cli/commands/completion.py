"""Shell completion generators for the ``xpkg`` command."""

from __future__ import annotations

import sys

import typer
from typer.completion import get_completion_script

from xpkg.cli.shared import JsonOption, run_command


def _completion_script(shell: str) -> str:
    """Return the shell completion script for ``xpkg``.

    typer 0.26 vendors its own click as ``typer._click``, so generate the script
    through typer's own generator instead of the standalone ``click`` package.
    This keeps the script consistent with the installed CLI runtime and returns
    text directly (the standalone-click path wrote bytes).
    """
    return get_completion_script(
        prog_name="xpkg",
        complete_var="_XPKG_COMPLETE",
        shell=shell,
    )


def build_app(root_app: typer.Typer) -> typer.Typer:
    """Build the completion command group for the given root app."""
    app = typer.Typer(
        add_completion=False,
        help="Generate shell completion scripts.",
        no_args_is_help=True,
        rich_markup_mode="rich",
    )

    def show_completion(shell: str, *, json_output: bool) -> None:
        def action() -> dict[str, str]:
            script = _completion_script(shell)
            return {"shell": shell, "script": script}

        def human_output(payload: dict[str, str]) -> None:
            sys.stdout.write(payload["script"])

        run_command(json_output=json_output, action=action, human_output=human_output)

    @app.command("bash")
    def completion_bash(json_output: JsonOption = False) -> None:
        """Generate bash completion script."""
        show_completion("bash", json_output=json_output)

    @app.command("fish")
    def completion_fish(json_output: JsonOption = False) -> None:
        """Generate fish completion script."""
        show_completion("fish", json_output=json_output)

    @app.command("zsh")
    def completion_zsh(json_output: JsonOption = False) -> None:
        """Generate zsh completion script."""
        show_completion("zsh", json_output=json_output)

    return app
