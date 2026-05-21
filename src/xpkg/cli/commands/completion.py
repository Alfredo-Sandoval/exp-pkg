"""Shell completion generators for the ``xpkg`` command."""

from __future__ import annotations

import contextlib
import io
import sys

import typer
from click.shell_completion import shell_complete

from xpkg.cli.shared import JsonOption, run_command


def _completion_script(root_app: typer.Typer, shell: str) -> str:
    command = typer.main.get_command(root_app)
    out_buffer = io.StringIO()
    err_buffer = io.StringIO()
    with contextlib.redirect_stdout(out_buffer), contextlib.redirect_stderr(err_buffer):
        exit_code = shell_complete(
            command,
            {},
            "xpkg",
            "_XPKG_COMPLETE",
            f"{shell}_source",
        )
    if exit_code != 0:
        detail = err_buffer.getvalue().strip()
        message = f"Could not generate {shell} completion script."
        if detail:
            message = f"{message} {detail}"
        raise RuntimeError(message)
    return out_buffer.getvalue()


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
            script = _completion_script(root_app, shell)
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
