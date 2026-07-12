"""Shell completion generators for the ``xpkg`` command."""

from __future__ import annotations

import sys

import typer

from xpkg.cli.shared import JsonOption, run_command

_COMPLETION_SCRIPTS = {
    "bash": """_xpkg_completion() {
    local IFS=$'\\n'
    COMPREPLY=( $( env COMP_WORDS="${COMP_WORDS[*]}" \\
                   COMP_CWORD=$COMP_CWORD \\
                   _XPKG_COMPLETE=complete_bash $1 ) )
    return 0
}

complete -o default -F _xpkg_completion xpkg""",
    "fish": (
        "complete --command xpkg --no-files --arguments "
        '"(env _XPKG_COMPLETE=complete_fish '
        "_TYPER_COMPLETE_FISH_ACTION=get-args "
        '_TYPER_COMPLETE_ARGS=(commandline -cp) xpkg)" --condition '
        '"env _XPKG_COMPLETE=complete_fish '
        "_TYPER_COMPLETE_FISH_ACTION=is-args "
        '_TYPER_COMPLETE_ARGS=(commandline -cp) xpkg"'
    ),
    "zsh": """#compdef xpkg

_xpkg_completion() {
  eval $(env _TYPER_COMPLETE_ARGS="${words[1,$CURRENT]}" _XPKG_COMPLETE=complete_zsh xpkg)
}

compdef _xpkg_completion xpkg""",
}


def _completion_script(shell: str) -> str:
    """Return the shell completion script for ``xpkg``."""
    try:
        return _COMPLETION_SCRIPTS[shell]
    except KeyError as exc:
        raise ValueError(f"Unsupported completion shell: {shell}") from exc


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
