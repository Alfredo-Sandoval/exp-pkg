"""Shared CLI contracts for JSON output, progress, validation, and errors."""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from enum import StrEnum
from typing import Annotated, Any, NoReturn, overload

import typer
from click.exceptions import ClickException, MissingParameter, NoSuchOption

from .._core.json_utils import dump_json

JsonOption = Annotated[
    bool,
    typer.Option(
        "--json",
        help="Emit machine-readable JSON to stdout with no progress or prose.",
    ),
]


class PackMedia(StrEnum):
    """CLI values for packed project media scope."""

    full = "full"
    package = "package"
    manifest = "manifest"


def write_json(payload: object, *, stderr: bool = False) -> None:
    """Write one JSON document to stdout or stderr."""
    stream = sys.stderr if stderr else sys.stdout
    stream.write(dump_json(payload, indent=2, sort_keys=False) + "\n")


def emit_progress(message: str) -> None:
    """Emit human-mode progress text."""
    sys.stdout.write(message + "\n")


def progress_callback(json_output: bool) -> Callable[[str], None]:
    """Return a progress callback that is silent in JSON mode."""
    if json_output:
        return lambda _message: None
    return emit_progress


def write_path(path: object) -> None:
    """Write a path-like object in human mode."""
    sys.stdout.write(f"{path}\n")


def _hint_for_exception(exc: BaseException) -> str:
    if isinstance(exc, FileNotFoundError):
        return "Check that the path exists and rerun the command."
    if isinstance(exc, ValueError):
        return "Check the required flags and input format, then rerun the command."
    return "Run `xpkg describe --json` to inspect the command contract."


def _error_code_for_exception(exc: BaseException) -> tuple[str, int]:
    if isinstance(exc, FileNotFoundError):
        return "not_found", 3
    if isinstance(exc, ValueError):
        return "invalid_input", 1
    return "runtime_error", 1


def raise_cli_error(exc: BaseException, *, json_output: bool) -> NoReturn:
    """Raise a Typer exit after emitting the contract error envelope."""
    code, exit_code = _error_code_for_exception(exc)
    message = str(exc).strip() or exc.__class__.__name__
    hint = _hint_for_exception(exc)
    if json_output:
        write_json(
            {
                "ok": False,
                "error": {
                    "code": code,
                    "message": message,
                    "hint": hint,
                },
            },
            stderr=True,
        )
    else:
        sys.stderr.write(f"Error: {message}\nHint: {hint}\n")
    raise typer.Exit(exit_code)


def run_command(
    *,
    json_output: bool,
    action: Callable[[], dict[str, Any]],
    human_output: Callable[[dict[str, Any]], None],
) -> None:
    """Run a command action under the shared JSON/prose output contract.

    In ``--json`` mode the action's return value is wrapped in the standard
    success envelope ``{"ok": true, "data": <payload>}`` before writing to
    stdout. The human output callback continues to receive the bare payload.
    """
    try:
        payload = action()
    except Exception as exc:
        raise_cli_error(exc, json_output=json_output)
    if json_output:
        write_json({"ok": True, "data": payload})
    else:
        human_output(payload)


def run_typer_app(
    app: typer.Typer,
    *,
    argv: Sequence[str] | None = None,
    prog_name: str,
) -> int:
    """Run a Typer app and normalize process-style exit handling."""
    args = list(argv) if argv is not None else None
    try:
        result = app(args=args, prog_name=prog_name, standalone_mode=False)
    except typer.Exit as exc:
        return int(exc.exit_code or 0)
    except ClickException as exc:
        payload, exit_code = usage_error_payload(exc)
        if argv_requests_json(args):
            write_json(payload, stderr=True)
            raise SystemExit(exit_code) from exc
        exc.show(file=sys.stderr)
        raise SystemExit(exit_code) from exc
    if isinstance(result, int):
        return result
    return 0


def require_likelihood_threshold(value: float) -> float:
    """Validate a likelihood threshold option."""
    if value < 0.0 or value > 1.0:
        raise typer.BadParameter("expected a likelihood threshold in [0, 1]")
    return value


def require_positive_int(value: int) -> int:
    """Validate a positive integer option."""
    if value <= 0:
        raise typer.BadParameter("expected a positive integer")
    return value


def require_nonnegative_int(value: int) -> int:
    """Validate a non-negative integer option."""
    if value < 0:
        raise typer.BadParameter("expected a non-negative integer")
    return value


def usage_error_payload(exc: ClickException) -> tuple[dict[str, Any], int]:
    """Return the structured error payload and exit code for Click parse errors."""
    message = exc.format_message()
    if isinstance(exc, NoSuchOption):
        code = "unknown_option"
        exit_code = 1
        hint = "Run `xpkg --help` or `xpkg describe --json` for supported options."
    elif message.startswith("No such command "):
        code = "not_found"
        exit_code = 3
        hint = "Run `xpkg --help` to see supported commands."
    else:
        code = "usage_error"
        exit_code = 1
        hint = "Run `xpkg --help` or `xpkg describe --json` for the command contract."
    return (
        {
            "ok": False,
            "error": {
                "code": code,
                "message": message,
                "hint": hint,
            },
        },
        exit_code,
    )


def argv_requests_json(argv: Sequence[str] | None) -> bool:
    """Return whether the invocation requested JSON mode."""
    if argv is None:
        return "--json" in sys.argv[1:]
    return "--json" in argv


@overload
def require_option_value(value: str | None, option_name: str) -> str: ...


@overload
def require_option_value[OptionValue: object](
    value: OptionValue | None, option_name: str
) -> OptionValue: ...


def require_option_value[OptionValue: object](
    value: OptionValue | None, option_name: str
) -> OptionValue:
    """Raise a Click-style missing-option error when Typer passes `None`."""
    if value is None:
        # Quote the hint to match click's canonical "Missing option '--x'." text.
        raise MissingParameter(param_hint=f"'{option_name}'", param_type="option")
    return value
