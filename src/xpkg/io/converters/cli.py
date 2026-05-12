"""Argparse helpers shared by converter ``__main__`` entry points."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence

CliRunner = Callable[[argparse.Namespace, argparse.ArgumentParser], int]


def build_cli_parser(description: str) -> argparse.ArgumentParser:
    """Create a converter CLI parser with the provided description."""
    return argparse.ArgumentParser(description=description)


def add_output_path_argument(
    parser: argparse.ArgumentParser,
    *,
    help_text: str,
) -> None:
    """Register the required ``--out`` argument with converter-specific help text."""
    parser.add_argument("--out", required=True, help=help_text)


def add_bool_toggle_arguments(
    parser: argparse.ArgumentParser,
    *,
    dest: str,
    true_flag: str,
    false_flag: str,
    true_help: str,
    false_help: str,
    default: bool,
) -> None:
    """Register paired boolean flags in a mutually exclusive group."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(true_flag, dest=dest, action="store_true", help=true_help)
    group.add_argument(false_flag, dest=dest, action="store_false", help=false_help)
    parser.set_defaults(**{dest: default})


def parse_and_run_cli(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None,
    runner: CliRunner,
) -> int:
    """Parse arguments and invoke the CLI runner."""
    args = parser.parse_args(argv)
    return runner(args, parser)


__all__ = [
    "CliRunner",
    "add_bool_toggle_arguments",
    "add_output_path_argument",
    "build_cli_parser",
    "parse_and_run_cli",
]
