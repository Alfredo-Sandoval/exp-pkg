import argparse
from collections.abc import Callable, Sequence

from posetta.io.converters import converter_helpers as _helpers


def run_converter_cli(
    *,
    description: str,
    output_help: str,
    argv: Sequence[str] | None,
    runner: Callable[[argparse.Namespace, argparse.ArgumentParser], int],
    configure_parser: Callable[[argparse.ArgumentParser], None],
) -> int:
    parser = _helpers.build_cli_parser(description=description)
    _helpers.add_output_path_argument(parser, help_text=output_help)
    configure_parser(parser)
    return _helpers.parse_and_run_cli(parser, argv, runner)
