"""High-level SLEAP adapter exports for native project archives."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import posetta.io.converters.sleap_import as _sleap_import
from posetta.io.converters.converter_helpers import (
    CliRunner,
    ConversionResult,
    add_bool_toggle_arguments,
    add_output_path_argument,
    build_cli_parser,
    parse_and_run_cli,
)
from posetta.io.converters.progress import (
    PercentProgressCallback as ProgressCallback,
)
from posetta.io.converters.progress import (
    bridge_progress_callback,
)


def convert_sleap_package(
    slp: str,
    out_dir: str,
    *,
    fps: int = 30,
    encode_videos: bool | None = None,
    progress_callback: ProgressCallback | None = None,
) -> ConversionResult:
    """Convert a SLEAP `.pkg.slp` archive into a native project."""

    return _sleap_import.convert_sleap_package(
        slp,
        out_dir,
        fps=int(fps),
        encode_videos=encode_videos,
        bundle_extension=".sta",
        progress_callback=bridge_progress_callback(
            progress_callback,
            _sleap_import.SLEAP_PACKAGE_PROGRESS_MARKERS,
        ),
    )


def _configure_cli_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--slp", required=True, help="Path to .pkg.slp")
    parser.add_argument("--fps", type=int, default=30, help="FPS for encoded videos")
    add_bool_toggle_arguments(
        parser,
        dest="encode_videos",
        true_flag="--videos",
        false_flag="--no-videos",
        true_help="Encode MP4 videos into the project (off by default)",
        false_help="Do not encode MP4 videos (default)",
        default=True,
    )


def _run_cli(args: argparse.Namespace, _parser: argparse.ArgumentParser) -> int:
    convert_sleap_package(
        args.slp,
        args.out,
        fps=int(args.fps),
        encode_videos=bool(args.encode_videos),
        progress_callback=None,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for SLEAP package -> native project conversion."""
    runner: CliRunner = _run_cli
    parser = build_cli_parser(description="Convert SLEAP .pkg.slp to a Posetta project")
    add_output_path_argument(parser, help_text="Output Posetta project root directory")
    _configure_cli_parser(parser)
    return parse_and_run_cli(parser, argv, runner)


__all__ = ["ConversionResult", "ProgressCallback", "convert_sleap_package", "main"]
