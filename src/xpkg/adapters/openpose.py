"""High-level OpenPose adapter exports for native project archives."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import xpkg.io.converters.openpose_import as _openpose_import
from xpkg.io.converters.converter_helpers import (
    CliRunner,
    ConversionResult,
    add_output_path_argument,
    build_cli_parser,
    parse_and_run_cli,
)
from xpkg.io.converters.progress import (
    PercentProgressCallback as ProgressCallback,
)
from xpkg.io.converters.progress import bridge_progress_callback


def convert_openpose_json(
    json_dir: Path | str,
    video_path: Path | str,
    out_path: Path | str,
    *,
    skeleton_name: str = "imported",
    likelihood_threshold: float = 0.0,
    progress_callback: ProgressCallback | None = None,
) -> ConversionResult:
    """Convert an OpenPose BODY_25 JSON directory plus its video into a native archive."""

    return _openpose_import.convert_openpose_json(
        json_dir,
        video_path,
        out_path,
        skeleton_name=skeleton_name,
        likelihood_threshold=float(likelihood_threshold),
        progress_callback=bridge_progress_callback(
            progress_callback,
            _openpose_import.OPENPOSE_JSON_PROGRESS_MARKERS,
        ),
    )


def _configure_cli_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--json",
        dest="json_dir",
        required=True,
        help="Path to an OpenPose --write_json output directory",
    )
    parser.add_argument("--video", required=True, help="Path to the matching video file")
    parser.add_argument(
        "--skeleton-name",
        default="imported",
        help="Skeleton name to store in the converted archive",
    )
    parser.add_argument(
        "--likelihood-threshold",
        type=float,
        default=0.0,
        help="Minimum OpenPose confidence required to keep a keypoint",
    )


def _run_cli(args: argparse.Namespace, _parser: argparse.ArgumentParser) -> int:
    convert_openpose_json(
        args.json_dir,
        args.video,
        args.out,
        skeleton_name=args.skeleton_name,
        likelihood_threshold=float(args.likelihood_threshold),
        progress_callback=None,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for OpenPose JSON -> native project conversion."""

    runner: CliRunner = _run_cli
    parser = build_cli_parser(
        description="Convert OpenPose BODY_25 JSON output plus a video into an xpkg project"
    )
    add_output_path_argument(parser, help_text="Output xpkg archive path")
    _configure_cli_parser(parser)
    return parse_and_run_cli(
        parser,
        argv,
        runner,
    )


__all__ = [
    "ConversionResult",
    "ProgressCallback",
    "convert_openpose_json",
    "main",
]
