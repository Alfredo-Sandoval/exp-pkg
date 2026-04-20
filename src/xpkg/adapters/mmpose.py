"""High-level MMPose adapter exports for native project archives."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import xpkg.io.converters.mmpose_import as _mmpose_import
from xpkg.io.archive_format.shared import CANONICAL_ARCHIVE_SUFFIX
from xpkg.io.converters.converter_helpers import (
    CliRunner,
    add_output_path_argument,
    build_cli_parser,
    parse_and_run_cli,
)
from xpkg.io.converters.progress import (
    PercentProgressCallback as ProgressCallback,
)
from xpkg.io.converters.progress import bridge_progress_callback

ConversionResult = _mmpose_import.ConversionResult


def convert_mmpose_topdown_json(
    json_path: Path | str,
    video_path: Path | str,
    out_path: Path | str,
    *,
    skeleton_name: str = "imported",
    instance_index: int = 0,
    likelihood_threshold: float = 0.0,
    progress_callback: ProgressCallback | None = None,
) -> ConversionResult:
    """Convert official MMPose top-down demo JSON predictions into a native archive."""

    return _mmpose_import.convert_mmpose_topdown_json(
        json_path,
        video_path,
        out_path,
        skeleton_name=skeleton_name,
        instance_index=int(instance_index),
        likelihood_threshold=float(likelihood_threshold),
        archive_extension=CANONICAL_ARCHIVE_SUFFIX,
        progress_callback=bridge_progress_callback(
            progress_callback,
            _mmpose_import.MMPOSE_TOPDOWN_JSON_PROGRESS_MARKERS,
        ),
    )


def _configure_cli_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--json",
        required=True,
        help="Path to an MMPose --save-predictions JSON export",
    )
    parser.add_argument("--video", required=True, help="Path to the matching video file")
    parser.add_argument(
        "--skeleton-name",
        default="imported",
        help="Skeleton name to store in the converted archive",
    )
    parser.add_argument(
        "--instance-index",
        type=int,
        default=0,
        help="Per-frame instance slot to import from the MMPose predictions",
    )
    parser.add_argument(
        "--likelihood-threshold",
        type=float,
        default=0.0,
        help="Minimum MMPose confidence required to keep a keypoint",
    )


def _run_cli(args: argparse.Namespace, _parser: argparse.ArgumentParser) -> int:
    convert_mmpose_topdown_json(
        args.json,
        args.video,
        args.out,
        skeleton_name=args.skeleton_name,
        instance_index=int(args.instance_index),
        likelihood_threshold=float(args.likelihood_threshold),
        progress_callback=None,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for MMPose top-down JSON conversion."""

    runner: CliRunner = _run_cli
    parser = build_cli_parser(
        description="Convert MMPose top-down JSON predictions plus a video into an xpkg archive"
    )
    add_output_path_argument(parser, help_text="Output xpkg archive path")
    _configure_cli_parser(parser)
    return parse_and_run_cli(parser, argv, runner)


__all__ = [
    "ConversionResult",
    "ProgressCallback",
    "convert_mmpose_topdown_json",
    "main",
]
