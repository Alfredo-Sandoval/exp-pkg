"""High-level DeepLabCut adapter exports for native project archives."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import xpkg.io.converters.dlc_import as _dlc_import
from xpkg.io.siesta_format.shared import CANONICAL_BUNDLE_SUFFIX
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
from xpkg.io.converters.progress import (
    bridge_progress_callback,
)

convert_dlc_csv = _dlc_import.convert_dlc_csv
convert_dlc_h5_project = _dlc_import.convert_dlc_h5_project
convert_dlc_project = _dlc_import.convert_dlc_project


def convert_dlc_h5(
    h5_path: Path | str,
    video_path: Path | str | Sequence[Path | str],
    project_root: Path | str,
    *,
    likelihood_threshold: float = 0.0,
    progress_callback: ProgressCallback | None = None,
) -> ConversionResult:
    """Convert one DLC H5 tracking file plus an explicit video into a native project."""

    normalized_video_paths = (
        [video_path] if isinstance(video_path, (str, Path)) else list(video_path)
    )
    return _dlc_import.convert_dlc_h5_project(
        h5_path,
        normalized_video_paths,
        project_root,
        likelihood_threshold=float(likelihood_threshold),
        progress_callback=bridge_progress_callback(
            progress_callback,
            _dlc_import.DLC_H5_PROJECT_PROGRESS_MARKERS,
        ),
        bundle_extension=CANONICAL_BUNDLE_SUFFIX,
    )


def _configure_cli_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--h5", required=True, help="Path to a DLC H5 tracking file")
    parser.add_argument("--video", required=True, help="Path to the video included in the project")
    parser.add_argument(
        "--likelihood-threshold",
        type=float,
        default=0.0,
        help="Minimum DLC likelihood required to keep a keypoint",
    )


def _run_cli(args: argparse.Namespace, _parser: argparse.ArgumentParser) -> int:
    convert_dlc_h5(
        args.h5,
        args.video,
        args.out,
        likelihood_threshold=float(args.likelihood_threshold),
        progress_callback=None,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for DLC H5 -> native project conversion."""
    runner: CliRunner = _run_cli
    parser = build_cli_parser(
        description="Convert DLC H5 tracking plus explicit videos into an xpkg project"
    )
    add_output_path_argument(parser, help_text="Output xpkg project root directory")
    _configure_cli_parser(parser)
    return parse_and_run_cli(parser, argv, runner)

__all__ = [
    "ConversionResult",
    "ProgressCallback",
    "convert_dlc_csv",
    "convert_dlc_h5",
    "convert_dlc_h5_project",
    "convert_dlc_project",
    "main",
]
