"""High-level SLEAP adapter exports for native project archives."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import xpkg.io.converters.sleap_import as _sleap_import
from xpkg.io.archive_format.shared import CANONICAL_ARCHIVE_SUFFIX
from xpkg.io.converters.converter_helpers import (
    CliRunner,
    ConversionResult,
    add_bool_toggle_arguments,
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
        archive_extension=CANONICAL_ARCHIVE_SUFFIX,
        progress_callback=bridge_progress_callback(
            progress_callback,
            _sleap_import.SLEAP_PACKAGE_PROGRESS_MARKERS,
        ),
    )


def convert_sleap_h5(
    h5_path: str,
    video_path: str,
    out_path: str,
    *,
    skeleton_name: str = "imported",
    likelihood_threshold: float = 0.0,
    progress_callback: ProgressCallback | None = None,
) -> ConversionResult:
    """Convert a SLEAP analysis H5 export plus its video into a native archive."""

    return _sleap_import.convert_sleap_h5(
        h5_path,
        video_path,
        out_path,
        skeleton_name=skeleton_name,
        likelihood_threshold=float(likelihood_threshold),
        archive_extension=CANONICAL_ARCHIVE_SUFFIX,
        progress_callback=bridge_progress_callback(
            progress_callback,
            _sleap_import.SLEAP_H5_PROGRESS_MARKERS,
        ),
    )


def _configure_cli_parser(parser: argparse.ArgumentParser) -> None:
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--slp", help="Path to .pkg.slp")
    source_group.add_argument("--h5", help="Path to SLEAP analysis H5")
    parser.add_argument("--video", help="Path to the matching video when using --h5")
    parser.add_argument(
        "--skeleton-name",
        default="imported",
        help="Skeleton name to use when importing --h5 tracking files",
    )
    parser.add_argument(
        "--likelihood-threshold",
        type=float,
        default=0.0,
        help="Minimum SLEAP confidence required to keep a keypoint for --h5 imports",
    )
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


def _run_cli(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.h5:
        if not args.video:
            parser.error("--video is required when using --h5")
        convert_sleap_h5(
            args.h5,
            args.video,
            args.out,
            skeleton_name=args.skeleton_name,
            likelihood_threshold=float(args.likelihood_threshold),
            progress_callback=None,
        )
        return 0

    convert_sleap_package(
        args.slp,
        args.out,
        fps=int(args.fps),
        encode_videos=bool(args.encode_videos),
        progress_callback=None,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for SLEAP package or analysis H5 conversion."""
    runner: CliRunner = _run_cli
    parser = build_cli_parser(description="Convert SLEAP package or analysis H5 data to xpkg")
    add_output_path_argument(
        parser,
        help_text="Output project root directory for --slp, or output .xpkg path for --h5",
    )
    _configure_cli_parser(parser)
    return parse_and_run_cli(parser, argv, runner)


__all__ = [
    "ConversionResult",
    "ProgressCallback",
    "convert_sleap_h5",
    "convert_sleap_package",
    "main",
]
