"""High-level MediaPipe adapter exports for native project archives."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import xpkg.io.converters.mediapipe_import as _mediapipe_import
from xpkg.io.archive_format.shared import CANONICAL_ARCHIVE_SUFFIX
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


def convert_mediapipe_pose_landmarks_json(
    json_path: str,
    video_path: str,
    out_path: str,
    *,
    skeleton_name: str = "mediapipe_pose",
    likelihood_threshold: float = 0.0,
    progress_callback: ProgressCallback | None = None,
) -> ConversionResult:
    """Convert MediaPipe pose-landmarks JSON plus its video into a native archive."""

    return _mediapipe_import.convert_mediapipe_pose_landmarks_json(
        json_path,
        video_path,
        out_path,
        skeleton_name=skeleton_name,
        likelihood_threshold=float(likelihood_threshold),
        archive_extension=CANONICAL_ARCHIVE_SUFFIX,
        progress_callback=bridge_progress_callback(
            progress_callback,
            _mediapipe_import.MEDIAPIPE_POSE_LANDMARKS_JSON_PROGRESS_MARKERS,
        ),
    )


def _configure_cli_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", required=True, help="Path to MediaPipe pose-landmarks JSON")
    parser.add_argument("--video", required=True, help="Path to the matching video")
    parser.add_argument(
        "--skeleton-name",
        default="mediapipe_pose",
        help="Skeleton name to store in the converted archive",
    )
    parser.add_argument(
        "--likelihood-threshold",
        type=float,
        default=0.0,
        help="Minimum MediaPipe visibility score required to keep a keypoint",
    )


def _run_cli(args: argparse.Namespace, _parser: argparse.ArgumentParser) -> int:
    convert_mediapipe_pose_landmarks_json(
        args.json,
        args.video,
        args.out,
        skeleton_name=args.skeleton_name,
        likelihood_threshold=float(args.likelihood_threshold),
        progress_callback=None,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for MediaPipe pose-landmarks JSON conversion."""

    runner: CliRunner = _run_cli
    parser = build_cli_parser(
        description="Convert MediaPipe pose-landmarks JSON plus a video into an xpkg archive"
    )
    add_output_path_argument(parser, help_text="Output .xpkg archive path")
    _configure_cli_parser(parser)
    return parse_and_run_cli(parser, argv, runner)


__all__ = [
    "ConversionResult",
    "ProgressCallback",
    "convert_mediapipe_pose_landmarks_json",
    "main",
]
