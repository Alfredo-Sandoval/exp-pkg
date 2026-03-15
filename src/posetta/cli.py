"""Posetta command-line interface."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

from posetta.io.converters.converter_helpers import ConversionResult
from posetta.io.converters.dlc_import import convert_dlc_csv, convert_dlc_h5, convert_dlc_project
from posetta.io.converters.sleap_import import convert_sleap_package
from posetta.version import __version__

CliCommand = Callable[[argparse.Namespace], int]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {parsed}")
    return parsed


def _likelihood_threshold(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0 or parsed > 1.0:
        raise argparse.ArgumentTypeError(
            f"expected a likelihood threshold in [0, 1], got {parsed}"
        )
    return parsed


def _emit_progress(message: str) -> None:
    sys.stdout.write(message + "\n")


def _write_result(result: ConversionResult) -> None:
    sys.stdout.write(f"Wrote {result.siesta_path}\n")


def _configure_tracking_parser(
    parser: argparse.ArgumentParser,
    *,
    data_flag: str,
    data_help: str,
    command: CliCommand,
) -> None:
    parser.add_argument(data_flag, required=True, help=data_help)
    parser.add_argument("--video", required=True, help="Path to the matching video file.")
    parser.add_argument("--out", required=True, help="Output .siesta file path.")
    parser.add_argument(
        "--skeleton-name",
        default="imported",
        help="Skeleton name to store in the converted archive.",
    )
    parser.add_argument(
        "--threshold",
        type=_likelihood_threshold,
        default=0.0,
        help="Likelihood threshold for including keypoints (0 to 1).",
    )
    parser.set_defaults(func=command)


def _add_dlc_parser(parent: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    dlc = parent.add_parser("dlc", help="Convert DeepLabCut tracking outputs.")
    dlc_subparsers = dlc.add_subparsers(dest="dlc_source", required=True)

    csv_parser = dlc_subparsers.add_parser("csv", help="Convert a DLC CSV tracking file.")
    _configure_tracking_parser(
        csv_parser,
        data_flag="--csv",
        data_help="Path to a DLC CSV tracking file.",
        command=_cmd_dlc_csv,
    )

    h5_parser = dlc_subparsers.add_parser("h5", help="Convert a DLC H5 tracking file.")
    _configure_tracking_parser(
        h5_parser,
        data_flag="--h5",
        data_help="Path to a DLC H5 tracking file.",
        command=_cmd_dlc_h5,
    )

    project_parser = dlc_subparsers.add_parser(
        "project",
        help="Convert every supported item in a DLC project directory.",
    )
    project_parser.add_argument("--project", required=True, help="Path to the DLC project root.")
    project_parser.add_argument("--out", required=True, help="Output directory for .siesta files.")
    project_parser.add_argument(
        "--threshold",
        type=_likelihood_threshold,
        default=0.0,
        help="Likelihood threshold for including keypoints (0 to 1).",
    )
    project_parser.set_defaults(func=_cmd_dlc_project)


def _add_sleap_parser(parent: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    sleap = parent.add_parser("sleap", help="Convert a SLEAP .pkg.slp archive.")
    sleap.add_argument("--slp", required=True, help="Path to the input .pkg.slp archive.")
    sleap.add_argument("--out", required=True, help="Output project directory.")
    sleap.add_argument(
        "--fps",
        type=_positive_int,
        default=30,
        help="Frame rate to use when encoding videos.",
    )
    encode_group = sleap.add_mutually_exclusive_group()
    encode_group.add_argument(
        "--videos",
        dest="encode_videos",
        action="store_true",
        help="Encode MP4 videos into the output project.",
    )
    encode_group.add_argument(
        "--no-videos",
        dest="encode_videos",
        action="store_false",
        help="Keep extracted frame sequences instead of encoding MP4 videos.",
    )
    sleap.set_defaults(func=_cmd_sleap, encode_videos=True)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="posetta", description="Posetta CLI tools")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    convert = subparsers.add_parser(
        "convert",
        help="Convert external pose formats into native .siesta archives.",
    )
    convert_subparsers = convert.add_subparsers(dest="format", required=True)
    _add_dlc_parser(convert_subparsers)
    _add_sleap_parser(convert_subparsers)
    return parser


def _cmd_dlc_csv(args: argparse.Namespace) -> int:
    result = convert_dlc_csv(
        args.csv,
        args.video,
        args.out,
        skeleton_name=args.skeleton_name,
        likelihood_threshold=args.threshold,
        progress_callback=_emit_progress,
    )
    _write_result(result)
    return 0


def _cmd_dlc_h5(args: argparse.Namespace) -> int:
    result = convert_dlc_h5(
        args.h5,
        args.video,
        args.out,
        skeleton_name=args.skeleton_name,
        likelihood_threshold=args.threshold,
        progress_callback=_emit_progress,
    )
    _write_result(result)
    return 0


def _cmd_dlc_project(args: argparse.Namespace) -> int:
    results = convert_dlc_project(
        args.project,
        args.out,
        likelihood_threshold=args.threshold,
        progress_callback=_emit_progress,
    )
    sys.stdout.write(f"Converted {len(results)} project item(s)\n")
    for result in results:
        _write_result(result)
    return 0


def _cmd_sleap(args: argparse.Namespace) -> int:
    result = convert_sleap_package(
        args.slp,
        args.out,
        fps=args.fps,
        encode_videos=args.encode_videos,
        progress_callback=_emit_progress,
    )
    _write_result(result)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


__all__ = ["main"]
