"""Compatibility Detectron2 adapter exports for direct archive workflows."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import xpkg.io.converters.detectron2_import as _detectron2_import
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


def convert_detectron2_coco(
    predictions_path: str,
    dataset_json_path: str,
    image_root: str,
    out_path: str,
    *,
    category_id: int | None = None,
    skeleton_name: str | None = None,
    likelihood_threshold: float = 0.0,
    progress_callback: ProgressCallback | None = None,
) -> ConversionResult:
    """Convert Detectron2 COCO keypoint predictions into a direct `.xpkg` archive."""

    return _detectron2_import.convert_detectron2_coco(
        predictions_path,
        dataset_json_path,
        image_root,
        out_path,
        category_id=category_id,
        skeleton_name=skeleton_name,
        likelihood_threshold=float(likelihood_threshold),
        progress_callback=bridge_progress_callback(
            progress_callback,
            _detectron2_import.DETECTRON2_COCO_PROGRESS_MARKERS,
        ),
    )


def _configure_cli_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to Detectron2 COCOEvaluator coco_instances_results.json",
    )
    parser.add_argument(
        "--dataset-json",
        required=True,
        help="Path to the COCO dataset JSON registered with Detectron2",
    )
    parser.add_argument(
        "--image-root",
        required=True,
        help="Image root paired with the COCO dataset JSON",
    )
    parser.add_argument(
        "--category-id",
        type=int,
        help="COCO category id to import when the dataset has multiple keypoint categories",
    )
    parser.add_argument(
        "--skeleton-name",
        help="Optional skeleton name override for the imported keypoints",
    )
    parser.add_argument(
        "--likelihood-threshold",
        type=float,
        default=0.0,
        help="Minimum keypoint confidence required to keep a point",
    )


def _run_cli(args: argparse.Namespace, _parser: argparse.ArgumentParser) -> int:
    convert_detectron2_coco(
        args.predictions,
        args.dataset_json,
        args.image_root,
        args.out,
        category_id=args.category_id,
        skeleton_name=args.skeleton_name,
        likelihood_threshold=float(args.likelihood_threshold),
        progress_callback=None,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for Detectron2 COCO keypoint conversion."""

    runner: CliRunner = _run_cli
    parser = build_cli_parser(
        description=(
            "Convert Detectron2 COCO keypoint results plus dataset metadata into an xpkg archive"
        )
    )
    add_output_path_argument(parser, help_text="Output xpkg archive path")
    _configure_cli_parser(parser)
    return parse_and_run_cli(parser, argv, runner)


__all__ = [
    "ConversionResult",
    "ProgressCallback",
    "convert_detectron2_coco",
    "main",
]
