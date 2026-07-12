"""CLI command for inspecting files and folders before import."""

from __future__ import annotations

import sys
from typing import Annotated, Any

import typer

from xpkg.cli.shared import JsonOption, require_likelihood_threshold, run_command


def inspect_path(path: str, *, confidence_threshold: float) -> Any:
    """Inspect a path through the heavy inspection module, imported lazily."""
    from xpkg.inspection import inspect_path as _inspect_path

    return _inspect_path(path, confidence_threshold=confidence_threshold)


def inspect_target(
    target: Annotated[str, typer.Argument(help="File, folder, project, or .expkg to inspect.")],
    confidence_threshold: Annotated[
        float,
        typer.Option(
            "--confidence-threshold",
            "--threshold",
            callback=require_likelihood_threshold,
            help="Confidence threshold used for pose QC summaries.",
        ),
    ] = 0.5,
    json_output: JsonOption = False,
) -> None:
    """Inspect an input path and suggest likely xpkg importers."""

    def action() -> dict[str, Any]:
        return inspect_path(target, confidence_threshold=confidence_threshold).to_dict()

    def human_output(payload: dict[str, Any]) -> None:
        sys.stdout.write(f"{payload['path']}\n")
        sys.stdout.write(f"Kind: {payload['description']}\n")
        importers = payload.get("likely_importers") or []
        if importers:
            sys.stdout.write(f"Likely importers: {', '.join(str(item) for item in importers)}\n")
        summary = payload.get("summary") or {}
        if isinstance(summary, dict):
            for key in ("frames", "keypoints", "tracks", "fps", "width", "height"):
                if key in summary and summary[key] is not None:
                    sys.stdout.write(f"{key}: {summary[key]}\n")
            confidence = summary.get("confidence")
            if isinstance(confidence, dict):
                sys.stdout.write(
                    "confidence below threshold: "
                    f"{confidence.get('below_threshold')} / {confidence.get('finite_values')}\n"
                )
        warning_records = payload.get("warning_records") or []
        for record in warning_records:
            if isinstance(record, dict):
                sys.stdout.write(f"Warning: {record.get('message', record)}\n")

    run_command(json_output=json_output, action=action, human_output=human_output)


__all__ = ["inspect_target"]
