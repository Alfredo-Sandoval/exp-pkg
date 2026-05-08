"""CLI commands for the xpkg project lifecycle and portable artifacts."""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer

from xpkg._core.json_utils import load_json_dict
from xpkg.cli.shared import JsonOption, PackMedia, require_option_value, run_command, write_path
from xpkg.model import AcquisitionMetadata, DatasetDatasheet, DatasetShareMetadata, ModelCard
from xpkg.project import (
    current_project_state_path,
    init_project,
    load_project_acquisition_metadata,
    load_project_dataset_share_metadata,
    load_project_datasheet,
    load_project_descriptor,
    load_project_model_card,
    pack_project,
    project_acquisition_metadata_path,
    project_artifacts_root,
    project_dataset_share_metadata_path,
    project_datasheet_path,
    project_descriptor_path,
    project_exports_root,
    project_media_root,
    project_model_card_path,
    project_state_root,
    project_store_root,
    resolve_project_root,
    save_project_acquisition_metadata,
    save_project_dataset_share_metadata,
    save_project_datasheet,
    save_project_model_card,
    unpack_project,
)
from xpkg.project import (
    validate_artifact as validate_artifact_target,
)

app = typer.Typer(
    add_completion=False,
    help="Create, inspect, validate, pack, and unpack project-first projects.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


def _project_describe_payload(path: str) -> dict[str, Any]:
    root = resolve_project_root(path)
    if root is None:
        raise FileNotFoundError(f"Not an xpkg project: {path}")
    descriptor = load_project_descriptor(root)
    current_state = current_project_state_path(root)
    return {
        "status": "described",
        "project": str(root),
        "descriptor": descriptor.to_dict(),
        "paths": {
            "descriptor": str(project_descriptor_path(root)),
            "store": str(project_store_root(root)),
            "artifacts": str(project_artifacts_root(root)),
            "state": str(project_state_root(root)),
            "media": str(project_media_root(root)),
            "exports": str(project_exports_root(root)),
            "current_state": str(current_state),
        },
        "has_current_state": current_state.exists(),
    }


def _emit_project_description(payload: dict[str, Any]) -> None:
    paths = payload["paths"]
    sys.stdout.write(f"Project {payload['project']}\n")
    sys.stdout.write(f"Descriptor {paths['descriptor']}\n")
    sys.stdout.write(f"Store {paths['store']}\n")
    sys.stdout.write(f"Artifacts {paths['artifacts']}\n")
    sys.stdout.write(f"State {paths['state']}\n")
    sys.stdout.write(f"Media {paths['media']}\n")
    sys.stdout.write(f"Exports {paths['exports']}\n")
    sys.stdout.write(f"Current state present: {payload['has_current_state']}\n")


def _load_metadata_payload(path: str) -> dict[str, Any]:
    return load_json_dict(path)


def _emit_saved_metadata(payload: dict[str, object]) -> None:
    sys.stdout.write(f"Saved {payload['metadata']} metadata for {payload['project']}\n")
    write_path(Path(str(payload["path"])))


def _emit_loaded_metadata(payload: dict[str, object]) -> None:
    metadata_kind = str(payload["metadata"])
    if payload.get(metadata_kind) is None:
        sys.stdout.write(f"No {metadata_kind} metadata set for {payload['project']}\n")
        return
    sys.stdout.write(f"{metadata_kind} metadata for {payload['project']}\n")
    write_path(Path(str(payload["path"])))


@app.command("describe")
def describe(
    project: Annotated[str, typer.Argument(help="Project directory to inspect.")],
    json_output: JsonOption = False,
) -> None:
    """Describe the normalized project layout and descriptor."""

    run_command(
        json_output=json_output,
        action=lambda: _project_describe_payload(project),
        human_output=_emit_project_description,
    )


@app.command("init")
def init(
    project: Annotated[str, typer.Argument(help="Project directory to create.")],
    title: Annotated[
        str | None,
        typer.Option("--title", help="Optional project title."),
    ] = None,
    project_id: Annotated[
        str | None,
        typer.Option("--id", help="Optional project identifier."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Allow initialization into an existing empty directory."),
    ] = False,
    json_output: JsonOption = False,
) -> None:
    """Create a new empty exp-pkg project."""

    def action() -> dict[str, object]:
        init_project(
            project,
            title=title,
            project_id=project_id,
            force=force,
        )
        return {
            "status": "initialized",
            "project": project,
            "title": title,
            "project_id": project_id,
        }

    def human_output(payload: dict[str, object]) -> None:
        sys.stdout.write(f"Initialized project {Path(str(payload['project']))}\n")

    run_command(json_output=json_output, action=action, human_output=human_output)


@dataclass(frozen=True, slots=True)
class _MetadataSlot:
    """Per-slot dispatch entry for ``xpkg project metadata`` set/show."""

    name: str  # canonical kebab-case slot name in the CLI surface
    payload_key: str  # snake_case key under which the slot value lives in JSON output
    coerce: Callable[[Mapping[str, Any]], Any]
    save: Callable[[str, Any], Path]
    load: Callable[[str], Any]
    path_for: Callable[[str], Path]


_METADATA_SLOTS: dict[str, _MetadataSlot] = {
    "acquisition": _MetadataSlot(
        name="acquisition",
        payload_key="acquisition",
        coerce=AcquisitionMetadata.from_dict,
        save=save_project_acquisition_metadata,
        load=load_project_acquisition_metadata,
        path_for=project_acquisition_metadata_path,
    ),
    "dataset-share": _MetadataSlot(
        name="dataset-share",
        payload_key="dataset_share",
        coerce=DatasetShareMetadata.from_dict,
        save=save_project_dataset_share_metadata,
        load=load_project_dataset_share_metadata,
        path_for=project_dataset_share_metadata_path,
    ),
    "datasheet": _MetadataSlot(
        name="datasheet",
        payload_key="datasheet",
        coerce=DatasetDatasheet.from_dict,
        save=save_project_datasheet,
        load=load_project_datasheet,
        path_for=project_datasheet_path,
    ),
    "model-card": _MetadataSlot(
        name="model-card",
        payload_key="model_card",
        coerce=ModelCard.from_dict,
        save=save_project_model_card,
        load=load_project_model_card,
        path_for=project_model_card_path,
    ),
}


def _resolve_metadata_slot(slot: str) -> _MetadataSlot:
    if slot not in _METADATA_SLOTS:
        choices = ", ".join(sorted(_METADATA_SLOTS))
        raise typer.BadParameter(
            f"Unknown metadata slot {slot!r}. Choose from: {choices}.",
            param_hint="SLOT",
        )
    return _METADATA_SLOTS[slot]


metadata_app = typer.Typer(
    add_completion=False,
    help="Show or update typed durable metadata slots stored under .xpkg/metadata/.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


@metadata_app.command("set")
def metadata_set(
    slot: Annotated[
        str,
        typer.Argument(
            help=(
                "Metadata slot to write: acquisition, dataset-share, datasheet, "
                "or model-card."
            ),
        ),
    ],
    project: Annotated[str, typer.Argument(help="Project directory to update.")],
    source: Annotated[
        str | None,
        typer.Option(
            "--from",
            "--input",
            help="Path to a JSON object matching the slot's typed schema.",
        ),
    ] = None,
    json_output: JsonOption = False,
) -> None:
    """Write one typed metadata slot from a JSON file."""
    entry = _resolve_metadata_slot(slot)
    source = require_option_value(source, "--from")

    def action() -> dict[str, object]:
        record = entry.coerce(_load_metadata_payload(source))
        saved_path = entry.save(project, record)
        return {
            "status": "saved",
            "metadata": entry.payload_key,
            "project": project,
            "path": str(saved_path),
            entry.payload_key: record.to_dict(),
        }

    run_command(json_output=json_output, action=action, human_output=_emit_saved_metadata)


@metadata_app.command("show")
def metadata_show(
    slot: Annotated[
        str,
        typer.Argument(
            help=(
                "Metadata slot to read: acquisition, dataset-share, datasheet, "
                "or model-card."
            ),
        ),
    ],
    project: Annotated[str, typer.Argument(help="Project directory to inspect.")],
    json_output: JsonOption = False,
) -> None:
    """Show one typed metadata slot."""
    entry = _resolve_metadata_slot(slot)

    def action() -> dict[str, object]:
        record = entry.load(project)
        return {
            "status": "loaded" if record is not None else "missing",
            "metadata": entry.payload_key,
            "project": project,
            "path": str(entry.path_for(project)),
            entry.payload_key: None if record is None else record.to_dict(),
        }

    run_command(json_output=json_output, action=action, human_output=_emit_loaded_metadata)


app.add_typer(metadata_app, name="metadata")


@app.command("pack")
def pack(
    project: Annotated[str, typer.Argument(help="Project directory to pack.")],
    out: Annotated[
        str | None,
        typer.Option("--out", help="Explicit output .expkg path."),
    ] = None,
    media: Annotated[
        PackMedia,
        typer.Option(
            "--media",
            help=(
                "Media scope: full includes all managed media, package omits video "
                "containers, manifest records media without storing bytes."
            ),
        ),
    ] = PackMedia.full,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Replace an existing output artifact."),
    ] = False,
    json_output: JsonOption = False,
) -> None:
    """Pack a project into a .expkg artifact."""

    def action() -> dict[str, object]:
        artifact_path = pack_project(
            project,
            out=out,
            media=media.value,
            overwrite=overwrite,
        )
        return {
            "status": "packed",
            "project": project,
            "artifact": str(artifact_path),
            "media": media.value,
        }

    def human_output(payload: dict[str, object]) -> None:
        sys.stdout.write(f"Packed {payload['project']}\n")
        write_path(Path(str(payload["artifact"])))

    run_command(json_output=json_output, action=action, human_output=human_output)


@app.command("unpack")
def unpack(
    artifact: Annotated[str, typer.Argument(help="Path to the .expkg artifact.")],
    out: Annotated[str, typer.Option("--out", help="Destination project directory.")],
    force: Annotated[
        bool,
        typer.Option("--force", help="Allow unpacking into an existing empty directory."),
    ] = False,
    rename: Annotated[
        str | None,
        typer.Option("--rename", help="Optional new project title."),
    ] = None,
    json_output: JsonOption = False,
) -> None:
    """Unpack a .expkg artifact into a project."""
    out = require_option_value(out, "--out")

    def action() -> dict[str, object]:
        project_path = unpack_project(
            artifact,
            out,
            force=force,
            rename_title=rename,
        )
        return {
            "status": "unpacked",
            "artifact": artifact,
            "project": str(project_path),
            "title": rename,
        }

    def human_output(payload: dict[str, object]) -> None:
        sys.stdout.write(f"Unpacked {payload['artifact']}\n")
        write_path(Path(str(payload["project"])))

    run_command(json_output=json_output, action=action, human_output=human_output)


@app.command("validate")
def validate(
    path: Annotated[str, typer.Argument(help="Project or .expkg artifact to validate.")],
    json_output: JsonOption = False,
) -> None:
    """Validate a project or packed .expkg artifact."""

    def action() -> dict[str, object]:
        validate_artifact_target(path)
        return {"status": "valid", "path": path}

    def human_output(payload: dict[str, object]) -> None:
        sys.stdout.write(f"Valid {payload['path']}\n")

    run_command(json_output=json_output, action=action, human_output=human_output)
