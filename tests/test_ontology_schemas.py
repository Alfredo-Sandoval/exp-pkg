from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

from xpkg.io.experiment_json import EXPERIMENT_SCHEMA_VERSION
from xpkg.io.session_json import RECORDING_SESSION_SCHEMA_VERSION
from xpkg.model import PoseTrack, RecordingSession, SubjectTrackAssignment
from xpkg.ontology import ONTOLOGY_SCHEMA_VERSION, ontology_schema_documents


def test_generated_ontology_schemas_match_canonical_source() -> None:
    for name, document in ontology_schema_documents().items():
        stored = json.loads((Path("schemas") / name).read_text(encoding="utf-8"))
        assert stored == document


def test_ontology_catalog_tracks_live_dataclass_properties() -> None:
    document = ontology_schema_documents()["ontology.json"]
    objects = {item["name"]: item for item in document["object_types"]}
    assert objects["RecordingSession"]["properties"] == [
        field.name for field in fields(RecordingSession)
    ]
    assert objects["SubjectTrackAssignment"]["properties"] == [
        field.name for field in fields(SubjectTrackAssignment)
    ]
    assert objects["PoseTrack"]["properties"] == [field.name for field in fields(PoseTrack)]
    assert objects["SessionPose"]["properties"] == [
        "name",
        "data",
        "videos",
        "calibration",
        "provenance",
        "metadata",
    ]
    object_names = set(objects)
    for link in document["link_types"]:
        assert link["source_type"] in object_names
        assert link["target_type"] in object_names


def test_persisted_ontology_versions_advance_together() -> None:
    assert ONTOLOGY_SCHEMA_VERSION == 4
    assert RECORDING_SESSION_SCHEMA_VERSION == ONTOLOGY_SCHEMA_VERSION
    assert EXPERIMENT_SCHEMA_VERSION == ONTOLOGY_SCHEMA_VERSION


def test_semantic_model_does_not_import_io_layer() -> None:
    model_root = Path("src/xpkg/model")
    offenders = [
        str(path)
        for path in model_root.rglob("*.py")
        if "xpkg.io" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []
