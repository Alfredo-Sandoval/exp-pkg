from __future__ import annotations

import json
from pathlib import Path


def test_project_schema_locks_required_v1_descriptor_fields() -> None:
    schema_path = Path("schemas/project.schema.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == [
        "format",
        "project_schema_version",
        "layout_version",
        "title",
        "project_id",
        "created_at",
        "updated_at",
        "store_path",
        "media_root",
        "exports_root",
        "default_pack_mode",
    ]

    properties = schema["properties"]
    assert properties["format"]["const"] == "xpkg-project"
    assert properties["project_schema_version"]["const"] == 1
    assert properties["layout_version"]["const"] == 1
    assert properties["store_path"]["const"] == ".xpkg"
    assert properties["media_root"]["const"] == "Media"
    assert properties["exports_root"]["const"] == "Exports"
    assert properties["default_pack_mode"]["enum"] == ["portable", "snapshot"]
