"""Generate packaged schema-4 ontology documents."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from xpkg.ontology import ontology_schema_documents

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_ROOT = ROOT / "schemas"


def _render(document: dict[str, object]) -> str:
    return json.dumps(document, indent=2, sort_keys=False, ensure_ascii=True) + "\n"


def main() -> int:
    check = sys.argv[1:] == ["--check"]
    if sys.argv[1:] not in ([], ["--check"]):
        raise SystemExit("usage: generate_ontology_schemas.py [--check]")
    stale: list[Path] = []
    for name, document in ontology_schema_documents().items():
        path = SCHEMA_ROOT / name
        expected = _render(document)
        if check:
            if not path.is_file() or path.read_text(encoding="utf-8") != expected:
                stale.append(path)
        else:
            path.write_text(expected, encoding="utf-8")
    for path in stale:
        print(f"Stale generated ontology schema: {path.relative_to(ROOT)}", file=sys.stderr)
    return bool(stale)


if __name__ == "__main__":
    raise SystemExit(main())
