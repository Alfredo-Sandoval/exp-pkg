# Performance Guidance

xpkg has two different usage modes:

- shallow description for project pickers, catalogs, startup checks, and agents
- full hydration or validation when a user opens, edits, validates, or publishes a project

Use the shallowest public surface that answers the current question. Project
state can hold dense frame labels, predictions, masks, and recording payloads,
so materializing it during a list refresh turns a cheap UI action into a full
data load.

## Project Pickers And Catalogs

Downstream GUIs should populate project rows from descriptor and layout
metadata, not from the full project payload.

| Job | Use | Avoid in hot list paths |
| --- | --- | --- |
| Show a project row | `xpkg project describe PATH --json`, `ProjectService.open(PATH).describe()`, `load_project_descriptor(PATH)` | `validate_project(PATH)` |
| Detect whether a path is an xpkg project | `xpkg inspect PATH --json`, `xpkg.inspection.inspect_path(PATH)` | `ProjectService.inspect()` when you do not need a validated state summary |
| Show current-state presence or size | `current_project_state_path(PATH).exists()` and `stat().st_size` | `load_project_payload(PATH)` |
| Read durable typed metadata | `ProjectService.open(PATH).metadata` | loading labels, media, predictions, or Vicon recordings |
| Open a selected project for editing | `ProjectService.open(PATH).load_labels()` or the relevant domain loader | doing this for every project in the catalog |
| Validate before publish or CI | `project.validate()` / `xpkg project validate PATH` | running validation on every startup refresh |

Treat `.xpkg/state/current.json` as project state, not as a list index. It can
be tens or hundreds of megabytes when predictions or dense annotations are
committed.

## Recommended Downstream Flow

For a GUI, agent, or automation tool:

1. At startup, discover candidate folders with normal filesystem scans.
2. For each candidate, call `xpkg project describe PATH --json` or
   `ProjectService.open(PATH).describe()` to get descriptor and managed paths.
3. Read only `PROJECT.json`, typed metadata slots, and current-state file stats
   for list rows.
4. When a user selects a project, open and hydrate only that project.
5. Run full validation only on explicit validate, pack, publish, or CI actions.

```python
from pathlib import Path

from xpkg.project import current_project_state_path, load_project_descriptor


def project_row(project_root: Path) -> dict[str, object]:
    descriptor = load_project_descriptor(project_root)
    state_path = current_project_state_path(project_root)
    state_bytes = state_path.stat().st_size if state_path.exists() else None
    return {
        "project_id": descriptor.project_id,
        "title": descriptor.title,
        "has_current_state": state_path.exists(),
        "state_bytes": state_bytes,
    }
```

That pattern is intentionally shallow: it does not parse frames, predictions,
media manifests, masks, or Vicon payloads.

## CLI Surfaces

Use the CLI the same way:

```bash
xpkg project describe "./My Project" --json
xpkg inspect "./My Project" --json
xpkg project validate "./My Project"
```

`describe` and `inspect` are for cheap identification and summary work.
`validate` is the full contract check. `pack` validates before creating a
portable `.expkg`.

When the input is a packed `.expkg`, inspect or validate the manifest before
unpacking. Unpack only when the caller needs the editable project state or
managed media.

## Dense Outputs

Do not store or repeatedly scan dense model outputs through a frame-by-frame
project-list path.

- Use Parquet mask tables with `write_mask_table` and
  `MaskTableReader` for dense instance-mask outputs.
- Prefer window reads over full-table materialization in GPU or batch
  pipelines.
- Save model cards, acquisition metadata, dataset-share metadata, and
  datasheets in typed metadata slots when that information belongs in catalog
  views.
- Keep frame labels and predictions for open/edit/analyze paths, not catalog
  refresh paths.

## API Naming Contract

New xpkg APIs should make their cost visible:

- `describe`, `inspect`, `list`, and `summary` surfaces should be shallow unless
  their docstring explicitly says otherwise.
- `load`, `hydrate`, `validate`, `pack`, and `unpack` surfaces may read full
  state, media, and artifacts.
- If a summary API must parse state payloads, document the reason and add a
  cheaper descriptor-level alternative.

This keeps downstream tools responsive and preserves one clear rule: user
selection hydrates state; project listing does not.
