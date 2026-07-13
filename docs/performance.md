# Performance Guidance

xpkg has two different usage modes:

- shallow description for project pickers, catalogs, startup checks, and agents
- full hydration or validation when a user opens, edits, validates, or publishes a project

## Enforced Budgets

Run the deterministic performance gate with:

```bash
make performance-check
```

The canonical limits and sample counts live in `performance-budgets.json` at
the repository root. The gate measures package import, CLI help startup,
project creation, shallow project description, and shallow path inspection. It
prints a versioned JSON verdict and exits with a nonzero status when any median
exceeds its declared budget. CI runs the same command on macOS and Linux.

Media decode, seek, encode, and transcode throughput depend on codecs, hardware,
and acceleration backends. Keep those measurements in the media benchmark lane
described in [Media I/O Architecture](architecture/media-io.md); do not disguise
machine-specific throughput as a portable CI threshold.

Use the shallowest public surface that answers the current question. Project
state can hold dense frame labels, predictions, masks, and media references,
so materializing it during a list refresh turns a cheap UI action into a full
data load.

## Project Pickers And Catalogs

Downstream GUIs should populate project rows from descriptor and layout
metadata, not from the full project payload. `PROJECT.json` stays the small
generated identity and layout descriptor; `.xpkg/indexes/project_summary.json`
is the generated inventory for "what is in this project".

| Job | Use | Avoid in hot list paths |
| --- | --- | --- |
| Show a project row | `xpkg project describe PATH --json`, `ProjectService.open(PATH).describe()`, `load_project_summary(PATH)` | `validate_project(PATH)` |
| Detect whether a path is an xpkg project | `xpkg inspect PATH --json`, `xpkg.inspection.inspect_path(PATH)`, `ProjectService.inspect()` | `validate_project(PATH)` |
| Show current-state presence or size | `current_project_state_path(PATH).exists()` and `stat().st_size` | `load_project_payload(PATH)` |
| Read durable typed metadata | `ProjectService.open(PATH).metadata` | loading labels, media, or predictions |
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
3. Read `PROJECT.json` plus `.xpkg/indexes/project_summary.json` for list rows.
4. When a user selects a project, open and hydrate only that project.
5. Run full validation only on explicit validate, pack, publish, or CI actions.

```python
from pathlib import Path

from xpkg.project import load_project_summary


def project_row(project_root: Path) -> dict[str, object]:
    summary = load_project_summary(project_root)
    return {
        "project_id": summary.project_id,
        "title": summary.title,
        "modalities": list(summary.modalities),
        "has_current_state": summary.has_current_state,
        "state_bytes": summary.state_bytes,
    }
```

That pattern is intentionally shallow: it does not parse frames, predictions,
media manifests or masks.

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
- Keep acquisition metadata on its recording session and dataset-sharing
  metadata on the experiment. Keep model cards and datasheets as compact
  documentation records when they belong in catalog views.
- Persist compact associated-media inventory during save/import, when labels
  and media descriptors are already in memory. Inspect can then re-check path
  presence and known frame-index ranges without decoding media or hydrating the
  label payload.
- Keep frame labels and predictions for open/edit/analyze paths, not catalog
  refresh paths.

## API Naming Contract

New xpkg APIs should make their cost visible:

- `describe`, `list`, and `summary` surfaces should be shallow unless their
  docstring explicitly says otherwise.
- `xpkg inspect PATH --json`, `xpkg.inspection.inspect_path(PATH)`, and
  `ProjectService.inspect()` are shallow for project directories.
- `load`, `hydrate`, `validate`, `pack`, and `unpack` surfaces may read full
  state, media, and artifacts.
- If a summary API must parse state payloads, document the reason and add a
  cheaper descriptor-level alternative.

This keeps downstream tools responsive and preserves one clear rule: user
selection hydrates state; project listing does not.
