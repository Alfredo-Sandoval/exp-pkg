# Formats

<div class="page-intro">
<p>
<code>xpkg.formats</code> documents the low-level legacy
<code>.siesta</code> archive APIs that remain during the transition to the
workspace-first public contract.
</p>
</div>

!!! note
    The public xpkg v1 artifact contract is workspace folder +
    <code>.expkg</code>. Use this module when you need the current
    compatibility APIs for legacy <code>.siesta</code> data, fixtures, tests,
    and migration workflows.

## Read and Write Legacy Archives

### `write_siesta(path, labels, predictions=None, suggestions=None, metadata=None, metrics=None, manifest=None)`

Create or overwrite a `.siesta` archive on disk.

### `read_siesta(path, *, lazy=False)`

Load a `.siesta` archive and return a dict:

| Key | Type | Contents |
| --- | --- | --- |
| `"labels"` | `Labels` | The main annotation container |
| `"metadata"` | `dict` | Archive-level metadata |
| `"videos"` | `list[Video]` | Video references |
| `"predictions"` | `dict` or `None` | Prediction payloads if present |

- `lazy=False` materializes archive arrays immediately.
- `lazy=True` keeps dataset-backed handles for larger reads.

Example:

```python
from pathlib import Path

from xpkg.formats import read_siesta, write_siesta
from xpkg.model import Labels

path = Path("example.siesta")
write_siesta(path, Labels())
payload = read_siesta(path, lazy=False)
labels = payload["labels"]
```

## Typical Compatibility Workflow

<div class="panel-grid panel-grid-3" markdown="1">

<div class="surface-card" markdown="1">
<div class="surface-kicker">READ</div>
Load an archive with `read_siesta(...)` to get labels, metadata, videos,
and prediction payloads in one call.
</div>

<div class="surface-card" markdown="1">
<div class="surface-kicker">WRITE</div>
Create or replace an archive with `write_siesta(...)` when your `Labels` object is
ready.
</div>

<div class="surface-card" markdown="1">
<div class="surface-kicker">UPDATE</div>
Use append, merge, and metrics helpers to modify archive contents in place.
</div>

</div>

## Update Existing Archives

### `update_labels_siesta(path, labels, *, journal=True, regenerate_predictions=False)`

Overwrite the labels portion of an existing `.siesta` archive while preserving
the rest of the archive structure.

## Append or Merge Predictions

### `append_predictions_siesta(path, batch, *, allow_max_inst_growth=False, journal=True, fsync=True, run_metadata=None) -> int`

Append new prediction rows to an existing archive.

### `merge_predictions_siesta(path, batch, *, allow_max_inst_growth=True, journal=True, fsync=True, run_metadata=None) -> int`

Merge predictions into already-existing frames in an archive.

Both functions operate on sequences of `PredictionAppendItem`.

### Prediction helper types

- `PredictionAppendItem`
- `SerializerPredictedInstance`
- `MaxInstancesExceededError`

## Validate and Summarize

### `summarize_project(path)`

Return a lightweight summary of an archive or project path.

### `validate_project(path)`

Run structural validation against the archive layout and raise if the file is not
valid.

## Metrics Tables

### `read_metrics_table(bundle_path, name) -> pandas.DataFrame`

Read one named metrics table from `/metrics/<name>`.

### `write_metrics_table(bundle_path, name, dataframe, *, mode="append")`

Write or append one metrics table into an archive.

Use `mode="replace"` to overwrite an existing table.

## Minimal Write Plus Metrics Example

```python
import pandas as pd

from xpkg.formats import read_siesta, write_metrics_table, write_siesta
from xpkg.model import Labels

archive_path = "session.siesta"
write_siesta(archive_path, Labels())
write_metrics_table(
    archive_path,
    "pose_eval",
    pd.DataFrame({"video": ["session.mp4"], "score": [0.94]}),
    mode="replace",
)

payload = read_siesta(archive_path, lazy=False)
print(payload["labels"])
```

## Experimental Durable Store

!!! warning
    The durable store is experimental private machinery, not a public artifact
    contract. In the v1 model it belongs under the workspace-owned
    <code>.xpkg/</code> state, even though the current prototype still wraps
    staged <code>.siesta</code> compatibility archives internally.

Use the store when you want commit-style recovery around staged archive writes.
Keep using `write_siesta(...)` and `read_siesta(...)` for the ordinary single-file
archive workflow.

### `create_store_from_archive(store_root, initial_archive) -> SiestaStore`

Create a directory-backed durable store from an existing `.siesta` archive.

### `create_store_from_sta(store_root, initial_sta) -> SiestaStore`

Compatibility alias for `create_store_from_archive(...)`.

### `open_store(store_root) -> SiestaStore`

Open a durable store and run recovery before returning it.

The mounted store then gives you:

- `current_archive_path()` to resolve the current committed archive
- `commit_new_archive(...)` to commit a staged archive as the new head

For the full workflow and on-disk layout, read
[Experimental Durable Store](../architecture/experimental-store.md).
