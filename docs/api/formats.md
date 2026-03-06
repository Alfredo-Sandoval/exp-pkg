# Formats

<div class="page-intro">
<p>
<code>posetta.formats</code> reads and writes native <code>.siesta</code> bundles.
</p>
</div>

## Read and Write Bundles

### `write_siesta(path, labels, predictions=None, suggestions=None, metadata=None, metrics=None, manifest=None)`

Create or overwrite a `.siesta` bundle on disk.

### `read_siesta(path, *, lazy=False)`

Load a `.siesta` bundle and return a dict:

| Key | Type | Contents |
| --- | --- | --- |
| `"labels"` | `Labels` | The main annotation container |
| `"metadata"` | `dict` | Bundle-level metadata |
| `"videos"` | `list[Video]` | Video references |
| `"predictions"` | `dict` or `None` | Prediction payloads if present |

- `lazy=False` materializes bundle arrays immediately.
- `lazy=True` keeps dataset-backed handles for larger reads.

Example:

```python
from pathlib import Path

from posetta.formats import read_siesta, write_siesta
from posetta.model import Labels

path = Path("example.siesta")
write_siesta(path, Labels())
payload = read_siesta(path, lazy=False)
labels = payload["labels"]
```

## Typical Workflow

<div class="panel-grid panel-grid-3" markdown="1">

<div class="surface-card" markdown="1">
<div class="surface-kicker">READ</div>
Load a bundle with `read_siesta(...)` to get labels, metadata, videos,
and prediction payloads in one call.
</div>

<div class="surface-card" markdown="1">
<div class="surface-kicker">WRITE</div>
Create or replace a bundle with `write_siesta(...)` when your `Labels` object is
ready.
</div>

<div class="surface-card" markdown="1">
<div class="surface-kicker">UPDATE</div>
Use append, merge, and metrics helpers to modify bundle contents in place.
</div>

</div>

## Update Existing Bundles

### `update_labels_siesta(path, labels, *, journal=True)`

Overwrite the labels portion of an existing `.siesta` archive while preserving
the rest of the bundle structure.

## Append or Merge Predictions

### `append_predictions_siesta(path, batch, *, allow_max_inst_growth=False, journal=True, fsync=True, run_metadata=None) -> int`

Append new prediction rows to an existing bundle.

### `merge_predictions_siesta(path, batch, *, allow_max_inst_growth=True, journal=True, fsync=True, run_metadata=None) -> int`

Merge predictions into already-existing frames in a bundle.

Both functions operate on sequences of `PredictionAppendItem`.

### Prediction helper types

- `PredictionAppendItem`
- `SerializerPredictedInstance`
- `MaxInstancesExceededError`

## Validate and Summarize

### `summarize_project(path)`

Return a lightweight summary of a bundle or project path.

### `validate_project(path)`

Run structural validation against the bundle layout and raise if the file is not
valid.

## Metrics Tables

### `read_metrics_table(bundle_path, name) -> pandas.DataFrame`

Read one named metrics table from `/metrics/<name>`.

### `write_metrics_table(bundle_path, name, dataframe, *, mode="append")`

Write or append one metrics table into a bundle.

Use `mode="replace"` to overwrite an existing table.

## Minimal Write Plus Metrics Example

```python
import pandas as pd

from posetta.formats import read_siesta, write_metrics_table, write_siesta
from posetta.model import Labels

bundle_path = "session.siesta"
write_siesta(bundle_path, Labels())
write_metrics_table(
    bundle_path,
    "pose_eval",
    pd.DataFrame({"video": ["session.mp4"], "score": [0.94]}),
    mode="replace",
)

payload = read_siesta(bundle_path, lazy=False)
print(payload["labels"])
```
