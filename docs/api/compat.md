# Compatibility

<div class="page-intro">
<p>
<code>xpkg.compat</code> is the edge compatibility layer for archive-first
flows. It exposes the canonical <code>.xpkg</code> helpers and keeps older
<code>.sta</code> / <code>.sta</code> names as aliases during the transition.
</p>
</div>

!!! warning
    This is not the preferred integration boundary for new code. Use
    <code>xpkg.formats</code>, <code>xpkg.api</code>, or
    <code>xpkg.services.WorkspaceService</code> for the workspace-first
    project contract. Reach for <code>xpkg.compat</code> only when you are
    migrating legacy bundles, writing fixtures, or touching edge archive flows.

## Canonical Archive Helpers

### `write_xpkg(path, labels, predictions=None, suggestions=None, metadata=None, metrics=None, manifest=None)`

Create or overwrite a canonical `.xpkg` archive on disk.

### `read_xpkg(path, *, lazy=False)`

Load a `.xpkg` archive and return a dict with:

| Key | Type | Contents |
| --- | --- | --- |
| `"labels"` | `Labels` | The main annotation container |
| `"metadata"` | `dict` | Archive-level metadata |
| `"videos"` | `list[Video]` | Video references |
| `"predictions"` | `dict` or `None` | Prediction payloads if present |

### `update_labels_xpkg(path, labels, *, journal=True, regenerate_predictions=False)`

Overwrite the labels portion of an existing archive while preserving the rest.

### `append_predictions_xpkg(path, batch, *, allow_max_inst_growth=False, journal=True, fsync=True, run_metadata=None) -> int`

Append predictions to an existing archive.

### `merge_predictions_xpkg(path, batch, *, allow_max_inst_growth=True, journal=True, fsync=True, run_metadata=None) -> int`

Merge predictions into existing archive frames.

### `summarize_xpkg(path)`

Return a lightweight summary of an archive or project path.

### `validate_xpkg(path)`

Run structural validation against the archive layout and raise if invalid.

## Metrics Tables

### `read_metrics_table(bundle_path, name) -> pandas.DataFrame`

Read one named metrics table from `/metrics/<name>`.

### `write_metrics_table(bundle_path, name, dataframe, *, mode="append")`

Write or append one metrics table into an archive.

## Legacy Alias Layer

The following names remain available as compatibility aliases:

- `read_sta`
- `write_sta`
- `update_labels_sta`
- `append_predictions_sta`
- `merge_predictions_sta`
- `read_archive`
- `write_archive`
- `update_labels_archive`
- `append_predictions_archive`
- `merge_predictions_archive`

Use them only when you need compatibility with older call sites or archived
tooling that still speaks in `.sta` terms.

## Durable Archive Store

### `create_store_from_archive(store_root, initial_archive) -> ArchiveStore`

Create a directory-backed durable store from an existing archive payload.

### `create_store_from_xpkg(store_root, initial_xpkg) -> ArchiveStore`

Canonical entrypoint that makes the `.xpkg` naming explicit.

### `create_store_from_sta(store_root, initial_sta) -> ArchiveStore`

Legacy alias that keeps older `.sta` naming available at the edge.

### `open_store(store_root) -> ArchiveStore`

Open a durable store and run recovery before returning it.

### `create_archive_store(store_root, initial_archive) -> ArchiveStore`

Alias for `create_store_from_archive(...)`.

### `open_archive_store(store_root) -> ArchiveStore`

Alias for `open_store(...)`.
