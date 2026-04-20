# Compatibility

<div class="page-intro">
<p>
<code>xpkg.compat</code> is the edge compatibility layer for archive-first
flows. It exposes the canonical <code>.xpkg</code> helpers for direct archive
reads, writes, and archive-level migration work.
</p>
</div>

!!! warning
    This is not the preferred integration boundary for new code. Use
    <code>xpkg.formats</code>, <code>xpkg.api</code>, or
    <code>xpkg.services.WorkspaceService</code> for the workspace-first
    project contract. Reach for <code>xpkg.compat</code> only when you are
    migrating archive-first workflows, writing fixtures, or touching edge
    archive flows.

Prefer the canonical <code>.xpkg</code>-named helpers on this module such as
<code>read_xpkg</code>, <code>write_xpkg</code>, and
<code>create_store_from_xpkg</code>. Older archive-shaped names remain
importable only as deprecated aliases for existing callers.

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

### `read_metrics_table(archive_path, name) -> pandas.DataFrame`

Read one named metrics table from `/metrics/<name>`.

### `write_metrics_table(archive_path, name, dataframe, *, mode="append")`

Write or append one metrics table into an archive.

## Durable Archive Store

### `create_store_from_xpkg(store_root, initial_xpkg) -> ArchiveStore`

Canonical entrypoint that makes the `.xpkg` naming explicit.

### `open_store(store_root) -> ArchiveStore`

Open a durable store and run recovery before returning it.

## Deprecated Legacy Aliases

The following older names intentionally remain importable for backward
compatibility, but new code should not start from them:

- `read_archive(...)` -> prefer `read_xpkg(...)`
- `write_archive(...)` -> prefer `write_xpkg(...)`
- `update_labels_archive(...)` -> prefer `update_labels_xpkg(...)`
- `append_predictions_archive(...)` -> prefer `append_predictions_xpkg(...)`
- `merge_predictions_archive(...)` -> prefer `merge_predictions_xpkg(...)`
- `validate_archive(...)` / `validate_project(...)` -> prefer `validate_xpkg(...)`
- `summarize_archive(...)` / `summarize_project(...)` -> prefer `summarize_xpkg(...)`
- `create_archive_store(...)` / `create_store_from_archive(...)` -> prefer `create_store_from_xpkg(...)`
