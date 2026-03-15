# Experimental Durable Store

<div class="page-intro">
<p>
Posetta now has an <strong>experimental</strong> durable store layer for crash-safe,
commit-oriented archive management. It does <em>not</em> replace normal
<code>.siesta</code> archive IO yet. It wraps that archive format in a directory-backed
store with immutable objects, dual superblocks, and journal recovery.
</p>
</div>

!!! warning
    This workflow is experimental. The stable interchange artifact is still the
    single-file <code>.siesta</code> archive produced by <code>write_siesta(...)</code>.
    Use the durable store when you want stronger recovery semantics around commit
    boundaries, not when you need the simplest on-disk exchange format.

## What Changed

The old and current stable workflow is a single HDF5 archive file:

```text
session.siesta
```

The new experimental workflow is a store root directory that manages committed
archives internally:

```text
session.siesta/
  superblock.a.json
  superblock.b.json
  LOCK
  journal/
    active.json
  commits/
    000000000001/commit.json
  objects/
    ab/cd/obj_<sha256>.siesta
  workspace/
  snapshots/
```

That means the suffix is the same, but the meaning is different:

- `session.siesta` as a file means a normal archive.
- `session.siesta/` as a directory means an experimental durable store root.

## Why You Would Use It

Use the durable store when you want commit-style persistence around autosave or
interactive editing boundaries:

- each committed archive is immutable
- the current head is selected by a superblock flip instead of in-place mutation
- recovery falls back to the last clean commit when a journal is left behind
- staged writes are separated from the committed archive head

This is aimed at protecting the last committed human-label state when a process
dies mid-save.

## Recommended Experimental Workflow

The durable store is designed to sit on top of the existing archive writer.

1. Produce a normal `.siesta` archive with the regular archive API.
2. Create a store root from that archive.
3. For each new save boundary, write a fresh staged archive.
4. Commit that staged archive into the store.
5. Reopen through the store when you want recovery semantics.

Example:

```python
from pathlib import Path

from posetta.formats import (
    create_store_from_archive,
    open_store,
    read_siesta,
    write_siesta,
)
from posetta.model import Labels

labels = Labels()

# 1. Create the first ordinary archive
seed_archive = Path("seed.siesta")
write_siesta(seed_archive, labels)

# 2. Wrap it in an experimental durable store
store = create_store_from_archive(Path("session.siesta"), seed_archive)

# 3. Later, stage a fresh archive with the normal writer
staged_archive = Path("session-next.siesta")
write_siesta(staged_archive, labels)

# 4. Commit the staged archive as the new durable head
store.commit_new_archive(staged_archive, reason="autosave")

# 5. Reopen with recovery semantics and resolve the current committed archive
store = open_store(Path("session.siesta"))
payload = read_siesta(store.current_archive_path(), lazy=False)
```

## Public Entry Points

The experimental format surface currently exposes:

- `create_store_from_archive(store_root, initial_archive)`
- `create_store_from_sta(store_root, initial_sta)`
  This is a compatibility alias. It still accepts `.siesta` archives.
- `open_store(store_root)`
- `SiestaStore.current_archive_path()`
- `SiestaStore.commit_new_archive(...)`

The store class also keeps compatibility aliases:

- `current_bundle_path()`
  This is a legacy alias for `current_archive_path()`.
- `commit_new_bundle(...)`
  This is a legacy alias for `commit_new_archive(...)`.

Those exist so we can layer the feature in without breaking older naming while
we transition the user-facing language toward `archive`.

## Recovery Model

`open_store(...)` calls recovery before returning a mounted store.

The recovery logic is intentionally narrow:

- if no journal exists, the highest valid superblock wins
- if one superblock is corrupt and the other is valid, the valid one wins
- if the journal is left in `staging` or `validating`, the store reverts to
  `last_clean_commit_id`
- if the journal is left in `committing` or `cleanup`, the store trusts the
  flipped head only when the referenced commit file exists; otherwise it reverts

This is why the feature is useful for autosave-style boundaries: it prefers a
known clean head over guessing about partially finished writes.

## What Is Still Stable Versus Experimental

Stable today:

- `write_siesta(...)`
- `read_siesta(...)`
- append/merge/update on the single-file archive path
- `.siesta` as the native archive contract

Experimental today:

- directory-backed `siesta_store`
- commit-oriented autosave flow
- superblock/journal durability layer
- direct application integration with staged archive commits

If you are building interchange, exports, fixtures, or test corpora, prefer the
single-file archive. If you are prototyping crash-safe editing or autosave
behavior, the experimental store is the right layer to evaluate.
