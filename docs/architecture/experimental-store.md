# Experimental Durable Store

<div class="page-intro">
<p>
xpkg now has an <strong>experimental</strong> durable store layer for crash-safe,
commit-oriented project state. In the locked v1 artifact contract this belongs
under the workspace-owned <code>.xpkg/</code> directory. The current prototype
still wraps staged compatibility archives internally while we harden the
private storage engine.
</p>
</div>

If you want the broader rationale for why the runtime still stages
legacy archive payloads at all, read [Storage Direction](storage-direction.md).

!!! warning
    This workflow is experimental private machinery. The public v1 artifact
    contract is workspace folder + <code>.expkg</code>. Use the durable store
    when you want stronger recovery semantics inside <code>.xpkg/</code>, not
    as a public interchange layer.

!!! info
    Status: current private prototype. The committed source of truth is the
    durable store head; <code>.xpkg/state/current.json</code> is only a cache.

## What Changed

The public contract is a workspace:

```text
My Project/
  PROJECT.json
  .xpkg/
  Media/
  Exports/
    My Project.expkg
```

Inside that workspace, the current experimental prototype manages committed
compatibility archives internally:

```text
My Project/.xpkg/
  superblock.a.json
  superblock.b.json
  LOCK
  journal/
    active.json
  commits/
    000000000001/commit.json
  objects/
    ab/cd/obj_<sha256>.xpkg
  workspace/
    tmp-<txn>.xpkg
  state/
    current.json
```

That internal layout is intentionally private and versioned. The only public
guarantee is that `.xpkg/` exists and is valid for the declared project
version.

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

The current prototype sits on top of the existing compatibility archive writer
while targeting the `.xpkg/` private state layer.

1. Produce a normal `.xpkg` compatibility archive with the regular archive API.
2. Create a store root from that archive.
3. For each new save boundary, write a fresh staged archive.
4. Commit that staged archive into the store.
5. Reopen through the store when you want recovery semantics.

Example:

```python
from pathlib import Path

from xpkg.compat import (
    create_store_from_archive,
    open_store,
    read_xpkg,
    write_xpkg,
)
from xpkg.model import Labels

labels = Labels()

# 1. Create the first staged compatibility archive
workspace_root = Path("My Project")
seed_archive = workspace_root / ".xpkg" / "workspace" / "seed.xpkg"
seed_archive.parent.mkdir(parents=True, exist_ok=True)
write_xpkg(seed_archive, labels)

# 2. Wrap it in the experimental private store root
store = create_store_from_archive(workspace_root / ".xpkg", seed_archive)

# 3. Later, stage a fresh compatibility archive with the normal writer
staged_archive = workspace_root / ".xpkg" / "workspace" / "session-next.xpkg"
write_xpkg(staged_archive, labels)

# 4. Commit the staged archive as the new durable head
store.commit_new_archive(staged_archive, reason="autosave")

# 5. Reopen with recovery semantics and resolve the current committed archive
store = open_store(workspace_root / ".xpkg")
payload = read_xpkg(store.current_archive_path(), lazy=False)
```

## Public Entry Points

The experimental format surface currently exposes:

- `create_store_from_archive(store_root, initial_archive)`
- `create_store_from_xpkg(store_root, initial_xpkg)`
- `create_store_from_sta(store_root, initial_sta)`
  This is a compatibility alias. It still accepts `.sta` / `.sta` archives.
- `open_store(store_root)`
- `ArchiveStore.current_archive_path()`
- `ArchiveStore.commit_new_archive(...)`

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

- public workspace contract: `PROJECT.json`, `.xpkg/`, `Media/`, `Exports/`
- `.expkg` as the portable project artifact
- `xpkg.compat` as the edge compatibility surface for `.xpkg` archives and
  legacy aliases

Experimental today:

- private `.xpkg/` durable-store machinery
- commit-oriented autosave flow
- superblock/journal durability layer
- direct application integration with staged archive commits

If you are building the public project contract, think in terms of workspace +
`.expkg`. If you are prototyping crash-safe editing or autosave behavior,
the experimental store is the right private layer to evaluate.
