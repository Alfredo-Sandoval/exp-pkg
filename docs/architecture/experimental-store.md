# Experimental Durable Store

<div class="page-intro">
<p>
xpkg now has an <strong>experimental</strong> durable store layer for crash-safe,
commit-oriented project state. In the locked v1 artifact contract this belongs
under the workspace-owned <code>.xpkg/</code> directory. Normal workspace
commits now store workspace-native snapshot roots in that private store.
</p>
</div>

If you want the broader rationale for the storage cutover, read
[Storage Direction](storage-direction.md).

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

Inside that workspace, the current private store manages committed root
payloads behind dual superblocks:

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
    ab/cd/obj_<sha256>.json
  workspace/
    tmp-<txn>
  state/
    current.json
```

That internal layout is intentionally private and versioned. The only public
guarantee is that `.xpkg/` exists and is valid for the declared project
version.

## Why You Would Use It

Use the durable store when you want commit-style persistence around autosave or
interactive editing boundaries:

- each committed root payload is immutable
- the current head is selected by a superblock flip instead of in-place mutation
- recovery falls back to the last clean commit when a journal is left behind
- staged writes are separated from the committed head

This is aimed at protecting the last committed human-label state when a process
dies mid-save.

## Recommended Experimental Workflow

Normal project-facing code should not call the store directly. Use the
workspace APIs; they commit snapshot roots into `.xpkg/` and refresh
`.xpkg/state/current.json` as a rebuildable cache.

Example:

```python
from xpkg.model import Labels
from xpkg.services import WorkspaceService

labels = Labels()

workspace = WorkspaceService.create("My Project", title="My Project")
workspace.save_labels(labels, metadata={"source": "manual"})
workspace.validate()
```

## Low-Level Entry Points

Low-level store helpers still exist for tests, recovery work, and private
workspace storage flows. They are intentionally outside the workspace-first
public contract and do not appear in `xpkg.api`, `xpkg.workspace`, or the CLI:

- `ArchiveStore.create_from_archive(store_root, initial_xpkg)`
- `ArchiveStore.open(store_root)`
- `ArchiveStore.current_archive_path()`
- `ArchiveStore.commit_new_archive(...)`

## Recovery Model

`ArchiveStore.open(...)` calls recovery before returning a mounted store.

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

Experimental today:

- private `.xpkg/` durable-store machinery
- commit-oriented autosave flow
- superblock/journal durability layer
- low-level `.xpkg` archive helpers under `xpkg.io.archive_format` for
  migrations, fixtures, and private storage internals

If you are building the public project contract, think in terms of workspace +
`.expkg`. If you are prototyping crash-safe editing or autosave behavior,
the experimental store is the right private layer to evaluate.
