# Workspace Durability

<div class="page-intro">
<p>
xpkg now has an <strong>experimental</strong> workspace durability layer for
crash-safe, commit-oriented project state. In the locked v1 artifact contract
this belongs under the workspace-owned <code>.xpkg/</code> directory. Normal
workspace commits now store workspace-native snapshot roots in that private
store.
</p>
</div>

If you want the broader rationale for the storage cutover, read
[Storage Direction](storage-direction.md).

!!! warning
This workflow is experimental private machinery. The public v1 artifact
contract is workspace folder + <code>.expkg</code>. Use the durability layer
when you want stronger recovery semantics inside <code>.xpkg/</code>, not
as a public interchange layer.

!!! info
Status: current private prototype. The committed source of truth is the
workspace durability head; <code>.xpkg/state/current.json</code> is only a
cache.

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
  state/
    current.json
```

That internal layout is intentionally private and versioned. The only public
guarantee is that `.xpkg/` exists and is valid for the declared project
version.

## Why You Would Use It

Use the durability layer when you want commit-style persistence around autosave or
interactive editing boundaries:

- each committed root payload is immutable
- the current head is selected by a superblock flip instead of in-place mutation
- recovery falls back to the last clean commit when a journal is left behind
- staged writes are separated from the committed head

This is aimed at protecting the last committed human-label state when a process
dies mid-save.

## Recommended Experimental Workflow

Normal project-facing code should not call the durability layer directly. Use the
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

Low-level durability helpers still exist for tests, recovery work, and private
workspace storage flows. They are intentionally outside the workspace-first
public contract and do not appear in `xpkg.api`, `xpkg.workspace`, or the CLI:

- `WorkspaceDurableStore.open(store_root)`
- `WorkspaceDurableStore.create_from_roots(store_root, {"snapshot": snapshot_path})`
- `WorkspaceDurableStore.current_root_path("snapshot")`
- `WorkspaceDurableStore.commit_new_roots({"snapshot": snapshot_path}, ...)`

## Recovery Model

`WorkspaceDurableStore.open(...)` calls recovery before returning a mounted store.

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

- private `.xpkg/` workspace durability machinery
- commit-oriented autosave flow
- superblock/journal durability layer
- generic committed roots for workspace-native snapshot payloads

If you are building the public project contract, think in terms of workspace +
`.expkg`. If you are prototyping crash-safe editing or autosave behavior,
the workspace durability layer is the right private layer to evaluate.
