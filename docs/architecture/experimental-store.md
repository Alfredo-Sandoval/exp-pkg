# Project Durability

<div class="page-intro">
<p>
xpkg now has an <strong>experimental</strong> project durability layer for
crash-safe, commit-oriented project state. In the locked v1 artifact contract
this belongs under the project-owned <code>.xpkg/</code> directory. Normal
project commits now store project-native snapshot roots in that private
store.
</p>
</div>

If you want the broader rationale for the storage cutover, read
[Storage Direction](storage-direction.md).

!!! warning
This workflow is experimental private machinery. The public v1 artifact
contract is project folder + <code>.expkg</code>. Use the durability layer
when you want stronger recovery semantics inside <code>.xpkg/</code>, not
as a public interchange layer.

!!! info
Status: current private prototype. The committed source of truth is the
project durability head; <code>.xpkg/state/current.json</code> is only a
cache.

## What Changed

The public contract is a project:

```text
My Project/
  PROJECT.json
  .xpkg/
  Media/
  Exports/
    My Project.expkg
```

Inside that project, the current private store manages committed root
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
project APIs; they commit snapshot roots into `.xpkg/` and refresh
`.xpkg/state/current.json` as a rebuildable cache.

Example:

```python
from xpkg.model import Labels
from xpkg.services import ProjectService

labels = Labels()

project = ProjectService.create("My Project", title="My Project")
project.save_labels(labels, metadata={"source": "manual"})
project.validate()
```

## Low-Level Entry Points

Low-level durability helpers still exist for tests, recovery work, and private
project storage flows. They are intentionally outside the project-first
public contract and do not appear in `xpkg.api`, `xpkg.project`, or the CLI:

- `ProjectDurableStore.open(store_root)`
- `ProjectDurableStore.create_from_roots(store_root, {"snapshot": snapshot_path})`
- `ProjectDurableStore.current_root_path("snapshot")`
- `ProjectDurableStore.commit_new_roots({"snapshot": snapshot_path}, ...)`

## Recovery Model

`ProjectDurableStore.open(...)` calls recovery before returning a mounted store.

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

- public project contract: `PROJECT.json`, `.xpkg/`, `Media/`, `Exports/`
- `.expkg` as the portable project artifact

Experimental today:

- private `.xpkg/` project durability machinery
- commit-oriented autosave flow
- superblock/journal durability layer
- generic committed roots for project-native snapshot payloads

If you are building the public project contract, think in terms of project +
`.expkg`. If you are prototyping crash-safe editing or autosave behavior,
the project durability layer is the right private layer to evaluate.
