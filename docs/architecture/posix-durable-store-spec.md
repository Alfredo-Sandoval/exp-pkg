# POSIX Durable Store Spec

This document scopes the `xpkg` durable store to macOS and Linux only.

The goal is to make committed workspace saves crash-safe on local POSIX-like
filesystems without turning `xpkg` into a large storage-engine project.

## Status

This is a target architecture spec for the private `.xpkg/` runtime store.

- Supported operating systems: macOS, Linux
- Explicitly out of scope: Windows
- Source of truth for committed state: durable store head
- Fast local cache: `.xpkg/state/current.json`

## Problem

The main failure we want to prevent is loss of the last committed human label
state after:

- process crashes
- forced quits
- power loss
- partial writes during save

The only acceptable loss is uncommitted work since the last successful commit.

## Design Goal

Committed saves must never rely on in-place mutation of the current archive or
current state file. A save produces a new immutable committed payload and then
atomically flips a root pointer to make that payload current.

This gives us a narrow and auditable contract:

- old committed state remains valid until the final pointer flip
- new committed state becomes visible only after the pointer flip
- recovery prefers the last known-clean commit over guessing

## Platform Contract

This spec assumes local POSIX-style filesystem semantics.

Supported:

- APFS on macOS
- ext4 and similar local Linux filesystems

Not guaranteed:

- NFS
- SMB
- cloud-sync folders such as Dropbox or iCloud Drive
- FUSE mounts
- any storage layer with weak rename or sync semantics

## Workspace Layout

The durable store lives under the existing workspace-owned `.xpkg/` directory.
This matches the current workspace layout in `src/xpkg/io/project_layout.py`.

```text
<workspace>/
  PROJECT.json
  .xpkg/
    superblock.a.json
    superblock.b.json
    LOCK
    journal/
      active.json
    commits/
      000000000001/
        commit.json
      000000000002/
        commit.json
    objects/
      ab/cd/obj_<sha256>.xpkg
    workspace/
      tmp-<txn>.xpkg
    state/
      current.json
  Media/
  Exports/
```

## Authoritative Data Model

### Durable truth

These files define committed state:

- `superblock.a.json`
- `superblock.b.json`
- `commits/*/commit.json`
- `objects/**/obj_<sha256>.xpkg`

### Rebuildable cache

`state/current.json` is not authoritative. It is a fast local projection of the
current committed head and must always be rebuildable from store metadata plus
the committed archive object.

If `current.json` is missing, stale, or invalid, `xpkg` must recover from the
committed head and regenerate it.

## Core Invariants

1. The selected superblock is the only authoritative pointer to the current
   committed state.
2. Objects are immutable and content-addressed.
3. Commit metadata is immutable.
4. `state/current.json` may lag the committed head, but it must never get ahead
   of it.
5. No commit mutates an existing committed `.xpkg` object in place.

## Storage Primitives

The implementation should rely on the smallest set of POSIX-friendly
primitives:

- temp file write in the target directory
- `flush()` plus `fsync()` on the temp file
- `os.replace()` for atomic path replacement
- directory `fsync()` on the containing directory
- advisory single-writer lock on `.xpkg/LOCK`

For macOS and Linux, prefer `fcntl.flock()` or `fcntl.lockf()` for the store
lock instead of a hard-link lock. A POSIX-only target lets us use the simpler
and more native locking model.

## Commit State Machine

The journal state machine should stay intentionally narrow:

- `staging`
- `committing`
- `cleanup`

No extra states should be added unless they provide a concrete recovery
distinction that changes behavior.

## Commit Write Path

All writes that participate in a commit happen while holding the exclusive
store lock.

### Step 1: prepare staged payload

- Normalize and copy media into the workspace if needed.
- Materialize a fresh staged `.xpkg` archive under `.xpkg/workspace/`.
- Flush and `fsync()` the staged archive before it is referenced by anything
  else.

### Step 2: create journal

Write `journal/active.json` with at least:

- `txn_id`
- `state="staging"`
- `base_commit_id`
- `target_generation`
- checksum

Then atomically persist it.

### Step 3: install immutable object

- Hash the staged archive.
- Move or copy it into `objects/` using the content-addressed object path.
- `fsync()` the object file.
- `fsync()` the object parent directory.

If the object already exists for the same digest, reuse it.

### Step 4: write commit metadata

Create `commits/<generation>/commit.json` with:

- commit id
- generation
- parent commit id
- creation timestamp
- reason
- created_by payload
- root archive object id and extension
- checksum

Then atomically persist it and `fsync()` its parent directory.

### Step 5: flip superblock

Create a new superblock pointing to the new commit and write it to the inactive
slot:

- if `a` is active, write `b`
- if `b` is active, write `a`

Then:

- `fsync()` the superblock file
- `fsync()` the `.xpkg/` directory

This is the only step that changes what is considered current.

### Step 6: clear journal

After the new superblock is durable, clear `journal/active.json`.

### Step 7: refresh cache

Rewrite `.xpkg/state/current.json` as a cache of the committed head.

Important ordering rule:

- update the superblock first
- update `current.json` second

That guarantees the cache may be stale after a crash, but never falsely ahead
of the committed head.

## Recovery Algorithm

Recovery runs on store open before normal load proceeds.

### Superblock selection

1. Read both superblocks.
2. Reject any superblock with invalid checksum or unsupported version.
3. If only one is valid, use it.
4. If both are valid, choose the higher generation.
5. If generations tie, choose the one with the later `updated_at`.

### Journal handling

If no journal exists, mount the selected head.

If a journal exists:

- if recovery cannot prove that the flipped head completed cleanly, revert to
  `last_clean_commit_id`
- if the referenced commit file and referenced object both exist and validate,
  accept the flipped head and clear the journal

Recovery must prefer a known-clean head over any attempt to infer partially
finished state.

### Cache handling

After selecting the committed head:

- if `state/current.json` is missing, regenerate it
- if `state/current.json` references a different commit id than the selected
  head, regenerate it
- if parsing or validation fails, regenerate it

## Public Behavior

The workspace-first API should continue to feel native:

- users load a workspace directory
- saves update workspace state
- `current.json` remains the fast path for normal local loads

Internally, the durable head becomes the source of truth for committed state.

That means the main save path in `src/xpkg/io/project_workspace.py` should
change from:

- "write current snapshot JSON directly"

to:

- "produce fresh `.xpkg` payload"
- "commit into durable store"
- "refresh `state/current.json` cache"

## Explicit Non-Goals For V1

The following are out of scope for the first POSIX-only implementation:

- unsaved-edit oplog replay
- multi-session merge or collaboration semantics
- background history browser
- object garbage collection and pruning policy
- network filesystem correctness guarantees
- Windows support

## Testing Requirements

The minimum confidence bar is:

- unit tests for superblock selection
- unit tests for checksum validation
- recovery tests for crash after object write, commit write, and superblock flip
- stale or corrupt `current.json` rebuild tests
- lock contention tests with two writer processes

Manual verification should cover:

- macOS on APFS
- Linux on ext4

## Recommended Implementation Order

1. Keep the current `ArchiveStore` metadata model and recovery shape.
2. Make the committed head authoritative for workspace saves.
3. Downgrade `state/current.json` from source of truth to rebuildable cache.
4. Keep `.xpkg` as the committed immutable payload for now.
5. Leave oplog and richer autosave recovery for a later phase.

## Bottom Line

This spec is worth doing because it targets the highest-value reliability
property for annotation work:

"a committed save should survive crashes."

It is intentionally narrow. The goal is not to build a general storage engine.
The goal is to make committed saves on macOS and Linux dependable.
