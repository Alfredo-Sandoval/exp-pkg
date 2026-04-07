# Storage Direction

<div class="page-intro">
<p>
If <code>.siesta</code> feels wrong in the current Posetta story, that reaction is
reasonable. The public product story is now workspace folder + private
<code>.posetta/</code> state + portable <code>.expkg</code> export, but the
implementation still relies on <code>.siesta</code> as the only complete archive
engine behind saves, migration, and durable commits.
</p>
</div>

## Current Truth

Today Posetta still has four storage ideas in play, but the live workspace path
has moved forward:

- workspace root as the editable project boundary
- `.posetta/` as the private mutable store boundary
- `.expkg` as the portable packed artifact
- `.siesta` as the internal compatibility archive that still carries the full
  round-trip payload

The normal workspace save/load/import/migrate flow now uses a native snapshot at
`.posetta/state/current.json` as the source of truth. Archive reads still
remain in the codebase for older workspaces, migration, fixtures, and explicit
bundle-facing workflows.

That split explains the current tension. The public contract is workspace-first,
but the code still treats a `.siesta` archive as the canonical thing it knows
how to stage, validate, and commit safely.

## Why `.siesta` Is Still Here

### 1. It is still the complete storage engine

The current round-trip serializer lives in the legacy archive layer:

- `posetta.io.siesta_format.write_siesta`
- `posetta.io.siesta_format.update_labels_siesta`
- `posetta.io.siesta_format.read_siesta`

Those functions already know how to carry labels, predictions, segmentation,
metrics, metadata, and manifest information together. The workspace layer does
not yet have an independent storage backend with the same coverage.

### 2. Workspace saves still stage archives

The workspace code is already public-facing, but its save path still runs
through staged `.siesta` files.

Archive dependency is now concentrated in compatibility and migration seams:

- `migrate_legacy_archive(...)`
- `import_dlc_csv_workspace(...)`
- `import_dlc_h5_workspace(...)`
- `import_sleap_package_workspace(...)`
- archive fallback when opening an older workspace that does not yet have a
  native snapshot

### 3. The durable store commits immutable archives

The new private store is not fake. It has real recovery semantics, journaled
commit boundaries, and immutable objects under `.posetta/`.

But the object it currently commits is still an archive file:

- `SiestaStore.create_from_archive(...)`
- `SiestaStore.current_archive_path()`
- `SiestaStore.commit_new_archive(...)`

So the store is newer than the archive format, but it still wraps the archive
format instead of replacing it.

### 4. Migration and compatibility still matter

Existing adapters, tests, fixtures, and migration flows still move through
`.siesta`. Keeping it available has practical value while the workspace-first
surface hardens.

That part is not irrational. The problem is not that `.siesta` exists at all.
The problem is that it still occupies too much conceptual space compared with
the current product direction.

## Why It Feels Wrong

The discomfort is structural, not cosmetic.

- The public story says workspace and experiment packaging, but the save engine
  still speaks archive.
- The naming is still inherited from an older era and sounds narrower than the
  current mission.
- The workspace API is thin and clean, but its implementation still depends on
  a lower-level compatibility format for the actual committed payload.
- The public contract has moved faster than the underlying storage cutover.

So when someone asks, "why do we still have `.siesta`?", the honest answer is:
because it is still the only fully implemented storage backend, even though it
is no longer the product we want to talk about.

## Recommended Position

Posetta should treat `.siesta` as a transition mechanism, not the product
identity.

That means:

- keep `.siesta` available for migration, fixtures, import, and low-level
  compatibility work
- stop expanding the public product story around `.siesta`
- keep `.siesta` out of the primary artifact contract
- be explicit that the workspace/store layer is the future-facing boundary

In other words: `.siesta` can stay for a while, but it should feel private and
transitional rather than native and aspirational.

## Cutover Paths

### Option A: Keep `.siesta` as a private internal payload for now

This is the least disruptive path.

- keep the current save path working
- reduce user-facing emphasis on `.siesta`
- make the workspace/store layer the primary public contract
- avoid designing new features directly against `.siesta`

This is the best short-term option if stability matters more than immediate
storage surgery.

### Option B: Rename the internal archive concept without changing semantics

This would keep the same basic payload shape but stop exposing the old name as
the dominant concept.

That helps messaging, but it does not actually reduce architectural coupling.
It is mostly a vocabulary cleanup.

### Option C: Replace archive-backed commits with workspace-native state

This is the real cutover.

- workspace save stops calling `write_siesta(...)` directly
- the store commits normalized workspace state instead of a single archive file
- labels, segmentation, predictions, and metadata get a backend-neutral storage
  contract
- the archive layer becomes export/import compatibility instead of the canonical
  save engine

This is the most aligned end-state, but it is a real refactor, not a wording
change.

## What Needs To Happen Before `.siesta` Can Shrink

The current code suggests a practical sequence.

1. Split `posetta.io.project_workspace` into smaller layers.
   Right now descriptor logic, media management, imports, migration, save
   staging, packing, and validation all live in one module.
2. Separate store protocol from archive semantics.
   `SiestaStore` currently has real durability logic, but its roots still point
   at immutable archive files.
3. Extract a backend-neutral save model.
   The workspace layer needs a storage contract for labels, segmentation,
   predictions, and metadata that does not require writing a `.siesta` archive
   first.
4. Broaden the workspace service beyond labels-only save/load semantics.
   The current service still exposes `load_labels()` and `save_labels(...)` as
   its main mutable operations.

Until those changes happen, `.siesta` will keep showing up because the runtime
still depends on it for the actual durable payload.

## Bottom Line

`.siesta` is still here because it remains the only complete round-trip archive
engine in the implementation.

That is useful in the short term, but it also explains why the code still feels
partly anchored to an older storage model. The right posture is to keep
`.siesta` as a compatibility substrate while moving product language, public
contracts, and future storage work toward the workspace/store boundary.
