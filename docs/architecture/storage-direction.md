# Storage Direction

<div class="page-intro">
<p>
xpkg is now workspace-first in both product language and the normal durable
write path. The editable contract is workspace folder + private
<code>.xpkg/</code> state + portable <code>.expkg</code> export.
</p>
</div>

!!! info
    Status: current implementation notes. Today the committed source of truth is
    the durable store head under <code>.xpkg/</code>. Normal workspace commits
    store a native snapshot root, while <code>.xpkg/state/current.json</code>
    remains a rebuildable cache keyed by the durable commit id.

## Current Truth

Today xpkg has three storage ideas in play:

- workspace root as the editable project boundary
- `.xpkg/` as the private durable store boundary
- `.expkg` as the portable packed artifact

Normal workspace save and import flows do not commit archive blobs first. They
commit a workspace-native snapshot into the durable store, then materialize
`.xpkg/state/current.json` as the local cache for fast reopening.

## Durable Store Contract

The durable store already had the right high-level shape:

- journaled commit boundaries
- immutable content-addressed objects
- commit metadata with generic `roots`
- a stable head commit id for stale-cache protection

The missing cutover was runtime behavior. The store now uses that generic
`roots` capability for normal workspace heads:

- workspace-native commits store `roots["snapshot"]`
- commit roots now hydrate through typed `RootEntry` values instead of raw
  root dictionaries
- the embedded `xpkg_commit_id` in `.xpkg/state/current.json` must match the
  durable head before the snapshot cache is trusted

That means the durable head stays authoritative while the cache stays cheap to
rebuild.

Workspace load/pack/validate flows now also reject a tampered
`.xpkg/state/current.json` cache even when its embedded `xpkg_commit_id`
matches the head. If the cache diverges from the committed snapshot payload,
xpkg rebuilds it from the durable store before continuing.

## Public Cutover Status

The public cleanup now matches that storage model:

- `WorkspaceService` and `workspace.imports.*` are the primary downstream path
- `xpkg.formats` keeps workspace lifecycle/import helpers
- package-level `xpkg.adapters` and the CLI `xpkg convert` surface were removed
- compatibility alias maps such as `current_project_archive_path(...)` were
  removed from the public facades

## Remaining Archive Uses

The archive engine is still valuable, but its role is narrower now.

Archive handling remains appropriate for:

- temporary internal conversion artifacts while workspace importers are being
  rewritten around native snapshots
- fixtures and compatibility coverage inside this repo

Archive handling is no longer the normal committed workspace write path or a
first-class public downstream surface.

## Why The Archive Layer Still Exists

The archive layer still provides the broadest round-trip surface for internal
conversion work:

- `xpkg.io.archive_format.write_archive`
- `xpkg.io.archive_format.update_labels_archive`
- `xpkg.io.archive_format.read_archive`

Those functions still matter internally because they already know how to carry
labels, predictions, segmentation, metrics, metadata, and manifests in a
portable archive-shaped payload.

## Recommended Position

xpkg should keep treating direct `.xpkg` archive handling as internal
conversion machinery, not the product identity.

That means:

- keep archive support for fixtures and explicit internal archive work
- avoid routing new workspace features through archive mutation
- keep the public storage story centered on workspace + `.xpkg/` + `.expkg`
- keep shrinking archive-first assumptions at the boundaries rather than adding
  new ones

## Bottom Line

The durable store head is workspace-native for normal workspace flows. Direct
archive handling is no longer a public migration path or a primary committed
storage contract.
