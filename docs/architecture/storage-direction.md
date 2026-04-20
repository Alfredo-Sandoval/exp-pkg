# Storage Direction

<div class="page-intro">
<p>
xpkg is now workspace-first in both product language and the normal durable
write path. The editable contract is workspace folder + private
<code>.xpkg/</code> state + portable <code>.expkg</code> export. Direct
<code>.xpkg</code> archive handling remains, but it is now a compatibility
surface rather than the primary committed source of truth.
</p>
</div>

!!! info
    Status: current implementation notes. Today the committed source of truth is
    the durable store head under <code>.xpkg/</code>. Normal workspace commits
    store a native snapshot root, while <code>.xpkg/state/current.json</code>
    remains a rebuildable cache keyed by the durable commit id.

## Current Truth

Today xpkg has four storage ideas in play:

- workspace root as the editable project boundary
- `.xpkg/` as the private durable store boundary
- `.expkg` as the portable packed artifact
- `.xpkg` as the direct archive compatibility format

The important shift is that normal workspace save/import/migrate flows no
longer commit archive blobs first. They commit a workspace-native snapshot into
the durable store, then materialize `.xpkg/state/current.json` as the local
cache for fast reopening.

## Durable Store Contract

The durable store already had the right high-level shape:

- journaled commit boundaries
- immutable content-addressed objects
- commit metadata with generic `roots`
- a stable head commit id for stale-cache protection

The missing cutover was runtime behavior. The store now uses that generic
`roots` capability for normal workspace heads:

- workspace-native commits store `roots["snapshot"]`
- legacy/archive-backed heads may still store `roots["archive"]`
- the embedded `xpkg_commit_id` in `.xpkg/state/current.json` must match the
  durable head before the snapshot cache is trusted

That means the durable head stays authoritative while the cache stays cheap to
rebuild.

## Remaining Archive Uses

The archive engine is still valuable, but its role is narrower now.

Archive handling remains appropriate for:

- explicit `.xpkg` archive workflows
- migration/import of older archives
- fixtures and compatibility coverage
- explicit archive materialization when an archive-facing helper is called

Archive handling is no longer the normal committed workspace write path.

## Why The Archive Layer Still Exists

The archive layer still provides the broadest round-trip surface for legacy and
interop work:

- `xpkg.io.archive_format.write_archive`
- `xpkg.io.archive_format.update_labels_archive`
- `xpkg.io.archive_format.read_archive`

Those functions still matter for compatibility because they already know how to
carry labels, predictions, segmentation, metrics, metadata, and manifests in a
portable archive-shaped payload.

## What Still Remains

The cutover is materially forward, but not finished.

Remaining seams include:

1. The workspace loader still needs archive fallback for older workspaces and
   legacy store heads.
2. Explicit archive-facing helpers still require lazy archive materialization
   from the committed workspace snapshot when no archive root exists.
3. Some architectural docs and tests still assume “store head equals archive
   blob” and must be retired as the new baseline hardens.

## Recommended Position

xpkg should keep treating direct `.xpkg` archive handling as a compatibility
mechanism, not the product identity.

That means:

- keep archive support for migration, fixtures, and explicit archive APIs
- avoid routing new workspace features through archive mutation
- keep the public storage story centered on workspace + `.xpkg/` + `.expkg`
- keep shrinking archive-first assumptions at the boundaries rather than adding
  new ones

## Bottom Line

The durable store head is now workspace-native for normal workspace flows.
Direct `.xpkg` archives still matter, but they now belong to compatibility and
interop seams instead of the primary committed storage contract.
