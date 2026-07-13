# xpkg Artifact Contract

This document defines the versioned on-disk artifact contract for xpkg.

The scope of this freeze is the on-disk durability contract only: the `.expkg`
zip container, its root `EXPKG.json` manifest, and the three-class layout
(editable project folder / private `.xpkg/` store / portable `.expkg` export).
Current writes use artifact schema version 2. Readers reject unsupported schema
versions instead of guessing or silently applying compatibility conventions.

The Python API and CLI command surface are a separate, pre-1.0 concern and may
still change before 1.0. The package itself is at 0.x (alpha). This document
freezes the artifact format, not the code surface.

The contract is project-first because xpkg is aimed at whole experiments with
one or many recording sessions. The public artifact holds more context than a single archive
blob, so the contract is defined around a project, a private store, and a
portable export.

It supersedes the older public framing that treated a direct HDF5 archive file
as the native single-file project artifact. `.xpkg/` is now only the private
project store directory. It is not a portable user-facing project contract.

## Artifact Classes

There are exactly three artifact classes.

### A. Project

The canonical editable project is a normal folder.

Example:

```text
My Project/
```

This is what users open, click into, back up, and work from.

### B. Internal Store

Inside the project, xpkg keeps a hidden implementation-owned store:

```text
My Project/.xpkg/
```

This is the authoritative mutable state. It is opaque to users.

### C. Portable Project Artifact

The move/share/export artifact is a single file:

```text
My Project.expkg
```

This is the only portable file type. It is not `.h5`. It is not `.zip`. It is
not `.archive`.

## Canonical Project Layout

This is the public, stable layout:

```text
My Project/
  PROJECT.json
  .xpkg/
  Media/
  Exports/
```

Rules:

- `PROJECT.json` is required.
- `.xpkg/` is required.
- `Media/` is optional but standard.
- `Exports/` is optional but standard.
- No xpkg-required symlinks exist anywhere in the contract.

## File and Folder Semantics

### `PROJECT.json`

This is the project descriptor and locator. It is not the annotation truth
store.

Required fields:

```json
{
  "format": "xpkg-project",
  "project_schema_version": 1,
  "layout_version": 1,
  "title": "My Project",
  "project_id": "uuid-or-ulid",
  "created_at": "2026-03-15T00:00:00Z",
  "updated_at": "2026-03-15T00:00:00Z",
  "store_path": ".xpkg",
  "media_root": "Media",
  "exports_root": "Exports"
}
```

Notes:

- `PROJECT.json` contains metadata and pointers only.
- It must never become a shadow copy of the full project state.
- Transactional state lives in `.xpkg/`.
- The machine-readable schema for this file lives at
  `schemas/project.schema.json`.

### `.xpkg/`

This is xpkg-owned internal state. The public guarantee is only that it
exists and is valid for the declared version.

Its internal sublayout is not part of the public long-term contract. That
preserves room to evolve journals, commits, objects, caches, and indexes
without freezing every internal subdirectory forever.

Current xpkg projects also use `.xpkg/artifacts/` for registered scientific
outputs such as figures, tables, analyses, reports, stats reports, and source
data. These artifact entries are still private project state, but their
manifests are portable JSON records that connect outputs back to inputs,
producer commands, stats reports, source data, and checksum-bearing file
records. The compact discovery index lives at `.xpkg/artifacts/index.json` and
can be rebuilt from individual manifests.

Domain packages may also keep derived outputs under caller-owned namespaces
such as `.xpkg/neuro-analysis/figures/` or
`.xpkg/neuro-analysis/tables/`. The same artifact manifest contract applies; the
namespace only controls where the files live inside the project. `xpkg` does
not reserve or hard-code downstream package names.

### `Media/`

This is the standard location for managed media. Any media copied into the
project for portability lives here.

### `Exports/`

This is the standard location for emitted `.expkg` files and other project
exports.

## Portable Artifact Semantics

`*.expkg` is a packed project artifact.

Rules:

- It represents a committed, validated project.
- Unpacking recreates a project layout with `PROJECT.json`, `.xpkg/`,
  `Media/`, and `Exports/`.
- It is a project artifact, not a raw storage engine.
- Users and third parties must treat it as opaque.
- v1 uses a zip container with a root `EXPKG.json` export manifest.
- Already-compressed media and common binary containers are stored without
  additional zip compression to avoid slow double-compression.

On unpack, xpkg reconstructs:

```text
<Project Name>/
  PROJECT.json
  .xpkg/
  Media/        # populated only when media was included
  Exports/
```

For portable artifacts the result must be logically identical, not
byte-identical. Locks, caches, temporary files, and machine-local scratch state
are excluded. Artifacts packed with omitted media preserve project state and
media references, but those unpacked projects are not guaranteed to load
media-backed labels on another machine until the matching media are restored.

## Open, Pack, Unpack, Import

xpkg should accept a project folder as the primary open target:

```text
My Project/
```

GUI behavior for `.expkg`:

- `File > Open` on a `.expkg` triggers unpack/import into a chosen folder,
  then opens that project.
- xpkg does not edit a packed file in place.

### Pack

`project pack` creates a packed project artifact from a project.

Example:

```bash
xpkg project pack "My Project"
# emits My Project/Exports/My Project.expkg
```

Media scope is controlled by `--media`:

```bash
xpkg project pack "My Project" --media full
xpkg project pack "My Project" --media package
xpkg project pack "My Project" --media manifest
```

### Unpack

`project unpack` creates a project from a portable artifact.

Example:

```bash
xpkg project unpack "My Project.expkg" --out "./My Project"
```

### Import

Foreign formats import into a project, not directly into an opaque legacy
HDF5 archive.

Examples:

```bash
xpkg import pose dlc-csv --path tracking.csv --video video.mp4 --out "./My Project"
xpkg import pose sleap-package --path labels.pkg.slp --out "./My Project"
xpkg import signals photometry-csv --path photometry.csv --out "./My Project"
```

The private state root contains versioned `xpkg.experiment` state. It stores
typed subjects, protocols, conditions, recording sessions, signals, videos,
pose, events, timebases, metadata, and provenance. Labels are nested pose data
inside a session, not a competing project-state kind.

The pre-1.0 command surface is documented in `docs/cli_command_spec_v1.md`.

## Media Policy

`.expkg` is the only packed project artifact. There is no separate state
artifact type and no alternate file extension.

Rules:

- All required media must be inside `Media/` before pack.
- If required media are external to `Media/`, pack fails loudly.
- No silent omission is allowed. Omitted media must be declared in `EXPKG.json`.
- `--media full` stores every managed media file in the `.expkg`.
- `--media package` stores package-sized media such as image sequences while
  manifesting video containers without storing their bytes.
- `--media manifest` stores no media bytes. It records managed media paths,
  sizes, and SHA-256 digests in `EXPKG.json`.
- `EXPKG.json` records the media mode, per-file inclusion status, sizes,
  SHA-256 digests, and compression policy.

Cloud or dataset-backed media workflows should be represented by future cloud
sync/import/export layers on top of this manifest, not by adding absolute paths
or hidden side channels to the `.expkg` contract.

## Path Rules

- All project-internal paths are stored relative to the project root.
- No xpkg-required absolute paths exist in the portable contract.
- No required symlink semantics exist.
- Symlinks may exist in user content, but official project validity must not
  depend on them.

## Legacy HDF5 Boundary Policy

`.xpkg/` is the private project store directory name, not a public project
artifact and not a single HDF5 file.

Policy:

- New projects are created as project folders.
- New portable exports are `.expkg`.
- No dedicated project-to-`.xpkg` export command is part of the v1 artifact contract.
- No new core features should depend on direct HDF5 archive handling as a
  project contract.

## What Is Public vs Private

### Public and Stable

- project root is a normal folder
- `PROJECT.json`
- hidden store path `.xpkg/`
- standard `Media/` and `Exports/`
- portable file suffix `.expkg`
- pack/unpack/import/open semantics
- media portability rules

### Private and Versioned

- internal `.xpkg/` subdirectory layout
- exact storage engine
- exact compression/container used inside `.expkg`
- cache structure
- lock implementation

This split is deliberate. It gives xpkg a stable user contract without
freezing internal storage machinery too early.

## Final Locked Decision

Use this:

```text
My Project/
  PROJECT.json
  .xpkg/
  Media/
  Exports/
    My Project.expkg
```

And this:

- editable project = project folder
- authoritative mutable state = `.xpkg/`
- portable artifact = `.expkg`
- legacy HDF5 archive compatibility = removed from the v1 artifact contract
- no required symlink layer
- no `.h5` public contract
