# xpkg v1 CLI Command Spec

This document defines the locked public CLI contract for the xpkg v1 artifact
model.

It describes the intended public commands for workspace-first project handling.
Current code may still expose legacy `.siesta` conversion helpers during the
transition. Those are compatibility interfaces, not the long-term public
artifact workflow.

## Command Surface

The v1 public command surface is:

```text
xpkg init
xpkg import
xpkg pack
xpkg unpack
xpkg migrate
```

## Shared Rules

- The primary editable unit is a workspace folder.
- `.expkg` is the only portable project artifact.
- xpkg never edits a `.expkg` file in place.
- Commands that create a project must produce a valid workspace containing
  `PROJECT.json` and `.xpkg/`.
- Project-internal paths are stored relative to the workspace root.
- Portable mode defaults to `portable`.
- Pack must fail loudly if required media are missing from a portable export.

## `xpkg init`

Create a new empty workspace with the canonical public layout.

### Synopsis

```bash
xpkg init "./My Project"
xpkg init "./My Project" --title "My Project"
xpkg init "./My Project" --pack-mode portable
```

### Required behavior

- Creates the workspace root if needed.
- Creates `PROJECT.json`.
- Creates `.xpkg/`.
- Creates `Media/` and `Exports/` when bootstrapping a new workspace.
- Initializes `project_id`, timestamps, and default descriptor fields.
- Refuses to overwrite a non-empty target unless an explicit future overwrite
  flag is added.

### Output

- A valid workspace folder at the requested output path.
- `PROJECT.json` conforming to `schemas/project.schema.json`.

## `xpkg import`

Import foreign or legacy data into a workspace.

### Synopsis

```bash
xpkg import dlc csv --csv tracking.csv --video video.mp4 --out "./My Project"
xpkg import dlc h5 --h5 tracking.h5 --video video.mp4 --out "./My Project"
xpkg import sleap --slp labels.pkg.slp --out "./My Project"
xpkg import legacy --file tracking.siesta --out "./My Project"
```

### Required behavior

- Imports into a workspace, never directly into a new opaque single-file native
  artifact.
- Creates the workspace if it does not already exist.
- Writes authoritative mutable state into `.xpkg/`.
- Populates `Media/` when the import produces managed media.
- Updates `PROJECT.json` metadata and timestamps.
- Accepts legacy `.siesta` input through the `legacy` importer.

### Non-goals

- Import is not an in-place update of packed `.expkg` artifacts.
- Import is not a version-to-version migration of an existing xpkg workspace.

## `xpkg pack`

Create a portable `.expkg` artifact from a workspace.

### Synopsis

```bash
xpkg pack "./My Project"
xpkg pack "./My Project" --out "./release/My Project.expkg"
xpkg pack "./My Project" --mode snapshot
```

### Required behavior

- Reads a workspace folder as input.
- Validates the workspace before packing.
- Emits a `.expkg` file.
- Defaults output to `./My Project/Exports/My Project.expkg` when `--out` is
  omitted.
- Uses `portable` mode by default.
- Fails if required media are external and unavailable for a portable pack.
- Declares the chosen mode in the packed artifact metadata.

### Mode semantics

- `portable`: includes or internalizes required media so the artifact can move
  across machines.
- `snapshot`: allows external media references and is not guaranteed to fully
  open elsewhere.

## `xpkg unpack`

Create a workspace from a `.expkg` artifact.

### Synopsis

```bash
xpkg unpack "./My Project.expkg" --out "./My Project"
```

### Required behavior

- Accepts a `.expkg` file as input.
- Creates a valid workspace folder as output.
- Reconstructs `PROJECT.json` and `.xpkg/`.
- Restores `Media/` when it is included in the artifact.
- Excludes temp files, locks, caches, and machine-local scratch state.
- Refuses to unpack into a conflicting non-empty directory unless an explicit
  future overwrite flag is added.

## `xpkg migrate`

Upgrade an existing xpkg artifact to the latest supported public contract
without changing the project’s logical contents.

### Synopsis

```bash
xpkg migrate "./My Project"
xpkg migrate "./My Project.expkg" --out "./My Project"
```

### Required behavior

- Operates on xpkg-owned artifacts only.
- Upgrades workspace descriptor/layout versions when needed.
- When the input is `.expkg`, unpacks first and then migrates into the output
  workspace.
- Preserves logical project contents across migration.
- Leaves foreign-format ingestion to `xpkg import`, not `xpkg migrate`.

### Non-goals

- `migrate` is not the primary entrypoint for importing DLC, SLEAP, or other
  third-party formats.
- `migrate` does not define or freeze the private internal `.xpkg/`
  sublayout. It upgrades it as needed behind the public contract.

## Open Behavior

GUI and shell tooling should treat the workspace folder as the primary open
target.

- Opening a workspace opens it directly.
- Opening a `.expkg` prompts for an unpack destination, then opens the
  resulting workspace.
- Packed artifacts remain immutable from the user’s perspective.

## Transition Guidance

- New project creation follows the workspace + `.expkg` contract.
- `.siesta` remains supported only for legacy import/read and optional explicit
  export during transition.
- No new public examples should frame `.siesta` as the native user-facing
  project artifact.
