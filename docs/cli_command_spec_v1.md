# xpkg v1 CLI Command Spec

This document defines the shipped CLI contract for the xpkg v1 project and
artifact workflow.

The current CLI is workspace-first for project creation, importing, packing,
unpacking, validation, and artifact inspection.

## Command Surface

The workspace-first command surface is:

```text
xpkg artifacts
xpkg completion
xpkg describe
xpkg import
xpkg workspace
```

## Shared Rules

- The primary editable unit is a workspace folder.
- `.expkg` is the only portable project artifact.
- xpkg never edits a `.expkg` file in place.
- Commands that create a project must produce a valid workspace containing
  `PROJECT.json` and `.xpkg/`.
- Project-internal paths are stored relative to the workspace root.
- Pack defaults to `--media full`.
- Pack must fail loudly if required media are external to `Media/`.
- Every canonical command supports `--json` for machine-readable output.
- In `--json` mode, success payloads are written to stdout as one JSON object.
- In `--json` mode, progress text is suppressed and errors are written to
  stderr under a top-level `error` object with `code`, `message`, and `hint`.
- Exit codes are `0` for success, `1` for usage or runtime errors, `2` reserved
  for future auth/config failures, and `3` for not found.
- `--json` is reserved for machine output. Commands that import JSON files use
  `--input-json` for those input paths.
- `xpkg describe --json` exposes the current command contract for agents.
- Shell completion is exposed through `xpkg completion bash`, `xpkg completion
  zsh`, and `xpkg completion fish`.

## `xpkg workspace`

Manage workspace-first project lifecycle operations.

### Commands

- `xpkg workspace describe`
- `xpkg workspace init`
- `xpkg workspace pack`
- `xpkg workspace unpack`
- `xpkg workspace validate`

## `xpkg workspace init`

Create a new empty workspace with the canonical public layout.

### Synopsis

```bash
xpkg workspace init "./My Project"
xpkg workspace init "./My Project" --title "My Project"
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

Import supported external data into a workspace.

### Synopsis

```bash
xpkg import dlc csv --csv tracking.csv --video video.mp4 --out "./My Project"
xpkg import dlc h5 --h5 tracking.h5 --video video.mp4 --out "./My Project"
xpkg import dlc project --project dlc_project --out "./My Project"
xpkg import lightning-pose --csv predictions.csv --video video.mp4 --out "./My Project"
xpkg import sleap package --slp labels.pkg.slp --out "./My Project"
xpkg import sleap h5 --h5 analysis.h5 --video video.mp4 --out "./My Project"
xpkg import vicon recording --recording trial.c3d --out "./My Project"
xpkg import vicon csv --csv trial.csv --out "./My Project"
xpkg import vicon c3d --c3d trial.c3d --out "./My Project"
xpkg import mmpose --input-json results.json --video video.mp4 --out "./My Project"
xpkg import mediapipe --input-json pose_landmarks.json --video video.mp4 --out "./My Project"
```

### Supported families today

- `xpkg import dlc csv`
- `xpkg import dlc h5`
- `xpkg import dlc project`
- `xpkg import lightning-pose`
- `xpkg import sleap package`
- `xpkg import sleap h5`
- `xpkg import vicon recording`
- `xpkg import vicon csv`
- `xpkg import vicon c3d`
- `xpkg import mmpose`
- `xpkg import mediapipe`

### Required behavior

- Imports into a workspace, never directly into a new opaque single-file native
  artifact.
- Creates the workspace if it does not already exist.
- Writes authoritative mutable state into `.xpkg/`.
- Populates `Media/` when the import produces managed media.
- Updates `PROJECT.json` metadata and timestamps.

### Non-goals

- Import is not an in-place update of packed `.expkg` artifacts.

## `xpkg artifacts`

Inspect and validate registered workspace output artifacts.

### Synopsis

```bash
xpkg artifacts list "./My Project"
xpkg artifacts list "./My Project" --kind figure
xpkg artifacts inspect "./My Project" session-summary-figure --kind figure
xpkg artifacts validate "./My Project" --kind figure
xpkg artifacts validate "./My Project" session-summary-figure --kind figure
xpkg artifacts rebuild-index "./My Project"
```

### Required behavior

- Reads artifact manifests from a workspace.
- Lists compact entries from `.xpkg/artifacts/index.json`, rebuilding the index
  if it is missing.
- Prints one full manifest for `inspect`.
- Validates referenced input, output, and stats files.
- Verifies recorded checksums and sizes when present.
- Supports optional artifact kind and caller-owned namespace filters.

### Non-goals

- `artifacts` does not render plots, compute statistics, or choose scientific
  models.
- `artifacts` does not make `.expkg` files mutable; use `xpkg workspace pack` after
  workspace changes.

## `xpkg workspace describe`

Describe the normalized workspace layout and descriptor.

### Synopsis

```bash
xpkg workspace describe "./My Project"
xpkg workspace describe "./My Project" --json
```

### Required behavior

- Resolves the owning workspace root from the supplied path.
- Returns the normalized managed paths for `PROJECT.json`, `.xpkg/`, `Media/`,
  `Exports/`, and the current state cache.
- Emits the current `PROJECT.json` descriptor in JSON mode.

## `xpkg workspace pack`

Create a portable `.expkg` artifact from a workspace.

### Synopsis

```bash
xpkg workspace pack "./My Project"
xpkg workspace pack "./My Project" --out "./release/My Project.expkg"
xpkg workspace pack "./My Project" --media package
xpkg workspace pack "./My Project" --media manifest
```

### Required behavior

- Reads a workspace folder as input.
- Validates the workspace before packing.
- Emits a `.expkg` file.
- Defaults output to `./My Project/Exports/My Project.expkg` when `--out` is
  omitted.
- Fails if required media are external to `Media/`.
- Supports `--media full`, `--media package`, and `--media manifest`.
- Defaults to `--media full`, storing every managed media file.
- `--media package` stores package-sized media such as image sequences while
  manifesting video containers without storing their bytes.
- `--media manifest` stores no media bytes and records managed media paths,
  sizes, and SHA-256 digests in `EXPKG.json`.
- Writes a root `EXPKG.json` manifest declaring member paths, member sizes,
  member SHA-256 digests, media inclusion status, and compression policy.
- Stores already-compressed media and common binary containers without
  additional zip compression.

## `xpkg workspace unpack`

Create a workspace from a `.expkg` artifact.

### Synopsis

```bash
xpkg workspace unpack "./My Project.expkg" --out "./My Project"
```

### Required behavior

- Accepts a `.expkg` file as input.
- Creates a workspace folder as output.
- Reconstructs `PROJECT.json` and `.xpkg/`.
- Restores included `Media/` files from the artifact.
- Creates an empty `Media/` root when media were manifested but not stored.
- Excludes temp files, locks, caches, and machine-local scratch state.
- Refuses to unpack into a conflicting non-empty directory unless an explicit
  future overwrite flag is added.

## `xpkg workspace validate`

Validate a workspace or packed `.expkg` artifact.

### Synopsis

```bash
xpkg workspace validate "./My Project"
xpkg workspace validate "./My Project.expkg"
```

### Required behavior

- Accepts a workspace folder or `.expkg` file.
- Fails loudly when the supplied path does not satisfy the corresponding
  contract.
- For `.expkg`, verifies `EXPKG.json`, member path safety, duplicate member
  absence, member sizes, member SHA-256 digests, and media/member consistency.
- Leaves the validated artifact unchanged.

## Open Behavior

GUI and shell tooling should treat the workspace folder as the primary open
target.

- Opening a workspace opens it directly.
- Opening a `.expkg` prompts for an unpack destination, then opens the
  resulting workspace.
- Packed artifacts remain immutable from the user’s perspective.

## Transition Guidance

- New project creation follows the workspace + `.expkg` contract.
- No public command should frame direct `.xpkg` conversion as the native or
  preferred project workflow.
