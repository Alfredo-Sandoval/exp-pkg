# xpkg v1 CLI Command Spec

This document defines the shipped CLI contract for the xpkg v1 project and
artifact workflow.

The current CLI is workspace-first for project creation, packing, unpacking,
and validation, while still exposing transition helpers for direct `.xpkg`
archives.

## Command Surface

The primary workspace-first command surface is:

```text
xpkg init
xpkg import
xpkg pack
xpkg unpack
xpkg validate
xpkg migrate
```

The CLI also ships one legacy compatibility helper during the transition:

```text
xpkg convert
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
xpkg import dlc project --project dlc_project --out "./My Project"
xpkg import sleap --slp labels.pkg.slp --out "./My Project"
xpkg import sleap --h5 analysis.h5 --video video.mp4 --out "./My Project"
xpkg import mmpose --json results.json --video video.mp4 --out "./My Project"
xpkg import mediapipe --json pose_landmarks.json --video video.mp4 --out "./My Project"
xpkg import openpose --json openpose_json --video video.mp4 --out "./My Project"
xpkg import detectron2 --predictions coco_instances_results.json --dataset-json dataset.json --image-root images --out "./My Project"
xpkg import legacy --file tracking.xpkg --out "./My Project"
```

### Supported families today

- `xpkg import dlc csv`
- `xpkg import dlc h5`
- `xpkg import dlc project`
- `xpkg import sleap --slp`
- `xpkg import sleap --h5`
- `xpkg import mmpose`
- `xpkg import mediapipe`
- `xpkg import openpose`
- `xpkg import detectron2`
- `xpkg import legacy`

### Required behavior

- Imports into a workspace, never directly into a new opaque single-file native
  artifact.
- Creates the workspace if it does not already exist.
- Writes authoritative mutable state into `.xpkg/`.
- Populates `Media/` when the import produces managed media.
- Updates `PROJECT.json` metadata and timestamps.
- Accepts canonical `.xpkg` input through the `legacy` importer.

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

## `xpkg validate`

Validate a workspace, packed `.expkg` artifact, or direct `.xpkg` archive.

### Synopsis

```bash
xpkg validate "./My Project"
xpkg validate "./My Project.expkg"
xpkg validate "./tracking.xpkg"
```

### Required behavior

- Accepts a workspace folder, `.expkg` file, or direct `.xpkg` archive.
- Fails loudly when the supplied path does not satisfy the corresponding
  contract.
- Leaves the validated artifact unchanged.

## `xpkg migrate`

Migrate a `.xpkg` archive into a workspace-first xpkg project.

### Synopsis

```bash
xpkg migrate "./tracking.xpkg" --out "./My Project"
```

### Required behavior

- Accepts a canonical `.xpkg` archive as input.
- Creates or updates a workspace at the requested output path.
- Preserves the logical project contents while rewriting them into the
  workspace-first xpkg layout.
- Leaves DLC, SLEAP, and other third-party ingestion to `xpkg import`.

### Non-goals

- `migrate` is not currently a general xpkg-to-xpkg upgrade command.
- `migrate` does not define or freeze the private internal `.xpkg/`
  sublayout.

## `xpkg convert`

Convert external tracking formats directly into edge `.xpkg` archives.

This command remains available for compatibility pipelines, fixtures, and
archive-first workflows that have not moved to workspace import yet. It is not
the recommended project-facing entrypoint for new integrations.

### Synopsis

```bash
xpkg convert dlc csv --csv tracking.csv --video video.mp4 --out tracking.xpkg
xpkg convert dlc h5 --h5 tracking.h5 --video video.mp4 --out tracking.xpkg
xpkg convert dlc project --project dlc_project --out exports
xpkg convert sleap --slp labels.pkg.slp --out sleap_project --fps 30 --no-videos
xpkg convert mmpose --json results.json --video video.mp4 --out tracking.xpkg
xpkg convert mediapipe --json pose_landmarks.json --video video.mp4 --out tracking.xpkg
xpkg convert openpose --json openpose_json --video video.mp4 --out tracking.xpkg
xpkg convert detectron2 --predictions coco_instances_results.json --dataset-json dataset.json --image-root images --out tracking.xpkg
```

### Supported families today

- `xpkg convert dlc csv`
- `xpkg convert dlc h5`
- `xpkg convert dlc project`
- `xpkg convert sleap`
- `xpkg convert mmpose`
- `xpkg convert mediapipe`
- `xpkg convert openpose`
- `xpkg convert detectron2`

## Open Behavior

GUI and shell tooling should treat the workspace folder as the primary open
target.

- Opening a workspace opens it directly.
- Opening a `.expkg` prompts for an unpack destination, then opens the
  resulting workspace.
- Packed artifacts remain immutable from the user’s perspective.

## Transition Guidance

- New project creation follows the workspace + `.expkg` contract.
- `.xpkg` is the canonical edge archive suffix during transition work.
- `xpkg convert` remains available as an edge compatibility helper, not as the
  primary project workflow.
- No new public examples should frame direct `.xpkg` archive handling as the
  native or preferred project workflow.
