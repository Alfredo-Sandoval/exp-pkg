# exp-pkg

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)
[![Version: 0.1.0](https://img.shields.io/badge/version-0.1.0-green.svg)](pyproject.toml)

**Canonical IO and artifact layer for experiment data, managed workspaces, and portable project artifacts.**

Import from `xpkg` in Python and use `xpkg` for the CLI.

exp-pkg exists to give other repos one stable boundary for experiment-data IO.
It imports external formats, normalizes them into canonical `xpkg` objects,
stores them in a workspace-first project contract, and emits portable `.expkg`
artifacts.

This repo is not an analysis platform. It is the IO layer that analysis tools,
GUIs, and automation can build on when they need a coherent project/workspace
surface instead of a pile of ad hoc CSV, H5, JSON, and archive files.

The public product contract is now intentionally narrow:

- editable project = workspace folder
- authoritative mutable state = private `.xpkg/`
- portable artifact = `.expkg`
- legacy `.xpkg` archives = migration input, not a first-class downstream target

## Positioning

The intended stack is:

- external pose / annotation formats at the edge
- canonical in-memory objects in the middle
- in-memory exchange helpers for arrays / tables / JSON payloads
- editable workspace + private store + portable artifact at the boundary

The current codebase should be read as a generic IO and packaging layer for
experiment projects, not as an analysis framework.

## Recommended Workspace API

`WorkspaceService` is the primary service boundary for workspace-based
projects. Use it when you want one object that can create, open, import into,
validate, pack, or unpack a project:

```python
from xpkg.services import WorkspaceService

workspace = WorkspaceService.create("./My Project", title="My Project")
workspace.imports.dlc_csv(
    "tracking.csv",
    "video.mp4",
    skeleton_name="subject",
)
layout = workspace.validate()
artifact = workspace.pack()
restored = WorkspaceService.unpack(artifact, "./Restored Project")
```

If you are wiring another repo into xpkg, this is the place to start.

Choose the public surface by job:

| Task | Preferred public entrypoint |
| --- | --- |
| Create, open, import into, validate, pack, or unpack a project | `xpkg.services.WorkspaceService` |
| Import foreign pose data into a project you already manage through the service | `workspace.imports.*` from `xpkg.services.WorkspaceService` |
| Register figures, tables, analyses, reports, or other output artifacts | `workspace.artifacts.*` from `xpkg.services.WorkspaceService` |
| Save figure outputs and their lineage manifests | `workspace.figures.*` from `xpkg.services.WorkspaceService` |
| Save or load frame-level segmentation masks | `workspace.segmentation.*` from `xpkg.services.WorkspaceService` |
| Import foreign pose data through explicit free functions | `xpkg.formats.import_*_workspace(...)` |
| Cut over a legacy `.xpkg` archive into the workspace contract | `xpkg migrate` or `xpkg.formats.migrate_legacy_archive(...)` |

The explicit `xpkg.formats.import_*_workspace(...)` helpers remain public when
you want a function-level API or need to import before reopening a workspace.

Artifacts use a generic registry under `.xpkg/artifacts/<kind>/`, with common
kind directories such as `figures`, `tables`, `analyses`, `reports`, and
`stats-reports`. Callers may also choose their own app namespace, such as
`.xpkg/analysis-app/figures/`, by passing `namespace="analysis-app"` to
`workspace.artifacts.register(...)` or `workspace.figures.save(...)`. `xpkg`
treats namespaces as caller-owned strings; it does not reserve or hard-code
downstream package names.

The shipped workspace import surface currently covers:

- Vicon CSV and C3D recordings
- DeepLabCut CSV, H5, and project imports
- Lightning Pose prediction CSV (DLC-style MultiIndex)
- SLEAP analysis H5 and `.pkg.slp`
- MMPose top-down demo JSON (`--save-predictions`)
- MediaPipe pose-landmarks JSON

## What It Does

- Imports external pose / annotation formats into canonical xpkg objects
- Defines a stable project contract: workspace folder + private `.xpkg/` + `.expkg`
- Manages workspace lifecycle: create, open, validate, pack, unpack
- Carries canonical containers such as `Labels`, `Skeleton`, `Instance`, and `Video`
- Registers output artifacts with portable manifests for inputs, producer metadata, stats, checksums, and source data
- Provides figure convenience helpers on top of the generic artifact registry
- Saves and loads frame-level segmentation masks through `workspace.segmentation`
- Exposes a clean in-memory exchange layer through `xpkg.exchange`
- Handles media-aware packaging and workspace-relative project state
- Ships one explicit legacy migration path for older `.xpkg` archives

## Current Scope vs Direction

Implemented today:

- canonical annotation and media data objects
- readers and workspace importers for external formats
- workspace/store/artifact lifecycle operations
- media-aware packaging and portable exports
- a narrow legacy archive migration seam

Mission direction:

- keep xpkg narrow as the stable IO and artifact boundary
- support more external ecosystems through workspace importers
- make downstream analysis and GUI repos depend on xpkg instead of inventing
  their own project formats
- keep direct archive handling narrow and clearly secondary to workspace flows

## Supported Formats

| Source | Format | Status |
|--------|--------|--------|
| Vicon | CSV | Supported |
| Vicon | C3D | Supported |
| DeepLabCut | CSV | Supported |
| DeepLabCut | H5 | Supported |
| DeepLabCut | Project | Supported |
| Lightning Pose | Prediction CSV (DLC-style MultiIndex) | Supported |
| SLEAP | Analysis H5 | Supported |
| SLEAP | `.pkg.slp` | Supported |
| MMPose | Top-down demo JSON (`--save-predictions`) | Supported |
| MediaPipe | Pose landmarks JSON | Supported |

## Install

Not on PyPI yet. Clone and install locally:

When published, the distribution name will be `exp-pkg`. The Python import
name and CLI command remain `xpkg`.

```bash
git clone https://github.com/Alfredo-Sandoval/exp-pkg.git
cd exp-pkg
make env
```

Fallback if you do not want the canonical setup target:

```bash
bash environment/setup.sh
```

Then use the local quality gates:

```bash
make qa
make ci-local
```

Before a PyPI/TestPyPI cut, run the local release gate against real lab data:

```bash
make release-check REAL_DATA_ROOT=/path/to/xpkg-real-data
```

There is intentionally no hosted CI requirement for this repo. The local
release gate runs linting, type checking, synthetic tests, package build/check,
strict docs build, and the opt-in real-data suite.

## Real Data Tests

The normal test suite uses deterministic synthetic fixtures. Production
readiness requires a private real-data corpus supplied through
`XPKG_REAL_DATA_ROOT` or `REAL_DATA_ROOT=...` when invoking `make`.

Create a manifest named `xpkg-real-data.json` at the corpus root, or point
`XPKG_REAL_DATA_MANIFEST` at a manifest file:

```json
{
  "schema_version": 1,
  "cases": [
    {
      "id": "dlc-session-001",
      "kind": "dlc",
      "tracking": "dlc/session_001/tracking.csv",
      "video": "dlc/session_001/video.mp4",
      "skeleton_name": "subject",
      "expect": {
        "state": "labels",
        "videos": 1,
        "skeletons": 1,
        "min_labeled_frames": 1
      }
    },
    {
      "id": "vicon-trial-001",
      "kind": "vicon",
      "recording": "vicon/trial_001.c3d",
      "expect": {
        "state": "vicon"
      }
    }
  ]
}
```

Supported real-data `kind` values are `vicon`, `dlc`, `lightning_pose`,
`sleap`, `mmpose`, and `mediapipe`. Use `kind: "vicon"` for both CSV and C3D
recordings; use `kind: "dlc"` with either `tracking` plus `video` for a single
CSV/H5 tracking file, or `project` for a full DLC project folder; use
`kind: "lightning_pose"` with `tracking` plus `video` for a Lightning Pose
prediction CSV produced by `litpose predict`; use
`kind: "sleap"` with a `labels` file ending in `.slp`, `.pkg.slp`, `.h5`, or
`.hdf5`. SLEAP analysis H5 cases also need a matching `video`. Each case
imports into a fresh workspace, validates, packs to `.expkg`, unpacks, and
validates again unless `"skip_pack": true` is set.

## Documentation

Build and serve the docs locally with MkDocs:

```bash
make docs-build    # build the static site
make docs-serve    # live preview at localhost:8123
```

## Public Artifact Contract

exp-pkg v1 defines exactly three product-facing artifact classes:

```text
My Project/
  PROJECT.json
  .xpkg/
  Media/
  Exports/
    My Project.expkg
```

- Editable project = workspace folder
- Authoritative mutable state = `.xpkg/`
- Portable artifact = `.expkg`

The artifact model is workspace-first so experiment state, managed media, and
future aligned modalities have a clear home in one project layout.

The locked spec lives in `docs/artifact_contract_v1.md`, with the matching
command surface in `docs/cli_command_spec_v1.md`.

## Legacy Migration

Older `.xpkg` archives are still supported as migration inputs, but they are no
longer the public project contract.

Use one of these explicit cutover paths:

```bash
xpkg migrate "./legacy.xpkg" --out "./My Project"
```

```python
from xpkg.formats import migrate_legacy_archive

snapshot_path = migrate_legacy_archive("./legacy.xpkg", "./My Project")
```

That migration writes workspace-native state into `.xpkg/`, refreshes the
rebuildable `.xpkg/state/current.json` cache, and leaves new work on the normal
workspace-first path.

## CLI

The primary workspace-first CLI surface is:

```bash
xpkg init "./My Project"
xpkg import dlc csv --csv tracking.csv --video video.mp4 --out "./My Project"
xpkg import lightning-pose --csv predictions.csv --video video.mp4 --out "./My Project"
xpkg pack "./My Project"
xpkg unpack "./My Project.expkg" --out "./My Project"
xpkg validate "./My Project"
xpkg migrate "./legacy.xpkg" --out "./My Project"
xpkg artifacts list "./My Project" --kind figure
xpkg artifacts inspect "./My Project" validation-figure-3 --kind figure
xpkg artifacts validate "./My Project" --kind figure
```

The same `xpkg import` command also ships source-specific workspace imports for
Vicon recordings, Lightning Pose CSV, SLEAP, MMPose JSON, and MediaPipe JSON.
Every canonical command supports `--json` for machine-readable output, and
`xpkg describe --json` reports the current command contract for agents.
Input files that are themselves JSON use `--input-json` so `--json` is reserved
for output mode.

## Vicon Recording API

Vicon support is intentionally narrow and mocap-native: it preserves marker
names, source labels, `(frames, markers, 3)` positions, validity/gaps, fps,
frame offsets, optional raw C3D `EVENT` metadata, optional analog channels,
analog units/descriptions, optional additional point channels, and sibling
`.xcp` / `.vsk` sidecars when present.

Use the low-level reader when another repo just needs to load a recording:

```python
from xpkg.api import read_vicon_recording

recording = read_vicon_recording("trial.c3d")
print(recording.marker_names)
print(recording.positions.shape)
print([(event.side, event.event_type, event.source_frame) for event in recording.gait_events])
print(recording.analog.channel_names if recording.analog is not None else ())
print(recording.analog.candidate_emg_channel_names if recording.analog is not None else ())
```

Use the workspace service when another repo wants an imported, portable project:

```python
from xpkg.api import WorkspaceService

workspace = WorkspaceService.create("./Vicon Project", title="Vicon Project")
workspace.imports.vicon("trial.c3d")
recording = workspace.load_vicon_recording()
artifact = workspace.pack()
```

The explicit free-function surface is also public:

```python
from xpkg.api import import_vicon_c3d_workspace, load_workspace_vicon_recording

import_vicon_c3d_workspace("trial.c3d", "./Vicon Project")
recording = load_workspace_vicon_recording("./Vicon Project")
```

CLI examples:

```bash
xpkg import vicon --recording trial.c3d --out "./Vicon Project"
xpkg import vicon --csv trial.csv --out "./Vicon Project"
xpkg import vicon --c3d trial.c3d --out "./Vicon Project"
```

## Development

If you want to add an importer for a new pose-estimation framework or improve
existing functionality:

1. Open an issue or local task describing the change.
2. Create a focused feature branch.
3. Run `make qa` for the fast gate.
4. Run `make release-check REAL_DATA_ROOT=/path/to/xpkg-real-data` before a
   package handoff or PyPI/TestPyPI cut.

Please follow the existing code style (enforced by [Ruff](https://docs.astral.sh/ruff/) with the settings in `pyproject.toml`).

## License

This project is licensed under the **BSD 3-Clause License**. See the [LICENSE](LICENSE) file for full terms.
