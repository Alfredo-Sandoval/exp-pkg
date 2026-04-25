# exp-pkg

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Proprietary](https://img.shields.io/badge/license-Proprietary-red.svg)](LICENSE)
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
- in-memory codecs for arrays / tables / JSON payloads
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
    skeleton_name="mouse",
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
| Import foreign pose data through explicit free functions | `xpkg.formats.import_*_workspace(...)` |
| Cut over a legacy `.xpkg` archive into the workspace contract | `xpkg migrate` or `xpkg.formats.migrate_legacy_archive(...)` |

The explicit `xpkg.formats.import_*_workspace(...)` helpers remain public when
you want a function-level API or need to import before reopening a workspace.

The shipped workspace import surface currently covers:

- Vicon CSV and C3D recordings
- DeepLabCut CSV, H5, and project imports
- SLEAP analysis H5 and `.pkg.slp`
- MMPose top-down demo JSON (`--save-predictions`)
- MediaPipe pose-landmarks JSON
- OpenPose BODY_25 `--write_json` directories
- Detectron2 COCO keypoint results plus dataset/image metadata

## What It Does

- Imports external pose / annotation formats into canonical xpkg objects
- Defines a stable project contract: workspace folder + private `.xpkg/` + `.expkg`
- Manages workspace lifecycle: create, open, validate, pack, unpack
- Carries canonical containers such as `Labels`, `Skeleton`, `Instance`, and `Video`
- Exposes a clean in-memory codec layer through `xpkg.codecs`
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
| SLEAP | Analysis H5 | Supported |
| SLEAP | `.pkg.slp` | Supported |
| MMPose | Top-down demo JSON (`--save-predictions`) | Supported |
| MediaPipe | Pose landmarks JSON | Supported |
| OpenPose | BODY_25 `--write_json` directory | Supported |
| Detectron2 | COCO keypoint results JSON + dataset bundle | Supported |

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
xpkg pack "./My Project"
xpkg unpack "./My Project.expkg" --out "./My Project"
xpkg validate "./My Project"
xpkg migrate "./legacy.xpkg" --out "./My Project"
```

The same `xpkg import` command also ships source-specific workspace imports for
Vicon recordings, SLEAP, MMPose JSON, MediaPipe JSON, OpenPose JSON, and
Detectron2 COCO input.

## Vicon Recording API

Vicon support is intentionally narrow and mocap-native: it preserves marker
names, source labels, `(frames, markers, 3)` positions, validity/gaps, fps,
frame offsets, optional raw C3D `EVENT` metadata, optional analog channels,
optional additional point channels, and sibling `.xcp` / `.vsk` sidecars when
present.

Use the low-level reader when another repo just needs to load a recording:

```python
from xpkg.api import read_vicon_recording

recording = read_vicon_recording("trial.c3d")
print(recording.marker_names)
print(recording.positions.shape)
print([(event.side, event.event_type, event.source_frame) for event in recording.gait_events])
print(recording.analog.channel_names if recording.analog is not None else ())
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

## Contributing

Contributions are welcome! If you'd like to add an importer for a new
pose-estimation framework or improve existing functionality:

1. Open an issue describing the change you'd like to make.
2. Fork the repo and create a feature branch.
3. Run `make qa` for the fast gate, and `make ci-local` before you hand off a larger change.
4. Submit a pull request.

Please follow the existing code style (enforced by [Ruff](https://docs.astral.sh/ruff/) with the settings in `pyproject.toml`).

## License

This project is released under a **Proprietary License**. See the [LICENSE](LICENSE) file for full terms. © 2026 Alfredo and Joseph Sandoval. All rights reserved.
