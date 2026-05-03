# exp-pkg

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

**Project-first IO for multimodal neuroscience experiments, managed projects, and portable artifacts.**

Import from `xpkg` in Python and use `xpkg` for the CLI.

exp-pkg exists to give neuroscience repos one stable boundary for
experiment-data IO. It is built for multimodal sessions: pose estimates,
synchronized video, behavioral events, signals such as fiber photometry, and
the metadata needed to keep those modalities aligned.

It imports external formats, normalizes them into canonical `xpkg` objects,
stores them in a project-first project contract, and emits portable `.expkg`
artifacts. The implemented import surface is strongest today for pose,
motion-capture, and video-associated formats; the package direction is a shared
session/timeline layer that can add photometry and other signals without turning
into an analysis platform.

This repo is not an analysis platform. It is the IO layer that analysis tools,
GUIs, and automation can build on when they need a coherent project/project
surface instead of a pile of ad hoc CSV, H5, JSON, and project sidecars.

The public product contract is now intentionally narrow:

- editable project = project folder
- authoritative mutable state = private `.xpkg/`
- portable artifact = `.expkg`

`.expkg` v1 is a portable zip container with a root `EXPKG.json` manifest.
By default it includes managed media and records member paths, sizes, and
SHA-256 hashes. Pack commands can also emit package-media or manifest-only
media exports when users do not want to store every managed video byte inside
the `.expkg`.

## Positioning

The intended stack is:

- external neuroscience formats at the edge
- canonical in-memory session objects in the middle
- shared timing, event, and signal contracts across modalities
- in-memory exchange helpers for arrays / tables / JSON payloads
- editable project + private store + portable artifact at the boundary

The current codebase should be read as a multimodal neuroscience IO and
packaging layer for experiment projects, not as an analysis framework.

## Recommended Project API

`ProjectService` is the primary service boundary for project-based
projects. Use it when you want one object that can create, open, import into,
validate, pack, or unpack a project:

```python
from xpkg.services import ProjectService

project = ProjectService.create("./My Project", title="My Project")
project.imports.dlc_csv(
    "tracking.csv",
    "video.mp4",
    skeleton_name="subject",
)
layout = project.validate()
artifact = project.pack()
restored = ProjectService.unpack(artifact, "./Restored Project")
```

If you are wiring another repo into xpkg, this is the place to start.

Choose the public surface by job:

| Task | Preferred public entrypoint |
| --- | --- |
| Create, open, import into, validate, pack, or unpack a project | `xpkg.services.ProjectService` |
| Import foreign pose data into a project you already manage through the service | `project.imports.*` from `xpkg.services.ProjectService` |
| Register figures, tables, analyses, reports, or other output artifacts | `project.artifacts.*` from `xpkg.services.ProjectService` |
| Save figure outputs and their lineage manifests | `project.figures.*` from `xpkg.services.ProjectService` |
| Save or load frame-level segmentation masks | `project.segmentation.*` from `xpkg.services.ProjectService` |
| Import foreign pose data through explicit free functions | `xpkg.project.import_*_project(...)` |

The explicit `xpkg.project.import_*_project(...)` helpers remain public when
you want a function-level API or need to import before reopening a project.

Artifacts use a generic registry under `.xpkg/artifacts/<kind>/`, with common
kind directories such as `figures`, `tables`, `analyses`, `reports`, and
`stats-reports`. Callers may also choose their own app namespace, such as
`.xpkg/neuro-analysis/figures/`, by passing `namespace="neuro-analysis"` to
`project.artifacts.register(...)` or `project.figures.save(...)`. `xpkg`
treats namespaces as caller-owned strings; it does not reserve or hard-code
downstream package names.

The shipped project import surface currently covers:

- Vicon CSV and C3D recordings
- DeepLabCut CSV, H5, and project imports
- Lightning Pose prediction CSV (DLC-style MultiIndex)
- SLEAP analysis H5 and `.pkg.slp`
- MMPose top-down demo JSON (`--save-predictions`)
- MediaPipe pose-landmarks JSON
- Generic photometry CSV and event CSV
- pMAT-compatible photometry/event CSV
- pyPhotometry PPD and CSV+JSON
- RWD OFRS CSV session bundles
- Neurophotometrics/Bonsai CSV
- Doric `.doric` photometry containers
- Teleopto H5 exports
- TDT tank/block photometry streams through the optional `tdt` package

## What It Does

- Imports external pose, motion-capture, and media-associated annotation formats into canonical xpkg objects
- Defines a stable project contract: project folder + private `.xpkg/` + `.expkg`
- Manages project lifecycle: create, open, validate, pack, unpack
- Validates `.expkg` manifests, archive member paths, sizes, and SHA-256 digests
- Carries canonical containers such as `Labels`, `Skeleton`, `Instance`, and `Video`
- Registers output artifacts with portable manifests for inputs, producer metadata, stats, checksums, and source data
- Provides figure convenience helpers on top of the generic artifact registry
- Saves and loads frame-level segmentation masks through `project.segmentation`
- Exposes a clean in-memory adapter layer through `xpkg.adapters`
- Handles media-aware packaging and project-relative project state

## Current Scope vs Direction

Implemented today:

- canonical annotation and media data objects
- readers and project importers for external formats
- project/store/artifact lifecycle operations
- media-aware packaging and portable exports

Mission direction:

- keep xpkg narrow as the stable multimodal neuroscience IO and artifact boundary
- support more external ecosystems through project importers
- add first-class timing, event, and signal models for pose, video,
  photometry, behavior, and synchronization data
- make downstream analysis and GUI repos depend on xpkg instead of inventing
  their own project formats
- keep project storage centered on projects and portable `.expkg` exports

## Supported Project

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
| Generic photometry | CSV | Supported |
| Generic events | CSV | Supported |
| pMAT | Photometry/event CSV | Supported |
| pyPhotometry | PPD, CSV+JSON | Supported |
| RWD OFRS | CSV session bundle | Supported |
| Neurophotometrics/Bonsai | CSV | Supported |
| Doric | `.doric` HDF5 photometry container | Supported |
| Teleopto | H5 export | Supported |
| TDT | Tank/block streams | Supported with optional `tdt` dependency |

The fiber-photometry layer intentionally does not claim Inscopix `.isx` /
`.isxd`, Blackrock NEV/NSx, or Neuralynx Cheetah support. Those are imaging or
electrophysiology surfaces, not fiber-photometry IO.

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

The base install keeps heavyweight media/model stacks optional. Add extras when
you need richer media or deep-learning functionality:

```bash
uv pip install -e ".[media-rich]"  # PyAV / rich FFmpeg media handling
uv pip install -e ".[dl]"          # PyTorch + TorchCodec + TorchVision
uv pip install -e ".[inference]"   # ONNX Runtime
uv pip install -e ".[mlx]"         # MLX / Metal acceleration
uv pip install -e ".[nvpkg]"       # nvpkg bridge for Linux NVIDIA media packages
uv pip install -e ".[nvidia]"      # PyTorch + TorchCodec for NVIDIA CUDA
uv pip install -e ".[vision]"      # Kornia + PyTorch
uv pip install -e ".[hardware-accel]"  # MLX + NVIDIA optional runtimes
uv pip install -e ".[media-dl]"    # Full optional media/deep-learning stack
```

Runtime code can inspect available stacks through `xpkg.media`:

```python
from xpkg.media import (
    available_hardware_accelerators,
    available_media_backends,
    require_hardware_acceleration,
    require_media_backend,
)
from xpkg.media.video import Video

print(available_media_backends())
print(available_hardware_accelerators())
require_media_backend("pyav")
require_hardware_acceleration("mlx")
require_hardware_acceleration("opencv-cuda")
video = Video.from_filename("session.mp4", backend="pyav")
```

On Linux NVIDIA hosts, use `nvpkg` as the provisioning layer for CUDA-enabled
media libraries, then let `xpkg.media` verify the result:

```bash
nvpkg system doctor
nvpkg package install ffmpeg
nvpkg package install opencv_cuda
nvpkg package install torchcodec_cuda
nvpkg package verify opencv_cuda --json
```

Then use the local quality gates:

```bash
make qa
make ci-local
```

Build and check the Python package:

```bash
make package-check
make build
uv pip install dist/exp_pkg-*.whl
```

After the first PyPI release, users will install the distribution directly:

```bash
uv pip install exp-pkg
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
imports into a fresh project, validates, packs to `.expkg`, unpacks, and
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

- Editable project = project folder
- Authoritative mutable state = `.xpkg/`
- Portable artifact = `.expkg`

The artifact model is project-first so experiment state, managed media, and
future aligned modalities have a clear home in one project layout.

The locked spec lives in `docs/artifact_contract_v1.md`, with the matching
command surface in `docs/cli_command_spec_v1.md`.

## CLI

The primary project-first CLI surface is:

```bash
xpkg project init "./My Project"
xpkg import dlc csv --csv tracking.csv --video video.mp4 --out "./My Project"
xpkg import lightning-pose --csv predictions.csv --video video.mp4 --out "./My Project"
xpkg import sleap package --slp labels.pkg.slp --out "./My Project"
xpkg project pack "./My Project"
xpkg project pack "./My Project" --media package
xpkg project unpack "./My Project.expkg" --out "./My Project"
xpkg project validate "./My Project"
xpkg project describe "./My Project" --json
xpkg artifacts list "./My Project" --kind figure
xpkg artifacts inspect "./My Project" session-summary-figure --kind figure
xpkg artifacts validate "./My Project" --kind figure
```

The same `xpkg import` command also ships source-specific project imports for
Vicon recordings, Lightning Pose CSV, SLEAP, MMPose JSON, and MediaPipe JSON.
Every canonical command supports `--json` for machine-readable output, and
`xpkg describe --json` reports the current command contract for agents.
Input files that are themselves JSON use `--input-json` so `--json` is reserved
for output mode.

`xpkg project pack` defaults to `--media full`, which stores all managed
`Media/` files in the `.expkg`. Use `--media package` to include package-sized
media such as image sequences while manifesting video containers without
storing them, or `--media manifest` to record managed media paths, sizes, and
hashes without storing media bytes.

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

Use the project service when another repo wants an imported, portable project:

```python
from xpkg.api import ProjectService

project = ProjectService.create("./Vicon Project", title="Vicon Project")
project.imports.vicon("trial.c3d")
recording = project.load_vicon_recording()
artifact = project.pack()
```

The explicit free-function surface is also public:

```python
from xpkg.api import import_vicon_c3d_project, load_project_vicon_recording

import_vicon_c3d_project("trial.c3d", "./Vicon Project")
recording = load_project_vicon_recording("./Vicon Project")
```

CLI examples:

```bash
xpkg import vicon recording --recording trial.c3d --out "./Vicon Project"
xpkg import vicon csv --csv trial.csv --out "./Vicon Project"
xpkg import vicon c3d --c3d trial.c3d --out "./Vicon Project"
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
