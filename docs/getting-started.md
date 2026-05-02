# Getting Started

<div class="page-intro">
<p>
xpkg is the canonical IO and artifact layer for multimodal neuroscience
experiment projects. It is not on PyPI yet; clone the repo and install locally.
</p>
</div>

## Install

```bash
git clone https://github.com/Alfredo-Sandoval/exp-pkg.git
cd exp-pkg
make env
```

Fallback if you do not want the canonical setup target:

```bash
bash environment/setup.sh
```

`make env` installs the local dev and docs toolchain. The main local checks are:

```bash
make qa
make ci-local
```

Build and smoke-test the Python package:

```bash
make package-check
make build
uv pip install dist/exp_pkg-*.whl
```

After the first PyPI release, users will install the distribution directly:

```bash
uv pip install exp-pkg
```

Before a package handoff or PyPI/TestPyPI cut, run the release gate against a
private real-data corpus:

```bash
make release-check REAL_DATA_ROOT=/path/to/xpkg-real-data
```

## Preview the docs locally

```bash
make docs-build
make docs-serve
```

## Start with the v1 artifact model

The public xpkg v1 artifact model is project-first:

```text
My Project/
  PROJECT.json
  .xpkg/
  Media/
  Exports/
    My Project.expkg
```

- You edit a normal project folder.
- xpkg owns authoritative mutable state inside `.xpkg/`.
- You move/share/export a single `.expkg` file.
- Older single-file HDF5 archives are not part of the ongoing project contract.

This matters because the project is the place where media, segmentation,
labels, and future experiment-side modalities can live together under one
contract.

Read [Artifact Contract v1](artifact_contract_v1.md) for the full public
contract and [CLI Command Spec v1](cli_command_spec_v1.md) for the locked
project command surface.

## Recommended project-first API

```python
from xpkg.services import ProjectService

project = ProjectService.create("./My Project", title="My Project")
project.imports.dlc_csv(
    "tracking.csv",
    "video.mp4",
    skeleton_name="subject",
)
project.validate()
artifact = project.pack()
restored = ProjectService.unpack(artifact, "./Restored Project")
```

`ProjectService` is the normal project boundary: create or open a project,
import through `project.imports`, validate, then pack only when you want a
portable artifact. The dedicated guide for that surface lives in
[Services](api/services.md).

By default `project.pack()` includes all managed media. Use
`project.pack(media="package")` or `project.pack(media="manifest")` when the
project should keep video bytes outside the `.expkg` while still recording the
managed media manifest.

## Work With The Session Model

Use `xpkg.model` when you want in-memory multimodal objects without creating a
project yet. The timing, event, signal, photometry, and session classes are
the foundation for the next direct-reader and project-import work.

```python
from xpkg.model import Event, EventTable, PhotometryRecording, RecordingSession, TimeSeries

series = TimeSeries.from_samples(
    [[1.0, 0.5], [1.1, 0.48], [1.2, 0.47]],
    sample_rate_hz=20.0,
    channel_names=["gcamp", "isosbestic"],
    units=["dff", "dff"],
    name="fiber",
)
photometry = PhotometryRecording(
    series=series,
    signal_channel="gcamp",
    reference_channel="isosbestic",
)

events = EventTable.from_events(
    [Event(kind="trial", start_s=0.0, duration_s=1.0, label="A")]
)

session = RecordingSession(session_id="session-001")
session = session.with_signal("fiber", photometry).with_events(events)
```

Read [Multimodal Session Model](architecture/multimodal-session.md) for the
current object contract and [Roadmap](roadmap.md) for the direct-reader and
project-import work still ahead.

## Save figure artifacts

Use `project.artifacts` when you want the generic registry for tables,
analyses, reports, stats, figures, or other output files. Your domain package
creates the scientific output; `xpkg` stores the files, records portable
lineage, and keeps a project-wide index.

```python
from xpkg.services import ProjectService

project = ProjectService.open("./My Project")

project.artifacts.register(
    artifact_id="session_001_summary",
    artifact_type="table",
    namespace="neuro-analysis",
    outputs={"summary.csv": "output/session_001_summary.csv"},
    inputs=[".xpkg/neuro-analysis/events/session_001/final_events.csv"],
    producer={"package": "neuro-analysis"},
)
```

Use `project.figures` as the figure-specific convenience layer after your
domain package or plotting script has created the actual figure files:

```python
from xpkg.services import ProjectService

project = ProjectService.open("./My Project")

project.figures.save(
    figure_id="session_summary_figure",
    title="Validation against reference annotations",
    namespace="neuro-analysis",
    outputs={
        "figure.svg": "output/session_summary_figure.svg",
        "figure.pdf": "output/session_summary_figure.pdf",
        "source_data.csv": "output/session_summary_figure_source_data.csv",
    },
    inputs=[".xpkg/neuro-analysis/events/session_001/final_events.csv"],
    producer={"package": "neuro-analysis"},
)
```

That stores the figure outputs and manifest under
`.xpkg/neuro-analysis/figures/session-summary-figure/`. Without a namespace, xpkg
uses `.xpkg/artifacts/figures/`. Namespace values are supplied by downstream
callers; xpkg does not know or reserve package-specific namespace names.

## Save segmentation masks

Segmentation masks are project state too. Downstream repos can save a mask
onto a frame without manually constructing the full labels payload:

```python
import numpy as np

from xpkg.model import SegmentationMask
from xpkg.services import ProjectService

project = ProjectService.open("./My Project")

binary = np.zeros((480, 640), dtype=np.uint8)
binary[120:260, 180:340] = 1
mask = SegmentationMask.from_binary_mask(binary, class_name="cell")

project.segmentation.save_masks(frame_index=42, masks=[mask])
loaded = project.segmentation.load_masks(frame_index=42)
```

Pick the surface by intent:

| Task | Preferred entrypoint |
| --- | --- |
| Project lifecycle and service-bound imports | `xpkg.services.ProjectService` |
| Register tables, figures, analyses, reports, or stats | `project.artifacts.*` |
| Save/load figure outputs with lineage | `project.figures.*` |
| Save/load frame segmentation masks | `project.segmentation.*` |
| Function-level project imports | `xpkg.project.import_*_project(...)` |

## Lifecycle-only example

```python
from xpkg.services import ProjectService

project = ProjectService.open("./My Project")
layout = project.validate()
artifact = project.pack()
```

Once a project already exists, the same service object keeps validation,
packing, and reopen flows on the same public contract.

## Additional project import coverage

The same project-first pattern is available for:

- `import_dlc_h5_project(...)` and `import_dlc_project_directory(...)`
- `import_lightning_pose_csv_project(...)`
- `import_sleap_h5_project(...)` and `import_sleap_package_project(...)`
- `import_mmpose_topdown_json_project(...)`
- `import_mediapipe_pose_landmarks_json_project(...)`

Use those project helpers as the primary integration surface for new code.
The underlying `xpkg.project.import_*_project(...)` functions remain public
when you want the explicit function form.

Photometry and event-table CSV readers are available as direct reader APIs.
They are not project imports yet. Sync CSV and the project import methods
are the next planned modality work on top of the timing/events/signals model
layer.

## In-memory Adapters API

```python
from xpkg.adapters import labels_from_json_payload, labels_to_json_payload
from xpkg.model import Labels

labels = Labels()
payload = labels_to_json_payload(labels)
roundtripped = labels_from_json_payload(payload)
```

Use `xpkg.adapters` when another repo needs an in-memory handoff boundary rather
than a project path.
