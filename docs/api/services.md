# Services

<div class="page-intro">
<p>
<code>xpkg.services</code> is the normal downstream API for xpkg projects.
Start here when you want one consumer-facing object that can create, open,
import into, validate, pack, or unpack a project-first project.
</p>
</div>

## Start Here

- Use <code>ProjectService</code> as the stable consumer contract for project
  lifecycle operations.
- Use <code>project.imports.*</code> when you want the supported external
  importers without dropping out of that service object.
- Use <code>project.artifacts.*</code> when you want to register figures,
  tables, analyses, reports, stats, or other output files with portable
  manifests and a project-wide index.
- Use <code>project.figures.*</code> when you want to save figure outputs
  with portable provenance manifests. This is a convenience layer over the
  generic artifact registry.
- Use <code>project.segmentation.*</code> when you want to save or load
  frame-level segmentation masks without manually rebuilding a
  <code>Labels</code> object.
- Use <code>xpkg.project.import_*_project(...)</code> when you explicitly
  want the same importers as free functions.

## Recommended Flow

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

This is the canonical downstream path:

- create or open a project
- import through <code>project.imports.*</code>
- validate the managed project state
- pack only when you want a portable <code>.expkg</code> artifact
- reopen with <code>ProjectService.open(...)</code> or
  <code>ProjectService.unpack(...)</code> as needed

`project.pack()` defaults to `media="full"`. Pass `media="package"` to store
package-sized media while manifesting video containers, or `media="manifest"`
to record managed media without storing media bytes.

## Lifecycle Surface

`ProjectService` keeps the normal project-first project lifecycle on one
object:

- `ProjectService.create(...)`
- `ProjectService.open(...)`
- `ProjectService.unpack(...)`
- `project.describe()`
- `project.validate()`
- `project.load_labels()`
- `project.load_metadata()`
- `project.load_metadata_field(...)`
- `project.save_labels(...)`
- `project.save_metadata(...)`
- `project.save_metadata_field(...)`
- `project.artifacts.register(...)`
- `project.artifacts.load(...)`
- `project.artifacts.list(...)`
- `project.artifacts.index(...)`
- `project.artifacts.validate(...)`
- `project.figures.save(...)`
- `project.figures.load(...)`
- `project.figures.list(...)`
- `project.segmentation.save_masks(...)`
- `project.segmentation.load_masks(...)`
- `project.segmentation.load_frames(...)`
- `project.pack(...)`

`project.validate()` returns a `ProjectLayout` with the normalized managed
paths and descriptor for the project.

For mapping-valued metadata blobs that callers update independently, prefer the
service-bound field helpers instead of rewriting the whole metadata payload:

```python
from xpkg.services import ProjectService

project = ProjectService.open("./My Project")

project.save_metadata_field(
    "session_json",
    {"active_frame_idx": 7},
    reason="app.save.session_state",
)
session_state = project.load_metadata_field("session_json")
```

## Service-Bound Import Surface

Each service-bound importer mirrors a public
<code>xpkg.project.import_*_project(...)</code> helper:

| Service method | Matching free function |
| --- | --- |
| `project.imports.dlc_csv(...)` | `xpkg.project.import_dlc_csv_project(...)` |
| `project.imports.dlc_h5(...)` | `xpkg.project.import_dlc_h5_project(...)` |
| `project.imports.dlc_project(...)` | `xpkg.project.import_dlc_project_directory(...)` |
| `project.imports.lightning_pose_csv(...)` | `xpkg.project.import_lightning_pose_csv_project(...)` |
| `project.imports.sleap_h5(...)` | `xpkg.project.import_sleap_h5_project(...)` |
| `project.imports.sleap_package(...)` | `xpkg.project.import_sleap_package_project(...)` |
| `project.imports.mmpose_topdown_json(...)` | `xpkg.project.import_mmpose_topdown_json_project(...)` |
| `project.imports.mediapipe_pose_landmarks_json(...)` | `xpkg.project.import_mediapipe_pose_landmarks_json_project(...)` |

The service-bound methods are the preferred path for new project-facing code.
The free functions remain public for explicit function-level integrations.

## Multimodal Reader And Import Plan

The session/time/events/signals model layer is public. Direct fiber-photometry
and event readers are available now:

```python
xpkg.read_photometry_csv(...)
xpkg.read_events_csv(...)
xpkg.read_pyphotometry_ppd(...)
xpkg.read_pyphotometry_csv(...)
xpkg.read_pmat_photometry_csv(...)
xpkg.read_pmat_events_csv(...)
xpkg.read_rwd_ofrs_session(...)
xpkg.read_neurophotometrics_csv(...)
xpkg.read_doric_photometry(...)
xpkg.read_teleopto_h5(...)
xpkg.read_tdt_photometry_block(...)
```

These service-bound imports are not implemented yet:

```python
project.imports.photometry_csv(...)
project.imports.events_csv(...)
project.imports.sync_csv(...)
```

The remaining direct reader planned in this family is:

```python
xpkg.read_sync_csv(...)
```

The fiber-photometry surface intentionally excludes imaging/miniscope and
electrophysiology formats such as Inscopix `.isx`, Blackrock NEV/NSx, and
Neuralynx Cheetah. Those belong to separate IO layers.

See [Multimodal Session Model](../architecture/multimodal-session.md) for the
model objects that will back those imports and [Roadmap](../roadmap.md) for the
implementation order.

## Generic Artifact Registry

`project.artifacts` is the first-class output registry for scientific
packages that build on xpkg. It records files and lineage; it does not decide
what a table means, which statistical model is correct, or how a figure should
look.

```python
from xpkg.services import ProjectService

project = ProjectService.open("./My Project")

table = project.artifacts.register(
    artifact_id="session_001_summary",
    artifact_type="table",
    title="Session 001 summary table",
    namespace="neuro-analysis",
    outputs={"summary.csv": "results/session_001_summary.csv"},
    inputs=[".xpkg/neuro-analysis/events/session_001/final_events.csv"],
    producer={
        "package": "neuro-analysis",
        "command": "neuro-analysis make-tables session_001",
        "git_commit": "...",
    },
    metadata={"unit_of_analysis": "event"},
)

project.artifacts.validate(table.artifact_id, kind="table", namespace="neuro-analysis")
```

Generic artifacts are stored under `.xpkg/artifacts/<kind>/<artifact_id>/`.
Namespaced artifacts are stored under
`.xpkg/<namespace>/<kind>/<artifact_id>/`. The project-wide index lives at
`.xpkg/artifacts/index.json` and can be rebuilt from manifests at any time.

Common artifact kinds map to readable directory names:

| Artifact type | Directory |
| --- | --- |
| `figure` | `figures` |
| `table` | `tables` |
| `analysis` | `analyses` |
| `report` | `reports` |
| `stats-report` | `stats-reports` |

## Figure Artifacts

Figures are saved as project artifacts, not plotted by `xpkg`. Your domain
package creates the figure and source-data files; `xpkg` copies those outputs
into the same project and writes a manifest that records inputs, producer
metadata, stats reports, and portable output paths.

```python
from xpkg.services import ProjectService

project = ProjectService.open("./My Project")

figure = project.figures.save(
    figure_id="session_summary_figure",
    title="Validation against reference annotations",
    namespace="neuro-analysis",
    outputs={
        "figure.svg": "output/session_summary_figure.svg",
        "figure.pdf": "output/session_summary_figure.pdf",
        "source_data.csv": "output/session_summary_figure_source_data.csv",
    },
    inputs=[
        ".xpkg/neuro-analysis/events/session_001/final_events.csv",
        ".xpkg/neuro-analysis/labels/session_001/reference_annotations.csv",
    ],
    stats=[
        ".xpkg/neuro-analysis/analysis/validation/stats_report.json",
    ],
    producer={
        "package": "neuro-analysis",
        "module": "neuro_analysis.figures.validation",
        "command": "neuro-analysis make-figures --analysis validation",
        "git_commit": "...",
    },
)

project.figures.validate(figure.artifact_id)
```

With `namespace="neuro-analysis"`, outputs are copied under
`.xpkg/neuro-analysis/figures/<figure_id>/`. Omit `namespace` to use the generic
`.xpkg/artifacts/figures/<figure_id>/` registry. Namespaces are caller-owned
strings; `xpkg` does not reserve or hard-code downstream package names. The
manifest is intentionally generic: `xpkg` tracks and packages the
claim-carrying artifact, while the downstream package still owns the scientific
or domain-specific meaning of the plot.

## Segmentation Masks

Segmentation masks are first-class project state. Attach them to frames
through the service instead of manually constructing a full labels payload:

```python
import numpy as np

from xpkg.model import SegmentationMask
from xpkg.services import ProjectService

project = ProjectService.open("./My Project")

binary = np.zeros((480, 640), dtype=np.uint8)
binary[120:260, 180:340] = 1
mask = SegmentationMask.from_binary_mask(
    binary,
    class_name="cell",
    confidence=0.94,
)

project.segmentation.save_masks(
    frame_index=42,
    masks=[mask],
)

masks = project.segmentation.load_masks(frame_index=42)
```

Use `mode="append"` to add masks to a frame without replacing the existing
ones. Use `project.segmentation.load_frames(...)` when you want every frame
that currently has segmentation masks, with optional filters such as
`predicted=True` or `class_name="cell"`.

## Secondary Public Surfaces

- Use [Project](project.md) when you want the same project-first behavior as
  explicit free functions.
