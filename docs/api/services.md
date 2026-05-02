# Services

<div class="page-intro">
<p>
<code>xpkg.services</code> is the normal downstream API for xpkg projects.
Start here when you want one consumer-facing object that can create, open,
import into, validate, pack, or unpack a workspace-first project.
</p>
</div>

## Start Here

- Use <code>WorkspaceService</code> as the stable consumer contract for project
  lifecycle operations.
- Use <code>workspace.imports.*</code> when you want the supported external
  importers without dropping out of that service object.
- Use <code>workspace.artifacts.*</code> when you want to register figures,
  tables, analyses, reports, stats, or other output files with portable
  manifests and a workspace-wide index.
- Use <code>workspace.figures.*</code> when you want to save figure outputs
  with portable provenance manifests. This is a convenience layer over the
  generic artifact registry.
- Use <code>workspace.segmentation.*</code> when you want to save or load
  frame-level segmentation masks without manually rebuilding a
  <code>Labels</code> object.
- Use <code>xpkg.formats.import_*_workspace(...)</code> when you explicitly
  want the same importers as free functions.

## Recommended Flow

```python
from xpkg.services import WorkspaceService

workspace = WorkspaceService.create("./My Project", title="My Project")
workspace.imports.dlc_csv(
    "tracking.csv",
    "video.mp4",
    skeleton_name="subject",
)
workspace.validate()
artifact = workspace.pack()
restored = WorkspaceService.unpack(artifact, "./Restored Project")
```

This is the canonical downstream path:

- create or open a workspace
- import through <code>workspace.imports.*</code>
- validate the managed workspace state
- pack only when you want a portable <code>.expkg</code> artifact
- reopen with <code>WorkspaceService.open(...)</code> or
  <code>WorkspaceService.unpack(...)</code> as needed

## Lifecycle Surface

`WorkspaceService` keeps the normal workspace-first project lifecycle on one
object:

- `WorkspaceService.create(...)`
- `WorkspaceService.open(...)`
- `WorkspaceService.unpack(...)`
- `workspace.describe()`
- `workspace.validate()`
- `workspace.load_labels()`
- `workspace.save_labels(...)`
- `workspace.artifacts.register(...)`
- `workspace.artifacts.load(...)`
- `workspace.artifacts.list(...)`
- `workspace.artifacts.index(...)`
- `workspace.artifacts.validate(...)`
- `workspace.figures.save(...)`
- `workspace.figures.load(...)`
- `workspace.figures.list(...)`
- `workspace.segmentation.save_masks(...)`
- `workspace.segmentation.load_masks(...)`
- `workspace.segmentation.load_frames(...)`
- `workspace.pack(...)`

`workspace.validate()` returns a `WorkspaceLayout` with the normalized managed
paths and descriptor for the workspace.

## Service-Bound Import Surface

Each service-bound importer mirrors a public
<code>xpkg.formats.import_*_workspace(...)</code> helper:

| Service method | Matching free function |
| --- | --- |
| `workspace.imports.dlc_csv(...)` | `xpkg.formats.import_dlc_csv_workspace(...)` |
| `workspace.imports.dlc_h5(...)` | `xpkg.formats.import_dlc_h5_workspace(...)` |
| `workspace.imports.dlc_project(...)` | `xpkg.formats.import_dlc_project_workspace(...)` |
| `workspace.imports.lightning_pose_csv(...)` | `xpkg.formats.import_lightning_pose_csv_workspace(...)` |
| `workspace.imports.sleap_h5(...)` | `xpkg.formats.import_sleap_h5_workspace(...)` |
| `workspace.imports.sleap_package(...)` | `xpkg.formats.import_sleap_package_workspace(...)` |
| `workspace.imports.mmpose_topdown_json(...)` | `xpkg.formats.import_mmpose_topdown_json_workspace(...)` |
| `workspace.imports.mediapipe_pose_landmarks_json(...)` | `xpkg.formats.import_mediapipe_pose_landmarks_json_workspace(...)` |

The service-bound methods are the preferred path for new project-facing code.
The free functions remain public for explicit function-level integrations.

## Multimodal Reader And Import Plan

The session/time/events/signals model layer is public. Direct CSV readers are
available now:

```python
xpkg.read_photometry_csv(...)
xpkg.read_events_csv(...)
xpkg.read_pyphotometry_ppd(...)
```

These service-bound imports are not implemented yet:

```python
workspace.imports.photometry_csv(...)
workspace.imports.events_csv(...)
workspace.imports.sync_csv(...)
```

The remaining direct reader planned in this family is:

```python
xpkg.read_sync_csv(...)
```

See [Multimodal Session Model](../architecture/multimodal-session.md) for the
model objects that will back those imports and [Roadmap](../roadmap.md) for the
implementation order.

## Generic Artifact Registry

`workspace.artifacts` is the first-class output registry for scientific
packages that build on xpkg. It records files and lineage; it does not decide
what a table means, which statistical model is correct, or how a figure should
look.

```python
from xpkg.services import WorkspaceService

workspace = WorkspaceService.open("./My Project")

table = workspace.artifacts.register(
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

workspace.artifacts.validate(table.artifact_id, kind="table", namespace="neuro-analysis")
```

Generic artifacts are stored under `.xpkg/artifacts/<kind>/<artifact_id>/`.
Namespaced artifacts are stored under
`.xpkg/<namespace>/<kind>/<artifact_id>/`. The workspace-wide index lives at
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

Figures are saved as workspace artifacts, not plotted by `xpkg`. Your domain
package creates the figure and source-data files; `xpkg` copies those outputs
into the same workspace and writes a manifest that records inputs, producer
metadata, stats reports, and portable output paths.

```python
from xpkg.services import WorkspaceService

workspace = WorkspaceService.open("./My Project")

figure = workspace.figures.save(
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

workspace.figures.validate(figure.artifact_id)
```

With `namespace="neuro-analysis"`, outputs are copied under
`.xpkg/neuro-analysis/figures/<figure_id>/`. Omit `namespace` to use the generic
`.xpkg/artifacts/figures/<figure_id>/` registry. Namespaces are caller-owned
strings; `xpkg` does not reserve or hard-code downstream package names. The
manifest is intentionally generic: `xpkg` tracks and packages the
claim-carrying artifact, while the downstream package still owns the scientific
or domain-specific meaning of the plot.

## Segmentation Masks

Segmentation masks are first-class workspace state. Attach them to frames
through the service instead of manually constructing a full labels payload:

```python
import numpy as np

from xpkg.model import SegmentationMask
from xpkg.services import WorkspaceService

workspace = WorkspaceService.open("./My Project")

binary = np.zeros((480, 640), dtype=np.uint8)
binary[120:260, 180:340] = 1
mask = SegmentationMask.from_binary_mask(
    binary,
    class_name="cell",
    confidence=0.94,
)

workspace.segmentation.save_masks(
    frame_index=42,
    masks=[mask],
)

masks = workspace.segmentation.load_masks(frame_index=42)
```

Use `mode="append"` to add masks to a frame without replacing the existing
ones. Use `workspace.segmentation.load_frames(...)` when you want every frame
that currently has segmentation masks, with optional filters such as
`predicted=True` or `class_name="cell"`.

## Secondary Public Surfaces

- Use [Formats](formats.md) when you want the same workspace-first behavior as
  explicit free functions.
