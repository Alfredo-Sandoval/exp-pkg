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
- Use <code>workspace.figures.*</code> when you want to save figure outputs
  with portable provenance manifests.
- Use <code>workspace.segmentation.*</code> when you want to save or load
  frame-level segmentation masks without manually rebuilding a
  <code>Labels</code> object.
- Use <code>xpkg.formats.import_*_workspace(...)</code> when you explicitly
  want the same importers as free functions.
- Use <code>xpkg.formats.migrate_legacy_archive(...)</code> or
  <code>xpkg migrate</code> only when you are cutting an older
  <code>.xpkg</code> archive over to the workspace contract.

## Recommended Flow

```python
from xpkg.services import WorkspaceService

workspace = WorkspaceService.create("./My Project", title="My Project")
workspace.imports.dlc_csv(
    "tracking.csv",
    "video.mp4",
    skeleton_name="mouse",
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
| `workspace.imports.sleap_h5(...)` | `xpkg.formats.import_sleap_h5_workspace(...)` |
| `workspace.imports.sleap_package(...)` | `xpkg.formats.import_sleap_package_workspace(...)` |
| `workspace.imports.mmpose_topdown_json(...)` | `xpkg.formats.import_mmpose_topdown_json_workspace(...)` |
| `workspace.imports.mediapipe_pose_landmarks_json(...)` | `xpkg.formats.import_mediapipe_pose_landmarks_json_workspace(...)` |
| `workspace.imports.openpose_json(...)` | `xpkg.formats.import_openpose_json_workspace(...)` |
| `workspace.imports.detectron2_coco(...)` | `xpkg.formats.import_detectron2_coco_workspace(...)` |

The service-bound methods are the preferred path for new project-facing code.
The free functions remain public for explicit function-level integrations.

## Figure Artifacts

Figures are saved as workspace artifacts, not plotted by `xpkg`. Your domain
package creates the figure and source-data files; `xpkg` copies those outputs
into the same workspace and writes a manifest that records inputs, producer
metadata, stats reports, and portable output paths.

```python
from xpkg.services import WorkspaceService

workspace = WorkspaceService.open("./My Project")

figure = workspace.figures.save(
    figure_id="openoperant_validation_figure_3",
    title="Validation against human labels",
    namespace="openoperant",
    outputs={
        "figure.svg": "output/validation_figure_3.svg",
        "figure.pdf": "output/validation_figure_3.pdf",
        "source_data.csv": "output/validation_figure_3_source_data.csv",
    },
    inputs=[
        ".xpkg/openoperant/events/session_001/final_events.csv",
        ".xpkg/openoperant/labels/session_001/human_labels.csv",
    ],
    stats=[
        ".xpkg/openoperant/analysis/validation/stats_report.json",
    ],
    producer={
        "package": "openoperant",
        "module": "openoperant.figures.validation",
        "command": "openoperant make-figures --analysis validation",
        "git_commit": "...",
    },
)

workspace.figures.validate(figure.artifact_id)
```

With `namespace="openoperant"`, outputs are copied under
`.xpkg/openoperant/figures/<figure_id>/`. Omit `namespace` to use the generic
`.xpkg/artifacts/figures/<figure_id>/` registry. The manifest is intentionally
generic: `xpkg` tracks and packages the claim-carrying artifact; OpenOperant,
PHRASE, FIESTA, or another downstream package still owns the scientific
meaning of the plot.

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
- Use `xpkg.formats.migrate_legacy_archive(...)` or `xpkg migrate` when you
  need the one retained legacy `.xpkg` cutover seam.
