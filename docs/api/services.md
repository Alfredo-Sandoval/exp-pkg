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
- Use <code>xpkg.formats.import_*_workspace(...)</code> when you explicitly
  want the same importers as free functions.
- Use <code>xpkg.compat</code> and <code>xpkg.adapters</code> only for
  low-level direct-archive workflows and migration edges.

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
| `workspace.imports.legacy_archive(...)` | `xpkg.formats.import_legacy_archive(...)` |

The service-bound methods are the preferred path for new project-facing code.
The free functions remain public for explicit function-level integrations.

## Secondary Public Surfaces

- Use [Formats](formats.md) when you want the same workspace-first behavior as
  explicit free functions.
- Use [Compatibility](compat.md) when you need low-level direct
  <code>.xpkg</code> archive access on purpose.
- Use [Adapters](adapters.md) when you explicitly need compatibility
  conversion outputs rather than a managed workspace lifecycle.
