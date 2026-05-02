# Getting Started

<div class="page-intro">
<p>
xpkg is the canonical IO and artifact layer for experiment-data projects. It is
not on PyPI yet; clone the repo and install locally.
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

The public xpkg v1 artifact model is workspace-first:

```text
My Project/
  PROJECT.json
  .xpkg/
  Media/
  Exports/
    My Project.expkg
```

- You edit a normal workspace folder.
- xpkg owns authoritative mutable state inside `.xpkg/`.
- You move/share/export a single `.expkg` file.
- Older `.xpkg` archives are migration inputs, not the ongoing project contract.

This matters because the workspace is the place where media, segmentation,
labels, and future experiment-side modalities can live together under one
contract.

Read [Artifact Contract v1](artifact_contract_v1.md) for the full public
contract and [CLI Command Spec v1](cli_command_spec_v1.md) for the locked
workspace command surface.

## Recommended workspace-first API

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

`WorkspaceService` is the normal project boundary: create or open a workspace,
import through `workspace.imports`, validate, then pack only when you want a
portable artifact. The dedicated guide for that surface lives in
[Services](api/services.md).

## Save figure artifacts

Use `workspace.artifacts` when you want the generic registry for tables,
analyses, reports, stats, figures, or other output files. Your domain package
creates the scientific output; `xpkg` stores the files, records portable
lineage, and keeps a workspace-wide index.

```python
from xpkg.services import WorkspaceService

workspace = WorkspaceService.open("./My Project")

workspace.artifacts.register(
    artifact_id="session_001_summary",
    artifact_type="table",
    namespace="analysis-app",
    outputs={"summary.csv": "output/session_001_summary.csv"},
    inputs=[".xpkg/analysis-app/events/session_001/final_events.csv"],
    producer={"package": "analysis-app"},
)
```

Use `workspace.figures` as the figure-specific convenience layer after your
domain package or plotting script has created the actual figure files:

```python
from xpkg.services import WorkspaceService

workspace = WorkspaceService.open("./My Project")

workspace.figures.save(
    figure_id="validation_figure_3",
    title="Validation against reviewer labels",
    namespace="analysis-app",
    outputs={
        "figure.svg": "output/validation_figure_3.svg",
        "figure.pdf": "output/validation_figure_3.pdf",
        "source_data.csv": "output/validation_figure_3_source_data.csv",
    },
    inputs=[".xpkg/analysis-app/events/session_001/final_events.csv"],
    producer={"package": "analysis-app"},
)
```

That stores the figure outputs and manifest under
`.xpkg/analysis-app/figures/validation-figure-3/`. Without a namespace, xpkg
uses `.xpkg/artifacts/figures/`. Namespace values are supplied by downstream
callers; xpkg does not know or reserve package-specific namespace names.

## Save segmentation masks

Segmentation masks are workspace state too. Downstream repos can save a mask
onto a frame without manually constructing the full labels payload:

```python
import numpy as np

from xpkg.model import SegmentationMask
from xpkg.services import WorkspaceService

workspace = WorkspaceService.open("./My Project")

binary = np.zeros((480, 640), dtype=np.uint8)
binary[120:260, 180:340] = 1
mask = SegmentationMask.from_binary_mask(binary, class_name="cell")

workspace.segmentation.save_masks(frame_index=42, masks=[mask])
loaded = workspace.segmentation.load_masks(frame_index=42)
```

Pick the surface by intent:

| Task | Preferred entrypoint |
| --- | --- |
| Workspace lifecycle and service-bound imports | `xpkg.services.WorkspaceService` |
| Register tables, figures, analyses, reports, or stats | `workspace.artifacts.*` |
| Save/load figure outputs with lineage | `workspace.figures.*` |
| Save/load frame segmentation masks | `workspace.segmentation.*` |
| Function-level workspace imports | `xpkg.formats.import_*_workspace(...)` |
| Legacy `.xpkg` cutover | `xpkg migrate` or `xpkg.formats.migrate_legacy_archive(...)` |

## Lifecycle-only example

```python
from xpkg.services import WorkspaceService

workspace = WorkspaceService.open("./My Project")
layout = workspace.validate()
artifact = workspace.pack()
```

Once a workspace already exists, the same service object keeps validation,
packing, and reopen flows on the same public contract.

## Additional workspace import coverage

The same workspace-first pattern is available for:

- `import_dlc_h5_workspace(...)` and `import_dlc_project_workspace(...)`
- `import_lightning_pose_csv_workspace(...)`
- `import_sleap_h5_workspace(...)` and `import_sleap_package_workspace(...)`
- `import_mmpose_topdown_json_workspace(...)`
- `import_mediapipe_pose_landmarks_json_workspace(...)`

Use those workspace helpers as the primary integration surface for new code.
The underlying `xpkg.formats.import_*_workspace(...)` functions remain public
when you want the explicit function form.

## Legacy migration example

When you need to cut over an older `.xpkg` archive, use the explicit migration
helper instead of treating archive IO as a normal downstream contract.

```bash
xpkg migrate "./legacy.xpkg" --out "./My Project"
```

```python
from xpkg.formats import migrate_legacy_archive

snapshot_path = migrate_legacy_archive("./legacy.xpkg", "./My Project")
```

That migration writes workspace-native state into the private store and refreshes
the rebuildable `.xpkg/state/current.json` cache for normal workspace use.

## In-memory Exchange API

```python
from xpkg.exchange import labels_from_json_payload, labels_to_json_payload
from xpkg.model import Labels

labels = Labels()
payload = labels_to_json_payload(labels)
roundtripped = labels_from_json_payload(payload)
```

Use `xpkg.exchange` when another repo needs an in-memory handoff boundary rather
than a workspace path.
