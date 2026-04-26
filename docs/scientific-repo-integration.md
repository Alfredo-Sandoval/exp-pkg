# Scientific Repo Integration

`xpkg` is the shared workspace and artifact substrate for lab science repos.
It should make data, masks, tables, stats, figures, reports, and exports
portable and inspectable without becoming the domain-specific analysis brain.

The clean split is:

- `xpkg` owns workspace layout, import/export, portable state, artifact
  manifests, media references, provenance records, and pack/unpack behavior.
- Domain repos own scientific meaning: metrics, validation rules, event
  semantics, model choices, plotting code, and biological interpretation.

## Namespace Rule

When a repo stores app-specific outputs in a shared workspace, use an app
namespace:

```python
from xpkg.services import WorkspaceService

workspace = WorkspaceService.open("./Experiment Workspace")

workspace.figures.save(
    figure_id="summary",
    namespace="phrase",
    outputs={"figure.svg": "results/summary.svg"},
    inputs=[".xpkg/phrase/analysis/source_data.csv"],
    producer={
        "package": "phrase",
        "module": "phrase.figures.behavior.summary",
        "command": "phrase make-figures summary",
    },
)
```

Namespaced figures are stored under:

```text
.xpkg/<namespace>/figures/<figure_id>/
  manifest.json
  figure.svg
```

Without a namespace, xpkg stores generic artifacts under:

```text
.xpkg/artifacts/figures/<figure_id>/
```

The namespace only controls workspace organization. The manifest contract stays
the same, so a packed `.expkg` can still validate and export the lineage.

## Local Repo Map

These are the expected integration boundaries for the current lab repos.

### OpenOperant

Local checkout:

```text
/home/lab/Documents/Github/OpenOperant
```

Role:

- Owns operant-specific event semantics, chamber-object segmentation, cue-light
  traces, contact-window review, evidence fusion, and validation figures.
- Uses `xpkg` as the canonical workspace and portable artifact surface.

Use xpkg for:

- `PROJECT.json`, `.xpkg/`, `.expkg` pack/unpack.
- Imported upstream pose labels/tracks/predictions.
- Frame-level masks through `workspace.segmentation`.
- Validation figures through `workspace.figures.save(namespace="openoperant")`.
- Figure manifests that link figures back to reviewed events, labels, stats,
  source media, pose, and masks.

OpenOperant should continue to own the plotting script or figure service. After
rendering SVG/PDF/PNG/source-data files, it should register those files with
`workspace.figures.save(...)` instead of maintaining a parallel figure manifest
schema.

### FIESTA

Local checkout:

```text
/home/lab/Documents/Github/fiesta
```

Role:

- Owns SAM2/SAM3 segmentation and tracking runtime behavior.
- Produces masks, overlays, run summaries, benchmark outputs, and runtime
  comparison figures.

Use xpkg for:

- Workspace-backed segmentation masks through `workspace.segmentation`.
- Portable figure artifacts through `workspace.figures.save(namespace="fiesta")`.
- Registered visualization outputs for segmentation benchmarks and runtime
  reports.
- Future generic run/report artifacts for non-figure benchmark bundles.

FIESTA should keep model/runtime decisions inside FIESTA. `xpkg` should record
what was produced, from which inputs, by which command/config/commit.

### Siesta

Local checkout:

```text
/home/lab/Documents/Github/siesta
```

Role:

- Owns pose GUI orchestration, training/inference backends, model artifact
  policy, and project UX.
- Already treats xpkg workspaces as the editable project format.

Use xpkg for:

- Project identity and workspace lifecycle.
- Labels, predictions, workspace metadata, and `.expkg` export.
- QC figures through `workspace.figures.save(namespace="siesta")`.
- Future model/report artifact registration for training bundles and inference
  profiles.

Siesta training checkpoints remain Siesta-owned model artifacts. `xpkg` should
register their provenance and exported summaries; it should not decide model
selection policy.

### PHRASE

Local checkout:

```text
/home/lab/Documents/Github/most_recent_blackbox-external_drive/phrase
```

Role:

- Owns pain/gait feature semantics, biological interpretation, behavior-state
  summaries, figure scripts, and PHRASE-specific manifests.

Use xpkg for:

- Recording/workspace identity when PHRASE data are packaged for reuse.
- Paw and body segmentation masks through `workspace.segmentation` when masks
  are frame-bound.
- Paper and report figures through `workspace.figures.save(namespace="phrase")`.
- Source data and stats files linked from each figure manifest.

PHRASE may keep domain manifests for local analysis while migrating the
portable, claim-carrying output registry into `.xpkg/phrase/...`.

### Vicon / Kinematics

Local checkout found:

```text
/home/lab/Documents/Github/most_recent_blackbox-external_drive/sensory_vicon
```

Role:

- Owns gait-event detection, spatiotemporal metrics, skeleton visualization,
  and kinematics interpretation.

Use xpkg for:

- Vicon CSV/C3D import through `workspace.imports.vicon(...)` or
  `xpkg.formats.import_vicon_*_workspace(...)`.
- Portable Vicon recording state and sidecars.
- Kinematics overview figures through `workspace.figures.save(namespace="vicon")`
  or a repo-specific namespace such as `sensory_vicon`.
- Future table/report artifacts for spatiotemporal metrics and diagnostics.

The local `tempo` repo was not found in the scanned checkout tree. It should
use the same pattern once its workspace root is known.

## Output-Type Contract

Use this rule of thumb when wiring repos into xpkg:

- Raw/imported data: xpkg import adapters and workspace state.
- Labels, predictions, pose, tracking, and frame masks: xpkg managed state.
- Domain analysis tables: domain repo computes them; xpkg stores or registers
  them as claim-carrying artifacts.
- Stats reports: domain repo chooses the model/test; xpkg records inputs,
  command, software context, and outputs.
- Figures: domain repo renders; xpkg registers the figure outputs, source data,
  stats, inputs, and producer metadata.
- Reports/manuscripts/export bundles: xpkg packages and validates lineage;
  domain repos own narrative and interpretation.

## Minimal Figure Manifest

Every saved figure artifact should answer:

- What is the stable figure id?
- Which output files are part of the artifact?
- Which input files and source-data tables support the claim?
- Which stats reports are linked?
- Which package/module/command produced it?
- Which app namespace owns the scientific semantics?

That is the shared contract that lets OpenOperant, FIESTA, Siesta, PHRASE, and
Vicon/Kinematics all work inside the same workspace without making xpkg a
plotting library.
