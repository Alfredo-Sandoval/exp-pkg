# Artifact Namespaces

`xpkg` supports optional artifact namespaces so multiple downstream packages can
write outputs into the same workspace without colliding. A namespace is an
ordinary caller-owned string. It is not a plugin registry, package registry, or
list of names known to xpkg.

The split is:

- `xpkg` owns workspace layout, import/export, portable state, artifact
  manifests, media references, provenance records, and pack/unpack behavior.
- Downstream packages own scientific or domain-specific meaning: metrics,
  validation rules, event semantics, model choices, plotting code, and
  interpretation.

## Generic and Namespaced Artifacts

Without a namespace, artifacts are stored in the generic artifact registry:

```text
.xpkg/artifacts/<kind>/<artifact_id>/
  manifest.json
  output-file.ext
```

With a namespace, artifacts are stored under the caller's namespace:

```text
.xpkg/<namespace>/<kind>/<artifact_id>/
  manifest.json
  output-file.ext
```

Common kinds use readable plural directories:

| Artifact type | Directory |
| --- | --- |
| `figure` | `figures` |
| `table` | `tables` |
| `analysis` | `analyses` |
| `report` | `reports` |
| `stats-report` | `stats-reports` |

For example:

```python
from xpkg.services import WorkspaceService

workspace = WorkspaceService.open("./Experiment Workspace")

workspace.artifacts.register(
    artifact_id="session_summary_figure",
    artifact_type="figure",
    namespace="neuro-analysis",
    outputs={"figure.svg": "results/session_summary_figure.svg"},
    inputs=[".xpkg/neuro-analysis/analysis/source_data.csv"],
    producer={
        "package": "neuro-analysis",
        "module": "neuro_analysis.figures.validation",
        "command": "neuro-analysis make-figures validation",
    },
)
```

That writes:

```text
.xpkg/neuro-analysis/figures/session-summary-figure/
  manifest.json
  figure.svg
```

For figures, the convenience API is equivalent:

```python
workspace.figures.save(
    figure_id="session_summary_figure",
    namespace="neuro-analysis",
    outputs={"figure.svg": "results/session_summary_figure.svg"},
)
```

## Namespace Rules

Namespace values are normalized into path-safe slugs. A caller may choose names
like `neuro-analysis`, `review-ui`, or `qc-runner`; xpkg treats them all the same.

Rules:

- Namespaces are optional.
- Namespaces only affect artifact placement inside `.xpkg/`.
- Namespaces do not change the artifact manifest schema.
- Namespaces are not hard-coded, reserved, discovered, or interpreted by xpkg.
- If the same artifact id exists in multiple namespaces or kinds, callers must
  pass `namespace=...` and/or `kind=...` when loading or validating it.

## Output-Type Contract

Use this rule of thumb when wiring downstream packages into xpkg:

- Raw/imported data: xpkg import adapters and workspace state.
- Labels, predictions, pose, tracking, and frame masks: xpkg managed state.
- Domain analysis tables: the downstream package computes them; xpkg stores or
  registers them as claim-carrying artifacts.
- Stats reports: the downstream package chooses the model/test; xpkg records
  inputs, command, software context, and outputs.
- Figures: the downstream package renders; xpkg registers the figure outputs,
  source data, stats, inputs, and producer metadata.
- Reports/manuscripts/export bundles: xpkg packages and validates lineage; the
  downstream package owns narrative and interpretation.

## Minimal Artifact Manifest

Every saved artifact should answer:

- What is the stable figure id?
- Which output files are part of the artifact?
- Which input files and source-data tables support the claim?
- Which stats reports are linked?
- Which package/module/command produced it?
- Which caller-owned namespace, if any, owns the domain semantics?

That contract lets independent packages share one workspace while keeping xpkg
as a portable artifact system rather than a domain-specific analysis library.

## Workspace Index

xpkg maintains a compact index at:

```text
.xpkg/artifacts/index.json
```

The index is for discovery and CLI listing. The individual `manifest.json`
files remain authoritative, so the index can be rebuilt:

```bash
xpkg artifacts rebuild-index "./Experiment Workspace"
```
