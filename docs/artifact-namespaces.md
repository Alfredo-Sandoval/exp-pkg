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

## Generic and Namespaced Figures

Without a namespace, figures are stored in the generic artifact registry:

```text
.xpkg/artifacts/figures/<figure_id>/
  manifest.json
  figure.svg
```

With a namespace, figures are stored under the caller's namespace:

```text
.xpkg/<namespace>/figures/<figure_id>/
  manifest.json
  figure.svg
```

For example:

```python
from xpkg.services import WorkspaceService

workspace = WorkspaceService.open("./Experiment Workspace")

workspace.figures.save(
    figure_id="validation_figure_3",
    namespace="analysis-app",
    outputs={"figure.svg": "results/validation_figure_3.svg"},
    inputs=[".xpkg/analysis-app/analysis/source_data.csv"],
    producer={
        "package": "analysis-app",
        "module": "analysis_app.figures.validation",
        "command": "analysis-app make-figures validation",
    },
)
```

That writes:

```text
.xpkg/analysis-app/figures/validation-figure-3/
  manifest.json
  figure.svg
```

## Namespace Rules

Namespace values are normalized into path-safe slugs. A caller may choose names
like `analysis-app`, `review-ui`, or `qc-runner`; xpkg treats them all the same.

Rules:

- Namespaces are optional.
- Namespaces only affect artifact placement inside `.xpkg/`.
- Namespaces do not change the figure manifest schema.
- Namespaces are not hard-coded, reserved, discovered, or interpreted by xpkg.
- If the same `figure_id` exists in multiple namespaces, callers must pass
  `namespace=...` when loading or validating that figure.

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

## Minimal Figure Manifest

Every saved figure artifact should answer:

- What is the stable figure id?
- Which output files are part of the artifact?
- Which input files and source-data tables support the claim?
- Which stats reports are linked?
- Which package/module/command produced it?
- Which caller-owned namespace, if any, owns the domain semantics?

That contract lets independent packages share one workspace while keeping xpkg
as a portable artifact system rather than a domain-specific analysis library.
