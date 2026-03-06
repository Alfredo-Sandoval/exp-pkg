---
hide:
  - toc
---

<div class="manual-head" markdown="1">

<div class="manual-kicker">POSE IO LIBRARY</div>

# Posetta

<p class="manual-deck">
Posetta reads and writes <code>.siesta</code> bundles (HDF5-based pose archives)
and converts DLC and SLEAP tracking into that format.
Three modules: <code>posetta.model</code>, <code>posetta.formats</code>, and
<code>posetta.adapters</code>.
</p>

</div>

<div class="spec-grid spec-grid-2" markdown="1">

<div class="spec-panel" markdown="1">
### At a Glance

| Item | Value |
| --- | --- |
| Native format | `.siesta` (HDF5 archive) |
| External adapters | DLC, SLEAP |
| Pose objects | `posetta.model` |
| Bundle IO | `posetta.formats` |
| Import tools | `posetta.adapters` |
</div>

<div class="spec-panel" markdown="1">
### Choose by Task

- Use `posetta.model` when you need `Labels`, `Skeleton`, `Instance`, or `Video`.
- Use `posetta.formats` when you need to read, write, or update `.siesta`.
- Use `posetta.adapters` when you need to import DLC or SLEAP.
</div>

</div>

## Current Coverage

<div class="spec-grid spec-grid-3" markdown="1">

<div class="spec-panel" markdown="1">
### Model

- `Labels`
- `LabeledFrame`
- `Instance`, `PredictedInstance`
- `Point`, `PredictedPoint`
- `Skeleton`, `Keypoint`, `Track`
- `Video`
</div>

<div class="spec-panel" markdown="1">
### Native Bundle IO

- `read_siesta`
- `write_siesta`
- `update_labels_siesta`
- prediction append and merge
- metrics table IO
- validation and summary
</div>

<div class="spec-panel" markdown="1">
### Adapters

- `convert_dlc_csv`
- `convert_dlc_h5`
- `convert_dlc_project`
- `convert_sleap_package`
- `ConversionResult`
</div>

</div>

## Minimal Roundtrip

```python
from posetta.formats import read_siesta, write_siesta
from posetta.model import Labels

labels = Labels()
write_siesta("empty.siesta", labels)

payload = read_siesta("empty.siesta", lazy=False)
loaded = payload["labels"]
assert isinstance(loaded, Labels)
```

## Navigation

<div class="quick-links" markdown="1">

- Start with [Getting Started](getting-started.md) for install and first-use examples.
- Read [Model](api/model.md) for the pose object graph.
- Read [Formats](api/formats.md) for native `.siesta` operations.
- Read [Adapters](api/adapters.md) for DLC and SLEAP conversion.
- Use the reference pages when you need exact signatures and docstrings.

</div>
