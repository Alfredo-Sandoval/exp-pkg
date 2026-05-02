# Adapters

<div class="page-intro">
<p>
<code>xpkg.adapters</code> is the in-memory conversion surface for canonical
xpkg objects. It exists for downstream tools that need arrays, tables, or
JSON-friendly payloads without coupling to workspace layout or archive-format
storage tooling.
</p>
</div>

!!! note
    Use <code>xpkg.model</code> for the object graph and
    <code>xpkg.services</code> or <code>xpkg.workspace</code> for the
    workspace/project contract.
    Use <code>xpkg.adapters</code> when you want pure in-memory conversions.

## Current Surface

### `labels_numpy(labels, *, video=None, all_frames=True, untracked=False, return_confidence=False)`

Convert labels into a dense NumPy tensor.

### `labels_to_dataframe(labels, *, video=None, scorer="xpkg")`

Convert labels into a tabular DeepLabCut-style dataframe.

### `labels_to_json_payload(labels, *, metadata=None)`

Convert labels into the canonical JSON-friendly payload document used by xpkg's
JSON interchange path.

### `labels_from_json_payload(document_or_payload)`

Hydrate `Labels` back from either a full JSON interchange document or its inner
payload mapping.

## Example

```python
from xpkg.adapters import labels_from_json_payload, labels_to_json_payload
from xpkg.model import Labels

labels = Labels()
payload = labels_to_json_payload(labels)
roundtripped = labels_from_json_payload(payload)
```

The important boundary is that none of these helpers require a workspace root
or portable `.expkg` artifact.
