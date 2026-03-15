# Getting Started

<div class="page-intro">
<p>
Posetta is not on PyPI yet. Clone the repo and install locally.
</p>
</div>

## Install

```bash
git clone https://github.com/Alfredo-Sandoval/Posetta.git
cd Posetta
pip install -e .
```

For the documentation toolchain:

```bash
pip install -e '.[docs]'
```

## Preview the docs locally

```bash
make docs-build
make docs-serve
```

## Start with the v1 artifact model

The public Posetta v1 artifact model is workspace-first:

```text
My Project/
  PROJECT.json
  .posetta/
  Media/
  Exports/
    My Project.poseproj
```

- You edit a normal workspace folder.
- Posetta owns authoritative mutable state inside `.posetta/`.
- You move/share/export a single `.poseproj` file.
- `.siesta` remains a legacy import/read compatibility format during
  transition.

Read [Artifact Contract v1](artifact_contract_v1.md) for the full public
contract and [CLI Command Spec v1](cli_command_spec_v1.md) for the locked
workspace command surface.

## Current compatibility API

```python
from posetta.formats import read_siesta, write_siesta
from posetta.model import Labels

labels = Labels()
write_siesta("empty.siesta", labels)

payload = read_siesta("empty.siesta", lazy=False)
loaded = payload["labels"]
assert isinstance(loaded, Labels)
```

These `.siesta` helpers remain useful for legacy import/read paths, fixtures,
and transition work. `read_siesta` returns a dict with these keys:

| Key | Type | Contents |
| --- | --- | --- |
| `"labels"` | `Labels` | The main annotation container |
| `"metadata"` | `dict` | Archive-level metadata |
| `"videos"` | `list[Video]` | Video references |
| `"predictions"` | `dict` or `None` | Prediction payloads if present |

## Current adapter import example

```python
from posetta.adapters import convert_dlc_csv

result = convert_dlc_csv(
    "tracking.csv",
    "video.mp4",
    "tracking.siesta",
    skeleton_name="mouse",
    likelihood_threshold=0.25,
)

print(result.siesta_path)
```
