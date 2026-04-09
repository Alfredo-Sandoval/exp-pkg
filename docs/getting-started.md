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

For the documentation toolchain:

```bash
mamba run -n xpkg uv pip install -e '.[docs]'
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
- `.sta` is the canonical archive suffix for the compatibility layer.
- `.siesta` remains a legacy alias during transition.

This matters because the workspace is the place where media, segmentation,
labels, and future experiment-side modalities can live together under one
contract.

Read [Artifact Contract v1](artifact_contract_v1.md) for the full public
contract and [CLI Command Spec v1](cli_command_spec_v1.md) for the locked
workspace command surface.

## Edge compatibility API

```python
from xpkg.compat import read_sta, write_sta
from xpkg.model import Labels

labels = Labels()
write_sta("empty.sta", labels)

payload = read_sta("empty.sta", lazy=False)
loaded = payload["labels"]
assert isinstance(loaded, Labels)
```

These compatibility helpers remain useful for legacy import/read paths,
fixtures, and transition work. `read_sta` returns a dict with these keys:

| Key | Type | Contents |
| --- | --- | --- |
| `"labels"` | `Labels` | The main annotation container |
| `"metadata"` | `dict` | Archive-level metadata |
| `"videos"` | `list[Video]` | Video references |
| `"predictions"` | `dict` or `None` | Prediction payloads if present |

## Current adapter import example

```python
from xpkg.adapters import convert_dlc_csv

result = convert_dlc_csv(
    "tracking.csv",
    "video.mp4",
    "tracking.sta",
    skeleton_name="mouse",
    likelihood_threshold=0.25,
)

print(result.bundle_path)
```

This is still a compatibility-oriented import example. New project-facing code
should think in terms of workspace creation, import into a workspace, and
portable `.expkg` export rather than a direct archive-first workflow.
