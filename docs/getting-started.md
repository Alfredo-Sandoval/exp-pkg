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

## Write and read your first archive

A `.siesta` file is an HDF5 archive that stores pose annotations, videos,
skeletons, and metrics in one archive.

```python
from posetta.formats import read_siesta, write_siesta
from posetta.model import Labels

labels = Labels()
write_siesta("empty.siesta", labels)

payload = read_siesta("empty.siesta", lazy=False)
loaded = payload["labels"]
assert isinstance(loaded, Labels)
```

`read_siesta` returns a dict with these keys:

| Key | Type | Contents |
| --- | --- | --- |
| `"labels"` | `Labels` | The main annotation container |
| `"metadata"` | `dict` | Archive-level metadata |
| `"videos"` | `list[Video]` | Video references |
| `"predictions"` | `dict` or `None` | Prediction payloads if present |

## Import your first external dataset

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
