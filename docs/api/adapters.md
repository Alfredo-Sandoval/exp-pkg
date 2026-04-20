# Adapters

<div class="page-intro">
<p>
<code>xpkg.adapters</code> converts DLC and SLEAP tracking into xpkg data
structures and current <code>.xpkg</code> archive outputs while the
workspace-first v1 project workflow is being wired in.
</p>
</div>

!!! note
    The locked public artifact contract is workspace folder +
    <code>.expkg</code>. The adapter functions documented here currently emit
    edge <code>.xpkg</code> archive outputs.

## Return Type

### `ConversionResult`

All adapter functions return a `ConversionResult` (or a list of them).

The returned object includes the original source path, the output project
directory, associated videos, and the emitted `.xpkg` archive path.

## DeepLabCut

### `convert_dlc_csv(csv_path, video_path, out_path, *, skeleton_name="imported", likelihood_threshold=0.0, progress_callback=None) -> ConversionResult`

Convert one DLC CSV tracking file and its matching video into a direct
`.xpkg` archive.

### `convert_dlc_h5(h5_path, video_path, out_path, *, skeleton_name="imported", likelihood_threshold=0.0, progress_callback=None) -> ConversionResult`

Convert one DLC-style H5 tracking file and its matching video.

### `convert_dlc_project(project_dir, out_dir, *, likelihood_threshold=0.0, progress_callback=None) -> list[ConversionResult]`

Convert a DLC project directory by scanning its `labeled-data/` and `videos/`
subdirectories.

Example:

```python
from xpkg.adapters import convert_dlc_csv

result = convert_dlc_csv(
    "CollectedData_mouse.csv",
    "session.mp4",
    "session.xpkg",
    skeleton_name="mouse_topdown",
    likelihood_threshold=0.2,
)

print(result.project_root)
```

## SLEAP

### `convert_sleap_h5(h5_path, video_path, out_path, *, skeleton_name="imported", likelihood_threshold=0.0, progress_callback=None) -> ConversionResult`

Convert a SLEAP analysis H5 tracking export and its matching video into a
direct `.xpkg` archive.

### `convert_sleap_package(slp, out_dir, *, fps=30, encode_videos=None, progress_callback=None) -> ConversionResult`

Convert a SLEAP `.pkg.slp` archive into a current `.xpkg` archive output.

- `fps` controls MP4 encoding frame rate when videos are emitted.
- `encode_videos=False` keeps extracted frame sequences instead of MP4 output.

Example:

```python
from xpkg.adapters import convert_sleap_h5

result = convert_sleap_h5(
    "analysis.h5",
    "session.mp4",
    "analysis.xpkg",
    skeleton_name="mouse_topdown",
    likelihood_threshold=0.1,
)
```

Example:

```python
from xpkg.adapters import convert_sleap_package

result = convert_sleap_package(
    "labels.pkg.slp",
    "sleap_export",
    fps=30,
    encode_videos=False,
)
```

## Progress Reporting

Every adapter accepts an optional `progress_callback: Callable[[str], None]`.
The callback receives short status strings during conversion (e.g.
`"Converting frame 10/500"`).

```python
from xpkg.adapters import convert_dlc_csv

def on_progress(msg: str) -> None:
    print(msg)

result = convert_dlc_csv(
    "tracking.csv", "video.mp4", "out.xpkg",
    progress_callback=on_progress,
)
```
