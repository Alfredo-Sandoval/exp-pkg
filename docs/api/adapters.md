# Adapters

<div class="page-intro">
<p>
<code>posetta.adapters</code> converts DLC and SLEAP tracking into native
<code>.siesta</code> bundles.
</p>
</div>

## Return Type

### `ConversionResult`

All adapter functions return a `ConversionResult` (or a list of them).

Fields:

- `source_dir`: the original source path or directory
- `project_root`: the output project directory
- `videos`: output video paths associated with the conversion
- `siesta_path`: the main `.siesta` bundle path

## DeepLabCut

### `convert_dlc_csv(csv_path, video_path, out_path, *, skeleton_name="imported", likelihood_threshold=0.0, progress_callback=None) -> ConversionResult`

Convert one DLC CSV tracking file and its matching video into a `.siesta`
bundle.

### `convert_dlc_h5(h5_path, video_path, out_path, *, skeleton_name="imported", likelihood_threshold=0.0, progress_callback=None) -> ConversionResult`

Convert one DLC-style H5 tracking file and its matching video.

### `convert_dlc_project(project_dir, out_dir, *, likelihood_threshold=0.0, progress_callback=None) -> list[ConversionResult]`

Convert a DLC project directory by scanning its `labeled-data/` and `videos/`
subdirectories.

Example:

```python
from posetta.adapters import convert_dlc_csv

result = convert_dlc_csv(
    "CollectedData_mouse.csv",
    "session.mp4",
    "session.siesta",
    skeleton_name="mouse_topdown",
    likelihood_threshold=0.2,
)

print(result.siesta_path)
```

## SLEAP

### `convert_sleap_package(slp, out_dir, *, fps=30, encode_videos=None, progress_callback=None) -> ConversionResult`

Convert a SLEAP `.pkg.slp` archive into a `.siesta` project.

- `fps` controls MP4 encoding frame rate when videos are emitted.
- `encode_videos=False` keeps extracted frame sequences instead of MP4 output.

Example:

```python
from posetta.adapters import convert_sleap_package

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
from posetta.adapters import convert_dlc_csv

def on_progress(msg: str) -> None:
    print(msg)

result = convert_dlc_csv(
    "tracking.csv", "video.mp4", "out.siesta",
    progress_callback=on_progress,
)
```
