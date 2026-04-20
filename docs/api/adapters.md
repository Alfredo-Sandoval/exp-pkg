# Adapters

<div class="page-intro">
<p>
<code>xpkg.adapters</code> converts DeepLabCut, SLEAP, MMPose, MediaPipe,
OpenPose, and Detectron2 pose exports into xpkg data structures and
compatibility <code>.xpkg</code> archive outputs.
</p>
</div>

!!! note
    The locked public artifact contract is workspace folder +
    <code>.expkg</code>. The adapter functions documented here currently emit
    edge <code>.xpkg</code> archive outputs. Project-facing code should prefer
    the matching <code>xpkg.formats.import_*_workspace(...)</code> helpers
    whenever a workspace lifecycle is the real goal. Each compatibility adapter
    here has a workspace-first counterpart on <code>xpkg.formats</code>.

## Preferred Project Path

If you are wiring a downstream repo into xpkg for the first time, start with
<code>WorkspaceService</code> and keep the import on the same service object:

```python
from xpkg.services import WorkspaceService

workspace = WorkspaceService.create("./My Project", title="My Project")
workspace.imports.dlc_csv(
    "tracking.csv",
    "video.mp4",
    skeleton_name="mouse",
)
artifact = workspace.pack()
```

The functions below remain public for explicit direct-archive compatibility
workflows.

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

## MMPose

### `convert_mmpose_topdown_json(json_path, video_path, out_path, *, skeleton_name="imported", instance_index=0, likelihood_threshold=0.0, progress_callback=None) -> ConversionResult`

Convert an official MMPose top-down demo JSON export written by
`topdown_demo_with_mmdet.py --save-predictions` plus its matching video into a
direct `.xpkg` archive.

- `instance_index` selects one per-frame prediction slot from the saved result.

## MediaPipe

### `convert_mediapipe_pose_landmarks_json(json_path, video_path, out_path, *, skeleton_name="mediapipe_pose", likelihood_threshold=0.0, progress_callback=None) -> ConversionResult`

Convert the supported serialized MediaPipe pose-landmarks JSON contract plus
its matching video into a direct `.xpkg` archive.

- The minimal supported contract is a single-pose JSON export derived from the
  official Pose Landmarker result fields.

## OpenPose

### `convert_openpose_json(json_dir, video_path, out_path, *, skeleton_name="imported", likelihood_threshold=0.0, progress_callback=None) -> ConversionResult`

Convert an OpenPose `--write_json` BODY_25 directory plus its matching video
into a direct `.xpkg` archive.

## Detectron2

### `convert_detectron2_coco(predictions_path, dataset_json_path, image_root, out_path, *, category_id=None, skeleton_name=None, likelihood_threshold=0.0, progress_callback=None) -> ConversionResult`

Convert Detectron2 COCO keypoint predictions written by `COCOEvaluator`
(`coco_instances_results.json`) plus the paired COCO dataset JSON and
`image_root` into a direct `.xpkg` archive.

- `category_id` is required when the dataset JSON defines more than one
  keypoint category.

## Progress Reporting

Every adapter accepts an optional
`progress_callback: Callable[[int, str], None]`. The callback receives a
best-effort percentage plus the underlying converter status string.

```python
from xpkg.adapters import convert_dlc_csv

def on_progress(percent: int, msg: str) -> None:
    print(percent, msg)

result = convert_dlc_csv(
    "tracking.csv", "video.mp4", "out.xpkg",
    progress_callback=on_progress,
)
```
