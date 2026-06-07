# 3D Pose Design

`xpkg` should treat 3D pose as a coordinate-frame and provenance contract, not
as `z` added to every 2D label point.

## Design Decisions

- Use the existing `PoseTrajectory` view for skeletal or named-marker
  trajectories in either 2D or 3D. Do not add a separate `PoseTrajectory3D`
  unless runtime storage needs diverge from the current `(frames, keypoints,
  dims)` contract.
- Reserve a future `SpatialPointSeries` for non-skeletal point streams that are
  not naturally keypoint trajectories, such as unlabeled point clouds, force
  platform corners, or reconstructed landmarks without a stable skeleton.
- Keep `Labels`, `Point`, and `PredictedPoint` image-space and 2D. 3D importers
  should produce or adapt to trajectory/project-level records rather than
  widening every annotation point with `z`.

## Coordinate Frames

Every 3D trajectory needs an explicit coordinate-frame declaration. The minimum
frame vocabulary should distinguish:

- `image_pixel`: 2D image coordinates tied to one source video or camera.
- `camera`: 3D coordinates in one calibrated camera frame.
- `calibration_world`: triangulated or reconstructed coordinates in a recorded
  calibration world frame.
- `marker_world`: marker-based coordinates in the source tracking
  lab frame.
- `lifted_model`: model-lifted 3D predictions whose frame is learned or
  source-defined rather than directly calibrated.

Frame metadata should record the frame name, handedness/axis convention when
known, and a short description when the source format provides one. Unknown
axis conventions should stay unknown instead of being inferred.

## Units

Units belong at trajectory or coordinate-frame scope, not on individual point
fields. A trajectory should use one coordinate unit such as `px`, `mm`, or `m`.
If a source file mixes units, import should either split the data into separate
series or fail with a clear validation error.

## Calibration Provenance

3D coordinates that depend on camera calibration should reference calibration
provenance by project metadata identity, not duplicate calibration payloads on
each point. The reference should include:

- calibration artifact or metadata slot name
- camera names or camera group used
- calibration source tool and version when known
- imported source path/checksum when available

The existing `Calibration`, `CalibrationSource`, `WorldFrame`, and
`CalibrationQuality` models already cover the calibration side of this contract.

## Quality And Scores

Keep quality fields separate from coordinates:

- `valid` or `visibility` remains a boolean mask at frame/keypoint scope.
- `confidence` remains numeric source-tool confidence when provided.
- `reprojection_error_px` belongs in calibration quality summaries and, when a
  tool provides dense values, in a per-frame/keypoint/camera quality sidecar.
- source-tool metadata belongs in trajectory/project provenance, mirroring
  `PoseModelProvenance` and `CalibrationSource`, with row-level metadata only
  for source columns that are actually row-specific.

This keeps downstream consumers from mistaking a coordinate value for a quality
or provenance value.

## Importer Implication

Anipose, DANNCE, DeepFly3D, and other 3D importers should wait until this
contract has concrete runtime storage. They should preserve source-tool
metadata and calibration references, but they should not force 3D data through
the 2D `Labels` point schema.
