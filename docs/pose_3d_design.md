# 2D and 3D Pose Contract

`xpkg` represents skeletal 2D and 3D results with one `PoseTrajectory` type.
Dimensionality, units, coordinate frame, tracks, validity, confidence, and
provenance are explicit properties. `Labels` remains the annotation-rich 2D
image-space representation.

## Array contract

`PoseTrajectory.positions` has shape
`(frames, tracks, keypoints, dimensions)`. `dimensions` is 2 or 3.
`valid` and optional `confidence` have shape
`(frames, tracks, keypoints)`. `track_ids` and `keypoint_names` name the two
semantic axes.

The track axis is present even for single-animal data. Omitting it was rejected
because it makes multi-animal 3D results impossible to represent without a
second type or an untyped metadata convention.

## Coordinate frames

Every trajectory carries one `PoseCoordinateFrame`:

- `image_pixel` for 2D image coordinates
- `camera` for 3D coordinates in a calibrated camera frame
- `calibration_world` for triangulated coordinates in a calibration world
- `marker_world` for a source marker or laboratory frame
- `lifted_model` for learned or source-defined 3D coordinates

Two-dimensional trajectories must use `image_pixel`. Three-dimensional
trajectories cannot use `image_pixel`. Units belong to the coordinate frame,
not individual points. Axis convention and frame description remain unknown
when the source does not provide them.

## Calibration relationship

A camera-frame or calibration-world pose link names the `SessionCalibration`
that defines its geometry. The calibration object stores camera intrinsics,
extrinsics, world frame, source, and quality. `CalibrationCameraLink` connects
each calibrated camera to the corresponding acquisition `CameraMetadata`
object. Calibration payloads are not copied into pose points or project
sidecars.

## Identity relationship

Trajectory track IDs are technical identities. They are not biological subject
IDs. `SubjectTrackLink` assigns a participating `Subject` to a named pose and
track. Optional `IdentityProvenanceRecord` evidence records source, confidence
spans, swaps, and proofreading. The experiment rejects assignments to unknown
subjects, poses, or tracks.

## Quality and provenance

- `valid` indicates whether a coordinate is usable.
- `confidence` stores source-tool confidence when supplied.
- calibration reprojection error belongs to `CalibrationQuality`.
- pose-model details belong to `SessionPose.provenance`.
- source-specific row metadata belongs at row scope only when it is genuinely
  row-specific.

## Importer rule

Anipose, DANNCE, DeepFly3D, and other 3D importers must parse source arrays into
this contract at the boundary. They must preserve track IDs, keypoint names,
coordinate frame, units, calibration relationship, confidence, and model
provenance. They must fail when required spatial semantics cannot be
established.
