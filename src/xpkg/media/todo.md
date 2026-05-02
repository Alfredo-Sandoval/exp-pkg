# Media TODO

This package owns exp-pkg's canonical media primitives. Media handling belongs
here because downstream projects should not each reinvent video opening, image
sequence handling, frame indexing, color policy, and media export rules.

## Current Scope

- Keep image decode/color policy in `images.py`.
- Keep video and image-sequence primitives in `video.py`.
- Treat media paths as project-relative wherever possible.
- Keep OpenCV/imageio details behind this package boundary.

## Near-Term Cleanup

- Split `video.py` into smaller modules once the public contract is clearer:
  `types.py`, `readers.py`, `writers.py`, and `transforms.py` are likely
  candidates.
- Decide whether `Video` remains the public class name or becomes a lighter
  descriptor plus explicit reader objects.
- Add focused media tests for image sequences, single-image videos, color
  conversion, writer round trips, and missing/corrupt files.
- Move any remaining media primitives out of `xpkg.io`.
- Keep domain-specific measurements, such as cue-light or masked-brightness
  extraction, out of this package unless they become generic media operations.

## Contract Questions

- What is the canonical frame color order returned by `Video.get_frame()`?
- Should image sequences support mixed extensions or require one detected type?
- How should media hashes, dimensions, frame counts, and fps be cached in project
  state?
- Which operations are pure media IO, and which belong to downstream analysis
  packages?
- What optional backends should exist beyond OpenCV/imageio?

## Not Here

- Behavioral event inference from video.
- OpenOperant-specific cue-light extraction.
- Photometry analysis.
- Pose or segmentation model inference.
- Project packaging or durable-store state.
