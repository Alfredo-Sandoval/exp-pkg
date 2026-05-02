# Canonical Media IO Stack

<div class="page-intro">
<p>
xpkg should own the canonical media IO layer for multimodal neuroscience
workflows. Downstream GUI apps should consume that layer and keep only
GUI/runtime orchestration that is truly app-specific.
</p>
</div>

## Status

This document is the target architecture for media ownership after the current
adapter and workspace IO cutover work. It is intentionally opinionated: the goal
is to stop splitting core media behavior across two repositories.

## Governing Idea

If xpkg is the IO layer, it must own media IO as a first-class subsystem
alongside labels, manifests, and converters. That means file videos, image
sequences, frame decode, frame encode, media capability discovery, and
deterministic backend behavior all belong in xpkg.

Downstream GUI apps should remain responsible for application behavior:

- GUI playback scheduling
- Qt worker orchestration
- live-session buffering and prefetch policy
- user settings and app preferences
- progress reporting wired to interactive workflows

The key rule is simple: optimize backends in xpkg, not semantics in downstream GUI apps.

## Why This Spec Exists

Today the repositories are split in an unhealthy way:

- xpkg already owns canonical labels IO, workspace state IO, and external
  adapter logic.
- xpkg also has a small media layer in `xpkg.io.video`.
- downstream GUI apps still own a richer and partially duplicated media stack for playback,
  writer selection, backend routing, and threaded export.

That duplication creates three recurring problems:

1. Ownership is unclear. It is not obvious which repository defines the true
   behavior of `Video`, `VideoReader`, or `write_video`.
2. Performance work lands in the wrong place. Faster CUDA or macOS paths added
   only in downstream GUI apps do not help the canonical IO layer.
3. Semantic drift becomes likely. Two stacks that both claim to read or write
   the same media will eventually disagree on indexing, colors, exact seek
   rules, error behavior, or writer defaults.

The fix is not more wrappers. The fix is one canonical media stack in xpkg
with explicit app-level consumers in downstream GUI apps.

## Ownership Boundary

| Concern | Owner | Notes |
| --- | --- | --- |
| File video open/close | xpkg | Includes path normalization and media validation. |
| Image sequence handling | xpkg | Directory and single-image media should use the same canonical contract. |
| Frame indexing, decode, and seek behavior | xpkg | Exact semantics must be defined once. |
| Batch frame access and frame iteration | xpkg | Converters and apps should share the same read primitives. |
| Video writing and codec capability probing | xpkg | Includes CPU and hardware-accelerated paths. |
| Media backend capability discovery | xpkg | No hidden fallback or app-only backend rules. |
| Headless transcode/export helpers | xpkg | Reusable outside the GUI. |
| Qt worker threads and interactive progress UI | downstream GUI app | App orchestration, not canonical IO. |
| Live playback cache and session buffering | downstream GUI app | UI responsiveness policy stays app-side. |
| User preferences about app behavior | downstream GUI app | Preferences should map onto xpkg capability requests. |

One important split inside downstream GUI applications deserves to stay
explicit: shared decode/container leasing is a media concern, but live
latest-frame hub state is an application concern. If container sharing remains
necessary for performance, the headless leasing primitive should move into
xpkg. Session-level live frame buffers should remain downstream.

## Canonical Public Contract

Near-term, xpkg should keep the existing public names to avoid needless API
churn:

- `Video`
- `VideoReader`
- `VideoWriter`
- `write_video`
- `gui_playback_backend_for_path`

The implementation underneath those names should move to a dedicated media
subsystem, but callers should not need to learn a second naming scheme during
the migration.

### Media Inputs

The canonical contract should treat these as first-class inputs:

- file-backed videos such as `.mp4`, `.avi`, `.mov`, and `.mkv`
- image-sequence directories
- single images treated as one-frame videos

Every input type must normalize into the same semantic model:

- `frames`
- `fps`
- `width`
- `height`
- `channels`
- `get_frame(i)`
- `iter_frames()`

### Frame Semantics

These rules should be backend-independent:

- Frame indices are zero-based.
- `Video.frames` is authoritative for valid index range.
- `get_frame(i)` must fail fast on invalid indices.
- Canonical in-memory frames are `numpy.ndarray` values in `uint8`.
- Default color order is BGR for compatibility with the current stack.
- Grayscale frames use shape `H x W x 1`.
- Image sequences and file videos must obey the same indexing rules.
- Exact and approximate seek behavior must be explicit, never inferred.

If a backend internally uses RGB, planar YUV, GPU surfaces, or hardware decoder
frames, that is an implementation detail. The public contract must still present
the canonical frame representation unless and until a separate explicit GPU API
is added.

### Writer Semantics

Writer behavior also needs a stable contract:

- explicit output path
- explicit output fps
- explicit output dimensions derived from the first written frame
- explicit backend choice or explicit `auto`
- deterministic frame ordering
- fail-fast behavior when a requested codec or hardware path is unavailable

`auto` is allowed to select the best validated backend for the host, but an
explicit backend request must never silently degrade to something else.

## Backend Model

xpkg should expose one semantic contract with multiple backend
implementations.

### Required backends

| Backend | Purpose | Platforms | Role |
| --- | --- | --- | --- |
| `images` | Image sequences and single-image videos | macOS, Linux | Reference support for still-frame workflows |
| `opencv` | Baseline portable decode/write path | macOS, Linux | Reference backend and broad compatibility |
| `ffmpeg-cpu` | Better codec coverage on CPU | macOS, Linux | Deterministic codec-heavy fallback |
| `cuda` | Hardware decode/encode on NVIDIA systems | Linux first | High-throughput path for large video workloads |
| `macos` | Apple media acceleration | macOS | Native optimized path for Apple hardware |

PyAV may still be used internally where appropriate, but it should be treated
as an implementation choice, not the final architecture boundary.

### Capability discovery

xpkg should add explicit capability reporting so callers can reason about the
host instead of guessing:

- which backends are available
- whether hardware decode is available
- whether hardware encode is available
- supported output codecs
- whether exact random seek is validated
- whether shared-container decode is supported

The important design point is that capability discovery belongs to xpkg.
Downstream GUI apps should not maintain their own independent truth about which
video backends exist or how they behave.

## Performance Requirements

The target stack should be optimized for two hardware families:

- NVIDIA/CUDA systems on Linux
- macOS systems using Apple media acceleration

That optimization work belongs in xpkg because the same fast path should
benefit:

- DLC/SLEAP conversion
- workspace import/export
- offline frame sampling
- future benchmarking tools
- GUI tools and other downstream applications

### CUDA goals

The CUDA backend should prioritize:

- fast sequential decode for long videos
- efficient sparse/random frame access when supported
- NVDEC/NVENC-backed transcode and export paths
- low-copy host handoff into canonical `numpy` frames
- explicit failure when requested hardware capabilities are not available

### macOS goals

The macOS backend should prioritize:

- hardware-assisted decode and encode
- strong performance on Apple Silicon laptops and desktops
- robust handling of common camera/container formats used in labs
- parity with the canonical frame contract used on Linux

### Reference-backend rule

Every accelerated backend must be checked against a portable reference backend.
Performance can vary by host, but semantics may not.

## Validation And Quality Gates

xpkg should own a backend conformance suite before downstream GUI apps delete
their duplicate stack.

### Required conformance checks

- frame count parity
- fps parity
- width/height/channel parity
- exact frame seek parity
- image sequence parity
- single-image parity
- color order parity
- grayscale shape parity
- writer roundtrip sanity checks
- clear failure behavior for unsupported codecs or missing accelerators

### Required benchmark coverage

- sequential decode throughput
- sparse frame sampling latency
- exact random seek latency
- encode throughput
- end-to-end transcode/export throughput

Accelerated backends should only become the default `auto` choice after they
pass conformance and beat the reference backend on the workload they are meant
to accelerate.

## Migration Plan

### Phase 1: Make xpkg authoritative

- Keep `Video`, `VideoReader`, `VideoWriter`, and `write_video` as the public
  surface in xpkg.
- Move or reimplement the generic backend logic currently duplicated in downstream GUI apps.
- Add capability discovery and explicit backend naming in xpkg.
- Keep downstream GUI apps consuming their current stack until xpkg parity is demonstrated.

### Phase 2: Port generic performance features

- Move generic image-sequence handling, exact-seek behavior, and writer
  capability logic into xpkg.
- Add accelerated Linux/CUDA and macOS backends behind the same contract.
- Add conformance tests that compare all supported backends to the reference
  backend.

### Phase 3: Collapse Downstream Duplication

- Switch downstream GUI app imports from local generic media modules to xpkg.
- Delete or shrink downstream GUI app modules that only duplicate canonical media logic.
- Keep only app-specific orchestration in downstream GUI apps.

## What Needs To Change Downstream

Downstream GUI apps should move from owning a media engine to consuming one.

### Modules that should stop owning canonical media behavior

These modules should be deleted, collapsed, or reduced to temporary adapters
after xpkg reaches feature parity:

- `<gui-app>/io/video_model.py`
- `<gui-app>/io/video_reader.py`
- `<gui-app>/io/video_writers.py`
- `<gui-app>/io/video_writer_backend.py`
- backend-generic portions of `<gui-app>/io/video_backends.py`
- backend-generic portions of `<gui-app>/io/video_pipeline.py`

### Modules that should remain app-side

These concerns stay app-side:

- Qt worker orchestration and task lifecycle
- GUI playback scheduling
- live frame buffering for active sessions
- progress routing into the GUI
- user preference mapping for playback/export policy

`<gui-app>/io/framehub.py` should likely be split: reusable container leasing
can move into xpkg if still needed, while the live hub buffer remains in the
downstream GUI app.

### Import strategy

Because downstream GUI apps may still be unreleased, the preferred end state is
direct imports from xpkg rather than long-term compatibility shims. In other
words, code that needs canonical media IO should eventually import xpkg
directly instead of pretending the implementation is still local to
`<gui-app>.io`.

### Testing strategy

Downstream GUI apps should stop testing their own duplicate video semantics and
instead test:

- integration with xpkg media contracts
- GUI-specific orchestration around those contracts
- app-specific performance policy where it differs intentionally

## Decision Summary

xpkg should become the authoritative media IO stack for multimodal neuroscience
workflows. Downstream GUI apps should not keep a second general-purpose video
subsystem. The right split is:

- xpkg owns canonical media semantics and optimized backends.
- downstream GUI apps own application behavior on top of that media layer.

That gives the project one place to optimize CUDA and macOS paths, one place to
define frame semantics, and one place to test backend parity.
