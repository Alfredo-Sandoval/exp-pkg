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
adapter and project IO cutover work. It is intentionally opinionated: the goal
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

- xpkg already owns canonical labels IO, project state IO, and external
  adapter logic.
- xpkg also has a small media layer in `xpkg.media.video`.
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

### Required and optional backends

| Backend | Purpose | Platforms | Role |
| --- | --- | --- | --- |
| `images` | Image sequences and single-image videos | macOS, Linux | Reference support for still-frame workflows |
| `opencv` | Baseline portable decode/write path | macOS, Linux | Reference backend and broad compatibility |
| `imageio` | Baseline FFmpeg-backed writer path | macOS, Linux | Current non-AVI writer fallback |
| `pyav` | Rich FFmpeg container, stream, codec, metadata, and filter control | macOS, Linux | Optional `media-rich` backend |
| `torchcodec` | PyTorch-native video/audio decode and encode | macOS, Linux | Optional `dl` backend for tensor pipelines |
| `onnxruntime` | Portable exported-model inference | macOS, Linux | Optional `inference` backend |
| `kornia` | Differentiable computer-vision tensor operations | macOS, Linux | Optional `vision` backend |
| `mlx` | Apple MLX tensor runtime | macOS, Linux | Optional `mlx` backend for model/tensor acceleration |
| `nvpkg` | NVIDIA media-stack provisioning and verification | Linux | Optional `nvpkg` bridge for CUDA media packages |

PyAV and TorchCodec should be explicit implementation backends, not hidden
dependencies. PyAV is the better fit for FFmpeg-level media ownership and
metadata. TorchCodec is the better fit when the next consumer is a PyTorch model
or tensor transform. ONNX Runtime and Kornia are model-adjacent rather than
general video readers, but they should still be discoverable through the same
capability surface because downstream workflows need to reason about them.

The package extras encode this split:

| Extra | Dependencies | Use |
| --- | --- | --- |
| `media-rich` | `av` | Rich FFmpeg media handling |
| `dl` | `torch`, `torchcodec`, `torchvision` | PyTorch tensor pipelines |
| `inference` | `onnxruntime` | Exported model inference |
| `mlx` | `mlx` | Apple/Metal tensor acceleration |
| `nvidia` | `torch`, `torchcodec`, `torchvision` | NVIDIA CUDA tensor/video pipelines |
| `vision` | `kornia`, `torch` | Differentiable tensor CV |
| `hardware-accel` | `mlx`, `torch`, `torchcodec`, `torchvision` | MLX plus NVIDIA optional runtimes |
| `media-dl` | all of the above plus `decord` where wheels exist | Full media/deep-learning stack |

This choice follows the current upstream direction:

- Decord is platform-gated in the install metadata because public wheels are
  not available on every supported host architecture.

- PyTorch positions TorchCodec as the video-to-tensor path for PyTorch model
  workflows:
  <https://pytorch.org/blog/torchcodec/>
- TorchCodec publishes an explicit compatibility matrix tying `torchcodec 0.11`
  to `torch 2.11`:
  <https://github.com/meta-pytorch/torchcodec#installing-torchcodec>
- TorchVision deprecated video decode/encode in `0.22` and removed it by
  `0.24`, so `torchvision.io` should not be the xpkg video backend:
  <https://docs.pytorch.org/vision/main/generated/torchvision.io.read_video.html>
- ImageIO's PyAV plugin is the richer replacement direction for its FFmpeg
  plugin:
  <https://imageio.readthedocs.io/en/v2.31.4/_autosummary/imageio.plugins.pyav.html>
- ONNX Runtime is the exported-model inference path, with CPU packages
  appropriate for macOS and Arm hosts:
  <https://onnxruntime.ai/docs/get-started/with-python.html>
- Kornia is the PyTorch-native differentiable computer-vision layer:
  <https://www.kornia.org/>
- MLX is Apple's array framework for machine learning on Apple silicon and is
  installed from PyPI with `mlx`:
  <https://pypi.org/project/mlx/>
- MLX exposes Metal availability through `mlx.core.metal.is_available()`:
  <https://ml-explore-mlx.mintlify.app/api/metal>
- PyTorch exposes CUDA availability through `torch.cuda.is_available()`:
  <https://docs.pytorch.org/docs/stable/cuda.html>
- TorchCodec supports CUDA/NVDEC decoding for NVIDIA video workloads:
  <https://meta-pytorch.org/torchcodec/stable/generated_examples/decoding/basic_cuda_example.html>
- nvpkg is the intended Linux NVIDIA provisioning bridge for FFmpeg CUDA,
  OpenCV CUDA, PyAV CUDA, Decord CUDA, TorchCodec CUDA, and DALI CUDA:
  <https://github.com/Alfredo-Sandoval/nvpkg>

### Capability discovery

xpkg should add explicit capability reporting so callers can reason about the
host instead of guessing:

- which backends are available
- which optional extra enables a missing backend
- whether hardware decode is available
- whether hardware encode is available
- supported output codecs
- whether exact random seek is validated
- whether shared-container decode is supported

The important design point is that capability discovery belongs to xpkg.
Downstream GUI apps should not maintain their own independent truth about which
video backends exist or how they behave.

The first implementation layer is `xpkg.media.backends`, which reports installed
core and optional backends without importing heavyweight runtime packages during
normal package import. It also reports hardware accelerators:

| Accelerator | Meaning |
| --- | --- |
| `mlx-metal` | MLX can use Metal on the current host |
| `torch-cuda` | PyTorch can use NVIDIA CUDA tensors |
| `torchcodec-cuda` | TorchCodec is importable and PyTorch CUDA is available |
| `ffmpeg-nvidia` | Host FFmpeg exposes NVDEC/NVENC support |
| `opencv-cuda` | OpenCV exposes `cv2.cuda` with a visible CUDA device |
| `pyav-cuda` | nvpkg verifies PyAV against the CUDA-capable media stack |
| `decord-cuda` | nvpkg verifies Decord GPU video loading |
| `dali-cuda` | nvpkg verifies NVIDIA DALI video-reader/data-loading support |

PyAV is also available as an explicit reader backend via
`Video.from_filename(..., backend="pyav")` when `exp-pkg[media-rich]` is
installed. It must remain explicit until conformance coverage proves it can join
`auto`.

## Performance Requirements

The target stack should be optimized for two hardware families:

- NVIDIA/CUDA systems on Linux
- macOS systems using Apple media acceleration

That optimization work belongs in xpkg because the same fast path should
benefit:

- DLC/SLEAP conversion
- project import/export
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

`nvpkg` owns the Linux NVIDIA provisioning layer so xpkg does not need to carry
CUDA build scripts or host-specific wheel logic. It is installed and managed
outside exp-pkg extras until a public package line is available. The expected
setup sequence is:

```bash
nvpkg --help
nvpkg system doctor
nvpkg package install ffmpeg
nvpkg package install opencv_cuda
nvpkg package install torchcodec_cuda
nvpkg package verify opencv_cuda --json
```

Additional nvpkg packages can be installed when a workflow needs them:
`pyav_cuda` for FFmpeg-level container control, `decord_cuda` for
throughput-oriented video loading, and `dali_cuda` for batched data-loading
pipelines. xpkg should consume those capabilities through explicit accelerator
names rather than auto-selecting them silently.

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

### Current Cutover State

xpkg now owns the generic media behavior that downstream GUI apps were
duplicating: directory and single-image-as-video inputs, BGR/uint8 frame
contracts, exact frame access, safe batch reads, strided and batched iteration,
explicit `opencv` / `pyav` / `decord-gpu` backend requests, writer factory
selection, FFmpeg encoder probing, and deterministic selected-frame extraction.
The `PyAVVideoResource` hook lets downstream apps attach their own PyAV container
lease policy while xpkg still owns the generic decode semantics.

Downstream apps should import these from `xpkg.media.*` directly. App-side code
should keep only GUI scheduling, worker/progress routing, live latest-frame
buffers, and product-specific inference/training integration.

### Phase 1: Make xpkg authoritative

- Keep `Video`, `VideoReader`, `VideoWriter`, and `write_video` as the public
  surface in xpkg.
- Keep generic backend logic and explicit backend naming in xpkg.
- Move downstream GUI apps to direct xpkg imports once their targeted integration
  tests pass.

### Phase 2: Port generic performance features

- Keep generic image-sequence handling, exact-seek behavior, and writer
  capability logic in xpkg.
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
