# Posetta

A Rosetta Stone for pose estimation and segmentation data.

Posetta defines the **Siesta format** (`.siesta`) — a compact, hierarchical dataset format for keypoints, skeletons, and segmentation masks — and provides **adapters** to convert to and from every major pose estimation and segmentation library.

## Supported Libraries

### Pose Estimation
- **MMPose** (OpenMMLab)
- **DeepLabCut**
- **MediaPipe**
- **OpenPose**
- **SLEAP**
- **ViTPose**
- **RTMPose**

### Segmentation
- **Detectron2**
- **MMSegmentation**
- **SAM / SAM 2** (Segment Anything)
- **YOLO** (Ultralytics)

### Dataset Formats
- COCO Keypoints / COCO Panoptic
- MPII Human Pose
- Human3.6M
- WFLW
- AP-10K
- AnimalPose

## Siesta Format

HDF5-based (`.siesta` / `.h5`) with a standardized schema for:

- 2D/3D keypoints and confidence scores
- Skeleton connectivity graphs
- Instance and semantic segmentation masks
- Rich metadata (subject, session, camera, frame)

## Installation

```bash
pip install posetta
```

## Quick Start

```python
import posetta

# Read any supported format
dataset = posetta.read("annotations.json", format="coco-keypoints")

# Convert to Siesta
posetta.write(dataset, "output.siesta")

# Export to another format
posetta.write(dataset, "export.json", format="deeplabcut")
```