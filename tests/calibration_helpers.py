from __future__ import annotations

from pathlib import Path


def write_anipose_toml(path: Path) -> Path:
    path.write_text(
        """
[cam_top]
name = "cam_top"
size = [1920, 1080]
matrix = [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]]
distortions = [0.1, -0.01, 0.001, 0.002, 0.0]
rotation = [0.0, 0.0, 0.0]
translation = [1.0, 2.0, 3.0]
fisheye = false
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def write_opencv_stereo_yaml(path: Path, *, distortion_key: str = "D2") -> Path:
    path.write_text(
        f"""
%YAML:1.0
---
image_width: 640
image_height: 480
M1: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ 1000.0, 0.0, 320.0, 0.0, 1001.0, 240.0, 0.0, 0.0, 1.0 ]
D1: !!opencv-matrix
   rows: 1
   cols: 5
   dt: d
   data: [ 0.1, -0.01, 0.001, 0.002, 0.0 ]
M2: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ 990.0, 0.0, 300.0, 0.0, 991.0, 230.0, 0.0, 0.0, 1.0 ]
{distortion_key}: !!opencv-matrix
   rows: 1
   cols: 4
   dt: d
   data: [ 0.02, -0.03, 0.004, -0.005 ]
R: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0 ]
T: !!opencv-matrix
   rows: 3
   cols: 1
   dt: d
   data: [ 10.0, 20.0, 30.0 ]
E: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ 0.0, -30.0, 20.0, 30.0, 0.0, -10.0, -20.0, 10.0, 0.0 ]
F: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ 0.0, -0.03, 0.02, 0.03, 0.0, -0.01, -0.02, 0.01, 0.0 ]
""".lstrip(),
        encoding="utf-8",
    )
    return path
