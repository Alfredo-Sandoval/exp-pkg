from __future__ import annotations

from pathlib import Path

import c3d
import numpy as np


def write_sample_vicon_csv(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "Trajectories",
                "100",
                ",,Mouse:center,,,Mouse:R_foot,,,Mouse:L_foot",
                "Frame,Sub Frame,X,Y,Z,X,Y,Z,X,Y,Z",
                ",,mm,mm,mm,mm,mm,mm,mm,mm,mm",
                "101,0,1,2,3,4,5,6,7,8,9",
                "102,0,1.5,2.5,3.5,4.5,5.5,6.5,,,",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_sample_vsk(path: Path) -> None:
    path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<KinematicModel MODEL="Sample Mouse">
  <MarkerSet>
    <Markers>
      <Marker NAME="center" />
      <Marker NAME="R_foot" />
      <Marker NAME="L_foot" />
    </Markers>
    <Sticks>
      <Stick MARKER1="center" MARKER2="R_foot" />
      <Stick MARKER1="center" MARKER2="L_foot" />
    </Sticks>
  </MarkerSet>
</KinematicModel>
""",
        encoding="utf-8",
    )


def write_sample_xcp(path: Path) -> None:
    path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<CameraCalibration>
  <Camera DEVICEID="1" USERID="2" SENSOR="Bonita" SENSOR_SIZE="1024 768">
    <KeyFrames>
      <KeyFrame
        POSITION="1 2 3"
        ORIENTATION="0 0 0 1"
        FOCAL_LENGTH="1250"
        IMAGE_ERROR="0.2"
        WORLD_ERROR="1.1"
      />
    </KeyFrames>
  </Camera>
  <Camera DEVICEID="2" USERID="1" SENSOR="Vero" SENSOR_SIZE="2048 1088">
    <KeyFrames>
      <KeyFrame
        POSITION="4 5 6"
        ORIENTATION="0 0.5 0 0.8660254"
        FOCAL_LENGTH="1100"
        IMAGE_ERROR="0.1"
        WORLD_ERROR="0.9"
      />
    </KeyFrames>
  </Camera>
</CameraCalibration>
""",
        encoding="utf-8",
    )


def write_sample_vicon_c3d(path: Path) -> None:
    writer = c3d.Writer(point_rate=100.0, analog_rate=200.0)
    writer.set_start_frame(11)
    writer.set_point_labels(["Mouse:center", "Mouse:R_foot", "Model:HipMoment"])
    writer.set_analog_labels(["Fx", "Fy", "Fz"])
    writer.set_analog_scales([1.0, 1.0, 1.0])
    writer.set_analog_offsets([0, 0, 0])

    for frame_idx in range(2):
        points = np.zeros((3, 5), dtype=np.float32)
        points[0, :3] = [1.0 + frame_idx, 2.0, 3.0]
        points[0, 3] = 0.0
        points[0, 4] = 3.0

        points[1, :3] = [4.0, 5.0 + frame_idx, 6.0]
        points[1, 3] = 0.0 if frame_idx == 0 else -1.0
        points[1, 4] = 2.0

        points[2, :3] = [7.0, 8.0, 9.0 + frame_idx]
        points[2, 3] = 0.0
        points[2, 4] = 1.0

        analog = np.array(
            [
                [10.0 + frame_idx, 20.0 + frame_idx],
                [30.0 + frame_idx, 40.0 + frame_idx],
                [50.0 + frame_idx, 60.0 + frame_idx],
            ],
            dtype=np.float32,
        )
        frame = np.empty((1, 2), dtype=object)
        frame[0, 0] = points
        frame[0, 1] = analog
        writer.add_frames(frame)
    with path.open("wb") as handle:
        writer.write(handle)
