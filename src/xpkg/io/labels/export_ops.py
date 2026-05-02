"""Export and keypoint transformation helpers for `Labels`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from xpkg.io.labels.video_types import VideoProtocol
from xpkg.pose.annotations import Instance, LabeledFrame, PredictedInstance

if TYPE_CHECKING:
    from xpkg.io.labels.model import Labels


def labels_numpy(
    labels: Labels,
    video: VideoProtocol | int | None = None,
    *,
    all_frames: bool = True,
    untracked: bool = False,
    return_confidence: bool = False,
) -> np.ndarray:
    """Return `np.ndarray` data for the selected `video` and frame range."""
    if not labels.videos:
        raise IndexError("There are no videos in this project. No points matrix to return.")

    tgt_video: VideoProtocol
    if video is None:
        tgt_video = labels.videos[0]
    elif isinstance(video, int):
        tgt_video = labels.videos[video]
    else:
        tgt_video = video

    lfs: list[LabeledFrame] = labels.query.find(video=tgt_video)
    frame_idxs = sorted([lf.frame_idx for lf in lfs])
    first_frame = 0 if all_frames else (frame_idxs[0] if frame_idxs else 0)
    last_frame = (
        max(int(tgt_video.frames) - 1, 0) if all_frames else (frame_idxs[-1] if frame_idxs else 0)
    )

    n_insts = max(
        [
            (lf.user_instance_count if lf.user_instance_count > 0 else lf.predicted_instance_count)
            for lf in lfs
        ],
        default=1,
    )

    untracked = untracked or n_insts == 1
    n_tracks = n_insts if untracked else len(labels.tracks)
    n_frames = max(0, last_frame - first_frame + 1)
    n_keypoints = len(labels.skeleton.keypoints)

    point_dims = 3 if return_confidence else 2
    shape = (n_frames, n_tracks, n_keypoints, point_dims)
    tracks = np.full(shape, np.nan, dtype="float32")

    track_map = {t: i for i, t in enumerate(labels.tracks)} if not untracked else {}

    def _set_track(inst: Instance | PredictedInstance, track_arr: np.ndarray) -> np.ndarray:
        if return_confidence:
            if isinstance(inst, PredictedInstance):
                return inst.points_and_scores_array
            track_arr[:, :-1] = inst.numpy()
            return track_arr
        return inst.numpy()

    for lf in lfs:
        i = lf.frame_idx - first_frame
        if i < 0 or i >= n_frames:
            raise IndexError(
                "Labeled frame index out of range: "
                f"{lf.frame_idx} not in [{first_frame}, {last_frame}]"
            )

        lf_insts: list[Instance] | list[PredictedInstance] = (
            lf.user_instances if lf.user_instance_count > 0 else lf.predicted_instances
        )

        if untracked:
            for j, inst in enumerate(lf_insts):
                tracks[i, j] = _set_track(inst, tracks[i, j])
        else:
            for inst in lf_insts:
                track = inst.track
                if track is None:
                    continue
                j = track_map.get(track)
                if j is None:
                    raise ValueError(
                        "Track missing from labels.tracks: "
                        f"{track.name} (spawned_on={track.spawned_on})"
                    )
                tracks[i, j] = _set_track(inst, tracks[i, j])

    return tracks


def labels_to_dataframe(
    labels: Labels,
    video: VideoProtocol | int | None = None,
    scorer: str = "xpkg",
):
    """Convert labels for a video to a DeepLabCut-style MultiIndex DataFrame."""
    import pandas as pd

    coords = labels_numpy(
        labels,
        video=video,
        all_frames=True,
        untracked=True,
        return_confidence=True,
    )
    n_frames, _, _, _ = coords.shape

    single_animal_coords = coords[:, 0, :, :]

    bodyparts = [kp.name for kp in labels.skeleton.keypoints]
    columns = pd.MultiIndex.from_product(
        [[scorer], bodyparts, ["x", "y", "likelihood"]],
        names=["scorer", "bodyparts", "coords"],
    )

    flat_coords = single_animal_coords.reshape(n_frames, -1)
    return pd.DataFrame(flat_coords, columns=columns)


def merge_keypoints(labels: Labels, base_keypoint: str, merge_keypoint: str) -> None:
    """Merge `merge_keypoint` into `base_keypoint` across the dataset."""
    merge_index = labels.skeleton.keypoint_to_index(merge_keypoint)
    merge_obj = labels.skeleton.get_keypoint_by_name(merge_keypoint)
    for inst in labels.instances():
        inst._merge_keypoints_data(base_keypoint, merge_keypoint)

    drop_keypoint_heatmaps(labels, merge_index)
    labels.skeleton.remove_keypoint(merge_keypoint)

    for inst in labels.instances():
        inst.realign_points()

    if merge_obj in labels.keypoints:
        labels.keypoints.remove(merge_obj)


def drop_keypoint_heatmaps(labels: Labels, keypoint_index: int) -> None:
    """Remove one keypoint channel from per-frame heatmaps."""
    kp_index = int(keypoint_index)
    updates: list[tuple[LabeledFrame, np.ndarray]] = []
    for lf in labels.labeled_frames:
        heatmaps = lf.heatmaps
        if heatmaps is None:
            continue
        heatmaps_arr = np.asarray(heatmaps)
        if heatmaps_arr.ndim != 3:
            raise ValueError("Heatmaps must be shaped (K, H, W) to drop a keypoint channel")
        if kp_index < 0 or kp_index >= int(heatmaps_arr.shape[0]):
            raise ValueError(
                f"Heatmaps keypoint index {kp_index} out of range for frame {lf.frame_idx}"
            )
        updates.append(
            (
                lf,
                np.concatenate((heatmaps_arr[:kp_index], heatmaps_arr[kp_index + 1 :]), axis=0),
            )
        )
    for lf, heatmaps in updates:
        lf.heatmaps = heatmaps


__all__ = [
    "drop_keypoint_heatmaps",
    "labels_numpy",
    "labels_to_dataframe",
    "merge_keypoints",
]
