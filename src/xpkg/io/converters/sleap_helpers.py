"""Shared SLEAP ``.pkg.slp`` extraction routines for converter modules."""

from __future__ import annotations

import os
from typing import Any, cast

import cv2
import h5py
import numpy as np
import pandas as pd

from xpkg._core.json_utils import parse_json, parse_json_dict
from xpkg._core.logging_utils import get_logger
from xpkg._core.path_registry import ensure_dir

_LOGGER = get_logger(__name__)

_ERR_NO_VIDEOS = "No videos found in SLEAP package"


def _warn(msg: str) -> None:
    if _LOGGER.hasHandlers():
        _LOGGER.warning(msg.rstrip())
    else:
        import sys as _sys

        _sys.stderr.write(msg.rstrip() + "\n")


def _column_index(column_names: list[str]) -> pd.Index:
    return pd.Index(column_names)


def _get_field(record: Any, field: str) -> Any:
    return record[field]


def _video_groups(hdf: h5py.File) -> dict[str, str]:
    videos: dict[str, str] = {}
    for group in hdf.keys():
        if not group.startswith("video"):
            continue
        sv = f"{group}/source_video"
        if sv in hdf:
            js = hdf[sv].attrs.get("json", "")
            if js:
                meta = parse_json_dict(cast(str | bytes | bytearray, js))
                fn = meta.get("backend", {}).get("filename")
                if fn:
                    videos[group] = fn
    return videos


def _group_video_indices_from_json(hdf: h5py.File) -> dict[str, int]:
    if "videos_json" not in hdf:
        return {}
    ds = cast(h5py.Dataset, hdf["videos_json"])
    mapping: dict[str, int] = {}
    for idx in range(len(ds)):
        raw = ds[idx]
        if isinstance(raw, bytes | np.bytes_):
            payload_raw = parse_json(bytes(raw))
        else:
            payload_raw = parse_json(str(raw))
        if not isinstance(payload_raw, dict):
            raise TypeError("videos_json entries must be JSON objects")
        payload: dict[str, object] = {str(key): value for key, value in payload_raw.items()}
        backend_raw = payload.get("backend")
        backend: dict[str, object] | None = None
        if isinstance(backend_raw, dict):
            backend = {str(key): value for key, value in backend_raw.items()}
        dataset = backend.get("dataset") if backend is not None else None
        if dataset is None:
            dataset = payload.get("dataset")
        if not isinstance(dataset, str) or not dataset.strip():
            raise ValueError("videos_json entry missing backend.dataset")
        group = dataset.split("/", 1)[0]
        if not group:
            raise ValueError("videos_json entry missing dataset group")
        mapping[group] = idx
    return mapping


def extract_frames(
    slp_path: str,
    out_dir: str,
) -> None:
    ensure_dir(out_dir)
    labeled = os.path.join(out_dir, "labeled-data")
    ensure_dir(labeled)

    with h5py.File(slp_path, "r") as hdf:
        vids = _video_groups(hdf)
        if not vids:
            raise RuntimeError(_ERR_NO_VIDEOS)
        for vg, fn in vids.items():
            base = os.path.splitext(os.path.basename(fn))[0]
            out_vdir = os.path.join(labeled, base)
            ensure_dir(out_vdir)

            if f"{vg}/video" not in hdf:
                continue
            frames = cast(Any, hdf[f"{vg}/video"])
            frame_numbers = cast(Any, hdf[f"{vg}/frame_numbers"])
            for img_bytes, frame_num in zip(frames, frame_numbers, strict=False):
                from xpkg.media.images import read_rgb_bytes as _read_rgb_bytes

                rgb = _read_rgb_bytes(bytes(img_bytes))
                name = f"img{int(frame_num):08d}.png"
                dst = os.path.join(out_vdir, name)
                cv2.imwrite(dst, rgb)


def extract_labels_step4(
    slp_path: str,
    out_dir: str,
) -> pd.DataFrame:
    """Build the flattened Step-4 labels table without writing a CSV file."""

    ensure_dir(out_dir)
    labeled = os.path.join(out_dir, "labeled-data")
    ensure_dir(labeled)

    with h5py.File(slp_path, "r") as hdf:
        vids = _video_groups(hdf)
        if not vids:
            raise RuntimeError(_ERR_NO_VIDEOS)

        md = parse_json_dict(cast(str | bytes | bytearray, hdf["metadata"].attrs.get("json", "{}")))
        nodes = md.get("nodes", [])
        sk = (md.get("skeletons", []) or [{}])[0]
        id2name: dict[int, str] = {}
        for i, n in enumerate(nodes):
            nm = str(n.get("name") or "").strip()
            if nm:
                id2name[int(i)] = nm
        sk_nodes = sk.get("nodes") or []
        order_ids: list[int] = []
        for item in sk_nodes:
            if isinstance(item, dict):
                cand_node = item.get("id")
                if isinstance(cand_node, dict) and "id" in cand_node:
                    cand_node = cand_node["id"]
                if isinstance(cand_node, int | str) and str(cand_node).isdigit():
                    order_ids.append(int(cand_node))
            elif isinstance(item, int | str) and str(item).isdigit():
                order_ids.append(int(item))
        node_ids_order = (
            order_ids if order_ids and all(i in id2name for i in order_ids) else sorted(id2name)
        )
        kp_ordered = [id2name[i] for i in node_ids_order]
        column_names = ["frame"]
        for kp in kp_ordered:
            column_names.extend([f"{kp}_x", f"{kp}_y"])

        frames_ds = cast(h5py.Dataset, hdf["frames"])
        points_ds = cast(h5py.Dataset, hdf["points"])
        inst_ds = cast(h5py.Dataset, hdf["instances"])

        fr_names = set(frames_ds.dtype.names or ())
        vid_to_idx: dict[int, set[int]] = {}
        for fr in frames_ds:
            vid_val = int(fr["video"])
            if "frame_idx" in fr_names:
                fidx = int(fr["frame_idx"])
            elif "frame" in fr_names:
                fidx = int(fr["frame"])
            else:
                fidx = int(fr["frame_id"])
            vid_to_idx.setdefault(vid_val, set()).add(fidx)

        g_idx_by_group: dict[str, set[int]] = {}
        for vg in vids.keys():
            g_idxs: set[int] = set()
            if f"{vg}/frame_numbers" in hdf:
                g_idxs = set(int(x) for x in cast(h5py.Dataset, hdf[f"{vg}/frame_numbers"]))
            g_idx_by_group[vg] = g_idxs

        group_to_vid = _group_video_indices_from_json(hdf)
        if group_to_vid:
            missing = [vid for vid in group_to_vid.values() if vid not in vid_to_idx]
            if missing:
                raise ValueError(
                    f"videos_json video indices missing from frames table: {sorted(set(missing))}"
                )

        for vg, g_idxs in g_idx_by_group.items():
            if vg in group_to_vid:
                continue
            eq_matches = [vid for vid, idxs in vid_to_idx.items() if idxs == g_idxs]
            if len(eq_matches) == 1:
                group_to_vid[vg] = int(eq_matches[0])
                continue
            best_vid = None
            best_score = -1.0
            for vid, idxs in vid_to_idx.items():
                inter = len(g_idxs & idxs)
                if inter <= 0:
                    continue
                union = len(g_idxs | idxs) or 1
                jacc = inter / union
                if jacc > best_score:
                    best_score = jacc
                    best_vid = int(vid)
            if best_vid is not None:
                group_to_vid[vg] = int(best_vid)
            else:
                _warn(f"XPKG_IMPORT WARN: No labeled-frame match for {vg}; skipping group.")

        dfs: list[pd.DataFrame] = []
        pts_fields = set(points_ds.dtype.names or ())
        compact_points = {"x", "y"}.issubset(pts_fields) and not (
            {
                "instance",
                "instance_id",
                "node",
                "node_id",
                "frame",
                "frame_idx",
                "frame_id",
                "video",
            }
            & pts_fields
        )

        if compact_points:
            for vg, _fn in vids.items():
                if vg not in group_to_vid:
                    continue
                vid_val = int(group_to_vid[vg])
                g_idxs = g_idx_by_group.get(vg, set())
                base = os.path.splitext(os.path.basename(_fn))[0]
                rows: list[list[float | None]] = []
                for fr in frames_ds:
                    if int(fr["video"]) != vid_val:
                        continue
                    fidx = int(fr["frame_idx"]) if "frame_idx" in fr_names else int(fr["frame"])
                    if fidx not in g_idxs:
                        continue
                    inst_indices: list[int] = []
                    if "instance_id_start" in fr_names and "instance_id_end" in fr_names:
                        s = int(fr["instance_id_start"])
                        e = int(fr["instance_id_end"])
                        if e > s:
                            inst_indices = list(range(s, e))
                    else:
                        for i, inst in enumerate(cast(Any, inst_ds)):
                            if int(_get_field(inst, "frame_id")) == int(fr["frame_id"]):
                                inst_indices.append(i)
                    if not inst_indices:
                        rows.append([float(fidx)] + [None] * (2 * len(kp_ordered)))
                        continue
                    best_flat: list[float | None] | None = None
                    best_count = -1
                    for inst_idx in inst_indices:
                        inst = inst_ds[inst_idx]
                        inst_names = set(inst.dtype.names or ())
                        if "instance_type" in inst_names and int(inst["instance_type"]) != 0:
                            continue
                        if "point_id_start" not in inst_names or "point_id_end" not in inst_names:
                            raise KeyError("SLEAP instances missing point_id_start/point_id_end")
                        pstart = int(inst["point_id_start"])
                        pend = int(inst["point_id_end"])
                        if pend <= pstart:
                            continue
                        cand = cast(list[Any], list(points_ds[pstart:pend]))
                        flat: list[float | None] = []
                        valid_points = 0
                        for i_idx, p in enumerate(cand):
                            if i_idx >= len(node_ids_order):
                                break
                            xv = float(p["x"])
                            yv = float(p["y"])
                            if "visible" in pts_fields and not bool(p["visible"]):
                                flat.extend([None, None])
                                continue
                            if not np.isfinite(xv) or not np.isfinite(yv):
                                flat.extend([None, None])
                                continue
                            flat.extend([xv, yv])
                            valid_points += 1
                        need = len(kp_ordered) * 2
                        if len(flat) < need:
                            flat.extend([None] * (need - len(flat)))
                        elif len(flat) > need:
                            flat = flat[:need]
                        if valid_points > best_count:
                            best_flat = flat
                            best_count = valid_points
                    if best_flat is None:
                        rows.append([float(fidx)] + [None] * (2 * len(kp_ordered)))
                        continue
                    rows.append([float(fidx), *best_flat])

                if rows:
                    df = pd.DataFrame(rows, columns=_column_index(column_names))
                    df["frame"] = df["frame"].apply(
                        lambda x, _base=base: f"labeled-data/{_base}/img{int(x):08d}.png"
                    )
                    dfs.append(df)
        else:
            for vg, _fn in vids.items():
                if vg not in group_to_vid:
                    continue
                vid = int(group_to_vid.get(vg, 0))
                g_idxs = g_idx_by_group.get(vg, set())
                fmap = {}
                fmap_inv = {}
                for fr in frames_ds:
                    if int(fr["video"]) != vid:
                        continue
                    fid = int(fr["frame_id"]) if "frame_id" in fr_names else int(fr["frame"])
                    fidx = int(fr["frame_idx"]) if "frame_idx" in fr_names else int(fr["frame"])
                    if fidx not in g_idxs:
                        continue
                    fmap[fid] = fidx
                    fmap_inv[fidx] = fid
                insts_by_frame: dict[int, list[Any]] = {}
                inst_names_all = set(inst_ds.dtype.names or ())
                inst_has_video = "video" in inst_names_all
                for inst in inst_ds:
                    inst_names = set(inst.dtype.names or ())
                    if inst_has_video and int(inst["video"]) != vid:
                        continue
                    if "frame_id" in inst_names:
                        fid = int(inst["frame_id"])
                    elif "frame_idx" in inst_names:
                        fidx = int(inst["frame_idx"])
                        fid = int(fmap_inv.get(fidx, -1))
                    elif "frame" in inst_names:
                        fidx = int(inst["frame"])
                        fid = int(fmap_inv.get(fidx, -1))
                    else:
                        fid = -1
                    if fid in fmap:
                        insts_by_frame.setdefault(fid, []).append(inst)
                rows: list[list[float | None]] = []
                for fid, fidx in fmap.items():
                    inst_list = insts_by_frame.get(fid, [])
                    if not inst_list:
                        rows.append([float(fidx)] + [None] * (2 * len(kp_ordered)))
                        continue
                    p_names = set(points_ds.dtype.names or ())
                    best_flat: list[float | None] | None = None
                    best_count = -1
                    for inst in inst_list:
                        inst_names = set(inst.dtype.names or ())
                        if "instance_type" in inst_names and int(inst["instance_type"]) != 0:
                            continue
                        inst_id: int | None = None
                        for key in ("id", "instance", "instance_id", "inst_id"):
                            if key in inst_names:
                                inst_id = int(inst[key])
                                break
                        cand: list[Any] = []
                        if inst_id is not None and (
                            "instance" in p_names or "instance_id" in p_names
                        ):
                            p_inst_key = "instance" if "instance" in p_names else "instance_id"
                            if "frame_id" in p_names:
                                p_frame_key = "frame_id"
                            elif "frame_idx" in p_names:
                                p_frame_key = "frame_idx"
                            elif "frame" in p_names:
                                p_frame_key = "frame"
                            else:
                                p_frame_key = None
                            p_has_video = "video" in p_names
                            for p in points_ds:
                                if p_inst_key in p and int(p[p_inst_key]) != inst_id:
                                    continue
                                if p_has_video and "video" in p and int(p["video"]) != vid:
                                    continue
                                if p_frame_key is not None:
                                    if p_frame_key == "frame_id":
                                        if p_frame_key in p and int(p[p_frame_key]) != int(fid):
                                            continue
                                    elif p_frame_key in ("frame_idx", "frame"):
                                        cur_idx = int(fmap.get(int(fid), -1))
                                        if p_frame_key in p and int(p[p_frame_key]) != cur_idx:
                                            continue
                                cand.append(p)

                        if (
                            not cand
                            and {"x", "y"}.issubset(p_names)
                            and not (
                                {
                                    "instance",
                                    "instance_id",
                                    "node",
                                    "node_id",
                                    "frame",
                                    "frame_idx",
                                    "frame_id",
                                }
                                & p_names
                            )
                        ):
                            pstart = None
                            pend = None
                            if "point_id_start" in inst_names:
                                val = _get_field(inst, "point_id_start")
                                if isinstance(val, int | np.integer):
                                    pstart = int(val)
                            if "point_id_end" in inst_names:
                                val = _get_field(inst, "point_id_end")
                                if isinstance(val, int | np.integer):
                                    pend = int(val)
                            if (
                                pstart is not None
                                and pend is not None
                                and 0 <= pstart <= pend
                                and pend <= len(points_ds)
                            ):
                                cand = cast(list[np.void], list(points_ds[pstart:pend]))

                        pts_by_node: dict[int, dict[str, float | int | bool | None]] = {}
                        if cand:
                            node_key = (
                                "node"
                                if "node" in p_names
                                else ("node_id" if "node_id" in p_names else None)
                            )
                            if node_key is not None:
                                for p in cand:
                                    node_id = int(p[node_key])
                                    rec: dict[str, float | int | bool | None] = {}
                                    for k in p_names:
                                        rec[k] = p[k] if k in p else None
                                    pts_by_node[node_id] = rec
                            else:
                                for idx, p in enumerate(cand):
                                    if idx >= len(node_ids_order):
                                        break
                                    node_id = int(node_ids_order[idx])
                                    rec: dict[str, float | int | bool | None] = {}
                                    for k in p_names:
                                        rec[k] = p[k] if k in p else None
                                    pts_by_node[node_id] = rec
                        if not pts_by_node:
                            if "frame_id" in p_names:
                                p_frame_key = "frame_id"
                            elif "frame_idx" in p_names:
                                p_frame_key = "frame_idx"
                            elif "frame" in p_names:
                                p_frame_key = "frame"
                            else:
                                p_frame_key = None
                            p_has_video = "video" in p_names
                            node_key = (
                                "node"
                                if "node" in p_names
                                else ("node_id" if "node_id" in p_names else None)
                            )
                            by_inst: dict[int, dict[int, dict[str, float | int | bool | None]]] = {}
                            for p in points_ds:
                                if p_has_video and "video" in p and int(p["video"]) != vid:
                                    continue
                                if p_frame_key is not None:
                                    if p_frame_key == "frame_id":
                                        if p_frame_key in p and int(p[p_frame_key]) != int(fid):
                                            continue
                                    else:
                                        cur_idx = int(fmap.get(int(fid), -1))
                                        if p_frame_key in p and int(p[p_frame_key]) != cur_idx:
                                            continue
                                if node_key is None:
                                    continue
                                if node_key not in p:
                                    continue
                                node_val = p[node_key]
                                if not isinstance(node_val, int | np.integer):
                                    continue
                                node_id = int(node_val)
                                inst_key = None
                                for key in ("instance", "instance_id"):
                                    if key in p_names:
                                        inst_key = key
                                        break
                                inst_val = int(p[inst_key]) if inst_key else 0
                                rec: dict[str, float | int | bool | None] = {}
                                for k in p_names:
                                    rec[k] = p[k] if k in p else None
                                by_inst.setdefault(inst_val, {})[node_id] = rec
                            if by_inst:
                                inst_choice = max(by_inst.items(), key=lambda kv: len(kv[1]))[0]
                                pts_by_node = by_inst[inst_choice]

                        visibility_key = None
                        if "visible" in p_names:
                            visibility_key = "visible"
                        elif "is_visible" in p_names:
                            visibility_key = "is_visible"
                        has_vis = visibility_key is not None
                        flat: list[float | None] = []
                        valid_points = 0
                        for node_id in node_ids_order:
                            rec_d = pts_by_node.get(int(node_id))
                            if not isinstance(rec_d, dict):
                                flat.extend([None, None])
                                continue
                            x = rec_d.get("x")
                            y = rec_d.get("y")
                            if has_vis:
                                vis_key = str(visibility_key)
                                vis = bool(rec_d.get(vis_key, True))
                            else:
                                vis = True
                            xv = float(x) if x is not None else None
                            yv = float(y) if y is not None else None
                            if xv is None or yv is None:
                                flat.extend([None, None])
                            elif has_vis and not bool(vis):
                                flat.extend([None, None])
                            elif not np.isfinite(xv) or not np.isfinite(yv):
                                flat.extend([None, None])
                            else:
                                flat.extend([xv, yv])
                                valid_points += 1
                        need = len(kp_ordered) * 2
                        if len(flat) < need:
                            flat.extend([None] * (need - len(flat)))
                        elif len(flat) > need:
                            flat = flat[:need]
                        if valid_points > best_count:
                            best_flat = flat
                            best_count = valid_points
                    if best_flat is None:
                        rows.append([float(fidx)] + [None] * (2 * len(kp_ordered)))
                        continue
                    rows.append([float(fidx), *best_flat])
                if rows:
                    df = pd.DataFrame(rows, columns=_column_index(column_names))
                    base = os.path.splitext(os.path.basename(_fn))[0]
                    df["frame"] = df["frame"].apply(
                        lambda x, _base=base: f"labeled-data/{_base}/img{int(x):08d}.png"
                    )
                    dfs.append(df)
        if not dfs:
            return pd.DataFrame(columns=_column_index(column_names))
        row_blocks = [df.to_numpy(copy=False) for df in dfs if not df.empty]
        if not row_blocks:
            return pd.DataFrame(columns=_column_index(column_names))
        out = np.concatenate(row_blocks, axis=0)
        return pd.DataFrame(out, columns=_column_index(column_names))


__all__ = [
    "extract_frames",
    "extract_labels_step4",
]
