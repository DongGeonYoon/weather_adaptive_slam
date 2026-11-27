#!/usr/bin/env python3
"""Evaluate FAST-LIO2 runs (sim/run_experiments.sh) on the simulated weather bags.

usage: evaluate_sim.py --results <RES> [--modes baseline original proposed]
       expects <RES>/data (dataset) and <RES>/runs/<weather>/<method>/rec

Metrics per (weather, mode):
  ATE RMSE      - absolute trajectory error after SE(3) Umeyama alignment [m]
  RPE trans     - relative translation error over 10 m of travel [% of distance]
  RPE rot       - relative rotation error over 10 m of travel [deg/10 m]
  end drift     - final position error with the first pose anchored to ground truth [m]
  map noise     - fraction of map-candidate points (/cloud_weather_filtered) that are
                  farther than 0.3 m from any true surface (pose error removed per frame,
                  so this isolates weather artefacts that would be inserted into the map)
  inserted noise- absolute number of such weather points offered to the map
  time          - mean total processing time per scan [ms]
  voxel IoU     - |V_est ∩ V_gt| / |V_est ∪ V_gt| of occupied 0.5 m voxels (proposal Sec. 4).
                  V_est: all map-candidate points (/cloud_weather_filtered), SE(3)-aligned
                  like ATE.  V_gt: noise-free ray casts from the GT poses (every 2nd scan).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from scipy.spatial.transform import Rotation as Rot
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry

import sim_world as sw

WEATHERS = ["clear", "rain", "fog", "snow", "all_weather"]
MODES = ["baseline", "original", "proposed", "proposed_cov"]   # missing runs are skipped
NOISE_DIST = 0.3
VOXEL = 0.5


def voxel_keys(pts, vs=VOXEL):
    k = np.floor(pts / vs).astype(np.int64) + (1 << 20)
    return np.unique((k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2])


def gt_voxels(res, gt, world, step=2):
    """Occupied voxels of the noise-free scene as seen from the GT poses (cached)."""
    cache = res / "data" / f"gt_voxels_{VOXEL}.npy"
    if cache.exists():
        return np.load(cache)
    gt_t, gt_p, gt_r = gt
    scan_t = np.arange(gt_t[0], gt_t[-1] - 0.1, 0.1)[::step]
    dirs = sw.beam_directions().reshape(-1, 3)
    keys = []
    for ts in scan_t:
        p, r = gt_at(gt, np.array([ts]))
        d = r[0].apply(dirs)
        rng, _, _ = world.raycast(np.repeat(p, len(d), 0), d)
        ok = np.isfinite(rng) & (rng > 1.0)
        keys.append(voxel_keys(p + d[ok] * rng[ok, None]))
    v = np.unique(np.concatenate(keys))
    np.save(cache, v)
    return v


def read_bag(path, topics):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(path), storage_id="sqlite3"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {"/Odometry": Odometry, "/cloud_weather_filtered": PointCloud2}
    out = {t: [] for t in topics}
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic in out:
            out[topic].append(deserialize_message(data, types[topic]))
    return out


def cloud_xyz(msg):
    off = {f.name: f.offset for f in msg.fields}
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(-1, msg.point_step)
    xyz = np.column_stack([raw[:, off[a]:off[a] + 4].copy().view("<f4")[:, 0] for a in "xyz"])
    return xyz.astype(np.float64)


def stamp_sec(h):
    return h.stamp.sec + h.stamp.nanosec * 1e-9


def load_run(run_dir):
    """Estimated poses (t, p, R) and per-frame map-candidate clouds from the recorded bag."""
    rec = read_bag(run_dir / "rec", ["/Odometry", "/cloud_weather_filtered"])
    od = rec["/Odometry"]
    t = np.array([stamp_sec(m.header) for m in od])
    p = np.array([[m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z] for m in od])
    q = np.array([[m.pose.pose.orientation.x, m.pose.pose.orientation.y,
                   m.pose.pose.orientation.z, m.pose.pose.orientation.w] for m in od])
    clouds = [(stamp_sec(m.header), cloud_xyz(m)) for m in rec["/cloud_weather_filtered"]]
    return t, p, Rot.from_quat(q), clouds


def load_gt(path):
    g = np.loadtxt(path)
    return g[:, 0], g[:, 1:4], Rot.from_quat(g[:, 4:8])


def gt_at(gt, times):
    gt_t, gt_p, gt_r = gt
    times = np.clip(times, gt_t[0], gt_t[-1])
    p = np.column_stack([np.interp(times, gt_t, gt_p[:, k]) for k in range(3)])
    idx = np.clip(np.searchsorted(gt_t, times) - 1, 0, len(gt_t) - 2)
    frac = (times - gt_t[idx]) / (gt_t[idx + 1] - gt_t[idx])
    r = gt_r[idx] * Rot.from_rotvec((gt_r[idx].inv() * gt_r[idx + 1]).as_rotvec() * frac[:, None])
    return p, r


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    u, _, vt = np.linalg.svd((dst - mu_d).T @ (src - mu_s))
    s = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        s[2, 2] = -1
    r = u @ s @ vt
    return r, mu_d - r @ mu_s


def anchored(est_p, est_r, gt_p0, gt_r0):
    """Express estimated poses in the GT world frame using only the first GT pose."""
    T_r = gt_r0 * est_r[0].inv()
    p = T_r.apply(est_p - est_p[0]) + gt_p0
    return p, T_r * est_r


def rpe(est_p, est_r, gt_p, gt_r, seg=10.0):
    d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(gt_p, axis=0), axis=1))])
    te, re = [], []
    for i in range(len(d)):
        j = np.searchsorted(d, d[i] + seg)
        if j >= len(d):
            break
        dg = gt_r[i].inv().apply(gt_p[j] - gt_p[i])
        de = est_r[i].inv().apply(est_p[j] - est_p[i])
        te.append(np.linalg.norm(dg - de) / seg * 100.0)
        rg = gt_r[i].inv() * gt_r[j]
        rr = est_r[i].inv() * est_r[j]
        re.append(np.degrees(np.linalg.norm((rg.inv() * rr).as_rotvec())))
    return float(np.mean(te)), float(np.mean(re))


def evaluate_run(run_dir, gt, world, v_gt=None):
    t, p, r, clouds = load_run(run_dir)
    if len(t) < 10:
        return {"error": "too few poses"}
    gp, gr = gt_at(gt, t)
    R, tr = umeyama(p, gp)
    ate = float(np.sqrt(np.mean(np.sum((p @ R.T + tr - gp) ** 2, axis=1))))
    ap, ar = anchored(p, r, gp[0], gr[0])
    err_anchor = np.linalg.norm(ap - gp, axis=1)
    rpe_t, rpe_r = rpe(p, r, gp, gr)

    # weather points offered to the map, with the frame's pose error removed
    n_total = n_noise = 0
    pose_index = {round(tt, 4): i for i, tt in enumerate(t)}
    per_frame_noise = []
    for ts, xyz in clouds:
        i = pose_index.get(round(ts, 4))
        if i is None or len(xyz) == 0:
            continue
        corr = gr[i] * r[i].inv()
        pw = corr.apply(xyz - p[i]) + gp[i]
        d = world.surface_distance(pw)
        k = int((d > NOISE_DIST).sum())
        n_total += len(xyz)
        n_noise += k
        per_frame_noise.append((ts, k, len(xyz)))

    iou = float("nan")
    if v_gt is not None and clouds:
        pts = np.concatenate([xyz for _, xyz in clouds if len(xyz)])
        v_est = voxel_keys(pts @ R.T + tr)
        inter = np.intersect1d(v_est, v_gt, assume_unique=True).size
        iou = inter / (v_est.size + v_gt.size - inter)

    import csv
    m = list(csv.DictReader(open(run_dir / "metrics.csv")))
    proc = np.array([float(x["total_processing_time"]) for x in m]) * 1000.0
    return {
        "frames": len(t),
        "ate_rmse_m": ate,
        "rpe_trans_pct": rpe_t,
        "rpe_rot_deg_per_10m": rpe_r,
        "end_drift_m": float(err_anchor[-1]),
        "max_err_anchored_m": float(err_anchor.max()),
        "map_candidates": n_total,
        "map_noise_points": n_noise,
        "map_noise_ratio": n_noise / max(1, n_total),
        "voxel_iou": iou,
        "proc_time_ms": float(proc.mean()),
        "_err_t": (t - t[0]).tolist(),
        "_err": err_anchor.tolist(),
        "_noise_frames": per_frame_noise,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--modes", nargs="+", default=MODES)
    args = ap.parse_args()
    res = Path(args.results)
    gt = load_gt(res / "data" / "gt_tum.txt")
    world = sw.World(seed=int((res / 'data' / 'world_seed.txt').read_text()) if (res / 'data' / 'world_seed.txt').exists() else 7)

    v_gt = gt_voxels(res, gt, world)
    summary = {}
    for w in WEATHERS:
        for m in args.modes:
            d = res / "runs" / w / m
            if not (d / "rec").exists():
                continue
            print(f"evaluating {w}/{m} ...", flush=True)
            summary.setdefault(w, {})[m] = evaluate_run(d, gt, world, v_gt)

    (res / "evaluation_full.json").write_text(json.dumps(summary))
    compact = {w: {m: {k: v for k, v in r.items() if not k.startswith("_")} for m, r in ms.items()}
               for w, ms in summary.items()}
    (res / "evaluation.json").write_text(json.dumps(compact, indent=2))

    hdr = (f"{'weather':12s} {'mode':17s} {'ATE[m]':>8s} {'RPE[%]':>7s} {'RPEr[deg]':>9s} "
           f"{'endDrift':>9s} {'mapNoise':>9s} {'noisePts':>9s} {'IoU':>6s} {'ms/scan':>8s}")
    lines = [hdr, "-" * len(hdr)]
    for w, ms in compact.items():
        for m, r in ms.items():
            if "error" in r:
                lines.append(f"{w:12s} {m:17s} {r['error']}")
                continue
            lines.append(f"{w:12s} {m:17s} {r['ate_rmse_m']:8.3f} {r['rpe_trans_pct']:7.2f} "
                         f"{r['rpe_rot_deg_per_10m']:9.3f} {r['end_drift_m']:9.3f} "
                         f"{r['map_noise_ratio'] * 100:8.2f}% {r['map_noise_points']:9d} {r['voxel_iou']:6.3f} "
                         f"{r['proc_time_ms']:8.1f}")
    table = "\n".join(lines)
    (res / "evaluation_table.txt").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
