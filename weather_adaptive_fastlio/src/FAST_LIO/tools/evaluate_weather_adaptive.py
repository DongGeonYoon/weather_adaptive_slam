#!/usr/bin/env python3
"""Compare baseline FAST-LIO2 and Weather-Adaptive FAST-LIO2 experiments."""

import argparse
import csv
import json
import math
import struct
from pathlib import Path

import numpy as np


def load_metrics(path):
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    data = np.atleast_1d(data)
    if data.dtype.names is None:
        raise ValueError(f"Metrics file has no CSV header: {path}")
    return data


def load_boreas_gt(path):
    """Load Boreas ground truth from official CSV or TUM trajectory text.

    Official Boreas applanix CSV files usually contain GPSTime/easting/northing/
    altitude columns. Some local conversion pipelines export the same reference
    trajectory in TUM format:

        timestamp tx ty tz qx qy qz qw

    Supporting both keeps ATE/RMSE evaluation tied to the actual sequence file
    instead of hard-coding one result format.
    """
    path = Path(path)
    with path.open(newline="") as stream:
        first_line = stream.readline()
        stream.seek(0)
        if "GPSTime" in first_line and "easting" in first_line:
            rows = list(csv.DictReader(stream))
            timestamps = np.array([float(row["GPSTime"]) * 1e-6 for row in rows])
            positions = np.array([[float(row["easting"]), float(row["northing"]), float(row["altitude"])]
                                  for row in rows])
            return timestamps, positions

    data = np.loadtxt(path, comments="#")
    data = np.atleast_2d(data)
    if data.shape[1] < 4:
        raise ValueError(f"Unsupported ground-truth format: {path}")
    return data[:, 0].astype(float), data[:, 1:4].astype(float)


def interpolate_positions(query_t, source_t, source_p):
    return np.column_stack([np.interp(query_t, source_t, source_p[:, axis]) for axis in range(3)])


def rigid_align(source, target):
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - source_center).T @ (target - target_center))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    return (source - source_center) @ rotation.T + target_center


def trajectory_metrics(metrics, gt_t, gt_p, rte_interval=1.0, time_bounds=None):
    if len(metrics) == 0:
        return {"error": "empty metrics"}
    timestamps = metrics["timestamp"].astype(float)
    estimated = np.column_stack([metrics["pose_x"], metrics["pose_y"], metrics["pose_z"]]).astype(float)
    valid = (timestamps >= gt_t[0]) & (timestamps <= gt_t[-1])
    if time_bounds is not None:
        valid &= (timestamps >= time_bounds[0]) & (timestamps <= time_bounds[1])
    timestamps, estimated = timestamps[valid], estimated[valid]
    if len(timestamps) < 3:
        return {"error": "insufficient timestamp overlap"}

    ground_truth = interpolate_positions(timestamps, gt_t, gt_p)
    aligned = rigid_align(estimated, ground_truth)
    errors = np.linalg.norm(aligned - ground_truth, axis=1)

    relative_errors = []
    for index, timestamp in enumerate(timestamps):
        end = np.searchsorted(timestamps, timestamp + rte_interval)
        if end >= len(timestamps):
            break
        est_delta = aligned[end] - aligned[index]
        gt_delta = ground_truth[end] - ground_truth[index]
        relative_errors.append(np.linalg.norm(est_delta - gt_delta))
    relative_errors = np.asarray(relative_errors)

    gt_steps = np.linalg.norm(np.diff(ground_truth, axis=0), axis=1)
    gt_path_length = float(gt_steps.sum())
    end_error = float(np.linalg.norm((aligned[-1] - aligned[0]) -
                                     (ground_truth[-1] - ground_truth[0])))
    return {
        "samples": int(len(timestamps)),
        "ate_rmse_m": float(np.sqrt(np.mean(errors ** 2))),
        "ate_mean_m": float(errors.mean()),
        "rte_rmse_m": float(np.sqrt(np.mean(relative_errors ** 2))) if len(relative_errors) else None,
        "rte_interval_s": rte_interval,
        "end_to_end_drift_m": end_error,
        "end_to_end_drift_percent": 100.0 * end_error / max(gt_path_length, 1e-9),
        "gt_path_length_m": gt_path_length,
    }


def experiment_metrics(metrics):
    if len(metrics) == 0:
        return {"error": "empty metrics"}
    confidence = metrics["lidar_confidence"].astype(float)
    additions = metrics["map_update_point_count"].astype(float)
    total_additions = float(additions.sum())
    low_conf_additions = float(additions[confidence < 0.5].sum())
    return {
        "residual_variance_mean": float(np.mean(metrics["residual_variance"])),
        "residual_mean": float(np.mean(metrics["residual_mean"])),
        "runtime_mean_ms": 1000.0 * float(np.mean(metrics["total_processing_time"])),
        "runtime_p95_ms": 1000.0 * float(np.percentile(metrics["total_processing_time"], 95)),
        "iekf_update_mean_ms": 1000.0 * float(np.mean(metrics["iekf_update_time"])),
        "mean_confidence": float(confidence.mean()),
        "mean_degradation": float(np.mean(metrics["degradation_score"])),
        "rank_valid_ratio_mean": float(np.mean(metrics["rank_valid_ratio"])),
        "rank_valid_ratio_min": float(np.min(metrics["rank_valid_ratio"])),
        "low_confidence_map_addition_ratio": low_conf_additions / max(total_additions, 1.0),
        "map_update_points": int(total_additions),
    }


def pcd_xyz_sample(path, max_points=500000):
    path = Path(path)
    with path.open("rb") as stream:
        header, offset = [], 0
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"Invalid PCD header: {path}")
            offset += len(line)
            decoded = line.decode("latin1").strip()
            header.append(decoded)
            if decoded.startswith("DATA "):
                break
    values = {line.split()[0]: line.split()[1:] for line in header if line.split()}
    if values["DATA"][0] != "binary":
        raise ValueError("Only binary PCD is supported")
    fields, sizes, types = values["FIELDS"], map(int, values["SIZE"]), values["TYPE"]
    dtype = []
    for name, size, kind in zip(fields, sizes, types):
        code = {("F", 4): "<f4", ("F", 8): "<f8", ("U", 2): "<u2", ("U", 4): "<u4",
                ("I", 4): "<i4"}.get((kind, size), f"V{size}")
        dtype.append((name, code))
    count = int(values["POINTS"][0])
    cloud = np.memmap(path, dtype=np.dtype(dtype), mode="r", offset=offset, shape=(count,))
    stride = max(1, count // max_points)
    sample = cloud[::stride]
    xyz = np.column_stack([sample[axis].astype(float) for axis in ("x", "y", "z")])
    return xyz[np.isfinite(xyz).all(axis=1)]


def map_metrics(path):
    points = pcd_xyz_sample(path)
    result = {"sample_points": int(len(points))}
    try:
        from scipy.spatial import cKDTree
        distances, _ = cKDTree(points).query(points, k=2, workers=-1)
        nearest = distances[:, 1]
        median = float(np.median(nearest))
        mad = float(np.median(np.abs(nearest - median)))
        result.update({
            "nearest_neighbor_median_m": median,
            "nearest_neighbor_p95_m": float(np.percentile(nearest, 95)),
            "map_noise_outlier_ratio": float(np.mean(nearest > median + 3.0 * max(mad, 1e-6))),
        })
    except ImportError:
        result["warning"] = "scipy unavailable; nearest-neighbor map noise skipped"
    p01, p99 = np.percentile(points, [1, 99], axis=0)
    result["robust_extent_xyz_m"] = (p99 - p01).tolist()
    return result, points


def quaternion_from_rpy_degrees(roll, pitch, yaw):
    roll, pitch, yaw = np.deg2rad([roll, pitch, yaw]) * 0.5
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return (sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy)


def export_tum(metrics, path):
    with open(path, "w") as stream:
        for row in metrics:
            quaternion = quaternion_from_rpy_degrees(row["roll"], row["pitch"], row["yaw"])
            stream.write(f"{row['timestamp']:.9f} {row['pose_x']:.9f} {row['pose_y']:.9f} {row['pose_z']:.9f} "
                         f"{quaternion[0]:.9f} {quaternion[1]:.9f} {quaternion[2]:.9f} {quaternion[3]:.9f}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--proposed", required=True)
    parser.add_argument("--ground-truth")
    parser.add_argument("--baseline-map")
    parser.add_argument("--proposed-map")
    parser.add_argument("--output", default="weather_adaptive_evaluation.json")
    parser.add_argument("--tum-dir")
    parser.add_argument("--plot", help="Save baseline/proposed top and side map comparison")
    args = parser.parse_args()

    baseline, proposed = load_metrics(args.baseline), load_metrics(args.proposed)
    report = {"baseline": experiment_metrics(baseline), "proposed": experiment_metrics(proposed)}
    if "runtime_mean_ms" in report["baseline"] and "runtime_mean_ms" in report["proposed"]:
        report["runtime_overhead_percent"] = 100.0 * (report["proposed"]["runtime_mean_ms"] /
                                                       max(report["baseline"]["runtime_mean_ms"], 1e-9) - 1.0)

    if args.ground_truth:
        gt_t, gt_p = load_boreas_gt(args.ground_truth)
        if len(baseline) and len(proposed):
            common_start = max(float(np.min(baseline["timestamp"])),
                               float(np.min(proposed["timestamp"])), gt_t[0])
            common_end = min(float(np.max(baseline["timestamp"])),
                             float(np.max(proposed["timestamp"])), gt_t[-1])
            common_bounds = (common_start, common_end)
            report["comparison_interval"] = {
                "start": common_start,
                "end": common_end,
                "duration_s": max(0.0, common_end - common_start),
            }
        else:
            common_bounds = None
            report["comparison_interval"] = {"error": "baseline/proposed metrics must both contain data rows"}
        report["baseline"]["trajectory"] = trajectory_metrics(baseline, gt_t, gt_p, time_bounds=common_bounds)
        report["proposed"]["trajectory"] = trajectory_metrics(proposed, gt_t, gt_p, time_bounds=common_bounds)
    map_points = {}
    for name, path in (("baseline", args.baseline_map), ("proposed", args.proposed_map)):
        if path:
            report[name]["map"], map_points[name] = map_metrics(path)
    if args.tum_dir:
        directory = Path(args.tum_dir)
        directory.mkdir(parents=True, exist_ok=True)
        export_tum(baseline, directory / "baseline.tum")
        export_tum(proposed, directory / "proposed.tum")
    if args.plot and len(map_points) == 2:
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(2, 2, figsize=(14, 10))
        for column, name in enumerate(("baseline", "proposed")):
            points = map_points[name][::max(1, len(map_points[name]) // 150000)]
            axes[0, column].scatter(points[:, 0], points[:, 1], s=0.1)
            axes[0, column].set_title(f"{name} top view")
            axes[0, column].set_aspect("equal")
            axes[1, column].scatter(points[:, 0], points[:, 2], s=0.1)
            axes[1, column].set_title(f"{name} side view")
        figure.tight_layout()
        figure.savefig(args.plot, dpi=180)

    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
