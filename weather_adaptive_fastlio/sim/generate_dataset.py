#!/usr/bin/env python3
"""Generate ROS 2 bags (clear / rain / fog / snow / all_weather) from the synthetic world.

Outputs in --out:
  bags/<condition>/            rosbag2 (sqlite3): /velodyne_points, /imu/data
  gt_tum.txt                   ground-truth sensor pose (TUM, 100 Hz, world frame)
  scans_<condition>.npz        decimated sensor-frame points + weather-noise labels
                               (for visualisation / evaluation only)
  world.npz                    scene primitives
"""

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import rosbag2_py
from builtin_interfaces.msg import Time
from rclpy.serialization import serialize_message
from sensor_msgs.msg import Imu, PointCloud2, PointField

import sim_world as sw
from weather_models import CONDITIONS, apply_weather

T0 = 1_700_000_000.0          # sim epoch (non-zero ROS time)
DISPLAY_DECIMATION = 3

POINT_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4"),
                        ("time", "<f4"), ("ring", "<u2"), ("_pad", "<u2")])


def stamp(t):
    sec = int(np.floor(t))
    return Time(sec=sec, nanosec=int(round((t - sec) * 1e9)) % 1_000_000_000)


def make_cloud(points, t_start):
    msg = PointCloud2()
    msg.header.stamp = stamp(t_start)
    msg.header.frame_id = "lidar"
    msg.height = 1
    msg.width = len(points)
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        PointField(name="time", offset=16, datatype=PointField.FLOAT32, count=1),
        PointField(name="ring", offset=20, datatype=PointField.UINT16, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = POINT_DTYPE.itemsize
    msg.row_step = msg.point_step * len(points)
    msg.is_dense = True
    msg.data = points.tobytes()
    return msg


def open_writer(path):
    if path.exists():
        shutil.rmtree(path)
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(path), storage_id="sqlite3"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    for name, typ in [("/velodyne_points", "sensor_msgs/msg/PointCloud2"),
                      ("/imu/data", "sensor_msgs/msg/Imu")]:
        writer.create_topic(rosbag2_py.TopicMetadata(name=name, type=typ, serialization_format="cdr"))
    return writer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--duration", type=float, default=None, help="truncate trajectory (s)")
    ap.add_argument("--world-seed", type=int, default=7)
    ap.add_argument("--speed", type=float, default=6.0)
    ap.add_argument("--seed-offset", type=int, default=0, help="shifts IMU/LiDAR/weather noise seeds")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "bags").mkdir(parents=True, exist_ok=True)

    world = sw.World(seed=args.world_seed)
    traj = sw.Trajectory(cruise_speed=args.speed)
    (out / "world_seed.txt").write_text(str(args.world_seed))
    duration = traj.duration if args.duration is None else min(args.duration, traj.duration)
    print(f"world: {len(world.boxes)} boxes, {len(world.cyls)} cylinders; "
          f"loop {traj.loop_len:.0f} m; duration {duration:.1f} s")
    np.savez(out / "world.npz", boxes=world.boxes, cyls=world.cyls)

    # ground truth (TUM, 100 Hz)
    gt_t = np.arange(0.0, duration, 0.01)
    gp, gr = traj.pose_at(gt_t)
    q = gr.as_quat()
    np.savetxt(out / "gt_tum.txt", np.column_stack([gt_t + T0, gp, q]), fmt="%.9f")

    imu_t, gyro, acc = traj.imu(seed=11 + args.seed_offset)
    keep = imu_t < duration
    imu_t, gyro, acc = imu_t[keep], gyro[keep], acc[keep]

    writers = {c: open_writer(out / "bags" / c) for c in args.conditions}
    rngs = {c: np.random.default_rng(100 + i + 1000 * args.seed_offset) for i, c in enumerate(args.conditions)}
    display = {c: {"pts": [], "noise": [], "t": []} for c in args.conditions}
    stats = {c: {"points": 0, "noise": 0} for c in args.conditions}

    dirs = sw.beam_directions()
    scan_starts = np.arange(0.0, duration - sw.SCAN_PERIOD, sw.SCAN_PERIOD)
    imu_i = 0
    clean_rng = np.random.default_rng(5 + args.seed_offset)
    tic = time.time()
    for k, ts in enumerate(scan_starts):
        te = ts + sw.SCAN_PERIOD
        # IMU messages up to the end of this scan
        while imu_i < len(imu_t) and imu_t[imu_i] <= te:
            m = Imu()
            m.header.stamp = stamp(imu_t[imu_i] + T0)
            m.header.frame_id = "imu"
            m.orientation_covariance[0] = -1.0
            m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = map(float, gyro[imu_i])
            m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = map(float, acc[imu_i])
            data = serialize_message(m)
            ns = int((imu_t[imu_i] + T0) * 1e9)
            for w in writers.values():
                w.write("/imu/data", data, ns)
            imu_i += 1

        scan = sw.simulate_clean_scan(world, traj, ts, dirs, clean_rng)
        base_noise = clean_rng.normal(0.0, sw.RANGE_NOISE_STD, len(scan["range"]))
        scan["range"] = scan["range"] + base_noise
        for c in args.conditions:
            r, inten, noise = apply_weather(scan, c, ts, rngs[c])
            ok = np.isfinite(r)
            p = scan["dir"][ok] * r[ok, None]
            pts = np.zeros(ok.sum(), dtype=POINT_DTYPE)
            pts["x"], pts["y"], pts["z"] = p[:, 0], p[:, 1], p[:, 2]
            pts["intensity"] = np.clip(inten[ok] * 255.0, 0, 255)
            pts["time"] = scan["time"][ok]
            pts["ring"] = scan["ring"][ok]
            writers[c].write("/velodyne_points", serialize_message(make_cloud(pts, ts + T0)),
                             int((te + T0) * 1e9))
            stats[c]["points"] += len(pts)
            stats[c]["noise"] += int(noise[ok].sum())
            sel = slice(None, None, DISPLAY_DECIMATION)
            display[c]["pts"].append(p[sel].astype(np.float16))
            display[c]["noise"].append(noise[ok][sel])
            display[c]["t"].append(ts + T0)
        if k % 50 == 0:
            print(f"scan {k}/{len(scan_starts)}  ({time.time() - tic:.0f}s)", flush=True)

    for c in args.conditions:
        del writers[c]
        d = display[c]
        lens = np.array([len(p) for p in d["pts"]])
        np.savez_compressed(out / f"scans_{c}.npz", pts=np.concatenate(d["pts"]),
                            noise=np.concatenate(d["noise"]), lens=lens, t=np.array(d["t"]))
        s = stats[c]
        print(f"{c:12s} mean pts/scan {s['points'] / len(scan_starts):8.0f}  "
              f"weather-noise ratio {s['noise'] / max(1, s['points']):.3f}")


if __name__ == "__main__":
    main()
