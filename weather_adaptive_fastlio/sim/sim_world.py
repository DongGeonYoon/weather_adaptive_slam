#!/usr/bin/env python3
"""Synthetic urban world, vehicle trajectory, 32-beam spinning LiDAR and IMU.

Everything is deterministic (fixed seeds) so that every weather condition is
generated from the *same* clean world, trajectory and IMU stream.  Only the
LiDAR returns differ between weather conditions (see weather_models.py).

Coordinates: world z-up, ground plane z = 0.  The LiDAR frame and the IMU
frame coincide (identity extrinsic), mounted SENSOR_HEIGHT above the ground.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as Rot

SENSOR_HEIGHT = 1.9
GRAVITY = 9.81

# ---------------------------------------------------------------- LiDAR model
N_RINGS = 32
ELEV_DEG = np.linspace(-25.0, 15.0, N_RINGS)          # VLP-32C-like vertical FOV
H_RES_DEG = 0.2
N_COLS = int(round(360.0 / H_RES_DEG))                 # 1800 columns / revolution
SCAN_PERIOD = 0.1                                      # 10 Hz
MAX_RANGE = 100.0
MIN_RANGE = 0.3
RANGE_NOISE_STD = 0.02

IMU_RATE = 200.0
STATIC_START = 2.0       # seconds standing still (IMU initialisation)


# ---------------------------------------------------------------- the world
class World:
    """Ground plane + axis-aligned boxes + vertical cylinders.

    Every primitive has a reflectivity in [0, 1] used for return intensity.
    """

    def __init__(self, seed=7):
        rng = np.random.default_rng(seed)
        boxes = []   # (xmin, ymin, zmin, xmax, ymax, zmax, reflectivity)
        cyls = []    # (cx, cy, radius, height, reflectivity)

        # Road loop is a rounded rectangle x in [-60, 60], y in [-40, 40].
        # Buildings line both sides of the road with irregular setbacks and
        # heights so that the geometry is not degenerate.
        def building_row(x0, x1, y_face, outward, depth_rng=(8, 18)):
            x = x0
            while x < x1:
                w = rng.uniform(7, 16)
                gap = rng.uniform(1.5, 6.0)
                setback = rng.uniform(0.0, 4.0)
                depth = rng.uniform(*depth_rng)
                h = rng.uniform(5, 22)
                xa, xb = x, min(x + w, x1)
                if outward > 0:
                    ya, yb = y_face + setback, y_face + setback + depth
                else:
                    ya, yb = y_face - setback - depth, y_face - setback
                boxes.append((xa, ya, 0.0, xb, yb, h, rng.uniform(0.35, 0.7)))
                x += w + gap

        def building_col(y0, y1, x_face, outward, depth_rng=(8, 18)):
            y = y0
            while y < y1:
                w = rng.uniform(7, 16)
                gap = rng.uniform(1.5, 6.0)
                setback = rng.uniform(0.0, 4.0)
                depth = rng.uniform(*depth_rng)
                h = rng.uniform(5, 22)
                ya, yb = y, min(y + w, y1)
                if outward > 0:
                    xa, xb = x_face + setback, x_face + setback + depth
                else:
                    xa, xb = x_face - setback - depth, x_face - setback
                boxes.append((xa, ya, 0.0, xb, yb, h, rng.uniform(0.35, 0.7)))
                y += w + gap

        road_half = 7.0
        # outer ring of buildings
        building_row(-80, 80, 40 + road_half, +1)
        building_row(-80, 80, -40 - road_half, -1)
        building_col(-40, 40, 60 + road_half, +1)
        building_col(-40, 40, -60 - road_half, -1)
        # inner block (inside the loop)
        building_row(-50, 50, 40 - road_half, -1, depth_rng=(10, 14))
        building_row(-50, 50, -40 + road_half, +1, depth_rng=(10, 14))
        building_col(-25, 25, 60 - road_half, -1, depth_rng=(10, 14))
        building_col(-25, 25, -60 + road_half, +1, depth_rng=(10, 14))

        # parked cars along the curbs
        for _ in range(40):
            side = rng.integers(4)
            if side < 2:
                x = rng.uniform(-50, 50)
                y = (40 if side == 0 else -40) + rng.choice([-1, 1]) * (road_half - 1.3)
                boxes.append((x - 2.2, y - 0.9, 0.0, x + 2.2, y + 0.9, rng.uniform(1.4, 1.9),
                              rng.uniform(0.2, 0.8)))
            else:
                y = rng.uniform(-28, 28)
                x = (60 if side == 2 else -60) + rng.choice([-1, 1]) * (road_half - 1.3)
                boxes.append((x - 0.9, y - 2.2, 0.0, x + 0.9, y + 2.2, rng.uniform(1.4, 1.9),
                              rng.uniform(0.2, 0.8)))

        # poles / tree trunks / bollards on the sidewalks
        for _ in range(90):
            side = rng.integers(4)
            off = rng.choice([-1, 1]) * (road_half + rng.uniform(0.3, 1.5))
            if side < 2:
                x, y = rng.uniform(-70, 70), (40 if side == 0 else -40) + off
            else:
                x, y = (60 if side == 2 else -60) + off, rng.uniform(-38, 38)
            if rng.random() < 0.5:
                cyls.append((x, y, rng.uniform(0.08, 0.15), rng.uniform(4, 8), 0.6))   # pole
            else:
                cyls.append((x, y, rng.uniform(0.2, 0.4), rng.uniform(2.5, 5), 0.3))   # trunk

        self.boxes = np.array(boxes, dtype=np.float64)
        self.cyls = np.array(cyls, dtype=np.float64)
        self.ground_reflectivity = 0.25

    # ---------------------------------------------------------- ray casting
    def raycast(self, origins, dirs, max_range=MAX_RANGE):
        """origins (N,3), dirs (N,3) unit. Returns range (N,), reflectivity (N,), normal (N,3)."""
        n = len(dirs)
        best = np.full(n, np.inf)
        refl = np.zeros(n)
        normal = np.zeros((n, 3))

        # ground plane
        dz = dirs[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            tg = -origins[:, 2] / dz
        ok = (dz < -1e-6) & (tg > 0)
        best[ok] = tg[ok]
        refl[ok] = self.ground_reflectivity
        normal[ok] = (0, 0, 1)

        # cull boxes farther than max_range from the (approximately shared) origin
        o0 = origins.mean(axis=0)
        bx = self.boxes
        cx = np.clip(o0[0], bx[:, 0], bx[:, 3]) - o0[0]
        cy = np.clip(o0[1], bx[:, 1], bx[:, 4]) - o0[1]
        near = np.hypot(cx, cy) < max_range + 5
        bx = bx[near]
        inv = 1.0 / np.where(np.abs(dirs) < 1e-12, 1e-12, dirs)
        for b in bx:   # loop over boxes, vectorised over rays
            t1 = (b[0:3] - origins) * inv
            t2 = (b[3:6] - origins) * inv
            tmin = np.minimum(t1, t2)
            tmax = np.maximum(t1, t2)
            tn = tmin.max(axis=1)
            tf = tmax.min(axis=1)
            hit = (tf >= tn) & (tf > 0) & (tn > 0) & (tn < best)
            if not hit.any():
                continue
            best[hit] = tn[hit]
            refl[hit] = b[6]
            axis = tmin[hit].argmax(axis=1)
            nrm = np.zeros((hit.sum(), 3))
            nrm[np.arange(len(axis)), axis] = -np.sign(dirs[hit][np.arange(len(axis)), axis])
            normal[hit] = nrm

        cy_ = self.cyls
        near = np.hypot(cy_[:, 0] - o0[0], cy_[:, 1] - o0[1]) < max_range + 2
        for c in cy_[near]:
            ox = origins[:, 0] - c[0]
            oy = origins[:, 1] - c[1]
            a = dirs[:, 0] ** 2 + dirs[:, 1] ** 2
            bq = 2 * (ox * dirs[:, 0] + oy * dirs[:, 1])
            cq = ox ** 2 + oy ** 2 - c[2] ** 2
            disc = bq ** 2 - 4 * a * cq
            ok = disc > 0
            if not ok.any():
                continue
            t = np.full(n, np.inf)
            t[ok] = (-bq[ok] - np.sqrt(disc[ok])) / (2 * a[ok])
            z = origins[:, 2] + t * dirs[:, 2]
            hit = ok & (t > 0) & (z >= 0) & (z <= c[3]) & (t < best)
            if not hit.any():
                continue
            best[hit] = t[hit]
            refl[hit] = c[4]
            p = origins[hit] + t[hit, None] * dirs[hit]
            nr = np.column_stack([p[:, 0] - c[0], p[:, 1] - c[1], np.zeros(hit.sum())])
            normal[hit] = nr / np.linalg.norm(nr, axis=1, keepdims=True)

        best[best > max_range] = np.inf
        return best, refl, normal

    # ---------------------------------------------------------- distance field
    def surface_distance(self, pts):
        """Unsigned distance (m) from world points (N,3) to the nearest surface."""
        d = np.abs(pts[:, 2])                                   # ground
        for b in self.boxes:
            q = np.maximum(np.maximum(b[0:3] - pts, pts - b[3:6]), 0.0)
            outside = np.linalg.norm(q, axis=1)
            dist = np.where(outside == 0,
                            np.minimum(pts - b[0:3], b[3:6] - pts).min(axis=1), outside)
            d = np.minimum(d, dist)
        for c in self.cyls:
            r = np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1])
            dr = np.abs(r - c[2])
            dz = np.maximum(np.maximum(-pts[:, 2], pts[:, 2] - c[3]), 0)
            d = np.minimum(d, np.hypot(dr, dz))
        return d


# ---------------------------------------------------------------- trajectory
class Trajectory:
    """Smooth closed loop on the road with a static start, plus mild roll/pitch."""

    def __init__(self, cruise_speed=6.0, laps=1.15, dt=0.001):
        # waypoints of a rounded rectangle centred on the road loop
        corners = []
        r = 12.0
        for cx, cy, a0 in [(60 - r, 40 - r, 0), (-60 + r, 40 - r, 90),
                           (-60 + r, -40 + r, 180), (60 - r, -40 + r, 270)]:
            for a in np.linspace(a0, a0 + 90, 7):
                corners.append((cx + r * np.cos(np.radians(a)), cy + r * np.sin(np.radians(a))))
        loop = np.array(corners)
        loop = np.vstack([loop, loop[:1]])
        seg = np.linalg.norm(np.diff(loop, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])
        self.loop_len = s[-1]
        self.spline = CubicSpline(s, loop, bc_type="periodic")

        # arc-length profile: static, smooth acceleration, cruise
        accel_time = 5.0
        total_s = laps * self.loop_len
        cruise_time = (total_s - 0.5 * cruise_speed * accel_time) / cruise_speed
        self.duration = STATIC_START + accel_time + cruise_time
        t = np.arange(0.0, self.duration + 1e-9, dt)
        v = np.where(t < STATIC_START, 0.0,
                     np.where(t < STATIC_START + accel_time,
                              cruise_speed * 0.5 * (1 - np.cos(np.pi * (t - STATIC_START) / accel_time)),
                              cruise_speed))
        arc = np.cumsum(v) * dt
        arc -= arc[0]
        # start at the middle of the east straight (x=60, y=0): arc offset of that point
        s_dense = np.linspace(0, self.loop_len, 20000)
        p_dense = self.spline(s_dense)
        s0 = s_dense[np.argmin(np.hypot(p_dense[:, 0] - 60, p_dense[:, 1] - 0))]
        arc_abs = (arc + s0) % self.loop_len

        xy = self.spline(arc_abs)
        dxy = self.spline(arc_abs, 1)
        yaw = np.unwrap(np.arctan2(dxy[:, 1], dxy[:, 0]))
        # mild suspension motion (zero during the static phase)
        moving = np.clip((t - STATIC_START) / accel_time, 0, 1)
        roll = moving * (0.012 * np.sin(2 * np.pi * 0.7 * t) + 0.006 * np.sin(2 * np.pi * 1.9 * t + 1))
        pitch = moving * (0.010 * np.sin(2 * np.pi * 0.5 * t + 0.3) + 0.005 * np.sin(2 * np.pi * 2.3 * t))
        z = SENSOR_HEIGHT + moving * 0.03 * np.sin(2 * np.pi * 1.1 * t)

        self.t = t
        self.pos = np.column_stack([xy, z])
        self.rot = Rot.from_euler("ZYX", np.column_stack([yaw, pitch, roll]))
        self.dt = dt

    def pose_at(self, times):
        """Interpolated (positions (N,3), Rotation) at arbitrary times."""
        times = np.clip(times, self.t[0], self.t[-1])
        pos = np.column_stack([np.interp(times, self.t, self.pos[:, k]) for k in range(3)])
        idx = np.clip(np.searchsorted(self.t, times) - 1, 0, len(self.t) - 2)
        frac = (times - self.t[idx]) / self.dt
        r0 = self.rot[idx]
        r1 = self.rot[idx + 1]
        delta = (r0.inv() * r1).as_rotvec()
        rot = r0 * Rot.from_rotvec(delta * frac[:, None])
        return pos, rot

    def imu(self, rate=IMU_RATE, seed=11):
        """Simulated IMU (gyro rad/s, acc m/s^2, body frame) with noise and bias."""
        rng = np.random.default_rng(seed)
        dt = self.dt
        vel = np.gradient(self.pos, dt, axis=0)
        acc_w = np.gradient(vel, dt, axis=0)
        rv = (self.rot[:-1].inv() * self.rot[1:]).as_rotvec() / dt
        gyro_dense = np.vstack([rv, rv[-1:]])
        f_w = acc_w + np.array([0, 0, GRAVITY])
        acc_dense = self.rot.inv().apply(f_w)

        times = np.arange(self.t[0], self.t[-1], 1.0 / rate)
        idx = np.round(times / dt).astype(int)
        gyro = gyro_dense[idx]
        acc = acc_dense[idx]
        gyro_bias = np.array([0.002, -0.0015, 0.001])
        acc_bias = np.array([0.03, -0.02, 0.04])
        gyro = gyro + gyro_bias + rng.normal(0, 0.003, gyro.shape)
        acc = acc + acc_bias + rng.normal(0, 0.04, acc.shape)
        return times, gyro, acc


# ---------------------------------------------------------------- lidar scan
def beam_directions():
    """Unit directions in sensor frame, shape (N_COLS, N_RINGS, 3), and column azimuths."""
    az = np.radians(np.arange(N_COLS) * H_RES_DEG)
    el = np.radians(ELEV_DEG)
    ca, sa = np.cos(az)[:, None], np.sin(az)[:, None]
    ce, se = np.cos(el)[None, :], np.sin(el)[None, :]
    d = np.stack([ca * ce, sa * ce, np.broadcast_to(se, (N_COLS, N_RINGS))], axis=-1)
    return d


def simulate_clean_scan(world, traj, t_start, dirs_sensor, rng):
    """One revolution starting at t_start with per-column motion distortion.

    Returns dict of flat arrays (only valid hits): range, ring, time offset,
    reflectivity, incidence cosine, sensor-frame unit direction.  Also returns the
    ray arrays for *all* rays (needed by weather models for empty rays).
    """
    col_t = t_start + np.arange(N_COLS) * (SCAN_PERIOD / N_COLS)
    pos, rot = traj.pose_at(col_t)
    mats = rot.as_matrix()                                  # (C,3,3)
    dirs_w = np.einsum("cij,crj->cri", mats, dirs_sensor)    # (C,R,3)
    origins = np.repeat(pos[:, None, :], N_RINGS, axis=1)
    rng_, refl, normal = world.raycast(origins.reshape(-1, 3), dirs_w.reshape(-1, 3))
    cos_inc = np.abs(np.einsum("ij,ij->i", normal, dirs_w.reshape(-1, 3)))
    ring = np.tile(np.arange(N_RINGS), N_COLS)
    toff = np.repeat(np.arange(N_COLS) * (SCAN_PERIOD / N_COLS), N_RINGS)
    return {
        "range": rng_,                       # inf where no hit
        "refl": refl,
        "cos_inc": cos_inc,
        "ring": ring,
        "time": toff,
        "dir": dirs_sensor.reshape(-1, 3),
    }
