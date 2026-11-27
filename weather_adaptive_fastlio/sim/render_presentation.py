#!/usr/bin/env python3
"""Presentation videos: baseline FAST-LIO2 vs proposed weather-adaptive FAST-LIO2 (3D maps).

Per weather (1920x1080, 10 fps, Korean labels):
  [ baseline 3D map              | proposed 3D map              ]
  [ raw scan | position error over time | live metrics           ]
followed by a 4 s summary card (ATE, drift, RPE, map noise, voxel IoU).

Also writes an intro card, an overall summary chart (PNG + clip) and a montage of
everything:  <out>/{compare_<dataset>_<weather>.mp4, summary.png, montage.mp4}

The 3D panels accumulate each run's /cloud_weather_filtered (the map the method
outputs), placed with that run's estimated pose anchored at the first GT pose, so
drift shows up as a smeared map.  Points are voxelised (0.25 m) and splatted with
a depth buffer from a fixed oblique camera; structure is coloured by height,
points off every true surface (judged with the pose error removed) are weather
artefacts and drawn red.

usage: render_presentation.py --results R1 [R2 ...] --labels "검증 데이터셋 C" ["..."] --out <dir>
       then sim/make_highlight.sh <dir> for a ~1.5 min cut
"""

import argparse
import json
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
import numpy as np  # noqa: E402

import sim_world as sw  # noqa: E402
from evaluate_sim import NOISE_DIST, anchored, gt_at, load_gt, load_run  # noqa: E402

for f in ["/usr/share/fonts/truetype/nanum/NanumSquare_acR.ttf",
          "/usr/share/fonts/truetype/nanum/NanumSquare_acB.ttf"]:
    if Path(f).exists():
        font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = "NanumSquare_ac"
plt.rcParams["axes.unicode_minus"] = False

FPS = 10
SIZE = (19.2, 10.8)
MODES = ["baseline", "proposed"]                         # run directory names
NAME = {"baseline": "Baseline FAST-LIO2", "proposed": "제안 방법 (Weather-Adaptive)"}
COLOR = {"baseline": "#d9730d", "proposed": "#1f5fbf"}
WEATHER_KO = {"clear": "맑음 (CLEAR)", "rain": "비 (RAIN)", "fog": "안개 (FOG)",
              "snow": "눈 (SNOW)", "all_weather": "복합 악천후 (ALL WEATHER)"}
WEATHERS = list(WEATHER_KO)

# ----------------------------------------------------------------------------- 3D splatting
VOX = 0.25
BOUNDS = np.array([[-92.0, 92.0], [-72.0, 72.0], [-1.0, 14.0]])
PW, PH = 940, 640                      # panel raster size [px]
NOISE_RGB = np.array([0.90, 0.12, 0.18], np.float32)


class Camera:
    """Fixed oblique perspective camera looking at the scene centre."""

    def __init__(self, azim=-58.0, elev=38.0, dist=230.0, target=(0.0, -4.0, 0.0)):
        a, e = np.radians(azim), np.radians(elev)
        fwd = -np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        self.pos = np.array(target) - fwd * dist
        right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        self.R = np.stack([right, up, fwd])            # world -> camera rows
        corners = np.array(np.meshgrid(*BOUNDS)).reshape(3, -1).T
        c = (corners - self.pos) @ self.R.T
        u, v = c[:, 0] / c[:, 2], c[:, 1] / c[:, 2]
        self.f = 1.08 * min(PW / (u.max() - u.min()), PH / (v.max() - v.min()))
        self.cu = PW / 2 - self.f * (u.max() + u.min()) / 2
        self.cv = PH / 2 + self.f * (v.max() + v.min()) / 2

    def project(self, pts):
        c = (pts - self.pos) @ self.R.T
        u = self.cu + self.f * c[:, 0] / c[:, 2]
        v = self.cv - self.f * c[:, 1] / c[:, 2]
        return u, v, c[:, 2]


def height_rgb(z):
    """Ground light grey, structure grey-blue getting darker with height."""
    t = np.clip(z / 12.0, 0, 1)[:, None]
    low = np.array([0.80, 0.82, 0.85], np.float32)
    high = np.array([0.10, 0.22, 0.42], np.float32)
    rgb = low * (1 - t) + high * t
    rgb[z < 0.3] = (0.86, 0.86, 0.86)
    return rgb


class Splat:
    """Incremental voxel map rendered into a depth-buffered image."""

    def __init__(self, cam):
        self.cam = cam
        self.seen = set()
        self.img = np.ones((PH, PW, 3), np.float32)
        self.depth = np.full((PH, PW), np.inf, np.float32)
        self.noise_pts = 0
        self.total_pts = 0

    def add(self, pts, noise):
        self.total_pts += len(pts)
        self.noise_pts += int(noise.sum())
        ok = np.all((pts >= BOUNDS[:, 0]) & (pts <= BOUNDS[:, 1]), axis=1)
        pts, noise = pts[ok], noise[ok]
        if not len(pts):
            return
        key = np.floor((pts - BOUNDS[:, 0]) / VOX).astype(np.int64)
        k1 = (key[:, 0] << 32) | (key[:, 1] << 16) | key[:, 2]
        k1, first = np.unique(k1, return_index=True)
        new = np.fromiter((k not in self.seen for k in k1.tolist()), bool, len(k1))
        if not new.any():
            return
        self.seen.update(k1[new].tolist())
        sel = first[new]
        centre = (key[sel] + 0.5) * VOX + BOUNDS[:, 0]
        rgb = np.where(noise[sel, None], NOISE_RGB, height_rgb(centre[:, 2]))
        u, v, d = self.cam.project(centre)
        for du, dv in ((0, 0), (1, 0), (0, 1), (1, 1)):        # 2x2 px splat
            iu = (u + du).astype(int); iv = (v + dv).astype(int)
            m = (iu >= 0) & (iu < PW) & (iv >= 0) & (iv < PH) & (d > 0)
            iu, iv, dd, cc = iu[m], iv[m], d[m], rgb[m]
            order = np.argsort(-dd)                             # far first, near written last
            iu, iv, dd, cc = iu[order], iv[order], dd[order], cc[order]
            closer = dd < self.depth[iv, iu]
            self.img[iv[closer], iu[closer]] = cc[closer]
            np.minimum.at(self.depth, (iv[closer], iu[closer]), dd[closer])


# ----------------------------------------------------------------------------- data
def prepare_mode(run_dir, gt, world):
    """Estimated poses (anchored at the first GT pose) and per-frame output-map points."""
    t, p, r, clouds = load_run(run_dir)
    gp, gr = gt_at(gt, t)
    ap, ar = anchored(p, r, gp[0], gr[0])
    idx = {round(tt, 4): i for i, tt in enumerate(t)}
    frames = {}
    for ts, xyz in clouds:
        i = idx.get(round(ts, 4))
        if i is None:
            continue
        local = r[i].inv().apply(xyz - p[i])            # body frame
        shown = ar[i].apply(local) + ap[i]               # where the SLAM placed it
        truth = gr[i].apply(local) + gp[i]               # pose error removed
        frames[round(ts, 4)] = (shown, world.surface_distance(truth) > NOISE_DIST)
    return {"t": t, "pos": ap, "err": np.linalg.norm(ap - gp, axis=1), "frames": frames}


def load_metrics(res):
    f = res / "evaluation.json"
    return json.loads(f.read_text()) if f.exists() else {}


def ffmpeg_writer(path):
    return subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
                             "-s", "1920x1080", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                             "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium", str(path)],
                            stdin=subprocess.PIPE)


def write_fig(proc, fig, n=1):
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba()).tobytes()
    for _ in range(n):
        proc.stdin.write(buf)


# ----------------------------------------------------------------------------- cards
def summary_card(fig, weather, label, met):
    fig.clf()
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.88, f"{WEATHER_KO[weather]} — 정량 지표", ha="center", fontsize=34, fontweight="bold")
    fig.text(0.5, 0.82, label, ha="center", fontsize=18, color="0.35")
    rows = [("ATE RMSE [m]", "ate_rmse_m", 1, False, "{:.3f}"),
            ("종단 드리프트 [m]", "end_drift_m", 1, False, "{:.2f}"),
            ("RPE [%]", "rpe_trans_pct", 1, False, "{:.3f}"),
            ("맵 노이즈 비율 [%]", "map_noise_ratio", 100, False, "{:.1f}"),
            ("Voxel IoU", "voxel_iou", 1, True, "{:.3f}")]
    x_cols = [0.25, 0.5, 0.75]
    y0, dy = 0.68, 0.105
    fig.text(x_cols[0], y0, "지표", ha="center", fontsize=24, fontweight="bold")
    for j, m in enumerate(MODES):
        fig.text(x_cols[j + 1], y0, NAME[m], ha="center", fontsize=24, fontweight="bold", color=COLOR[m])
    fig.add_artist(plt.Line2D([0.12, 0.88], [y0 - 0.03, y0 - 0.03], color="0.3", lw=1.5))
    for i, (title, key, scale, higher, fmt) in enumerate(rows):
        y = y0 - (i + 1) * dy
        fig.text(x_cols[0], y, title, ha="center", fontsize=22)
        vals = [met.get(m, {}).get(key) for m in MODES]
        if None in vals:
            continue
        better = vals[1] > vals[0] if higher else vals[1] < vals[0]
        for j, v in enumerate(vals):
            s = fmt.format(v * scale)
            if j == 1 and vals[0] > 0:
                ch = (v / vals[0] - 1) * 100
                s += f"  ({'+' if ch >= 0 else '-'}{abs(ch):.0f}%)"
            win = (j == 1) == better
            fig.text(x_cols[j + 1], y, s, ha="center", fontsize=22,
                     fontweight="bold" if win else "normal", color="#1a7f37" if win else "black")
    fig.text(0.5, 0.06, "굵은 초록색 = 더 좋은 값 · 괄호 = baseline 대비 변화율 · 맵 노이즈는 실제 표면에서 0.3 m 이상 "
             "떨어진 점의 비율 · IoU는 0.5 m voxel 기준", ha="center", fontsize=14, color="0.4")


def intro_card(fig, labels):
    fig.clf()
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.66, "Weather-Adaptive 3D LiDAR SLAM", ha="center", fontsize=46, fontweight="bold")
    fig.text(0.5, 0.57, "Baseline FAST-LIO2 vs 제안 방법 — 악천후 시뮬레이션 3D 지도 비교", ha="center", fontsize=28)
    fig.text(0.5, 0.42, " · ".join(labels), ha="center", fontsize=20, color="0.3")
    fig.text(0.5, 0.35, "튜닝에 사용하지 않은 새 장면(world seed·속도·노이즈 seed 변경), 파라미터 고정",
             ha="center", fontsize=18, color="0.4")
    fig.text(0.5, 0.20, "빨간 점 = 지도에 들어간 날씨 노이즈(비·눈·안개 반사)   ·   검은 점선 = 실제 경로(GT)",
             ha="center", fontsize=17, color="0.4")


# ----------------------------------------------------------------------------- per-weather video
def render_weather(args):
    weather, res, label, out, max_frames = args
    gt = load_gt(res / "data" / "gt_tum.txt")
    seed_f = res / "data" / "world_seed.txt"
    world = sw.World(seed=int(seed_f.read_text()) if seed_f.exists() else 7)
    modes = {m: prepare_mode(res / "runs" / weather / m, gt, world) for m in MODES}
    met = load_metrics(res).get(weather, {})
    raw = np.load(res / "data" / f"scans_{weather}.npz")
    raw_off = np.concatenate([[0], np.cumsum(raw["lens"])])
    raw_t_end = raw["t"] + sw.SCAN_PERIOD
    gt_t, gt_p, _ = gt

    cam = Camera()
    splat = {m: Splat(cam) for m in MODES}
    gt_u, gt_v, _ = cam.project(gt_p)
    traj_uv = {m: cam.project(d["pos"])[:2] for m, d in modes.items()}
    t_ref = modes["proposed"]["t"]
    t0 = min(d["t"][0] for d in modes.values())
    ymax = max(0.5, max(d["err"].max() for d in modes.values()) * 1.08)

    fig = plt.figure(figsize=SIZE, dpi=100)
    proc = ffmpeg_writer(out)
    top = fig.add_gridspec(1, 2, left=0.01, right=0.99, top=0.855, bottom=0.30, wspace=0.02)
    bot = fig.add_gridspec(1, 3, width_ratios=[0.55, 1.6, 0.85], left=0.03, right=0.99,
                           top=0.245, bottom=0.055, wspace=0.12)
    ax_m = {m: fig.add_subplot(top[0, j]) for j, m in enumerate(MODES)}
    ax_raw = fig.add_subplot(bot[0, 0])
    ax_err = fig.add_subplot(bot[0, 1])
    ax_txt = fig.add_subplot(bot[0, 2])
    fig.suptitle(f"{WEATHER_KO[weather]}", fontsize=30, fontweight="bold", y=0.99)
    fig.text(0.5, 0.94, f"{label}  ·  같은 센서 데이터를 두 방법에 입력 · 각 방법이 만든 3D 지도 "
             "(높이에 따라 색, 빨강 = 날씨 노이즈)", ha="center", fontsize=15, color="0.35")

    for fi, ts in enumerate(t_ref):
        if max_frames and fi >= max_frames:
            break
        key = round(ts, 4)
        for m in MODES:
            f = modes[m]["frames"].get(key)
            if f is not None:
                splat[m].add(*f)

        gmask = gt_t <= ts
        for m in MODES:
            d, ax, s = modes[m], ax_m[m], splat[m]
            ax.clear()
            ax.imshow(s.img, interpolation="bilinear")
            ax.plot(gt_u[gmask], gt_v[gmask], "--", color="black", lw=1.4, label="실제 경로 (GT)")
            j = np.searchsorted(d["t"], ts, side="right")
            u, v = traj_uv[m]
            ax.plot(u[:j], v[:j], color=COLOR[m], lw=2.8, label="추정 경로")
            if j:
                ax.plot(u[j - 1], v[j - 1], "o", color=COLOR[m], ms=10, mec="white", mew=1.5)
            ax.set_xlim(0, PW); ax.set_ylim(PH, 0); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(COLOR[m]); sp.set_linewidth(3)
            err = d["err"][j - 1] if j else 0.0
            nr = s.noise_pts / max(1, s.total_pts) * 100
            ax.set_title(f"{NAME[m]}\n현재 위치 오차 {err:5.2f} m   ·   지도 내 노이즈 {nr:4.1f}%",
                         fontsize=18, color=COLOR[m], fontweight="bold")
            ax.plot([], [], "s", color=NOISE_RGB, label="날씨 노이즈")
            ax.plot([], [], "s", color=(0.25, 0.35, 0.52), label="구조물 (높이별 색)")
            ax.legend(loc="lower right", fontsize=11, framealpha=0.92)

        # raw input (sensor frame, top view)
        ax_raw.clear()
        k = int(np.argmin(np.abs(raw_t_end - ts)))
        pts = raw["pts"][raw_off[k]:raw_off[k + 1]]
        nz = raw["noise"][raw_off[k]:raw_off[k + 1]]
        ax_raw.scatter(pts[~nz, 0], pts[~nz, 1], s=0.15, c="#333333", linewidths=0)
        ax_raw.scatter(pts[nz, 0], pts[nz, 1], s=0.8, c="#e63946", linewidths=0)
        ax_raw.set_xlim(-35, 35); ax_raw.set_ylim(-35, 35); ax_raw.set_aspect("equal")
        ax_raw.set_xticks([]); ax_raw.set_yticks([])
        ax_raw.set_title(f"입력 스캔 · 날씨 노이즈 {nz.mean() * 100:.1f}%", fontsize=13)

        # error plot
        ax_err.clear()
        for m in MODES:
            d = modes[m]
            j = np.searchsorted(d["t"], ts, side="right")
            ax_err.plot(d["t"][:j] - t0, d["err"][:j], color=COLOR[m], lw=2.6, label=NAME[m])
        ax_err.set_xlim(0, t_ref[-1] - t0); ax_err.set_ylim(0, ymax)
        ax_err.axvline(ts - t0, color="k", lw=0.8)
        ax_err.set_xlabel("시간 [s]", fontsize=12); ax_err.set_ylabel("위치 오차 [m]", fontsize=12)
        ax_err.set_title("시간에 따른 위치 오차 (시작 자세만 GT에 맞춤)", fontsize=13)
        ax_err.grid(alpha=0.3); ax_err.legend(loc="upper left", fontsize=11, ncol=2)

        # metric box (final values of the full run)
        ax_txt.clear(); ax_txt.axis("off")
        ax_txt.text(0.0, 0.98, "전체 주행 최종 지표", fontsize=15, fontweight="bold", va="top")
        rows = [("", "ATE [m]", "노이즈 [%]", "IoU")]
        for m in MODES:
            r = met.get(m)
            if r:
                rows.append((NAME[m].split(" (")[0], f"{r['ate_rmse_m']:.3f}",
                             f"{r['map_noise_ratio'] * 100:.1f}", f"{r['voxel_iou']:.3f}"))
        for i, row in enumerate(rows):
            y = 0.72 - i * 0.25
            c = COLOR[MODES[i - 1]] if i else "0.3"
            for x, s in zip([0.0, 0.5, 0.7, 0.9], row):
                ax_txt.text(x, y, s, fontsize=13, va="top", color=c,
                            fontweight="bold" if i == 2 else "normal")

        write_fig(proc, fig)

    hold = plt.figure(figsize=SIZE, dpi=100)
    summary_card(hold, weather, label, met)
    write_fig(proc, hold, n=4 * FPS)
    plt.close(hold)
    proc.stdin.close(); proc.wait(); plt.close(fig)
    return str(out)


# ----------------------------------------------------------------------------- overall summary
def overall_summary(datasets, out_png):
    metrics = [("ate_rmse_m", "ATE RMSE [m] (낮을수록 좋음)", 1),
               ("map_noise_ratio", "맵 노이즈 비율 [%] (낮을수록 좋음)", 100),
               ("voxel_iou", "Voxel IoU (높을수록 좋음)", 1)]
    fig, axes = plt.subplots(len(datasets), 3, figsize=SIZE, dpi=100, squeeze=False)
    fig.suptitle("검증 데이터셋 결과 요약 — 날씨별 비교", fontsize=28, fontweight="bold")
    x = np.arange(len(WEATHERS))
    wbar = 0.38
    for r, (label, res) in enumerate(datasets):
        ev = load_metrics(res)
        for c, (key, title, scale) in enumerate(metrics):
            ax = axes[r, c]
            for k, m in enumerate(MODES):
                v = [ev.get(w, {}).get(m, {}).get(key, np.nan) * scale for w in WEATHERS]
                ax.bar(x + (k - 0.5) * wbar, v, wbar, color=COLOR[m], label=NAME[m])
            ax.set_xticks(x)
            ax.set_xticklabels(["맑음", "비", "안개", "눈", "복합"], fontsize=13)
            ax.set_title(f"{label} — {title}", fontsize=14)
            ax.grid(axis="y", alpha=0.3)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, fontsize=15, bbox_to_anchor=(0.5, 0.935), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(out_png)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weathers", nargs="+", default=WEATHERS)
    ap.add_argument("--max-frames", type=int, default=0, help="preview: stop each clip after N frames")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    datasets = [(l, Path(r)) for l, r in zip(args.labels, args.results)]

    jobs = [(w, res, label, out / f"compare_{res.name}_{w}.mp4", args.max_frames)
            for label, res in datasets for w in args.weathers]
    with Pool(min(10, len(jobs))) as pool:
        for o in pool.imap_unordered(render_weather, jobs):
            print("wrote", o, flush=True)

    fig = plt.figure(figsize=SIZE, dpi=100)
    intro_card(fig, args.labels)
    fig.savefig(out / "intro.png")
    proc = ffmpeg_writer(out / "intro.mp4"); write_fig(proc, fig, n=5 * FPS); proc.stdin.close(); proc.wait()
    plt.close(fig)
    fig = overall_summary(datasets, out / "summary.png")
    proc = ffmpeg_writer(out / "summary.mp4"); write_fig(proc, fig, n=8 * FPS); proc.stdin.close(); proc.wait()
    plt.close(fig)

    lst = out / "concat.txt"
    lst.write_text("".join(f"file '{p.resolve()}'\n" for p in [out / "intro.mp4", *(j[3] for j in jobs),
                                                                  out / "summary.mp4"]))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c", "copy", str(out / "montage.mp4")], check=True)
    print("wrote", out / "montage.mp4")


if __name__ == "__main__":
    sys.exit(main())
