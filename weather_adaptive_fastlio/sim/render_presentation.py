#!/usr/bin/env python3
"""Presentation videos: baseline FAST-LIO2 vs proposed weather-adaptive FAST-LIO2.

Per weather (1920x1080, 10 fps, Korean labels):
  [ raw LiDAR input | baseline map | proposed map ]
  [ position error over time (baseline / original adaptive / proposed) | live metrics ]
followed by a 4 s summary card (ATE, drift, map noise, voxel IoU for all three methods).

Also writes an intro card, an overall summary chart (PNG + clip) and a montage of
everything:  <results>/presentation/{compare_<weather>.mp4, summary.png, montage.mp4}

Map panels accumulate each run's /cloud_weather_filtered (the points the method
outputs as map) placed with that run's estimated pose anchored at the first GT pose,
so drift shows up as a smeared map.  Points off every true surface (judged with the
pose error removed) are weather artefacts and are drawn red.

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
XLIM, YLIM, RES = (-90.0, 90.0), (-70.0, 70.0), 0.25
W = int((XLIM[1] - XLIM[0]) / RES)
H = int((YLIM[1] - YLIM[0]) / RES)

MODES = ["baseline", "original", "proposed"]           # run directory names
MAP_MODES = ["baseline", "proposed"]                      # drawn as map panels
NAME = {"baseline": "Baseline FAST-LIO2", "original": "기존 Weather-Adaptive", "proposed": "제안 방법 (수정본)"}
COLOR = {"baseline": "#d9730d", "original": "#8a8a8a", "proposed": "#1f5fbf"}
WEATHER_KO = {"clear": "맑음 (CLEAR)", "rain": "비 (RAIN)", "fog": "안개 (FOG)",
              "snow": "눈 (SNOW)", "all_weather": "복합 악천후 (ALL WEATHER)"}
WEATHERS = list(WEATHER_KO)


def raster_idx(xy):
    ix = ((xy[:, 0] - XLIM[0]) / RES).astype(int)
    iy = ((xy[:, 1] - YLIM[0]) / RES).astype(int)
    ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    return iy[ok], ix[ok]


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
        frames[round(ts, 4)] = (ts, shown, world.surface_distance(truth) > NOISE_DIST)
    return {"t": t, "pos": ap, "err": np.linalg.norm(ap - gp, axis=1), "frames": frames}


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


def load_metrics(res):
    f = res / "evaluation.json"
    return json.loads(f.read_text()) if f.exists() else {}


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
    x_cols = [0.22, 0.44, 0.63, 0.82]
    y0, dy = 0.68, 0.105
    fig.text(x_cols[0], y0, "지표", ha="center", fontsize=22, fontweight="bold")
    for j, m in enumerate(MODES):
        fig.text(x_cols[j + 1], y0, NAME[m], ha="center", fontsize=22, fontweight="bold", color=COLOR[m])
    fig.add_artist(plt.Line2D([0.1, 0.92], [y0 - 0.03, y0 - 0.03], color="0.3", lw=1.5))
    for i, (title, key, scale, higher, fmt) in enumerate(rows):
        y = y0 - (i + 1) * dy
        fig.text(x_cols[0], y, title, ha="center", fontsize=21)
        vals = [met.get(m, {}).get(key) for m in MODES]
        ok = [v for v in vals if v is not None]
        best = (max(ok) if higher else min(ok)) if ok else None
        for j, v in enumerate(vals):
            if v is None:
                fig.text(x_cols[j + 1], y, "-", ha="center", fontsize=21)
                continue
            s = fmt.format(v * scale)
            if MODES[j] == "proposed" and vals[0] is not None and vals[0] > 0:
                ch = (v / vals[0] - 1) * 100
                s += f"  ({'+' if ch >= 0 else '-'}{abs(ch):.0f}%)"
            is_best = v == best
            fig.text(x_cols[j + 1], y, s, ha="center", fontsize=21,
                     fontweight="bold" if is_best else "normal",
                     color="#1a7f37" if is_best else "black")
    fig.text(0.5, 0.06, "굵은 초록색 = 세 방법 중 최고 · 괄호 = baseline 대비 변화율 · 맵 노이즈는 실제 표면에서 0.3 m 이상 떨어진 점의 비율 · "
             "IoU는 0.5 m voxel 기준", ha="center", fontsize=14, color="0.4")


def intro_card(fig, labels):
    fig.clf()
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.66, "Weather-Adaptive 3D LiDAR SLAM", ha="center", fontsize=46, fontweight="bold")
    fig.text(0.5, 0.57, "Baseline FAST-LIO2 vs 제안 방법 — 악천후 시뮬레이션 비교", ha="center", fontsize=28)
    fig.text(0.5, 0.42, " · ".join(labels), ha="center", fontsize=20, color="0.3")
    fig.text(0.5, 0.35, "튜닝에 사용하지 않은 새 장면(world seed·속도·노이즈 seed 변경), 파라미터 고정",
             ha="center", fontsize=18, color="0.4")
    fig.text(0.5, 0.20, "빨간 점 = 날씨 노이즈(비·눈·안개 반사)   ·   검은 점선 = 실제 경로(GT)",
             ha="center", fontsize=17, color="0.4")


# ----------------------------------------------------------------------------- per-weather video
def render_weather(args):
    weather, res, label, out = args
    gt = load_gt(res / "data" / "gt_tum.txt")
    seed_f = res / "data" / "world_seed.txt"
    world = sw.World(seed=int(seed_f.read_text()) if seed_f.exists() else 7)
    modes = {m: prepare_mode(res / "runs" / weather / m, gt, world)
             for m in MODES if (res / "runs" / weather / m / "rec").exists()}
    met = load_metrics(res).get(weather, {})
    raw = np.load(res / "data" / f"scans_{weather}.npz")
    raw_off = np.concatenate([[0], np.cumsum(raw["lens"])])
    raw_t_end = raw["t"] + sw.SCAN_PERIOD
    gt_t, gt_p, _ = gt

    t_ref = modes["proposed"]["t"]
    t0 = min(d["t"][0] for d in modes.values())
    acc = {m: np.zeros((H, W, 2), np.float32) for m in MAP_MODES}
    cnt = {m: [0, 0] for m in MAP_MODES}         # [noise, total]

    fig = plt.figure(figsize=SIZE, dpi=100)
    proc = ffmpeg_writer(out)
    gs = fig.add_gridspec(2, 3, height_ratios=[4.0, 1.25], width_ratios=[0.78, 1.1, 1.1],
                          left=0.045, right=0.985, top=0.885, bottom=0.07, wspace=0.05, hspace=0.14)
    ax_raw = fig.add_subplot(gs[0, 0])
    ax_m = {"baseline": fig.add_subplot(gs[0, 1]), "proposed": fig.add_subplot(gs[0, 2])}
    ax_err = fig.add_subplot(gs[1, 0:2])
    ax_txt = fig.add_subplot(gs[1, 2])
    fig.suptitle(f"{WEATHER_KO[weather]}", fontsize=30, fontweight="bold", y=0.975)
    fig.text(0.5, 0.915, f"{label}  ·  같은 센서 데이터를 두 방법에 입력 (위에서 본 지도, 높이 오차는 아래 그래프)",
             ha="center", fontsize=15, color="0.35")
    ymax = max(0.5, max(d["err"].max() for d in modes.values()) * 1.08)

    for fi, ts in enumerate(t_ref):
        key = round(ts, 4)
        for m in MAP_MODES:
            f = modes[m]["frames"].get(key)
            if f is None:
                continue
            _, shown, noise = f
            iy, ix = raster_idx(shown[~noise, :2]); np.add.at(acc[m][..., 0], (iy, ix), 1)
            iy, ix = raster_idx(shown[noise, :2]); np.add.at(acc[m][..., 1], (iy, ix), 1)
            cnt[m][0] += int(noise.sum()); cnt[m][1] += len(noise)

        # raw input
        ax_raw.clear()
        k = int(np.argmin(np.abs(raw_t_end - ts)))
        pts = raw["pts"][raw_off[k]:raw_off[k + 1]]
        nz = raw["noise"][raw_off[k]:raw_off[k + 1]]
        ax_raw.scatter(pts[~nz, 0], pts[~nz, 1], s=0.25, c="#333333", linewidths=0)
        ax_raw.scatter(pts[nz, 0], pts[nz, 1], s=1.2, c="#e63946", linewidths=0)
        ax_raw.set_xlim(-40, 40); ax_raw.set_ylim(-40, 40); ax_raw.set_aspect("equal")
        ax_raw.set_xticks([]); ax_raw.set_yticks([])
        ax_raw.set_title(f"입력 LiDAR 스캔 (센서 기준, 반경 40 m)\n날씨 노이즈(빨강) {nz.mean() * 100:.1f}%",
                         fontsize=15)

        # maps
        gmask = gt_t <= ts
        for m in MAP_MODES:
            d, ax = modes[m], ax_m[m]
            ax.clear()
            a = acc[m]
            img = np.ones((H, W, 3), np.float32)
            s = np.clip(a[..., 0] / 3.0, 0, 1)[..., None]
            img = img * (1 - 0.85 * s)
            n = 0.6 * np.clip(a[..., 1] / 4.0, 0, 1)[..., None]
            img = img * (1 - n) + n * np.array([0.92, 0.25, 0.30])
            ax.imshow(img, origin="lower", extent=(*XLIM, *YLIM), interpolation="nearest")
            ax.plot(gt_p[gmask, 0], gt_p[gmask, 1], "--", color="black", lw=1.5, label="실제 경로 (GT)")
            j = np.searchsorted(d["t"], ts, side="right")
            ax.plot(d["pos"][:j, 0], d["pos"][:j, 1], color=COLOR[m], lw=2.8, label="추정 경로")
            if j:
                ax.plot(*d["pos"][j - 1, :2], "o", color=COLOR[m], ms=10, mec="white", mew=1.5)
            ax.set_xlim(*XLIM); ax.set_ylim(*YLIM); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(COLOR[m]); sp.set_linewidth(3)
            err = d["err"][j - 1] if j else 0.0
            nr = cnt[m][0] / max(1, cnt[m][1]) * 100
            ax.set_title(f"{NAME[m]}\n현재 위치 오차 {err:5.2f} m   ·   지도 내 노이즈 {nr:4.1f}%",
                         fontsize=17, color=COLOR[m], fontweight="bold")
            ax.plot([], [], "s", color=(0.92, 0.45, 0.48), label="지도에 들어간 날씨 노이즈")
            ax.plot([], [], "s", color="0.25", label="지도에 들어간 구조물")
            ax.legend(loc="lower right", fontsize=11, framealpha=0.92)

        # error plot
        ax_err.clear()
        for m in MODES:
            if m not in modes:
                continue
            d = modes[m]
            j = np.searchsorted(d["t"], ts, side="right")
            lw = 1.4 if m == "original" else 2.6
            ls = ":" if m == "original" else "-"
            ax_err.plot(d["t"][:j] - t0, d["err"][:j], ls, color=COLOR[m], lw=lw, label=NAME[m])
        ax_err.set_xlim(0, t_ref[-1] - t0); ax_err.set_ylim(0, ymax)
        ax_err.axvline(ts - t0, color="k", lw=0.8)
        ax_err.set_xlabel("시간 [s]", fontsize=13); ax_err.set_ylabel("위치 오차 [m]", fontsize=13)
        ax_err.set_title("시간에 따른 위치 오차 (시작 자세만 GT에 맞춤, 높이 오차 포함)", fontsize=14)
        ax_err.grid(alpha=0.3); ax_err.legend(loc="upper left", fontsize=12, ncol=3)

        # metric box (final values of the full run)
        ax_txt.clear(); ax_txt.axis("off")
        ax_txt.text(0.02, 0.92, "전체 주행 최종 지표", fontsize=16, fontweight="bold", va="top")
        lines = [("", "ATE [m]", "노이즈 [%]", "IoU")]
        for m in MODES:
            r = met.get(m)
            if r:
                lines.append((NAME[m], f"{r['ate_rmse_m']:.3f}", f"{r['map_noise_ratio'] * 100:.1f}",
                              f"{r['voxel_iou']:.3f}"))
        for i, row in enumerate(lines):
            y = 0.72 - i * 0.2
            c = COLOR.get(MODES[i - 1], "black") if i else "0.3"
            for x, s in zip([0.02, 0.55, 0.73, 0.9], row):
                ax_txt.text(x, y, s, fontsize=14, va="top", color=c,
                            fontweight="bold" if i and MODES[i - 1] == "proposed" else "normal")

        write_fig(proc, fig)

    hold_fig = plt.figure(figsize=SIZE, dpi=100)
    summary_card(hold_fig, weather, label, met)
    write_fig(proc, hold_fig, n=4 * FPS)
    plt.close(hold_fig)
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
    wbar = 0.27
    for r, (label, res) in enumerate(datasets):
        ev = load_metrics(res)
        for c, (key, title, scale) in enumerate(metrics):
            ax = axes[r, c]
            for k, m in enumerate(MODES):
                v = [ev.get(w, {}).get(m, {}).get(key, np.nan) * scale for w in WEATHERS]
                ax.bar(x + (k - 1) * wbar, v, wbar, color=COLOR[m], label=NAME[m])
            ax.set_xticks(x)
            ax.set_xticklabels(["맑음", "비", "안개", "눈", "복합"], fontsize=13)
            ax.set_title(f"{label} — {title}", fontsize=14)
            ax.grid(axis="y", alpha=0.3)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=15, bbox_to_anchor=(0.5, 0.935), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(out_png)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weathers", nargs="+", default=WEATHERS)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    datasets = [(l, Path(r)) for l, r in zip(args.labels, args.results)]

    jobs, clips = [], []
    for i, (label, res) in enumerate(datasets):
        tag = res.name
        for w in args.weathers:
            p = out / f"compare_{tag}_{w}.mp4"
            jobs.append((w, res, label, p))
            clips.append(p)
    with Pool(min(10, len(jobs))) as pool:
        for o in pool.imap_unordered(render_weather, jobs):
            print("wrote", o, flush=True)

    # intro + overall summary stills as clips
    fig = plt.figure(figsize=SIZE, dpi=100)
    intro_card(fig, args.labels)
    fig.savefig(out / "intro.png")
    proc = ffmpeg_writer(out / "intro.mp4"); write_fig(proc, fig, n=5 * FPS); proc.stdin.close(); proc.wait()
    plt.close(fig)
    fig = overall_summary(datasets, out / "summary.png")
    proc = ffmpeg_writer(out / "summary.mp4"); write_fig(proc, fig, n=8 * FPS); proc.stdin.close(); proc.wait()
    plt.close(fig)

    lst = out / "concat.txt"
    lst.write_text("".join(f"file '{p.resolve()}'\n" for p in [out / "intro.mp4", *clips, out / "summary.mp4"]))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c", "copy", str(out / "montage.mp4")], check=True)
    print("wrote", out / "montage.mp4")


if __name__ == "__main__":
    sys.exit(main())
