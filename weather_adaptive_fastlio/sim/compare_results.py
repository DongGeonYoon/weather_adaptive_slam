#!/usr/bin/env python3
"""Aggregate evaluation.json files over repetitions and datasets.

usage: compare_results.py --dataset C=../sim_results/valC --dataset D=../sim_results/valD \
                          [--dataset A=resA,resA_rep2 ...] [--methods baseline original proposed] \
                          [--out comparison.txt]

Prints per-dataset tables (mean over repetitions) and, for every method, the
geometric-mean ratio of each metric relative to 'baseline' and 'original' over
all (dataset, weather) cells together with the number of cells won.
"""

import argparse
import json
from pathlib import Path

import numpy as np

WEATHERS = ["clear", "rain", "fog", "snow", "all_weather"]
METRICS = [("ate_rmse_m", "ATE"), ("end_drift_m", "Drift"), ("rpe_trans_pct", "RPE"),
           ("rpe_rot_deg_per_10m", "RPErot"), ("map_noise_ratio", "Noise%"), ("voxel_iou", "IoU")]
HIGHER_BETTER = {"voxel_iou"}


def load(datasets):
    data = {}   # data[ds][weather][method][metric] -> list over repetitions
    for ds, dirs in datasets.items():
        for d in dirs:
            f = Path(d) / "evaluation.json"
            if not f.exists():
                print(f"[skip] {f} not found")
                continue
            for w, ms in json.loads(f.read_text()).items():
                for m, r in ms.items():
                    if "error" in r:
                        continue
                    for k, _ in METRICS:
                        if k in r:
                            data.setdefault(ds, {}).setdefault(w, {}).setdefault(m, {}).setdefault(k, []).append(r[k])
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", required=True, help="NAME=dir[,dir...]")
    ap.add_argument("--methods", nargs="+", default=["baseline", "original", "proposed", "proposed_cov"])
    ap.add_argument("--out")
    args = ap.parse_args()
    datasets = {s.split("=", 1)[0]: s.split("=", 1)[1].split(",") for s in args.dataset}
    data = load(datasets)
    methods = [m for m in args.methods if any(m in ws.get(w, {}) for ws in data.values() for w in WEATHERS)]

    out = []
    for ds, ws in data.items():
        out.append(f"\n=== dataset {ds} (mean over repetitions; n = runs) ===")
        hdr = f"{'weather':12s} {'method':13s} {'n':>2s} " + " ".join(f"{h:>8s}" for _, h in METRICS)
        out += [hdr, "-" * len(hdr)]
        for w in WEATHERS:
            for m in methods:
                r = ws.get(w, {}).get(m)
                if not r:
                    continue
                vals = [f"{np.mean(r[k]) * (100 if k == 'map_noise_ratio' else 1):8.3f}" if k in r else f"{'-':>8s}"
                        for k, _ in METRICS]
                out.append(f"{w:12s} {m:13s} {len(r['ate_rmse_m']):2d} " + " ".join(vals))
            out.append("")

    out.append("\n=== geometric-mean ratio over all (dataset, weather) cells; <1 is better (IoU: >1 is better) ===")
    for ref in [r for r in ["baseline", "original"] if r in methods]:
        out.append(f"\nreference = {ref}")
        hdr = f"{'method':13s} " + " ".join(f"{h:>8s} {'won':>5s}" for _, h in METRICS)
        out += [hdr, "-" * len(hdr)]
        for m in methods:
            if m == ref:
                continue
            cells = []
            for k, _ in METRICS:
                ratios, wins = [], 0
                for ws in data.values():
                    for w in WEATHERS:
                        a = ws.get(w, {}).get(m, {}).get(k)
                        b = ws.get(w, {}).get(ref, {}).get(k)
                        if not a or not b:
                            continue
                        ma, mb = np.mean(a), np.mean(b)
                        ratios.append(max(ma, 1e-9) / max(mb, 1e-9))
                        wins += (ma > mb) if k in HIGHER_BETTER else (ma < mb)
                if not ratios:
                    cells.append(f"{'-':>8s} {'-':>5s}")
                    continue
                cells.append(f"{np.exp(np.mean(np.log(ratios))):8.3f} {wins:2d}/{len(ratios):<2d}")
            out.append(f"{m:13s} " + " ".join(cells))
    text = "\n".join(out)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
