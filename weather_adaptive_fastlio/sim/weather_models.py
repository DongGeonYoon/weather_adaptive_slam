#!/usr/bin/env python3
"""Physics-inspired adverse-weather LiDAR models applied to a clean scan.

The models follow the structure of published LiDAR weather simulators:

* fog  - Beer-Lambert two-way attenuation exp(-2*alpha*R) limits the detection
         range and soft-scattering returns from the fog volume appear close to
         the sensor (Hahner et al., ICCV 2021, "Fog Simulation on Real LiDAR
         Point Clouds").
* rain - weak extinction, randomly located drop returns close to the sensor,
         extra range noise and dropout of grazing returns on the wet road
         (Kilic et al., "LiDAR Light Scattering Augmentation (LISA)", 2021).
* snow - stronger extinction and dense bright flake returns concentrated in the
         first ~15 m (Hahner et al., CVPR 2022, "LiDAR Snowfall Simulation").

Each condition also has a slowly varying severity with gust bursts so that the
degradation changes over time (useful for frame-adaptive methods).

All functions take per-ray arrays from sim_world.simulate_clean_scan and return
the measured range per ray (np.inf = no return), intensity and a boolean flag
telling whether the return is a weather artefact (ground-truth label, never
written into the bag).
"""

import numpy as np

CONDITIONS = {
    #               extinction  backscatter  drops/flakes  particle range         particle    range   wet-road
    #               alpha[1/m]  prob         prob          (offset, scale, max)   intensity   noise   dropout
    "clear":       dict(alpha=0.0,   bs=0.0,  part=0.0,  prange=(0, 1, 1),     pint=(0, 0),       noise=0.0,  wet=0.0),
    "rain":        dict(alpha=0.006, bs=0.0,  part=0.06, prange=(0.8, 5.0, 30), pint=(0.02, 0.15), noise=0.04, wet=0.5),
    "fog":         dict(alpha=0.040, bs=0.30, part=0.0,  prange=(0, 1, 1),     pint=(0.02, 0.08), noise=0.02, wet=0.0),
    "snow":        dict(alpha=0.015, bs=0.0,  part=0.12, prange=(0.8, 6.0, 35), pint=(0.10, 0.50), noise=0.03, wet=0.2),
    "all_weather": dict(alpha=0.030, bs=0.15, part=0.08, prange=(0.8, 5.5, 32), pint=(0.03, 0.40), noise=0.04, wet=0.4),
}

DETECTION_THRESHOLD = 0.012   # normalised received power needed for a return


def severity(t, condition, rng_seed=3):
    """Time-varying severity multiplier in [~0.6, ~1.6] with two gust bursts."""
    if condition == "clear":
        return 0.0
    base = 1.0 + 0.25 * np.sin(2 * np.pi * t / 23.0 + 0.7)
    bursts = 0.6 * np.exp(-0.5 * ((t - 18.0) / 2.5) ** 2) + 0.6 * np.exp(-0.5 * ((t - 41.0) / 3.0) ** 2)
    return base + bursts


def apply_weather(scan, condition, t_rel, rng):
    """Return range (inf = no return), intensity, is_weather_noise for each ray."""
    c = CONDITIONS[condition]
    r = scan["range"].copy()
    refl = scan["refl"]
    cos_inc = np.maximum(scan["cos_inc"], 0.05)
    n = len(r)
    sev = severity(t_rel, condition)
    noise_flag = np.zeros(n, dtype=bool)

    hit = np.isfinite(r)
    # received power of the hard target (simple 1/R^2-free normalised model; the
    # clear sensor sees every target up to MAX_RANGE)
    p_target = np.zeros(n)
    p_target[hit] = refl[hit] * cos_inc[hit]

    alpha = c["alpha"] * sev
    if alpha > 0:
        p_att = p_target * np.exp(-2.0 * alpha * np.where(hit, r, 0.0))
        # soft detection: probability rises smoothly around the threshold
        p_det = 1.0 / (1.0 + np.exp(-(np.log(np.maximum(p_att, 1e-9)) - np.log(DETECTION_THRESHOLD)) / 0.35))
        lost = hit & (rng.random(n) > p_det)
        r[lost] = np.inf
        hit = np.isfinite(r)
        p_target = np.where(hit, p_att, 0.0)

    intensity = np.where(hit, p_target, 0.0)

    # wet road: grazing ground returns are reflected away (specular)
    if c["wet"] > 0:
        grazing = hit & (cos_inc < 0.30) & (refl < 0.26)
        drop = grazing & (rng.random(n) < min(0.95, c["wet"] * sev))
        r[drop] = np.inf
        intensity[drop] = 0.0
        hit = np.isfinite(r)

    # range noise on remaining hard-target returns
    if c["noise"] > 0:
        r[hit] += rng.normal(0.0, c["noise"] * sev, hit.sum())

    # fog volume back-scatter: a near return replaces the target when the target
    # is weak (far / low reflectivity) - this is the dominant fog artefact
    if c["bs"] > 0:
        weak = np.where(hit, np.exp(-p_target / 0.05), 1.0)
        p_bs = np.clip(c["bs"] * sev * weak, 0, 0.9)
        bs = rng.random(n) < p_bs
        r_bs = 1.2 + rng.gamma(2.0, 1.6, bs.sum())
        keep = ~hit[bs] | (r_bs < r[bs])
        idx = np.flatnonzero(bs)[keep]
        r[idx] = r_bs[keep]
        intensity[idx] = rng.uniform(*c["pint"], len(idx))
        noise_flag[idx] = True

    # rain drops / snow flakes: discrete particles in front of the target
    if c["part"] > 0:
        off, scale, rmax = c["prange"]
        pp = rng.random(n) < min(0.6, c["part"] * sev)
        r_p = np.minimum(off + rng.exponential(scale, pp.sum()), rmax)
        closer = r_p < r[pp]
        idx = np.flatnonzero(pp)[closer]
        r[idx] = r_p[closer]
        intensity[idx] = rng.uniform(*c["pint"], len(idx))
        noise_flag[idx] = True

    valid = np.isfinite(r) & (r > 0.3)
    r[~valid] = np.inf
    noise_flag &= valid
    return r, intensity, noise_flag
