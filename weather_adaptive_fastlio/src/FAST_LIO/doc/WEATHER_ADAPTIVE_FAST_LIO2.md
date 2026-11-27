# Weather-Adaptive FAST-LIO2

This project extends the FAST-LIO2 baseline for weather-adaptive 3D LiDAR-inertial SLAM.

## Architecture

BOREAS-RT dataset -> ROS2 3D LiDAR and IMU input -> FAST-LIO2 IMU propagation ->
Fast Point Ranking -> confidence-weighted point-to-plane residual based LiDAR IEKF update
with an iKD-Tree local map -> LiDAR confidence `c_k` and degradation `D_k` -> adaptive
LiDAR residual covariance `R_k^LiDAR` -> FAST-LIO2 IEKF update -> confidence-aware
iKD-Tree map update -> trajectory and map evaluation.

The rank, Huber, and frame-confidence terms form `total_w`. Scaling both the Jacobian
and residual by `sqrt(total_w)` is a confidence-weighted measurement update; it is not
an explicit covariance update. `adaptive_lidar_cov` is the separate, explicit frame-level
LiDAR residual covariance passed to the FAST-LIO2 IEKF.

## BOREAS-RT Input Contract

- LiDAR topic: `/velodyne_points`, expected frame: `lidar`
- IMU topic: `/imu/data`, expected frame: `imu`
- Point fields: `x`, `y`, `z`, `intensity`, `ring`, `time`
- Point `time` unit: seconds (`preprocess.timestamp_unit: 0`)
- LiDAR/IMU timestamps are externally synchronized, so automatic synchronization is off.

For `rank_weight.ring_source: pointcloud2_ring`, preprocessing copies `ring` into
`PointType.normal_x`. The node checks field presence and preservation ratio and warns when
the configured minimum is not met. `normal_x` and `elevation` are available for other data.

`rank_weight.range_factor_mode` controls whether range changes point quality. BOREAS-RT
uses `inverse` so distant returns are not promoted solely because they are far away;
`linear_boost` preserves the earlier behavior and `none` supports ablation tests.

## Experiment Modes

- `config/boreas_rt_baseline.yaml`: FAST-LIO2 baseline, robustness and adaptive covariance off.
- `config/boreas_rt_weather_adaptive.yaml`: FPR, Huber, frame confidence, adaptive covariance,
  and confidence-aware map filtering on.

The per-frame CSV contains timestamps, point counts, rank validity, `c_k`, `D_k`, residual
statistics, adaptive covariance, map update count, pose, IEKF time, iKD-Tree time, and total time.

## Evaluation

`tools/evaluate_weather_adaptive.py` reports ATE, one-second RTE, end-to-end drift,
residual variance, map nearest-neighbor noise, low-confidence map accumulation, and runtime
overhead. It can export TUM trajectories for optional `evo_ape`/`evo_rpe` evaluation.

### Baseline run

```bash
ros2 launch fast_lio mapping_boreas.launch.py \
  config_path:=/home/yoon/FAST_LIO_ROS2/ros2_ws_from_container/install/fast_lio/share/fast_lio/config/boreas_rt_baseline.yaml \
  rank_weight_enable:=false rank_gpu_enable:=false rank_huber_enable:=false \
  rank_frame_confidence_enable:=false rank_map_filter_enable:=false \
  adaptive_lidar_cov_enable:=false bag_rate:=0.5
```

### Proposed run

```bash
ros2 launch fast_lio mapping_boreas.launch.py \
  config_path:=/home/yoon/FAST_LIO_ROS2/ros2_ws_from_container/install/fast_lio/share/fast_lio/config/boreas_rt_weather_adaptive.yaml \
  rank_weight_enable:=true rank_gpu_enable:=true rank_huber_enable:=true \
  rank_frame_confidence_enable:=true rank_map_filter_enable:=true \
  adaptive_lidar_cov_enable:=true bag_rate:=0.5
```

After each bag finishes, save the configured evaluation map before stopping the node:

```bash
ros2 service call /map_save std_srvs/srv/Trigger '{}'
```

Compare both runs:

```bash
python3 src/FAST_LIO/tools/evaluate_weather_adaptive.py \
  --baseline src/FAST_LIO/Log/baseline_metrics.csv \
  --proposed src/FAST_LIO/Log/proposed_metrics.csv \
  --ground-truth boreas/boreas-2021-01-26-11-22/applanix/lidar_poses.csv \
  --baseline-map src/FAST_LIO/PCD/baseline_map.pcd \
  --proposed-map src/FAST_LIO/PCD/proposed_map.pcd \
  --tum-dir src/FAST_LIO/traj/evo \
  --plot src/FAST_LIO/Log/map_comparison.png \
  --output src/FAST_LIO/Log/weather_adaptive_evaluation.json
```

## Weather reliability extension (`weather_reliability.*`)

All parameters default to off, which reproduces the behaviour above. The final simulation
configuration is `config/sim_weather_weather_adaptive.yaml`; results and discussion are in
the repository README.

| Parameter | Default | Meaning |
|---|---|---|
| `enable` | false | Turn the extension on. |
| `freeze_confidence` | true | Compute `c_k` (and thus `D_k`, `R_k^LiDAR`) once per scan, at the first IEKF iteration. |
| `alpha_rank`, `alpha_count`, `alpha_residual` | 0.4, 0.4, 0.2 | `c_k = (a_R R~ + a_N q_N + a_E q_E) / sum(a)`. |
| `ref_mean_rank` | 2.0 | Clear-weather mean `R_k,i`; `R~ = (mean R - 1) / (R_ref - 1)`. |
| `ref_struct_points` | 20000 | Clear-weather count of points with consistency >= threshold; `q_N = N / N_ref`. |
| `ref_residual_sigma` | 0.1 | `q_E = exp(-var / sigma0^2)`. |
| `ref_decay` | 0.002 | References follow a running maximum that decays at this rate. |
| `consistency_threshold` | 0.3 | Consistency (range-similarity term of `R_k,i`) separating structure from clutter. |
| `map_on_consistency` | true | Map filter scores points by consistency instead of `R_k,i` (the range factor would remove near ground first). |
| `adaptive_map_percent` | true | Map filter percentile `p_k = clamp(min + gain * 100 * max(0, clutter_share - ref_clutter_down), min, max)`. |
| `map_percent_min`, `map_percent_max`, `map_percent_gain`, `ref_clutter_down` | 5, 60, 1.0, 0.2 | Bounds and slope of `p_k`. |
| `separate_output_map` | false | ikd-tree keeps the light `rank_weight.map_filter_percent` filter on `R_k,i`; only the published output map (`/cloud_weather_filtered`) uses the adaptive `tau_k`. |

The per-frame CSV gains `wr_mean_rank, wr_struct_points, wr_clutter_ratio, wr_clutter_down,
map_filter_percent`.
