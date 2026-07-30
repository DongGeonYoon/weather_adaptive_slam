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
