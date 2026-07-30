#!/usr/bin/env bash
set -eo pipefail

mode="${1:-proposed}"
wall_seconds="${2:-120}"
workspace="/home/yoon/FAST_LIO_ROS2/ros2_ws_from_container"

cd "${workspace}"
source /opt/ros/humble/setup.bash
source install/setup.bash
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export ROS_LOG_DIR="${workspace}/validation/ros_logs"
mkdir -p "${ROS_LOG_DIR}"

if [[ "${mode}" == "baseline" ]]; then
  config="${workspace}/install/fast_lio/share/fast_lio/config/boreas_rt_baseline.yaml"
  flags=(rank_weight_enable:=false rank_gpu_enable:=false rank_huber_enable:=false
         rank_frame_confidence_enable:=false rank_map_filter_enable:=false
         adaptive_lidar_cov_enable:=false)
else
  config="${workspace}/install/fast_lio/share/fast_lio/config/boreas_rt_weather_adaptive.yaml"
  flags=(rank_weight_enable:=true rank_gpu_enable:=true rank_huber_enable:=true
         rank_frame_confidence_enable:=true rank_map_filter_enable:=true
         adaptive_lidar_cov_enable:=true)
fi

timeout --signal=INT --kill-after=10s "${wall_seconds}s" \
  ros2 launch fast_lio mapping_boreas.launch.py \
  rviz:=false bag_rate:=0.5 config_path:="${config}" "${flags[@]}" \
  2>&1 | tee "${workspace}/validation/${mode}_short.log"
