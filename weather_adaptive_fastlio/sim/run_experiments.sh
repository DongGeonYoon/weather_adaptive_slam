#!/usr/bin/env bash
# Run FAST-LIO2 methods on the simulated weather bags of one dataset.
#
# usage: RES=<results dir> [DATA=<dataset dir>] [METHODS="baseline original proposed"] \
#        sim/run_experiments.sh [weather ...]
# env:   RES      results directory; runs go to $RES/runs/<weather>/<method>
#        DATA     dataset directory made by generate_dataset.py (default $RES/data)
#        METHODS  subset of: baseline original proposed proposed_cov
#        DOMAIN   ROS_DOMAIN_ID (default 78); use different values to run datasets in parallel
#
# methods:
#   baseline      FAST-LIO2 (rank weighting, Huber, adaptive covariance, map filter all off)
#   original      weather-adaptive FAST-LIO2 before the weather_reliability changes
#   proposed      final configuration (sim_weather_weather_adaptive.yaml)
#   proposed_cov  proposed + adaptive LiDAR covariance R0(1 + gamma D_k) (ablation)
set -eo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RES="${RES:?set RES to a results directory}"
DATA="${DATA:-${RES}/data}"
METHODS=(${METHODS:-baseline original proposed})
WEATHERS=("$@")
[ ${#WEATHERS[@]} -eq 0 ] && WEATHERS=(clear rain fog snow all_weather)
CFG="${WS}/src/FAST_LIO/config"

source /opt/ros/humble/setup.bash
source "${HOME}/ws_livox/install/setup.bash"
source "${WS}/install/setup.bash"
set -u
export ROS_DOMAIN_ID="${DOMAIN:-78}"
export ROS_LOCALHOST_ONLY=1
export ROS_LOG_DIR="${RES}/ros_logs"
mkdir -p "${ROS_LOG_DIR}"

method_args() {   # prints: <config file> [extra ros args...]
    case "$1" in
        baseline)     echo "${CFG}/sim_weather_baseline.yaml -p rank_weight.gpu_enable:=false" ;;
        original)     echo "${CFG}/sim_weather_weather_adaptive_original.yaml" ;;
        proposed)     echo "${CFG}/sim_weather_weather_adaptive.yaml" ;;
        proposed_cov) echo "${CFG}/sim_weather_weather_adaptive.yaml -p adaptive_lidar_cov.enable:=true" ;;
        *) echo "unknown method $1" >&2; exit 1 ;;
    esac
}

stop_group() {   # SIGINT a process group, escalate if needed
    local pid=$1
    kill -INT "-${pid}" 2>/dev/null || true
    for _ in $(seq 1 20); do kill -0 "${pid}" 2>/dev/null || break; sleep 0.5; done
    kill -TERM "-${pid}" 2>/dev/null || true; sleep 1
    kill -KILL "-${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
}

run_one() {
    local weather=$1 method=$2
    local out="${RES}/runs/${weather}/${method}"
    rm -rf "${out}"; mkdir -p "${out}"
    read -r cfg_file extra <<< "$(method_args "${method}")"
    # shellcheck disable=SC2206
    local extra_arr=(${extra:-})
    echo "=== ${weather} / ${method} ==="

    setsid ros2 run fast_lio fastlio_mapping --ros-args \
        --params-file "${cfg_file}" \
        -p use_sim_time:=true \
        -p common.lidar_qos_depth:=2000 -p common.imu_qos_depth:=20000 \
        -p common.lidar_qos_reliable:=true -p common.imu_qos_reliable:=true \
        -p publish.range_image_en:=false -p publish.range_image_cloud_en:=false \
        -p publish.filtered_cloud_en:=true \
        -p experiment_log.path:="${out}/metrics.csv" \
        -p traj_save.traj_file_path:="${out}/traj_tum.txt" \
        -p map_file_path:="${out}/map.pcd" \
        "${extra_arr[@]}" > "${out}/node.log" 2>&1 &
    local node=$!
    setsid ros2 bag record -o "${out}/rec" /Odometry /cloud_weather_filtered > "${out}/record.log" 2>&1 &
    local rec=$!
    # wait for the node's LiDAR subscription, then give discovery time (--delay)
    for _ in $(seq 1 40); do
        ros2 topic info /velodyne_points 2>/dev/null | grep -q "Subscription count: [1-9]" && break
        sleep 0.5
    done
    sleep 1
    ros2 bag play "${DATA}/bags/${weather}" --clock 200 --rate 1.0 --delay 3 \
        --read-ahead-queue-size 2000 > "${out}/play.log" 2>&1 || echo "[WARN] bag play failed"
    sleep 3
    stop_group "${rec}"
    stop_group "${node}"
    echo "    processed frames: $(($(wc -l < "${out}/metrics.csv") - 1))"
}

for w in "${WEATHERS[@]}"; do
    for m in "${METHODS[@]}"; do
        run_one "$w" "$m"
    done
done
echo done
