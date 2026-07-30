#!/usr/bin/env bash
set -eo pipefail

# ============================================================
# FAST-LIO batch runner - tuned version
# clear, rain, fog, snow, all_weather
# baseline(off) + weather_on(on)
#
# tuned weather_on 설정:
# - rank-aware downsample ON
# - rank map filter ON
# - residual map gate ON
# - Huber OFF
# - frame confidence OFF
# - adaptive covariance OFF
#
# 목적:
# - 기존 결과에서 APE는 개선되지만 RPE가 소폭 악화됨
# - 따라서 필터 강도를 완화해서 local constraint 손실을 줄임
# ============================================================

WS=/home/yoon/FAST_LIO_ROS2/ros2_ws_from_container
DATA=/home/yoon/datasets/KITTI-360
PARAM_FILE=$WS/src/FAST_LIO/config/velodyne.yaml

source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"
set -u

cd "$WS"

# ============================================================
# rosbag 경로
# 실제 폴더명이 다르면 여기만 수정
# ============================================================
declare -A BAGS
BAGS[clear]="$DATA/kitti360_0003_fastlio_ros2bag"
BAGS[rain]="$DATA/kitti360_0003_fastlio_ros2bag_rain_heavy"
BAGS[fog]="$DATA/kitti360_0003_fastlio_ros2bag_fog_heavy"
BAGS[snow]="$DATA/kitti360_0003_fastlio_ros2bag_snow_heavy"
BAGS[all_weather]="$DATA/kitti360_0003_fastlio_ros2bag_all_weather_heavy"

WEATHERS=("clear" "rain" "fog" "snow" "all_weather")
MODES=("baseline" "weather_on")

stop_fastlio_node() {
    local NODE_PID=$1

    echo "[INFO] Bag finished. Stopping FAST-LIO node..."

    kill -INT "-$NODE_PID" 2>/dev/null || true

    # trajectory 저장을 위해 SIGINT 후 최대 20초 대기
    for i in {1..20}; do
        if ! kill -0 "$NODE_PID" 2>/dev/null; then
            break
        fi
        sleep 1
    done

    if kill -0 "$NODE_PID" 2>/dev/null; then
        echo "[WARN] FAST-LIO did not stop after SIGINT. Sending SIGTERM..."
        kill -TERM "-$NODE_PID" 2>/dev/null || true
        sleep 5
    fi

    if kill -0 "$NODE_PID" 2>/dev/null; then
        echo "[WARN] FAST-LIO still alive. Sending SIGKILL..."
        kill -KILL "-$NODE_PID" 2>/dev/null || true
    fi

    wait "$NODE_PID" 2>/dev/null || true
}

run_one() {
    local MODE=$1
    local WEATHER=$2
    local BAG=${BAGS[$WEATHER]}

    local TRAJ="$DATA/${MODE}_${WEATHER}_trajectory.txt"
    local METRICS="$DATA/${MODE}_${WEATHER}_metrics.csv"

    echo "============================================================"
    echo "[RUN] MODE=$MODE, WEATHER=$WEATHER"
    echo "BAG=$BAG"
    echo "TRAJ=$TRAJ"
    echo "METRICS=$METRICS"
    echo "============================================================"

    if [ ! -d "$BAG" ]; then
        echo "[ERROR] Bag directory not found: $BAG"
        echo "BAGS[$WEATHER] 경로를 실제 rosbag 폴더명으로 수정해야 함."
        exit 1
    fi

    # 기존 결과 덮어쓰기
    rm -f "$TRAJ" "$METRICS"

    if [ "$MODE" = "baseline" ]; then
        ROBUST_ARGS=(
            -p rank_weight.enable:=false
            -p rank_weight.gpu_enable:=false
            -p rank_weight.huber_enable:=false
            -p rank_weight.frame_confidence_enable:=false
            -p rank_weight.map_filter_enable:=false
            -p rank_weight.map_residual_gate_enable:=false
            -p adaptive_lidar_cov.enable:=false
        )
    else
        ROBUST_ARGS=(
            # ----------------------------------------------------
            # Core robust module
            # ----------------------------------------------------
            -p rank_weight.enable:=true
            -p rank_weight.gpu_enable:=false
            -p rank_weight.map_filter_enable:=true
            -p rank_weight.map_residual_gate_enable:=true

            # ----------------------------------------------------
            # RPE 악화 가능성이 있는 measurement weakening 계열 OFF
            # ----------------------------------------------------
            -p rank_weight.huber_enable:=false
            -p rank_weight.frame_confidence_enable:=false
            -p adaptive_lidar_cov.enable:=false

            # ----------------------------------------------------
            # Tuned numeric parameters
            # 기존보다 필터를 완화해서 local matching point를 더 보존
            # ----------------------------------------------------
            -p rank_weight.map_filter_percent:=10.0
            -p rank_weight.map_residual_gate:=0.35
            -p rank_weight.map_residual_sigma_factor:=4.0
            -p rank_weight.min_weight:=0.7
            -p rank_weight.max_weight:=1.3
            -p rank_weight.sigma_r:=1.2

            # FPR cylindrical neighborhood
            -p rank_weight.kernel_row_radius:=2
            -p rank_weight.kernel_col_radius:=2
        )
    fi

    setsid ros2 run fast_lio fastlio_mapping \
        --ros-args \
        --params-file "$PARAM_FILE" \
        -p common.lidar_qos_depth:=2000 \
        -p common.imu_qos_depth:=20000 \
        -p common.executor_threads:=8 \
        "${ROBUST_ARGS[@]}" \
        -p publish.scan_publish_en:=false \
        -p publish.dense_publish_en:=false \
        -p publish.scan_bodyframe_pub_en:=false \
        -p publish.map_publish_en:=false \
        -p traj_save.traj_save_en:=true \
        -p traj_save.traj_file_path:="$TRAJ" \
        -p experiment_log.enable:=true \
        -p experiment_log.path:="$METRICS" &

    NODE_PID=$!

    sleep 5

    set +e
    ros2 bag play "$BAG"
    BAG_STATUS=$?
    set -e

    stop_fastlio_node "$NODE_PID"

    sleep 2

    if [ "$BAG_STATUS" -ne 0 ]; then
        echo "[WARN] ros2 bag play returned non-zero status: $BAG_STATUS"
    fi

    if [ -f "$TRAJ" ]; then
        echo "[OK] Saved trajectory: $TRAJ"
    else
        echo "[WARN] Trajectory not saved: $TRAJ"
    fi

    if [ -f "$METRICS" ]; then
        echo "[OK] Saved metrics: $METRICS"
    else
        echo "[WARN] Metrics not saved: $METRICS"
    fi

    echo ""
}

for WEATHER in "${WEATHERS[@]}"; do
    for MODE in "${MODES[@]}"; do
        run_one "$MODE" "$WEATHER"
    done
done

echo "============================================================"
echo "[DONE] All weather bags finished."
echo "============================================================"
