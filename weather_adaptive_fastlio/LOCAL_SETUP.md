# Local ROS 2 Humble Setup

This workspace can run without Docker on Ubuntu 22.04 with ROS 2 Humble.
The local setup mirrors the Docker image in `src/docker/Dockerfile`.

## 1. Install dependencies

From the workspace root:

```bash
chmod +x scripts/setup_local_humble.sh scripts/build_local_humble.sh
./scripts/setup_local_humble.sh
```

The script installs:

- ROS 2 Humble desktop full
- PCL, Eigen, SuiteSparse, glog, GLEW, ATLAS
- ROS packages used by FAST-LIO, RViz, image transport, and PCL conversion
- Livox-SDK2 into `${HOME}/Livox-SDK2` unless `LIVOX_SDK2_DIR` is set

If ROS Humble apt packages are not available yet, install the official ROS 2 apt repository first.

## 2. Build the workspace

```bash
source /opt/ros/humble/setup.bash
./scripts/build_local_humble.sh
source install/setup.bash
export FAST_LIO_WS="$(pwd)"
export ROS_LOG_DIR="$(pwd)/log"
```

## 3. Run FAST-LIO locally

For the current Velodyne/Boreas-style setup:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
export FAST_LIO_WS="$(pwd)"
export ROS_LOG_DIR="$(pwd)/log"
ros2 launch fast_lio mapping_velodyne.launch.py use_sim_time:=true rviz:=true
```

In another terminal:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_LOG_DIR="$(pwd)/log"
ros2 bag play gate_01 --clock --topics /velodyne_points /handsfree/imu --remap /handsfree/imu:=/imu/data
```

For the Boreas bag mounted in the previous container flow:

```bash
ros2 bag play /bags/boreas_snow_300s_fastlio --clock --topics /velodyne_points /imu/data
```

## Notes

- Docker paths such as `/home/yoon/FAST_LIO_ROS2/ros2_ws_from_container` should become your local workspace path.
- `mapping_velodyne.launch.py` uses `FAST_LIO_WS` for the default PCD path. If it is unset, it falls back to `${HOME}/ros2_ws`.
- Set `ROS_LOG_DIR` to a writable directory if ROS tries to write logs under a read-only home directory.
- If RViz is slow or OpenGL fails on local GPU drivers, add `rviz_software_rendering:=true`.
- If you use a real LiDAR, keep `--net=host` Docker behavior in mind: locally this is already the default network namespace.
