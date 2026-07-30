#!/usr/bin/env bash
set -euo pipefail

WS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -r /opt/ros/humble/setup.bash ]]; then
  echo "ROS 2 Humble is not installed at /opt/ros/humble."
  echo "Run ./scripts/setup_local_humble.sh first."
  exit 1
fi

# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash

cd "${WS_DIR}"

if [[ ! -f src/livox_ros_driver2/package.xml ]]; then
  echo "src/livox_ros_driver2 is missing."
  exit 1
fi

# livox_ros_driver2 needs ROS_EDITION/HUMBLE_ROS cmake definitions. Its own
# build.sh also rewrites package.xml/launch folders, so we keep that behavior.
./src/livox_ros_driver2/build.sh humble

# Build local packages that are not always part of the upstream FAST-LIO tree.
colcon build --packages-select fast_lio weather_preprocess

echo
echo "Build complete."
echo "Use:"
echo "  source /opt/ros/humble/setup.bash"
echo "  source ${WS_DIR}/install/setup.bash"
echo "  export FAST_LIO_WS=${WS_DIR}"
echo "  export ROS_LOG_DIR=${WS_DIR}/log"
