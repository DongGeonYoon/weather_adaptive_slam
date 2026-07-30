#!/usr/bin/env bash
set -euo pipefail

# Install the same host-side dependencies that the Docker image provides.
# Run this on Ubuntu 22.04 with ROS 2 Humble available from apt.

if [[ "${EUID}" -eq 0 ]]; then
  SUDO=()
else
  SUDO=(sudo)
fi

if [[ -r /etc/os-release ]]; then
  # shellcheck disable=SC1091
  source /etc/os-release
else
  echo "Cannot detect OS. This script expects Ubuntu 22.04."
  exit 1
fi

if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "22.04" ]]; then
  echo "Warning: this workspace is pinned to ROS 2 Humble, normally on Ubuntu 22.04."
fi

if ! apt-cache policy ros-humble-desktop-full >/dev/null 2>&1; then
  echo "ROS Humble apt packages are not visible."
  echo "Install ROS 2 Humble apt repositories first:"
  echo "  https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html"
  exit 1
fi

"${SUDO[@]}" apt-get update
"${SUDO[@]}" apt-get install -y \
  build-essential \
  cmake \
  git \
  python3-colcon-common-extensions \
  python3-pip \
  wget \
  unzip \
  libatlas-base-dev \
  libeigen3-dev \
  libpcl-dev \
  libgoogle-glog-dev \
  libsuitesparse-dev \
  libglew-dev \
  ros-humble-desktop-full \
  ros-humble-tf2 \
  ros-humble-cv-bridge \
  ros-humble-pcl-conversions \
  ros-humble-xacro \
  ros-humble-robot-state-publisher \
  ros-humble-rviz2 \
  ros-humble-image-transport \
  ros-humble-image-transport-plugins \
  ros-humble-pcl-ros

SDK_DIR="${LIVOX_SDK2_DIR:-${HOME}/Livox-SDK2}"
if [[ ! -d "${SDK_DIR}/.git" ]]; then
  git clone https://github.com/Livox-SDK/Livox-SDK2.git "${SDK_DIR}"
fi

cmake -S "${SDK_DIR}" -B "${SDK_DIR}/build"
cmake --build "${SDK_DIR}/build" --parallel 2
"${SUDO[@]}" cmake --install "${SDK_DIR}/build"

echo
echo "Local dependencies are ready."
echo "Next:"
echo "  source /opt/ros/humble/setup.bash"
echo "  ./scripts/build_local_humble.sh"
