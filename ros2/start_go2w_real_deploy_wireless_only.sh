#!/bin/sh
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROS_SETUP=""
if [[ -n "${ROS_DISTRO:-}" && -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]]; then
  ROS_SETUP="/opt/ros/${ROS_DISTRO}/setup.bash"
elif [[ -f "/opt/ros/humble/setup.bash" ]]; then
  ROS_SETUP="/opt/ros/humble/setup.bash"
elif [[ -f "/opt/ros/foxy/setup.bash" ]]; then
  ROS_SETUP="/opt/ros/foxy/setup.bash"
fi

if [[ -z "${ROS_SETUP}" ]]; then
  echo "[go2w_real_deploy] Could not find a ROS2 setup.bash under /opt/ros" >&2
  exit 1
fi

UNITREE_SETUP="${UNITREE_ROS2_SETUP:-$HOME/unitree_ros2/setup_local.sh}"
if [[ ! -f "${UNITREE_SETUP}" ]]; then
  echo "[go2w_real_deploy] Missing Unitree setup script: ${UNITREE_SETUP}" >&2
  echo "[go2w_real_deploy] Set UNITREE_ROS2_SETUP to your setup_local.sh path." >&2
  exit 1
fi

INSTALL_SETUP="${SCRIPT_DIR}/install/setup.bash"
if [[ ! -f "${INSTALL_SETUP}" ]]; then
  echo "[go2w_real_deploy] Missing ${INSTALL_SETUP}" >&2
  echo "[go2w_real_deploy] Please build the ROS2 workspace first." >&2
  exit 1
fi

set +u
source "${ROS_SETUP}"
source "${UNITREE_SETUP}"
source "${INSTALL_SETUP}"
set -u

echo "[go2w_real_deploy] Launching with default policy paths"
echo "[go2w_real_deploy] Local USB gamepad disabled, using /wirelesscontroller only"

exec ros2 run go2w_vtm go2w_real_deploy -- use_local_gamepad=false "$@"
