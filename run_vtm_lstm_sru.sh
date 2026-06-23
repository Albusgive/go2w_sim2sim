#!/usr/bin/env bash
# Launch the go2w sim2sim MuJoCo viewer (lab2mj, ONNX backend).
#
# vtm_lstm_sru is Policy ID 2 (press key "3" after the window opens;
# keys 1/2/3/4 -> policy IDs 0/1/2/3). Visual policies default to a
# 1.0 m/s forward command (kPlayLikeDefaultCmd in mj_env.cpp).
#
# Override dependency locations via environment variables if your MuJoCo /
# ONNX Runtime live elsewhere:
#   MUJOCO_DIR  - MuJoCo SDK root (contains lib/ and include/)
#   ONNX_DIR    - ONNX Runtime root (contains lib/ and include/)
#   BUILD_DIR   - lab2mj build directory (default: mujoco/C++/build_local)
set -e

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MUJOCO_DIR="${MUJOCO_DIR:-$HOME/rl_robot/deps/mujoco-3.4.0}"
ONNX_DIR="${ONNX_DIR:-$HOME/rl_robot/deps/onnxruntime-linux-x64-gpu-1.23.2}"
BUILD_DIR="${BUILD_DIR:-$REPO/mujoco/C++/build_local}"

# ROS2 (ament/rclcpp) is a build/link dependency of lab2mj.
if [ -f /opt/ros/humble/setup.bash ]; then
  source /opt/ros/humble/setup.bash 2>/dev/null || true
fi

export LD_LIBRARY_PATH="$MUJOCO_DIR/lib:$ONNX_DIR/lib:$LD_LIBRARY_PATH"
export DISPLAY="${DISPLAY:-:1}"

if [ ! -x "$BUILD_DIR/lab2mj" ]; then
  echo "lab2mj not found in $BUILD_DIR. Build it first (see README -> Build)." >&2
  exit 1
fi

cd "$REPO/mujoco/C++"
exec "$BUILD_DIR/lab2mj" "$@"
