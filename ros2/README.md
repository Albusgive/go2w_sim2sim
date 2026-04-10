# ROS2 Nodes And Startup Guide

This directory builds the ROS2 package `go2w_vtm`.

The package currently installs three executables:

- `go2w_real_deploy`
- `go2w_stand_example`
- `deep_camera`

`go2w_real_deploy` is the main real-robot deployment node.
The other two are helper/debug nodes.

## Build

```bash
cd /home/albusgive2/go2w_sim2sim/ros2
source /opt/ros/humble/setup.bash
source ~/unitree_ros2/setup_local.sh
colcon build --packages-select go2w_vtm --cmake-args -DUSE_ONNX=ON -DBUILD_TESTING=OFF
source install/setup.bash
```

If ONNX Runtime is not in a standard system path, pass `ONNXRUNTIME_ROOT`:

```bash
colcon build --packages-select go2w_vtm --cmake-args \
  -DUSE_ONNX=ON \
  -DONNXRUNTIME_ROOT=/opt/onnxruntime \
  -DBUILD_TESTING=OFF
```

## Executables

### 1. `go2w_real_deploy`

Main deployment entry for real robot control and policy inference.

Run:

```bash
ros2 run go2w_vtm go2w_real_deploy
```

Main features:

- Loads 4 policies by default:
  - `motion_mlp`
  - `vtm`
  - `vtm_lstm_sru`
  - `vtm_gru_sru`
- Subscribes low-level robot state from `/lowstate`
- Publishes low-level commands to `/lowcmd`
- Subscribes remote controller from `/wirelesscontroller`
- Subscribes depth image from `/camera/depth/image_raw`
- Subscribes RGB image from `/camera/color/image_raw`
- Supports local USB gamepad through `cpp_gamepad`
- Supports fixed policy switching with D-pad for both local gamepad and Unitree wireless controller

Source:

- main: [go2w_real_deploy.cpp](/home/albusgive2/go2w_sim2sim/ros2/src/src/go2w_real_deploy/go2w_real_deploy.cpp)
- node: [go2w_real_deploy_node.cpp](/home/albusgive2/go2w_sim2sim/ros2/src/src/go2w_real_deploy/go2w_real_deploy_node.cpp)

#### Custom startup arguments

Custom arguments are passed after `--`, and each argument uses `key=value`.

Example:

```bash
ros2 run go2w_vtm go2w_real_deploy -- device=cuda use_local_gamepad=false
```

Supported arguments:

- `device=cpu|cuda|gpu`
  - Alias: `inference_device`
  - Default: `cpu`
  - Controls ONNX inference device

- `use_local_gamepad=true|false`
  - Alias: `local_gamepad`
  - Default: `true`
  - `false` means only `/wirelesscontroller` is used for command input

- `policy_perf_monitor=true|false`
  - Aliases: `monitor_policy_fps`, `log_policy_perf`
  - Default: `false`
  - Enables periodic policy inference performance logging

- `policy_perf_interval=<positive number>`
  - Alias: `policy_perf_interval_sec`
  - Default: `5.0`
  - Logging interval in seconds for performance summary

#### Policy path override arguments

These override the default policy directories compiled from CMake macros.

- `motion_mlp=/abs/path/to/motion_tracking`
- `vtm=/abs/path/to/vtm`
- `vtm_lstm_sru=/abs/path/to/vtm_lstm_sru`
- `vtm_gru_sru=/abs/path/to/vtm_gru_sru`

Example:

```bash
ros2 run go2w_vtm go2w_real_deploy -- \
  device=cuda \
  use_local_gamepad=false \
  policy_perf_monitor=true \
  policy_perf_interval=5 \
  motion_mlp=/data/policy/motion_tracking \
  vtm=/data/policy/vtm \
  vtm_lstm_sru=/data/policy/vtm_lstm_sru \
  vtm_gru_sru=/data/policy/vtm_gru_sru
```

#### Default policy mapping

Default startup policy is policy `0`, which is `motion_mlp`.

Fixed D-pad mapping:

- `D-pad Up` -> policy `0` -> `motion_mlp`
- `D-pad Right` -> policy `1` -> `vtm`
- `D-pad Down` -> policy `2` -> `vtm_lstm_sru`
- `D-pad Left` -> policy `3` -> `vtm_gru_sru`

This fixed mapping works for:

- local USB gamepad through `cpp_gamepad`
- Unitree `/wirelesscontroller`

Other controller mappings:

- `A` -> enable command output
- `B` -> controlled stop posture
- `X` -> next policy
- `Y` -> previous policy
- `LB` or wireless `L1` -> direct reset current policy state
- `RB` or wireless `R1` -> toggle sensor enable state
- `Menu` or wireless `Select` -> request runtime reset for current policy
- `Start` -> clear velocity command

#### Common run examples

Run with default 4 policies:

```bash
ros2 run go2w_vtm go2w_real_deploy
```

Run with GPU inference:

```bash
ros2 run go2w_vtm go2w_real_deploy -- device=cuda
```

Run without local USB gamepad:

```bash
ros2 run go2w_vtm go2w_real_deploy -- use_local_gamepad=false
```

Run with performance monitor:

```bash
ros2 run go2w_vtm go2w_real_deploy -- policy_perf_monitor=true policy_perf_interval=5
```

#### Notes

- Boolean values accept `true/false`, `on/off`, `yes/no`, `1/0`
- Custom arguments that are not recognized are ignored
- ROS2 standard arguments can still be added separately as needed

### 2. `go2w_stand_example`

Simple low-level standing posture node.

Run:

```bash
ros2 run go2w_vtm go2w_stand_example
```

Behavior:

- Publishes a fixed standing joint target to `/lowcmd`
- Subscribes `/lowstate`
- Uses a `2 ms` wall timer
- No custom `key=value` startup arguments are implemented

Source:

- [go2w_stand.cpp](/home/albusgive2/go2w_sim2sim/ros2/src/src/go2w_stand/go2w_stand.cpp)

### 3. `deep_camera`

Depth image visualization and processing debug node.

Run:

```bash
ros2 run go2w_vtm deep_camera
```

Behavior:

- Subscribes `/camera/camera/depth/image_rect_raw`
- Converts `16UC1` depth from millimeters to meters
- Clamps depth to `[0.25, 2.0]`
- Displays processed depth image with OpenCV windows
- No custom `key=value` startup arguments are implemented

Source:

- [deep_camera.cpp](/home/albusgive2/go2w_sim2sim/ros2/src/src/deep_camera/deep_camera.cpp)

## Installed Targets

These executables are defined in:

- [CMakeLists.txt](/home/albusgive2/go2w_sim2sim/ros2/src/CMakeLists.txt)

Current installed targets:

- `go2w_real_deploy`
- `go2w_stand_example`
- `deep_camera`

## Recommended Usage

For actual deployment, use only:

```bash
ros2 run go2w_vtm go2w_real_deploy
```

`go2w_stand_example` and `deep_camera` are better treated as auxiliary test/debug tools.
