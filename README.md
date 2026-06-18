# Go2W Sim2Sim

Go2W Sim2Sim is a MuJoCo-based simulation and policy playback workspace for the Go2W robot. The main executable is `lab2mj`, which loads the Go2W MJCF scene, runs ONNX/LibTorch policies, visualizes the robot and RayCaster camera, supports interactive perturbations, and records synchronized multi-view videos.

## Repository Layout

```text
mujoco/C++/                       C++ simulators and policy playback apps
mujoco/C++/mj_env.cpp             lab2mj environment and video recording logic
mujoco/C++/sim2sim_env.cpp        shared policy/control logic
robot/go2w_description/mjcf/      Go2W MJCF robot and scenes
robot/go2w_description/mjcf/scene_parkour.xml
                                  default parkour scene used by lab2mj
policy/                           policy checkpoints and deployment configs
utils/mujoco_thread/              MuJoCo window, input, camera, and recording backend
utils/mujoco_ray_caster/          external ray caster dependency
ros2/                             ROS2 integration workspace
```

## Dependencies

The C++ simulator currently expects a Linux desktop environment with OpenGL/GLFW support.

Required packages and libraries:

```bash
sudo apt-get install libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev \
  libudev-dev joystick libjsoncpp-dev libopencv-dev
```

ROS2-related dependencies are used by the `real2sim` target:

```bash
sudo apt-get install ros-${ROS_DISTRO}-rclcpp ros-${ROS_DISTRO}-std-msgs \
  ros-${ROS_DISTRO}-sensor-msgs ros-${ROS_DISTRO}-cv-bridge
```

MuJoCo is expected to be installed under `/opt/mujoco` and discoverable via:

```text
/opt/mujoco/lib/cmake
```

The CMake file uses:

```cmake
find_package(mujoco REQUIRED PATHS /opt/mujoco/lib/cmake NO_DEFAULT_PATH)
```

Install MuJoCo from source or a compatible release so that this CMake package exists.

## Ray Caster Dependency

The project uses an external ray caster utility under `utils/mujoco_ray_caster/`.

```bash
cd utils
git clone https://github.com/Albusgive/mujoco_ray_caster.git
```

This path is ignored by git.

## Inference Backend

The default build uses ONNX Runtime:

```cmake
option(USE_ONNX "Build with ONNX Runtime instead of LibTorch" ON)
```

Set `ONNXRUNTIME_ROOT` if ONNX Runtime is not installed in a default search path:

```bash
export ONNXRUNTIME_ROOT=/path/to/onnxruntime
```

Or pass it during configure:

```bash
cmake -S . -B build_onnx -DONNXRUNTIME_ROOT=/path/to/onnxruntime
```

To build with LibTorch instead:

```bash
cmake -S . -B build_libtorch -DUSE_ONNX=OFF -DTorch_DIR=/path/to/libtorch/share/cmake/Torch
```

## Build

The common build location used in this workspace is `mujoco/C++/build_onnx`.

```bash
cd mujoco/C++
cmake -S . -B build_onnx -DUSE_ONNX=ON
cmake --build build_onnx --target lab2mj -j2
```

Other available targets include:

```text
lab2mj        main policy playback simulator
camera_test   RayCaster camera test utility
camera_quat   camera quaternion helper
real2sim      ROS2-connected real-to-sim executable
```

## Run

```bash
cd mujoco/C++/build_onnx
./lab2mj
```

`lab2mj` loads:

```text
robot/go2w_description/mjcf/scene_parkour.xml
```

The scene path is compiled into the executable from `mujoco/C++/CMakeLists.txt` via `MJCF_PATH`.

## GitHub Pages Demo

This repository includes a static project page under `docs/` for GitHub Pages.
Configure Pages to deploy from the `main` branch and `/docs` folder. The hosted
demo is the `lab2mj` browser path only: MuJoCo WASM renders the Go2W MJCF scene
and ONNX Runtime Web runs the `motion_mlp` policy. It does not deploy ROS2,
`real2sim`, or a native simulator process. The visual VTM/SRU policies remain a
separate browser-porting step because they require the RayCaster image
observation and recurrent state pipeline.

Local preview:

```bash
cd docs
python3 -m http.server 8000
```

Open `http://localhost:8000/demo.html`.

## Default Scene

The default `scene_parkour.xml` contains a multi-lane parkour course:

- Progressive isolated climb boxes from `0.10m` to `0.60m`.
- U-turn lane transitions.
- A `0.4m`, `30deg` ramp connected to the continuous ditch section.
- Continuous ditch gaps from `0.20m` to `0.60m` width, with same-height recovery platforms.
- Low-to-high jump boxes.
- Suspended overhead boxes with support posts.
- Light blue grid floor and a muted skybox for video-friendly rendering.

The scene includes the Go2W robot from:

```text
robot/go2w_description/mjcf/go2w.xml
```

The robot has a fixed MJCF camera named:

```text
RayCasterCamera
```

This camera is used both by RayCaster depth observations and RGB video recording.

## Keyboard Controls

Policy command controls:

```text
w / s      increase / decrease forward command
a / d      increase / decrease lateral command
q / e      increase / decrease yaw command
space      zero velocity command
r          reset current policy state
1..9       switch policy by index
```

Policy/split recording controls:

```text
z          start / stop multi-view policy video recording
x          start split recording for current policy
v          mark split recording step
c          stop split recording
```

Simulation and camera controls:

```text
backspace  reset simulation
space      pause / resume stepping at the MuJoCo thread level
= / +      increase realtime factor
-          decrease realtime factor
[ / ]      switch MuJoCo fixed camera id
middle mouse click
           toggle look-at tracking for selected body
F          toggle relative-yaw tracking for selected body
F1         third-person over-shoulder forward view tracking base_link
F2         right-side view tracking base_link
V          low-level current-view recording to mujoco_videos/current_view.mp4
```

Mouse controls:

```text
left drag             rotate camera
right drag            pan camera
middle drag / scroll  zoom camera
shift + drag          alternate MuJoCo camera motion axis
double left click     select body
double right click    set camera target point
```

Perturbation controls follow MuJoCo simulate-style mapping:

```text
Ctrl + left drag   rotational perturbation / torque on selected body
Ctrl + right drag  translational perturbation / external force on selected body
```

## Camera Presets

Two runtime camera presets are built into `utils/mujoco_thread`:

```text
F1  base_link third-person over-shoulder view
    relative yaw: 15 deg, elevation: -18 deg, distance: 3.2

F2  base_link right-side view
    relative yaw: 90 deg, elevation: -5 deg, distance: 3.0
```

Both presets use relative-yaw tracking. The camera follows `base_link` position and keeps its yaw offset relative to the robot heading, without following the robot pitch or roll.

## Video Recording

Press `z` in `lab2mj` to start or stop multi-view recording.

Recordings are saved under the repository root, independent of the current working directory:

```text
policy_videos/<policy_name>_<timestamp>/
```

Example:

```text
policy_videos/vtm_20260512_153000/
```

The output directory is ignored by git via:

```gitignore
/policy_videos/
```

Each recording produces five synchronized MP4 files:

```text
third_person.mp4
right_view.mp4
raycaster_rgb.mp4
raycaster_rgb_clean.mp4
policy_depth.mp4
```

Stream meanings:

```text
third_person.mp4
  Offscreen RGB render from the base_link third-person over-shoulder camera.

right_view.mp4
  Offscreen RGB render from the base_link right-side camera.

raycaster_rgb.mp4
  RGB render from the MJCF fixed camera RayCasterCamera, including current custom
  draw overlays and MuJoCo camera visualization if enabled.

raycaster_rgb_clean.mp4
  RGB render from RayCasterCamera with custom draw overlays disabled and
  mjVIS_CAMERA temporarily hidden. This removes hit points and camera markers.

policy_depth.mp4
  The noisy RayCaster depth image used by the visual policy.
```

The current window view is not used for `z` recording, so UI overlays and the left table are not included in these videos.

There is also a lower-level `V` shortcut from `utils/mujoco_thread` that records the current view to:

```text
mujoco_videos/current_view.mp4
```

Use `z` for policy/debug recording and `V` only for quick current-window captures.

## RayCaster Camera Visualization

The environment draws RayCaster-related debug markers in `MJ_ENV::draw()`.

Gamepad RB toggles sensor/camera visualization by controlling:

```cpp
opt.flags[mjtVisFlag::mjVIS_CAMERA]
```

It does not disable the custom `draw()` and `draw_windows()` paths globally. The clean RGB recording stream disables these only for that stream's offscreen render.

## Gamepad

Gamepad support is implemented through `utils/cpp_gamepad`.

Main bindings in `Sim2SimEnv`:

```text
left stick   linear x/y command
right stick  yaw command
A/B/X/Y      switch policy presets
LB           request current policy reset
RB           toggle RayCaster sensor visualization
Menu         reset current policy state
```

Install useful joystick tooling with:

```bash
sudo apt-get install joystick libudev-dev
```

## Policy Directories

Policy paths are defined in `mujoco/C++/CMakeLists.txt`:

```text
policy/history_1/
policy/history_5/
policy/vtm/
policy/vtm_lstm_sru/
policy/vtm_gru_sru/
policy/motion_tracking/
policy/raycam/
```

Visual policies such as `vtm`, `vtm_lstm_sru`, and `vtm_gru_sru` use RayCaster observations and refresh visual observation buffers when the policy changes or resets.

## Generated Files

The following generated or local directories are ignored:

```text
build/
.cache/
mujoco/C++/build/
mujoco/C++/build_onnx/
utils/mujoco_ray_caster/
split_records/
policy_videos/
ros2/build/
ros2/install/
ros2/log/
```

## Useful Validation Commands

Build the main executable:

```bash
cd mujoco/C++
cmake --build build_onnx --target lab2mj -j2
```

Validate the parkour MJCF scene:

```bash
/opt/mujoco/bin/compile robot/go2w_description/mjcf/scene_parkour.xml /tmp/scene_parkour.mjb
```

Run from the build directory:

```bash
cd mujoco/C++/build_onnx
./lab2mj
```
