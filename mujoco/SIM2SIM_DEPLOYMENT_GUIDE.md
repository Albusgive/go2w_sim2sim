# Sim2Sim Deployment Guide

This document is for agents onboarding to the current `go2w_sim2sim` deployment stack.

It explains:

- how policy inference is wired into the MuJoCo runtime
- how environments and policies are registered
- how observation alignment is checked
- how SRU full/split deployment works under JIT and ONNX
- how runtime switching, reset, recording, and view capture work
- how to add a new policy or a new robot environment with minimal changes

This guide describes the current codebase as of the `vtm`, `vtm_lstm_sru`, and `vtm_gru_sru` setup.

## 1. Code Map

These are the core files to read first:

- [mujoco/C++/lab2mj.cpp](/home/albusgive2/go2w_sim2sim/mujoco/C++/lab2mj.cpp)
  Entry point. Builds the policy list, supports CLI path override, launches the environment.
- [mujoco/C++/sim2sim_env.h](/home/albusgive2/go2w_sim2sim/mujoco/C++/sim2sim_env.h)
- [mujoco/C++/sim2sim_env.cpp](/home/albusgive2/go2w_sim2sim/mujoco/C++/sim2sim_env.cpp)
  Base sim2sim runtime. Owns policy switching, command input, split recording, render-view recording, and common UI behavior.
- [mujoco/C++/mj_env.h](/home/albusgive2/go2w_sim2sim/mujoco/C++/mj_env.h)
- [mujoco/C++/mj_env.cpp](/home/albusgive2/go2w_sim2sim/mujoco/C++/mj_env.cpp)
  Robot-specific environment. Owns sensors, observation terms, action scaling, ray caster, and MuJoCo callbacks.
- [utils/cpp_manager_env/ManagerEnv.hpp](/home/albusgive2/go2w_sim2sim/utils/cpp_manager_env/ManagerEnv.hpp)
- [utils/cpp_manager_env/ManagerEnv.cpp](/home/albusgive2/go2w_sim2sim/utils/cpp_manager_env/ManagerEnv.cpp)
  Observation/action manager layer. Registers observation groups per policy, runs `obs -> policy -> action`, and performs the dummy dimension check at startup.
- [utils/cpp_manager_env/net.h](/home/albusgive2/go2w_sim2sim/utils/cpp_manager_env/net.h)
- [utils/cpp_manager_env/net.cpp](/home/albusgive2/go2w_sim2sim/utils/cpp_manager_env/net.cpp)
  Policy runtime. Loads JIT or ONNX models, handles SRU full/split deployment, recurrent state reset, and split snapshot capture.
- [utils/cpp_manager_env/Buffer.hpp](/home/albusgive2/go2w_sim2sim/utils/cpp_manager_env/Buffer.hpp)
  History buffer logic for vector and image observations.
- [utils/mujoco_thread/mujoco_thread.h](/home/albusgive2/go2w_sim2sim/utils/mujoco_thread/mujoco_thread.h)
- [utils/mujoco_thread/mujoco_thread.cpp](/home/albusgive2/go2w_sim2sim/utils/mujoco_thread/mujoco_thread.cpp)
  MuJoCo window, render loop, and render-frame capture support.

Useful runtime/analysis tools:

- [tools/open_split_record_heatmaps.sh](/home/albusgive2/go2w_sim2sim/tools/open_split_record_heatmaps.sh)
- [tools/play_split_record_heatmaps.py](/home/albusgive2/go2w_sim2sim/tools/play_split_record_heatmaps.py)
- [tools/render_frames_to_video.sh](/home/albusgive2/go2w_sim2sim/tools/render_frames_to_video.sh)
- [tools/SPLIT_RECORD_ANALYSIS_GUIDE.md](/home/albusgive2/go2w_sim2sim/tools/SPLIT_RECORD_ANALYSIS_GUIDE.md)
- [tools/SIM2SIM_ENV_BASE_GUIDE.md](/home/albusgive2/go2w_sim2sim/tools/SIM2SIM_ENV_BASE_GUIDE.md)

## 2. High-Level Architecture

The runtime is layered like this:

1. `lab2mj.cpp`
   Creates a list of `PolicySpec` entries and launches the environment.
2. `Sim2SimEnv`
   Provides common runtime behavior:
   - current policy ID
   - keyboard/gamepad switching
   - policy reset
   - sensor toggle
   - split recording
   - render-frame recording
3. `MJ_ENV`
   Provides robot-specific behavior:
   - MuJoCo model
   - sensor lookup
   - ray caster camera
   - observation term definitions
   - action scaling and default action
4. `ManagerBasedEnv`
   Owns the policy array and observation groups.
   Each policy has one registered observation group and one action term.
5. `Policy`
   Loads and runs one model or one split pipeline.

The most important design rule is:

- `Sim2SimEnv` is generic runtime infrastructure.
- `MJ_ENV` is robot-specific.
- `ManagerBasedEnv` is observation/action graph wiring.
- `Policy` is model/backend runtime.

If a new robot is being added, prefer creating a new environment subclass on top of `Sim2SimEnv`, not copying the old `MJ_ENV` monolith.

## 3. Runtime Flow

The current loop for the active policy is:

1. `MJ_ENV::step()`
2. `apply_pending_runtime_changes()`
3. `manager_step(policy_id)`
4. `handle_split_snapshot_after_step(d->time)`
5. write action into `d->ctrl`
6. update ray caster distance

Inside `manager_step(policy_id)`:

1. apply pending per-policy runtime reset
2. recompute observations for all registered policy groups
3. run the active policy
4. apply action clip, scale, and default action offset

Inside observation computation:

1. `ActionObsTerm` appends previous action history
2. each `ObservationTerm` computes or reuses its current value
3. all terms are flattened and concatenated
4. the final flattened tensor is passed to `Policy::get_action`

## 4. Policy Registration

Policies are registered in [lab2mj.cpp](/home/albusgive2/go2w_sim2sim/mujoco/C++/lab2mj.cpp).

Current example:

- `motion_mlp`
- `vtm`
- `vtm_lstm_sru`
- `vtm_gru_sru`

Current registration order:

```cpp
policy_list.push_back(PolicySpec::MLP(MOTION_POLICY_PATH, "motion_mlp"));
policy_list.push_back(PolicySpec::MLP(VTM_POLICY_PATH, "vtm"));
policy_list.push_back(
    PolicySpec::SRUSplit(VTM_LSTM_SRU_POLICY_PATH, "vtm_lstm_sru", 1, 128,
                         "lstm_sru"));
policy_list.push_back(
    PolicySpec::SRUSplit(VTM_GRU_SRU_POLICY_PATH, "vtm_gru_sru", 1, 128,
                         "gru_sru"));
```

Important rule:

- the `policy_list` order must match the order in which observation groups are registered in `MJ_ENV::initObsManager()`

Current `MJ_ENV::initObsManager()` order is:

1. `registerManager1()` for `motion_mlp`
2. `registerManager2()` for `vtm`
3. `registerManager3()` for `vtm_lstm_sru`
4. `registerManager4()` for `vtm_gru_sru`

If these two orders drift, the wrong observation set will be paired with the wrong policy.

Per-run path override is also supported from the `lab2mj` command line.

Format:

```bash
./lab2mj vtm_gru_sru=/abs/path/to/policy_dir
./lab2mj vtm=/abs/path/to/policy_dir
```

The key must match `PolicySpec.description`.

## 5. PolicySpec Semantics

`PolicySpec` is declared in [net.h](/home/albusgive2/go2w_sim2sim/utils/cpp_manager_env/net.h).

Available constructors:

- `PolicySpec::MLP(path, description)`
- `PolicySpec::SRU(path, description, num_layers, hidden_dim, memory_type="", mode=Auto)`
- `PolicySpec::SRUFull(...)`
- `PolicySpec::SRUSplit(...)`

Current SRU deployment mode behavior:

- `SRU(...)`
  Full-first. Try the full model first. If full loading fails and `student_deploy.json` exists, fall back to split.
- `SRUFull(...)`
  Force full deployment only.
- `SRUSplit(...)`
  Force split deployment only. Requires `*_deploy.json`.

`memory_type`:

- may be left empty and inferred from `student_info.json` or `student_deploy.json`
- may be set explicitly to enforce consistency
- current examples set it explicitly for clarity

## 6. Backend Selection

Backend is chosen at build time, not runtime.

- `-DUSE_ONNX=ON`
  Uses ONNX Runtime backend.
- `-DUSE_ONNX=OFF`
  Uses LibTorch backend.

Typical build directories in this repo:

- `mujoco/C++/build`
  LibTorch build
- `mujoco/C++/build_onnx`
  ONNX build

The same environment and `PolicySpec` configuration are shared by both backends.

## 7. Model File Discovery

If `PolicySpec.path` is a directory, the loader searches for files under that directory.

Preferred stems:

- MLP prefers `policy.*`, then `student.*`
- SRU prefers `student.*`, then `policy.*`

For SRU split deployment, the loader searches for `*_deploy.json`, typically:

- `student_deploy.json`

For full models, the typical files are:

- JIT full: `student.pt`
- ONNX full: `student.onnx`

For split models, the typical files are:

- `student_deploy.json`
- `student_info.json`
- `student_encoder.pt`
- `student_memory.pt`
- `student_actor.pt`
- `student_encoder.onnx`
- `student_memory.onnx`
- `student_actor.onnx`

Recommended layout:

```text
policy/<policy_name>/
  student.pt
  student.onnx
  student_info.json
  student_deploy.json
  student_encoder.pt
  student_memory.pt
  student_actor.pt
  student_encoder.onnx
  student_memory.onnx
  student_actor.onnx
```

## 8. SRU Runtime Semantics

### 8.1 Full JIT SRU

Expected exporter format:

- `forward(obs) -> action`
- exported `reset()`
- exported `reset_done(dones)`

Old TorchScript SRU with explicit `forward(obs, hidden, cell)` is no longer supported.

### 8.2 Split JIT SRU

Expected pipeline:

1. `student_encoder.pt`: `obs -> encoded_obs`
2. `student_memory.pt`: `encoded_obs -> latent`
3. `student_actor.pt`: `latent -> actions`

State management:

- recurrent state is internal to `student_memory.pt`
- deployment wrapper does not own hidden/cell tensors
- reset works by calling the memory module's `reset()` and `reset_done()`

### 8.3 Full ONNX SRU

Supported recurrent signatures are recognized from actual graph node names.

Current deployment accepts:

- LSTM-style:
  - inputs: `obs`, `hidden_state`, `cell_state`
  - outputs: `actions`, `next_hidden_state`, `next_cell_state`
- GRU-style:
  - inputs: `obs`, `hidden_state`
  - outputs: `actions`, `next_hidden_state`

Some current GRU exports still produce `next_cell_state` or still mention `cell_state` in metadata.
The runtime is tolerant to this, but exporter metadata should ideally match the actual graph.

State management:

- external state owned by C++ wrapper
- outer API still remains `obs -> action`
- wrapper internally allocates and updates recurrent state tensors

### 8.4 Split ONNX SRU

Expected conceptual pipeline:

1. `student_encoder.onnx`: `obs -> encoded_obs`
2. `student_memory.onnx`: recurrent step
3. `student_actor.onnx`: `latent -> actions`

State management:

- external recurrent state owned by C++ wrapper
- actual execution uses the graph's true input/output node names
- this allows current GRU exports to work even when `cell_state` is absent from the graph

### 8.5 Split Config as the Source of Truth

`student_deploy.json` is parsed by the deployment layer and used for:

- schema validation
- original full model names
- total observation input length
- input segment metadata
- encoded observation dimension
- latent dimension
- memory type, layer count, hidden dimension
- JIT pipeline step list
- ONNX pipeline step list
- state-management mode

Important limitation:

- environment-side observation construction is still defined in C++ registration code
- `student_deploy.json` drives split pipeline assembly, not automatic environment observation registration

## 9. Observation Registration and Alignment

Observation registration happens in [mj_env.cpp](/home/albusgive2/go2w_sim2sim/mujoco/C++/mj_env.cpp).

Each policy gets a vector of `ObservationTerm` objects and one `ActionTerm`.

Current examples:

- `registerManager1()`
  Motion policy observation set
- `registerManager2()`
  `vtm` observation set
- `registerManager3()`
  `vtm_lstm_sru` observation set
- `registerManager4()`
  `vtm_gru_sru` observation set

The startup alignment check happens in `ManagerBasedEnv::init_manager()`:

1. it evaluates each term's raw dimension
2. it computes the total flattened input dimension
3. it prints a table of all terms
4. it runs a dummy inference
5. if the model input dimension is wrong, startup fails immediately

This dummy inference is the first alignment check to trust.

## 10. Observation History Rules

Vector observations use `ObservationBuffer`.

Flattened dimension rule:

- vector term total dim = `raw_dim * history_length`

Image observations use `ImageObservationTerm`.

Flattened dimension rule:

- image term total dim = `raw_dim * (history_length + 1)`

Why `+1`:

- image history stores previous frames
- the current frame is appended separately

Special case:

- `history_length == 0` means "latest image only"
- no image history buffer is created
- `get_obs()` returns only the latest frame

This is the current setup for split SRU image input:

- raw image = `1 x 18 x 32 = 576`
- `history_length = 0`
- image contribution = `576`

This is the current setup for `vtm` CNN policy:

- raw image = `576`
- `history_length = 5`
- image contribution = `576 * (5 + 1) = 3456`

## 11. Current Observation Sets

### motion_mlp

Current total input dim:

- `187`

### vtm

Current total input dim:

- `3609`

Composition:

- vector history terms
- image term with `history_length = 5`

### vtm_lstm_sru / vtm_gru_sru

Current total input dim:

- `729`

Composition:

- `153` 1D values from vector terms
- `576` image values from latest `ray_caster`

These SRU policies expect the encoder inside the model to do:

1. 1D normalization
2. per-image CNN encode
3. concat encoded features
4. recurrent memory step
5. actor head

The C++ side should not pre-encode this into latent features.
It must supply the raw flattened observation in exactly the order used at training/export time.

## 12. Action Flow

The policy runtime returns a raw action tensor.

`ManagerBasedEnv::computeAction()` then:

1. stores raw action into `obs_actions[id]`
2. clones it
3. applies clip
4. applies scale
5. adds `default_action`

In `MJ_ENV`, `make_action_term()` currently selects between:

- `action_scale_vec`
- `action2_scale_vec`

The robot-specific PD/control semantics live here, not in the policy runtime.

## 13. Visual Policy Hooks

`Sim2SimEnv` provides hooks for sensor-driven policies:

- `uses_visual_policy(int policy_idx)`
- `apply_policy_defaults_for_policy(int policy_idx)`
- `refresh_visual_observations(bool warm_start_history)`
- `on_sensor_enabled_changed(bool enabled)`
- `on_env_reset()`

Current `MJ_ENV` implementation:

- policy switch can refresh visual observations
- reset can warm-start image history
- sensor enable/disable is forwarded to the ray caster

Important note:

- `uses_visual_policy()` is currently a manual policy-ID whitelist, not an automatic "has image input" detector
- when adding a new visual policy, do not forget to update this function if that policy needs warm-start or sensor-toggle behavior

When adding a new visual policy, update:

- observation registration
- `uses_visual_policy()`
- any default command or sensor behavior needed for that policy

## 14. Runtime Controls

Keyboard controls in `Sim2SimEnv::keyboard_press()`:

- `w/s`
  increase/decrease `cmd_x`
- `a/d`
  increase/decrease `cmd_y`
- `q/e`
  increase/decrease `cmd_yaw`
- `space`
  zero command
- `r`
  request policy recurrent-state reset
- `1` / `2` / `3` / `4`
  switch current policy
- `x`
  start split recording
- `v`
  add a manual mark to the current split recording
- `c`
  stop split recording

Gamepad controls in `Sim2SimEnv::init_gamepad()`:

- left stick / right stick
  continuous command control
- `A/B/Y/X`
  switch policy IDs `0/1/2/3`
- `LB`
  direct reset of current policy runtime state
- `RB`
  toggle sensor enabled state
- `Menu`
  request current policy recurrent-state reset

## 15. Reset Semantics

There are multiple reset layers.

Environment reset:

- handled in `Sim2SimEnv::reset_callback()`
- resets observation buffers
- resets policy states
- reapplies policy defaults
- may warm-start visual history

Policy reset:

- requested from keyboard/gamepad or during policy switching
- implemented through `Policy::reset_state()`

Per-env reset_done:

- exposed as `Policy::reset_state_done(dones)`
- for JIT full SRU: calls model `reset_done`
- for JIT split SRU: calls split memory module `reset_done`
- for ONNX full/split SRU: zeros wrapper-managed recurrent state for the done batch entries

There is no fixed-step automatic recurrent reset anymore.

## 16. Split Recording and Render Recording

Split recording is only available when the active policy is running in split mode.

When recording is active:

- policy runtime stores a snapshot of:
  - `obs`
  - `encoded_obs`
  - `latent`
  - `actions`
- `Sim2SimEnv` writes these to CSV every inference step
- MuJoCo render view is also captured and saved

Output directory:

- `<repo_root>/split_records/<policy_description>_<timestamp>/`

Files:

- `meta.json`
- `steps.csv`
- `obs.csv`
- `encoded_obs.csv`
- `latent.csv`
- `actions.csv`
- `events.csv`
- `render_frames.csv`
- `render_frames/*.jpg`

Important runtime behavior:

- switching away from the active recorded policy automatically stops and saves the recording
- recording stores manual marks as `manual_mark_0001`, `manual_mark_0002`, and so on

## 17. Current Tooling

Viewer and export tools live under `tools/`.

Common tasks:

- open heatmap viewer:
  - [tools/open_split_record_heatmaps.sh](/home/albusgive2/go2w_sim2sim/tools/open_split_record_heatmaps.sh)
- generate render video from recorded render frames:
  - [tools/render_frames_to_video.sh](/home/albusgive2/go2w_sim2sim/tools/render_frames_to_video.sh)
- heatmap viewer implementation:
  - [tools/play_split_record_heatmaps.py](/home/albusgive2/go2w_sim2sim/tools/play_split_record_heatmaps.py)

The viewer can:

- show recorded `obs`, `encoded_obs`, `latent`, `actions`
- show recorded render view
- show multiple manual marks
- switch between trajectories inside the web UI

## 18. How to Add a New Policy

If the robot and observation layout are already compatible, use this checklist.

1. Put the model files under `policy/<new_policy_name>/`.
2. If needed, add a new path macro in [mujoco/C++/CMakeLists.txt](/home/albusgive2/go2w_sim2sim/mujoco/C++/CMakeLists.txt).
3. Add a new `PolicySpec` entry in [lab2mj.cpp](/home/albusgive2/go2w_sim2sim/mujoco/C++/lab2mj.cpp).
4. Add or reuse a matching `registerManagerN()` in [mj_env.cpp](/home/albusgive2/go2w_sim2sim/mujoco/C++/mj_env.cpp).
5. Keep the policy registration order aligned with the manager registration order.
6. If the policy uses image observations with manual refresh behavior, update `uses_visual_policy()`.
7. Rebuild.
8. Start the binary and trust the startup dimension table plus dummy inference result.

If the new policy uses the exact same observation layout as an existing one:

- you can usually reuse the same `registerManagerN()` logic
- you still need a separate `PolicySpec` entry
- you may still need to update UI/gamepad switch mapping if you want direct selection

## 19. How to Add a New Robot Environment

Preferred path:

1. create a new environment subclass from `Sim2SimEnv`
2. keep policy/runtime/recording logic in the base class
3. implement only robot-specific parts:
   - sensor discovery
   - observation functions
   - action scaling
   - MuJoCo drawing
   - optional visual sensor hooks

Avoid copying the full old `MJ_ENV` if possible.

The goal is:

- `Sim2SimEnv` stays generic
- new robot subclasses stay small

## 20. Common Failure Modes

### Startup dummy inference fails

Usually means:

- total observation dimension does not match model input dimension
- observation order is wrong
- image history length is wrong
- wrong policy got paired with the wrong observation group

### Split deploy metadata mismatch

Typical causes:

- `PolicySpec` says `lstm_sru` but `student_deploy.json` says `gru_sru`
- `num_layers` or `hidden_dim` mismatch

### ONNX split signature mismatch

Typical causes:

- exporter graph signature and `student_deploy.json` do not match
- actual GRU ONNX graph dropped `cell_state` input

Current runtime is tolerant to recognized GRU variants, but exporter metadata should still be kept consistent.

### JIT full SRU rejected

Typical cause:

- full JIT model is still old explicit-state format and lacks `reset()` / `reset_done()`

### Recording ignored

Typical cause:

- current policy is not running in split mode

## 21. Recommended Workflow for New SRU Policies

If you are bringing in a new `StudentTeacherSRU`-style policy:

1. export both full and split variants
2. export both JIT and ONNX if possible
3. keep `student_info.json` and `student_deploy.json` next to the models
4. ensure `memory.type`, `num_layers`, and `hidden_dim` are correct
5. make sure full JIT exports `reset()` and `reset_done()`
6. make sure split JIT memory exports `reset()` and `reset_done()`
7. make sure ONNX graph names match the intended recurrent signature
8. prefer explicit `PolicySpec::SRUSplit(..., "lstm_sru")` or `(..., "gru_sru")` when wiring examples

## 22. Current Project-Specific Notes

Current `lab2mj` policies are:

- policy 0: `motion_mlp`
- policy 1: `vtm`
- policy 2: `vtm_lstm_sru`
- policy 3: `vtm_gru_sru`

Current `vtm_lstm_sru` and `vtm_gru_sru` observations are identical on the C++ side:

- same 1D vector terms
- same latest-frame image term
- same flattened input size `729`

Current runtime supports for both `vtm_lstm_sru` and `vtm_gru_sru`:

- JIT full
- JIT split
- ONNX full
- ONNX split

Split JIT and split ONNX are already exercised by the current example registration.

## 23. Minimal Mental Model

When in doubt, remember this:

- the environment constructs a flattened observation vector
- `ManagerBasedEnv` guarantees one observation group per policy
- `Policy` decides how to interpret that vector based on the deployment mode
- startup dummy inference is the first hard alignment gate
- split recording only exists for split runtime
- if adding a new robot, subclass `Sim2SimEnv`
- if adding a new policy on the same robot, update `PolicySpec` plus observation registration order
