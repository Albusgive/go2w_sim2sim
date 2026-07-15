# MuJoCo collector and key replay

`mujoco_data_collector` runs one generated terrain/command pair with either the
deployed `vtm_lstm_sru` or `vtm_gru_sru` policy. The executable owns environment
and recurrent-state resets, five-attempt retry behavior, terminal detection,
and atomic MJCF key output. Batch mode remains headless and runs without
real-time sleeps; `--visualize` opens a synchronized MuJoCo window and paces the
same control loop in real time.

Build the target with the same inference backend used by `lab2mj`. MuJoCo may
be supplied either as its installed CMake package or as a release archive.
MuJoCo 3.5 or newer is recommended because `vtm_lstm_sru` configures the
ray-caster's stereo-loss noise path; current MuJoCo 3.9 headers are supported:

```bash
cmake -S mujoco/C++ -B mujoco/C++/build_onnx \
  -DUSE_ONNX=ON \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime \
  -DMUJOCO_ROOT=/path/to/mujoco-release
cmake --build mujoco/C++/build_onnx \
  --target mujoco_data_collector mujoco_key_replayer -j
```

```bash
mujoco_data_collector \
  --terrain data_collection/single_platform/<terrain>/terrain.xml \
  --metadata data_collection/single_platform/<terrain>/terrain.json \
  --output data_collection/single_platform/<terrain>/single_platform-cmd_linv_x_0p50.xml \
  --speed 0.50 \
  --policy policy/vtm_lstm_sru \
  --policy-type lstm_sru \
  --result data_collection/single_platform/<terrain>/.collection_status/single_platform-cmd_linv_x_0p50.json
```

`--policy-type` accepts `lstm_sru` and `gru_sru`; pair it with the corresponding
policy directory. Every attempt begins with a complete reset of MuJoCo state,
observations/actions, commands, PID state, and recurrent hidden/cell state. The
optional `--reset-before-near-edge <meters>` performs one additional reset while
the robot is moving, when it reaches that distance before the terrain's near
edge. This in-motion reset clears only recurrent hidden/cell state: it does not
teleport the robot or reset MuJoCo, observations/actions, commands, or PID state.

The formal full-matrix profile always uses the LSTM policy with a recurrent-only
reset 1 m before each terrain edge. It is resumable and regenerates the report
and radar charts after collection:

```bash
python3 tools/recollect_all_near_reset.py --workers 4
```

To retry only ditch jobs that still have no valid key, run:

```bash
python3 tools/retry_failed_ditch_fallback.py --workers 4
```

The retry stages are ordered and stop at the first success: `vtm_lstm_sru` with
a recurrent-only reset 1 m before the ditch, then `vtm_gru_sru` without an
in-motion reset, then `vtm_gru_sru` with the same 1 m reset. Each stage allows
five attempts, and jobs that already have a valid key are left untouched.

Add `--visualize` to watch a single collection attempt. Space pauses/resumes;
`+` and `-` change the real-time rate; Escape or closing the window cancels the job. The
recommended selector is the standard-library Tk UI, which also preserves the
runner's schema-2 status and fingerprint semantics:

```bash
python3 tools/data_collection_ui.py
python3 tools/data_collection_ui.py --check
```

`mujoco_key_replayer --trajectory <key.xml> --metadata <terrain.json>` restores
recorded keyframes without calling `mj_step`, so replay cannot drift away from
the saved states. Its window supports Space, Left/Right, Home/End, `L`, `+/-`,
and Escape. The Tk UI exposes the same controls over newline-delimited JSON.

Use `--validate-only` with `--terrain` and `--metadata` to compile the terrain
and check its robot, sensor, timestep, and metadata contract without loading a
policy. Exit code `0` means success, `2` means all collection attempts failed,
and `1` means an argument, model, policy, or I/O error. The last stdout line is
always the corresponding compact JSON result for non-help invocations.

Platform terminals use an x center band plus minimum support height and lateral
bounds. Ditch terminals use a one-sided x distance. The forward command is set
to zero at the terminal and support must remain valid for the requested one
second hold. Stalls, non-finite state, and falling below the terrain are explicit
attempt failure reasons.

## Straight-path heading PID

Every attempt records the reset base position and yaw as a straight reference
path. At 50 Hz, lateral error in that initial heading frame produces a bounded
target-heading correction, and a reset-per-attempt PID writes only the policy's
yaw command. The forward command remains the selected `linv_x`, lateral velocity
is zero, and all three commands become zero during the terminal hold. High-level
commands and PID internals are result telemetry only; key XML remains standard
MJCF `time/qpos/qvel/act/ctrl` state.

Defaults are `cross_track_gain=1.25 rad/m`, heading correction `+/-0.35 rad`,
`Kp=1.20`, `Ki=0.05`, `Kd=0.10`, yaw command `+/-0.50 rad/s`, integral
`+/-0.50 rad*s`, and derivative low-pass new-sample weight `0.20`. Override
them per collector invocation with:

```text
--pid-kp
--pid-ki
--pid-kd
--pid-cross-track-gain
--pid-heading-limit
--pid-yaw-cmd-limit
--pid-integral-limit
--pid-derivative-alpha
```

Each attempt result reports `max_abs_cross_track_m` and
`final_heading_error_deg` (relative to the final cross-track-corrected target
heading, not simply the initial path yaw); the result root records the exact
PID configuration. Collection speeds are limited to `0.50-1.00 m/s` in
`0.05 m/s` increments.
