# Split Record Analysis Guide

This note is for other agents working in `/home/albusgive2/go2w_sim2sim`.
It describes the current SRU split-record collection pipeline, where data is
saved, how to open it, and how to analyze behavior around one or more marks.

## Scope

This guide is only about the `StudentTeacherSRU` split deployment debug path.
It does **not** apply to:

- full JIT `student.pt`
- full ONNX `student.onnx`
- non-SRU MLP policies

The recording path is active only when the runtime policy is actually using
`SRUSplit`.

## What Is Collected

When split recording is active, sim2sim stores one row per policy step for:

- `obs`
- `encoded_obs`
- `latent`
- `actions`

Current semantics are:

- `obs`
  - full policy observation seen by the split encoder
- `encoded_obs`
  - encoder output fed into the SRU memory module
- `latent`
  - SRU memory output fed into the actor
- `actions`
  - final actor output

This first version does **not** record SRU `hidden/cell` explicitly.

## Runtime Controls

Inside the MuJoCo runtime:

- `x`
  - start split recording
- `v`
  - add a manual mark
- `c`
  - stop recording and save

If the current policy is not an active split runtime, recording requests are
ignored and a warning is printed.

## Saved Directory

New recordings are written under:

- `/home/albusgive2/go2w_sim2sim/split_records`

Each session is stored as:

- `split_records/<policy_description>_<timestamp>/`

Example:

- `/home/albusgive2/go2w_sim2sim/split_records/vtm_sru_20260325_192941`

Older sessions may still exist under legacy locations:

- `/home/albusgive2/go2w_sim2sim/build/split_records`
- `/home/albusgive2/go2w_sim2sim/build_onnx/split_records`

The launcher script supports both new and legacy roots.

## Saved Files

Each recording directory contains:

- `meta.json`
- `steps.csv`
- `obs.csv`
- `encoded_obs.csv`
- `latent.csv`
- `actions.csv`
- `events.csv`

### File meanings

- `meta.json`
  - backend
  - shapes
  - written step count
  - marker count
- `steps.csv`
  - `inference_index`
  - `sim_time`
  - `policy_id`
  - `cmd_x`, `cmd_y`, `cmd_yaw`
- `events.csv`
  - `start_record`
  - `mark`
  - `policy_reset`
  - `env_reset`
  - `stop_record`

### Mark format

Manual marks currently look like:

- `manual_mark_0001`
- `manual_mark_0002`
- `manual_mark_0003`

To locate a mark, read `events.csv` and match on:

- `event == mark`
- `detail == manual_mark_xxxx`

The key join field across all CSVs is:

- `inference_index`

## Current Viewer Tools

Two tools exist under `/home/albusgive2/go2w_sim2sim/tools`:

- [open_split_record_heatmaps.sh](/home/albusgive2/go2w_sim2sim/tools/open_split_record_heatmaps.sh)
- [play_split_record_heatmaps.py](/home/albusgive2/go2w_sim2sim/tools/play_split_record_heatmaps.py)

### Recommended entry point

Use:

```bash
tools/open_split_record_heatmaps.sh
```

Current behavior:

- if no `record_dir` is passed, it opens the latest available recording
- if `--focus-mark` is omitted, it defaults to `--focus-mark all`
- if `--open` is omitted, it defaults to `--open`
- it supports both repo-root and legacy recording directories
- it pre-generates overview viewers for all known recordings so the HTML UI can switch trajectories

### Direct Python usage

Use:

```bash
python3 tools/play_split_record_heatmaps.py \
  /path/to/split_records/<session> \
  --focus-mark manual_mark_0003 \
  --window 160 \
  --open
```

`--focus-mark` supports:

- a single mark, for example `manual_mark_0003`
- multiple marks, for example `manual_mark_0002,manual_mark_0003`
- `all`

## Why The Viewer Is HTML, Not Matplotlib

The current local Python environment has a `numpy` / `matplotlib` ABI mismatch.
The viewer is intentionally implemented as:

- Python stdlib data loader
- self-contained generated HTML

This avoids runtime failures from local `matplotlib` imports.

If another agent wants to switch to `matplotlib`, verify the local environment
first.

## What The Viewer Shows

The generated HTML viewer currently visualizes:

- current `Obs Image`
- current `Obs 1D`
- current `Encoded Obs`
- current `Encoded Image Tail`
- current `Latent`
- current `Actions`
- temporal context heatmaps for:
  - `encoded_obs`
  - `latent`
  - `actions`

It also shows:

- all visible marks in the current selection window
- a trajectory selector for switching between recordings in the browser
- a timeline with mark positions
- per-frame summary text
- browser-side scale slider for compact viewing

### Viewer keyboard controls

- `Space`
  - play / pause
- `Left` / `Right`
  - previous / next frame
- `Shift + Left` / `Shift + Right`
  - jump 10 frames
- `[` / `]`
  - previous / next mark

## Observation Layout

Do not hardcode SRU split dimensions when analyzing. Read them from:

- `/home/albusgive2/go2w_sim2sim/policy/vtm_sru/student_deploy.json`

At the time of writing, the current `vtm_sru` deploy config is:

- `obs total_length = 729`
- `1D range = [0, 153)`
- `image range = [153, 729)`
- image shape = `1 x 18 x 32`
- `encoded_obs_dim = 217`
- `latent_dim = 128`
- `memory hidden_dim = 128`
- `num_layers = 1`
- action dim = `16`

For the current exporter structure:

- `encoded_obs[0:153]`
  - 1D branch contribution
- `encoded_obs[153:217]`
  - image feature tail

This mapping is inferred from:

- input 1D length = `153`
- encoder output length = `217`

If exporter semantics change later, re-check `student_deploy.json`.

## Suggested Analysis Workflow

When the goal is "the robot loses image sensitivity after a mark", use this
order:

1. Find the mark in `events.csv`.
2. Compare a window before and after the mark.
3. Inspect `obs` image first.
4. Inspect the encoded image tail next.
5. Inspect `latent` and `actions` last.

### Good default windows

For mark `m`:

- pre window: `m - 150` to `m - 30`
- transition: `m - 30` to `m + 30`
- post window: `m + 30` to `m + 150`

### What to look for

#### 1. Input failure

If the `Obs Image` becomes nearly constant, saturated, or clearly stale after a
mark, the issue is likely in observation generation or sensor update timing.

#### 2. Encoder stops using image

If the image still changes, but `Encoded Image Tail` becomes flat or nearly
constant, the encoder is no longer producing meaningful image features.

#### 3. Memory or actor ignores image

If `Encoded Image Tail` still reacts, but `Latent` and `Actions` collapse into a
low-variation regime, the SRU memory state or downstream actor is likely the
real failure point.

## Example Target Case

One current use case is:

- robot climbs onto a high platform
- jumps three gaps
- after `manual_mark_0003`, behavior suggests image information is no longer used

For this case, the recommended first command is:

```bash
tools/open_split_record_heatmaps.sh \
  build/split_records/vtm_sru_20260325_184212 \
  --focus-mark manual_mark_0003 \
  --window 160
```

Then compare:

- `Obs Image`
- `Encoded Image Tail`
- `Latent`
- `Actions`

before and after the marked transition.

## Notes For Future Agents

- Keep split recording logic separate from general policy inference.
- Do not assume recordings only live under `split_records/`; legacy sessions may
  still be in `build/split_records`.
- If `student_deploy.json` changes, prefer reading dimensions dynamically rather
  than updating analysis code by hand.
- If `lab2mj.cpp` starts failing on SRU init, verify that the `PolicySpec`
  memory config matches `student_deploy.json`.
