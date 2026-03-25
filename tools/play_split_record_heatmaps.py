#!/usr/bin/env python3
"""Generate a self-contained HTML heatmap viewer for SRU split recordings.

The viewer reads a split recording directory produced by sim2sim and exports
an interactive HTML page that can:
  - play / pause through recorded frames
  - jump to marks from events.csv
  - focus on a window around one or more marks
  - visualize:
      * current MuJoCo render view
      * current image observation
      * current 1D observation strip
      * current encoded_obs strip
      * current encoded image-feature strip
      * current latent strip
      * current actions strip
      * temporal context heatmaps for encoded_obs / latent / actions

The script intentionally uses only the Python standard library so it does not
depend on matplotlib in the local environment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_DEPLOY_JSON = Path("policy/vtm_sru/student_deploy.json")


@dataclass
class TensorTable:
    inference_index: list[int]
    sim_time: list[float]
    values: list[list[float]]

    @property
    def rows(self) -> int:
        return len(self.inference_index)

    @property
    def cols(self) -> int:
        return len(self.values[0]) if self.values else 0


def script_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an HTML heatmap player for split SRU recordings."
    )
    parser.add_argument(
        "record_dir",
        type=Path,
        help="Path to a split_records/<session> directory.",
    )
    parser.add_argument(
        "--deploy-json",
        type=Path,
        default=DEFAULT_DEPLOY_JSON,
        help="Path to student_deploy.json used to decode obs layout.",
    )
    parser.add_argument(
        "--focus-mark",
        type=str,
        default="",
        help=(
            "Optional mark selection. Use a mark label such as manual_mark_0003, "
            "a comma-separated list, or 'all' to visualize windows around all "
            "recorded marks."
        ),
    )
    parser.add_argument(
        "--window",
        type=int,
        default=180,
        help="Frames before/after --focus-mark to keep. Ignored if no mark is set.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Keep one frame every N frames when exporting the viewer.",
    )
    parser.add_argument(
        "--context-radius",
        type=int,
        default=40,
        help="Temporal context radius shown in the lower heatmaps.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=12.0,
        help="Initial playback FPS in the viewer.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML file. Defaults to <record_dir>/heatmap_viewer.html.",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the generated HTML in the default browser.",
    )
    return parser.parse_args()


def load_tensor_csv(path: Path) -> TensorTable:
    inference_index: list[int] = []
    sim_time: list[float] = []
    values: list[list[float]] = []

    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None or len(header) < 3:
            raise ValueError(f"Invalid tensor CSV: {path}")
        for row in reader:
            if not row:
                continue
            inference_index.append(int(row[0]))
            sim_time.append(float(row[1]))
            values.append([float(item) for item in row[2:]])

    return TensorTable(inference_index=inference_index, sim_time=sim_time, values=values)


def load_steps_csv(path: Path) -> dict[str, list[float | int]]:
    result = {
        "inference_index": [],
        "sim_time": [],
        "policy_id": [],
        "cmd_x": [],
        "cmd_y": [],
        "cmd_yaw": [],
    }
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            result["inference_index"].append(int(row["inference_index"]))
            result["sim_time"].append(float(row["sim_time"]))
            result["policy_id"].append(int(row["policy_id"]))
            result["cmd_x"].append(float(row["cmd_x"]))
            result["cmd_y"].append(float(row["cmd_y"]))
            result["cmd_yaw"].append(float(row["cmd_yaw"]))
    return result


def load_events_csv(path: Path) -> list[dict[str, str | float | int]]:
    events: list[dict[str, str | float | int]] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            events.append(
                {
                    "inference_index": int(row["inference_index"]),
                    "sim_time": float(row["sim_time"]),
                    "event": row["event"],
                    "detail": row["detail"],
                }
            )
    return events


def load_render_frames_csv(path: Path) -> dict[int, dict[str, str | int | float]]:
    if not path.exists():
        return {}

    render_frames: dict[int, dict[str, str | int | float]] = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            inference_index = int(row["inference_index"])
            render_frames[inference_index] = {
                "sim_time": float(row["sim_time"]),
                "render_frame_id": int(row["render_frame_id"]) if row["render_frame_id"] else 0,
                "width": int(row["width"]) if row["width"] else 0,
                "height": int(row["height"]) if row["height"] else 0,
                "file": row.get("file", ""),
            }
    return render_frames


def load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def collect_available_record_dirs(repo_root: Path) -> list[Path]:
    search_roots = [
        repo_root / "split_records",
        repo_root / "build" / "split_records",
        repo_root / "build_onnx" / "split_records",
    ]
    records: list[Path] = []
    for root in search_roots:
        if not root.is_dir():
            continue
        for entry in sorted(root.iterdir()):
            if entry.is_dir():
                records.append(entry.resolve())
    return sorted(records, reverse=True)


def summarize_record_dir(record_dir: Path) -> dict:
    meta_path = record_dir / "meta.json"
    backend = "unknown"
    steps = None
    marks = None
    if meta_path.exists():
        try:
            meta = load_json(meta_path)
            backend = str(meta.get("backend", backend))
            steps = meta.get("written_steps")
            marks = meta.get("marker_count")
        except Exception:
            pass

    return {
        "recordDir": str(record_dir.resolve()),
        "label": record_dir.name,
        "relativeDir": str(record_dir.resolve()).replace(str(script_repo_root()) + "/", ""),
        "backend": backend,
        "steps": steps,
        "marks": marks,
        "viewerUri": (record_dir / "heatmap_viewer_all.html").resolve().as_uri()
        if (record_dir / "heatmap_viewer_all.html").exists()
        else "",
    }


def validate_alignment(steps: dict[str, list], tensors: Iterable[TensorTable]) -> None:
    reference = steps["inference_index"]
    for tensor in tensors:
        if tensor.inference_index != reference:
            raise ValueError(
                "CSV alignment mismatch. steps.csv and tensor CSV files must have "
                "the same inference_index ordering."
            )


def build_frame_index_map(inference_index: list[int]) -> dict[int, int]:
    return {value: idx for idx, value in enumerate(inference_index)}


def is_user_mark_event(event: dict[str, str | float | int]) -> bool:
    detail = str(event.get("detail", ""))
    event_name = str(event.get("event", ""))
    return event_name == "mark" or detail.startswith("manual_mark_")


def parse_focus_mark_query(focus_mark: str) -> list[str]:
    if not focus_mark.strip():
        return []
    if focus_mark.strip().lower() == "all":
        return ["all"]
    return [item.strip() for item in focus_mark.split(",") if item.strip()]


def select_frame_indices(
    total_frames: int,
    mark_frames: list[int],
    window: int,
    stride: int,
    extra_frames: Iterable[int],
) -> list[int]:
    selected = set()
    if not mark_frames:
        ranges = [(0, total_frames - 1)]
    else:
        ranges = [
            (max(0, frame - window), min(total_frames - 1, frame + window))
            for frame in sorted(set(mark_frames))
        ]

    for start, end in ranges:
        for idx in range(start, end + 1):
            if (idx - start) % max(1, stride) == 0:
                selected.add(idx)
    for idx in extra_frames:
        if any(start <= idx <= end for start, end in ranges):
            selected.add(idx)
    for frame in mark_frames:
        selected.add(frame)
    return sorted(selected)


def gather_rows(values: list[list[float]], indices: list[int]) -> list[list[float]]:
    return [values[idx] for idx in indices]


def gather_scalar_rows(values: list[float | int], indices: list[int]) -> list[float | int]:
    return [values[idx] for idx in indices]


def compute_min_max(rows: Iterable[Iterable[float]]) -> tuple[float, float]:
    minimum = math.inf
    maximum = -math.inf
    found = False
    for row in rows:
        for value in row:
            minimum = min(minimum, value)
            maximum = max(maximum, value)
            found = True
    if not found:
        return 0.0, 1.0
    if minimum == maximum:
        epsilon = 1.0 if minimum == 0.0 else abs(minimum) * 0.05
        return minimum - epsilon, maximum + epsilon
    return minimum, maximum


def compute_abs_max(rows: Iterable[Iterable[float]]) -> float:
    abs_max = 0.0
    for row in rows:
        for value in row:
            abs_max = max(abs_max, abs(value))
    return abs_max if abs_max > 0.0 else 1.0


def sum_1d_obs_length(input_segments: list[dict]) -> int:
    total = 0
    for segment in input_segments:
        if segment.get("type") == "1D":
            start, end = segment["range"]
            total += end - start
    return total


def collect_image_stats(obs_rows: list[list[float]], image_segments: list[dict]) -> tuple[float, float]:
    minimum = math.inf
    maximum = -math.inf
    found = False
    for row in obs_rows:
        for segment in image_segments:
            start, end = segment["range"]
            for value in row[start:end]:
                minimum = min(minimum, value)
                maximum = max(maximum, value)
                found = True
    if not found:
        return 0.0, 1.0
    if minimum == maximum:
        epsilon = 1.0 if minimum == 0.0 else abs(minimum) * 0.05
        return minimum - epsilon, maximum + epsilon
    return minimum, maximum


def build_marks(
    events: list[dict[str, str | float | int]],
    frame_lookup: dict[int, int],
    selected_frame_set: set[int],
) -> list[dict[str, str | float | int]]:
    marks: list[dict[str, str | float | int]] = []
    for event in events:
        inference_index = int(event["inference_index"])
        frame = frame_lookup.get(inference_index)
        if frame is None or frame not in selected_frame_set:
            continue
        marks.append(
            {
                "frame": frame,
                "inference_index": inference_index,
                "sim_time": event["sim_time"],
                "event": event["event"],
                "detail": event["detail"],
            }
        )
    return marks


def remap_marks_to_selected_indices(
    marks: list[dict[str, str | float | int]], selected_indices: list[int]
) -> list[dict[str, str | float | int]]:
    selected_lookup = {frame: idx for idx, frame in enumerate(selected_indices)}
    remapped = []
    for mark in marks:
        frame = int(mark["frame"])
        remapped.append({**mark, "frame": selected_lookup[frame]})
    return remapped


def find_focus_mark_frames(
    events: list[dict[str, str | float | int]],
    frame_lookup: dict[int, int],
    focus_mark: str,
) -> list[int]:
    query = parse_focus_mark_query(focus_mark)
    if not query:
        return []

    if query == ["all"]:
        frames = [
            frame_lookup[int(event["inference_index"])]
            for event in events
            if is_user_mark_event(event) and int(event["inference_index"]) in frame_lookup
        ]
        return sorted(set(frames))

    matched_frames: list[int] = []
    missing: list[str] = []
    for target in query:
        target_found = False
        for event in events:
            detail = str(event["detail"])
            event_name = str(event["event"])
            if detail == target or event_name == target:
                inference_index = int(event["inference_index"])
                if inference_index in frame_lookup:
                    matched_frames.append(frame_lookup[inference_index])
                    target_found = True
        if not target_found:
            missing.append(target)
    if missing:
        raise ValueError(f'Focus mark(s) not found in events.csv: {", ".join(missing)}')
    return sorted(set(matched_frames))


def make_output_path(record_dir: Path, output: Path | None, focus_mark: str) -> Path:
    if output is not None:
        return output
    suffix = ""
    if focus_mark:
        safe_mark = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in focus_mark)
        suffix = f"_{safe_mark}"
    return record_dir / f"heatmap_viewer{suffix}.html"


def build_payload(
    record_dir: Path,
    deploy_json: Path,
    focus_mark: str,
    window: int,
    stride: int,
    context_radius: int,
    initial_fps: float,
) -> dict:
    repo_root = script_repo_root()
    obs = load_tensor_csv(record_dir / "obs.csv")
    encoded = load_tensor_csv(record_dir / "encoded_obs.csv")
    latent = load_tensor_csv(record_dir / "latent.csv")
    actions = load_tensor_csv(record_dir / "actions.csv")
    steps = load_steps_csv(record_dir / "steps.csv")
    events = load_events_csv(record_dir / "events.csv")
    render_frames = load_render_frames_csv(record_dir / "render_frames.csv")
    deploy = load_json(deploy_json)
    record_meta = load_json(record_dir / "meta.json") if (record_dir / "meta.json").exists() else {}

    validate_alignment(steps, [obs, encoded, latent, actions])

    frame_lookup = build_frame_index_map(steps["inference_index"])
    focus_frames = find_focus_mark_frames(events, frame_lookup, focus_mark)

    event_frames = [
        frame_lookup[int(event["inference_index"])]
        for event in events
        if int(event["inference_index"]) in frame_lookup
    ]
    selected_indices = select_frame_indices(
        total_frames=len(steps["inference_index"]),
        mark_frames=focus_frames,
        window=window,
        stride=max(1, stride),
        extra_frames=event_frames,
    )
    selected_index_set = set(selected_indices)

    marks = build_marks(events, frame_lookup, selected_index_set)
    marks = remap_marks_to_selected_indices(marks, selected_indices)

    input_segments = deploy["input"]["segments"]
    image_segments = [segment for segment in input_segments if segment.get("type") == "2D"]
    one_d_length = sum_1d_obs_length(input_segments)
    encoded_obs_dim = int(deploy["intermediate_tensors"]["encoded_obs_dim"])
    encoded_img_dim = max(0, encoded_obs_dim - one_d_length)

    obs_rows = gather_rows(obs.values, selected_indices)
    encoded_rows = gather_rows(encoded.values, selected_indices)
    latent_rows = gather_rows(latent.values, selected_indices)
    actions_rows = gather_rows(actions.values, selected_indices)
    selected_inference_indices = gather_scalar_rows(steps["inference_index"], selected_indices)

    render_rows = []
    for inference_index in selected_inference_indices:
        render_info = render_frames.get(int(inference_index))
        if not render_info:
            render_rows.append(None)
            continue
        file_value = str(render_info.get("file", "") or "")
        render_rows.append(
            {
                "uri": (record_dir / file_value).resolve().as_uri() if file_value else "",
                "file": file_value,
                "renderFrameId": int(render_info.get("render_frame_id", 0)),
                "width": int(render_info.get("width", 0)),
                "height": int(render_info.get("height", 0)),
            }
        )

    obs_min, obs_max = compute_min_max(obs_rows)
    image_min, image_max = collect_image_stats(obs_rows, image_segments)

    payload = {
        "repoRoot": str(repo_root),
        "recordDir": str(record_dir.resolve()),
        "deployJson": str(deploy_json.resolve()),
        "focusMark": focus_mark,
        "focusMarks": parse_focus_mark_query(focus_mark),
        "selectedFrameCount": len(selected_indices),
        "contextRadius": max(1, context_radius),
        "initialFps": initial_fps,
        "steps": {
            "inferenceIndex": selected_inference_indices,
            "simTime": gather_scalar_rows(steps["sim_time"], selected_indices),
            "policyId": gather_scalar_rows(steps["policy_id"], selected_indices),
            "cmdX": gather_scalar_rows(steps["cmd_x"], selected_indices),
            "cmdY": gather_scalar_rows(steps["cmd_y"], selected_indices),
            "cmdYaw": gather_scalar_rows(steps["cmd_yaw"], selected_indices),
        },
        "marks": marks,
        "renderFrames": render_rows,
        "obs": obs_rows,
        "encodedObs": encoded_rows,
        "latent": latent_rows,
        "actions": actions_rows,
        "availableTrajectories": [
            {
                **summary,
                "isCurrent": summary["recordDir"] == str(record_dir.resolve()),
            }
            for summary in (
                summarize_record_dir(path) for path in collect_available_record_dirs(repo_root)
            )
        ],
        "meta": {
            "record": record_meta,
            "deploySchema": deploy.get("schema", ""),
            "modelName": deploy.get("model_name", ""),
            "totalObsDim": int(deploy["input"]["total_length"]),
            "encodedObsDim": encoded_obs_dim,
            "latentDim": int(deploy["intermediate_tensors"]["latent_dim"]),
            "actionDim": actions.cols,
            "memory": deploy.get("memory", {}),
            "obsSegments": input_segments,
            "imageSegments": image_segments,
            "oneDObsLength": one_d_length,
            "encodedImageRange": (
                [one_d_length, one_d_length + encoded_img_dim]
                if encoded_img_dim > 0
                else None
            ),
            "renderViewAvailable": any(
                entry is not None and bool(entry.get("uri"))
                for entry in render_rows
            ),
            "scales": {
                "obs": {"min": obs_min, "max": obs_max},
                "obsImage": {"min": image_min, "max": image_max},
                "encodedObs": {"absMax": compute_abs_max(encoded_rows)},
                "encodedImage": {
                    "absMax": compute_abs_max(
                        row[one_d_length : one_d_length + encoded_img_dim]
                        for row in encoded_rows
                    )
                    if encoded_img_dim > 0
                    else 1.0
                },
                "latent": {"absMax": compute_abs_max(latent_rows)},
                "actions": {"absMax": compute_abs_max(actions_rows)},
            },
        },
    }
    return payload


def build_html(payload: dict) -> str:
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Split Record Heatmap Viewer</title>
  <style>
    :root {{
      --bg: #0f1116;
      --panel: #171b22;
      --panel-2: #202734;
      --text: #e7edf8;
      --muted: #a6b1c2;
      --accent: #5ec8ff;
      --mark: #ffb347;
      --border: #2c3443;
      --panel-scale: 0.86;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0 auto;
      max-width: 1760px;
      padding: 10px;
      background: radial-gradient(circle at top, #1a2230 0%, var(--bg) 52%);
      color: var(--text);
      font: 13px/1.35 "SFMono-Regular", "Menlo", "Consolas", monospace;
    }}
    h1, h2, h3 {{ margin: 0 0 6px 0; font-weight: 600; }}
    .top {{
      display: grid;
      gap: 10px;
      grid-template-columns: minmax(0, 1.7fr) minmax(250px, 0.9fr);
      margin-bottom: 10px;
    }}
    .panel {{
      background: rgba(23, 27, 34, 0.94);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 10px 12px;
      box-shadow: 0 18px 50px rgba(0, 0, 0, 0.25);
      min-width: 0;
    }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      margin-bottom: 8px;
    }}
    .controls label {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    button, select, input {{
      background: var(--panel-2);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 6px 9px;
      font: inherit;
    }}
    input[type="range"] {{
      width: 170px;
      padding: 0;
    }}
    #frameSlider {{
      width: 220px;
    }}
    .summary {{
      white-space: pre-wrap;
      color: var(--muted);
      min-height: 50px;
      font-size: 12px;
    }}
    .layout {{
      display: grid;
      gap: 10px;
      grid-template-columns: 1.02fr 1.02fr 0.72fr 0.72fr;
    }}
    .layout-wide {{
      display: grid;
      gap: 10px;
      grid-template-columns: 1.08fr 0.96fr 0.8fr;
      margin-top: 10px;
    }}
    .canvas-wrap {{
      display: flex;
      flex-direction: column;
      gap: 5px;
      min-width: 0;
    }}
    canvas {{
      width: 100%;
      display: block;
      background: #0a0d12;
      border: 1px solid var(--border);
      border-radius: 10px;
      image-rendering: pixelated;
    }}
    #timeline {{
      height: calc(80px * var(--panel-scale));
    }}
    #renderViewImage {{
      width: 100%;
      height: calc(168px * var(--panel-scale));
      display: block;
      background: #0a0d12;
      border: 1px solid var(--border);
      border-radius: 10px;
      object-fit: contain;
      image-rendering: auto;
    }}
    #obsImageCanvas {{
      height: calc(168px * var(--panel-scale));
    }}
    #obs1dCanvas,
    #encodedCanvas,
    #encodedImageCanvas,
    #latentCanvas,
    #actionsCanvas {{
      height: calc(44px * var(--panel-scale));
    }}
    #encodedContextCanvas,
    #latentContextCanvas,
    #actionsContextCanvas {{
      height: calc(176px * var(--panel-scale));
    }}
    .caption {{
      color: var(--muted);
      font-size: 11px;
    }}
    .status-line {{
      color: var(--muted);
      font-size: 11px;
      min-height: 16px;
    }}
    .small {{
      font-size: 12px;
      color: var(--muted);
    }}
    .mark-list {{
      max-height: calc(250px * var(--panel-scale));
      overflow: auto;
      line-height: 1.5;
      padding-right: 4px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}
    .mark-item {{
      padding: 6px 8px;
      color: var(--muted);
      background: #10151d;
      border: 1px solid var(--border);
      border-radius: 8px;
      text-align: left;
      cursor: pointer;
      font: inherit;
    }}
    .mark-item.current {{
      border-color: var(--mark);
      color: var(--text);
      box-shadow: inset 0 0 0 1px rgba(255, 179, 71, 0.25);
    }}
    .meta-grid {{
      display: grid;
      gap: 8px;
      grid-template-columns: 1fr;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      color: var(--muted);
    }}
    .scale-value {{
      color: var(--muted);
      min-width: 42px;
      text-align: right;
    }}
    .kbd {{
      display: inline-block;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 1px 6px;
      background: #10151d;
      color: var(--text);
    }}
    @media (max-width: 1100px) {{
      .top, .layout, .layout-wide {{
        grid-template-columns: 1fr;
      }}
      #renderViewImage,
      #obsImageCanvas,
      #encodedContextCanvas,
      #latentContextCanvas,
      #actionsContextCanvas {{
        height: auto;
      }}
    }}
  </style>
</head>
<body>
  <div class="top">
    <div class="panel">
      <h1>Split Record Heatmap Viewer</h1>
      <div class="controls">
        <button id="playPause">Play</button>
        <button id="prevFrame">Prev</button>
        <button id="nextFrame">Next</button>
        <button id="prevMark">Prev Mark</button>
        <button id="nextMark">Next Mark</button>
        <label>Trajectory
          <select id="trajectorySelect"></select>
        </label>
        <label>Frame
          <input id="frameSlider" type="range" min="0" value="0">
        </label>
        <label>FPS
          <input id="fpsInput" type="number" min="1" max="60" step="1">
        </label>
        <label>Scale
          <input id="scaleSlider" type="range" min="65" max="125" step="5" value="86">
          <span class="scale-value" id="scaleValue">86%</span>
        </label>
        <label>Jump Mark
          <select id="markSelect"></select>
        </label>
      </div>
      <canvas id="timeline" width="1200" height="120"></canvas>
      <div class="summary" id="summary"></div>
      <div class="small">
        Keys:
        <span class="kbd">Space</span> play / pause,
        <span class="kbd">←</span> / <span class="kbd">→</span> frame,
        <span class="kbd">Shift</span> + <span class="kbd">←</span> / <span class="kbd">→</span> jump 10 frames,
        <span class="kbd">[</span> / <span class="kbd">]</span> prev / next mark.
      </div>
    </div>
    <div class="panel">
      <h3>Marks</h3>
      <div class="small" id="markMeta"></div>
      <div class="mark-list" id="markList"></div>
    </div>
  </div>

  <div class="layout">
    <div class="panel canvas-wrap">
      <h3>Render View</h3>
      <img id="renderViewImage" alt="Recorded MuJoCo render view">
      <div class="status-line" id="renderViewStatus"></div>
      <div class="caption">Window framebuffer captured during split recording.</div>
    </div>
    <div class="panel canvas-wrap">
      <h3>Obs Image</h3>
      <canvas id="obsImageCanvas" width="640" height="360"></canvas>
      <div class="caption">Current 2D observation decoded from obs range.</div>
    </div>
    <div class="panel canvas-wrap">
      <h3>Obs 1D</h3>
      <canvas id="obs1dCanvas" width="640" height="72"></canvas>
      <div class="caption">Current 1D observation strip.</div>
    </div>
    <div class="panel canvas-wrap">
      <h3>Encoded Obs</h3>
      <canvas id="encodedCanvas" width="640" height="72"></canvas>
      <div class="caption">Full encoder output fed into SRU memory.</div>
    </div>
  </div>

  <div class="layout-wide">
    <div class="panel canvas-wrap">
      <h3>Encoded Image Tail</h3>
      <canvas id="encodedImageCanvas" width="640" height="72"></canvas>
      <div class="caption">Tail slice of encoded_obs attributed to image features.</div>
    </div>
    <div class="panel canvas-wrap">
      <h3>Latent</h3>
      <canvas id="latentCanvas" width="640" height="72"></canvas>
      <div class="caption">SRU output latent.</div>
    </div>
    <div class="panel canvas-wrap">
      <h3>Actions</h3>
      <canvas id="actionsCanvas" width="640" height="72"></canvas>
      <div class="caption">Final actor output.</div>
    </div>
  </div>

  <div class="layout-wide">
    <div class="panel canvas-wrap">
      <h3>Encoded Obs Context</h3>
      <canvas id="encodedContextCanvas" width="700" height="360"></canvas>
      <div class="caption">Rows are time around the current frame. Center row is highlighted.</div>
    </div>
    <div class="panel canvas-wrap">
      <h3>Latent Context</h3>
      <canvas id="latentContextCanvas" width="420" height="360"></canvas>
      <div class="caption">Temporal context for SRU latent.</div>
    </div>
    <div class="panel canvas-wrap">
      <h3>Action Context</h3>
      <canvas id="actionsContextCanvas" width="300" height="360"></canvas>
      <div class="caption">Temporal context for actor output.</div>
    </div>
  </div>

  <script>
  const payload = {payload_json};

  const frameCount = payload.steps.inferenceIndex.length;
  const contextRadius = payload.contextRadius;
  const marks = payload.marks;
  const scales = payload.meta.scales;
  const trajectories = payload.availableTrajectories || [];

  const state = {{
    frame: 0,
    playing: false,
    fps: payload.initialFps,
    timerId: null,
  }};

  const elements = {{
    playPause: document.getElementById("playPause"),
    prevFrame: document.getElementById("prevFrame"),
    nextFrame: document.getElementById("nextFrame"),
    prevMark: document.getElementById("prevMark"),
    nextMark: document.getElementById("nextMark"),
    trajectorySelect: document.getElementById("trajectorySelect"),
    frameSlider: document.getElementById("frameSlider"),
    fpsInput: document.getElementById("fpsInput"),
    scaleSlider: document.getElementById("scaleSlider"),
    scaleValue: document.getElementById("scaleValue"),
    markSelect: document.getElementById("markSelect"),
    markList: document.getElementById("markList"),
    markMeta: document.getElementById("markMeta"),
    summary: document.getElementById("summary"),
    timeline: document.getElementById("timeline"),
    renderViewImage: document.getElementById("renderViewImage"),
    renderViewStatus: document.getElementById("renderViewStatus"),
    obsImageCanvas: document.getElementById("obsImageCanvas"),
    obs1dCanvas: document.getElementById("obs1dCanvas"),
    encodedCanvas: document.getElementById("encodedCanvas"),
    encodedImageCanvas: document.getElementById("encodedImageCanvas"),
    latentCanvas: document.getElementById("latentCanvas"),
    actionsCanvas: document.getElementById("actionsCanvas"),
    encodedContextCanvas: document.getElementById("encodedContextCanvas"),
    latentContextCanvas: document.getElementById("latentContextCanvas"),
    actionsContextCanvas: document.getElementById("actionsContextCanvas"),
  }};

  elements.frameSlider.max = String(Math.max(0, frameCount - 1));
  elements.fpsInput.value = String(payload.initialFps);

  function applyScale(percent) {{
    const clamped = clamp(percent, 65, 125);
    document.documentElement.style.setProperty(
      "--panel-scale",
      (clamped / 100.0).toFixed(2)
    );
    elements.scaleSlider.value = String(clamped);
    elements.scaleValue.textContent = `${{clamped}}%`;
  }}

  function clamp(value, min, max) {{
    return Math.min(max, Math.max(min, value));
  }}

  function lerp(a, b, t) {{
    return a + (b - a) * t;
  }}

  function blendColor(a, b, t) {{
    return [
      Math.round(lerp(a[0], b[0], t)),
      Math.round(lerp(a[1], b[1], t)),
      Math.round(lerp(a[2], b[2], t)),
      255,
    ];
  }}

  function colorSymmetric(value, absMax) {{
    const safe = Math.max(absMax, 1e-6);
    const t = clamp(0.5 + value / (2 * safe), 0.0, 1.0);
    if (t < 0.5) {{
      return blendColor([36, 72, 132], [248, 248, 248], t / 0.5);
    }}
    return blendColor([248, 248, 248], [184, 58, 58], (t - 0.5) / 0.5);
  }}

  function colorSequential(value, minValue, maxValue) {{
    const span = Math.max(maxValue - minValue, 1e-6);
    const t = clamp((value - minValue) / span, 0.0, 1.0);
    if (t < 0.33) {{
      return blendColor([12, 18, 28], [44, 124, 191], t / 0.33);
    }}
    if (t < 0.66) {{
      return blendColor([44, 124, 191], [246, 211, 101], (t - 0.33) / 0.33);
    }}
    return blendColor([246, 211, 101], [255, 120, 88], (t - 0.66) / 0.34);
  }}

  function drawHeatmap(canvas, rows, cols, getValue, colorFn, highlightRow = null) {{
    const ctx = canvas.getContext("2d");
    const offscreen = document.createElement("canvas");
    offscreen.width = Math.max(1, cols);
    offscreen.height = Math.max(1, rows);
    const offCtx = offscreen.getContext("2d");
    const imageData = offCtx.createImageData(offscreen.width, offscreen.height);
    const data = imageData.data;

    let offset = 0;
    for (let r = 0; r < rows; ++r) {{
      for (let c = 0; c < cols; ++c) {{
        const color = colorFn(getValue(r, c));
        data[offset++] = color[0];
        data[offset++] = color[1];
        data[offset++] = color[2];
        data[offset++] = color[3];
      }}
    }}
    offCtx.putImageData(imageData, 0, 0);

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(offscreen, 0, 0, canvas.width, canvas.height);
    ctx.restore();

    if (highlightRow !== null) {{
      const rowHeight = canvas.height / rows;
      ctx.strokeStyle = "rgba(94, 200, 255, 0.95)";
      ctx.lineWidth = 2;
      ctx.strokeRect(0, highlightRow * rowHeight, canvas.width, rowHeight);
    }}
  }}

  function drawStrip(canvas, values, absMax) {{
    const rows = 1;
    const cols = Math.max(1, values.length);
    drawHeatmap(
      canvas,
      rows,
      cols,
      (_, c) => values[c],
      (v) => colorSymmetric(v, absMax)
    );
  }}

  function drawImageMatrix(canvas, matrix, rows, cols, minValue, maxValue) {{
    drawHeatmap(
      canvas,
      rows,
      cols,
      (r, c) => matrix[r][c],
      (v) => colorSequential(v, minValue, maxValue)
    );
  }}

  function buildObs1d(obsRow) {{
    const result = [];
    for (const segment of payload.meta.obsSegments) {{
      if (segment.type !== "1D") {{
        continue;
      }}
      const start = segment.range[0];
      const end = segment.range[1];
      for (let i = start; i < end; ++i) {{
        result.push(obsRow[i]);
      }}
    }}
    return result;
  }}

  function buildImageMatrix(obsRow) {{
    const tiles = [];
    let width = 0;
    for (const segment of payload.meta.imageSegments) {{
      const shape = segment.shape;
      let channels = 1;
      let height = 1;
      let cols = 1;
      if (shape.length === 3) {{
        channels = shape[0];
        height = shape[1];
        cols = shape[2];
      }} else if (shape.length === 2) {{
        height = shape[0];
        cols = shape[1];
      }} else {{
        continue;
      }}
      width = Math.max(width, cols);
      const segmentValues = obsRow.slice(segment.range[0], segment.range[1]);
      for (let ch = 0; ch < channels; ++ch) {{
        const base = ch * height * cols;
        for (let r = 0; r < height; ++r) {{
          const row = [];
          for (let c = 0; c < cols; ++c) {{
            row.push(segmentValues[base + r * cols + c]);
          }}
          tiles.push(row);
        }}
        if (ch !== channels - 1) {{
          tiles.push(new Array(cols).fill(0.0));
        }}
      }}
    }}
    if (tiles.length === 0) {{
      return {{
        rows: 1,
        cols: 1,
        matrix: [[0.0]],
      }};
    }}
    return {{
      rows: tiles.length,
      cols: width,
      matrix: tiles.map((row) => row.concat(new Array(width - row.length).fill(0.0))),
    }};
  }}

  function getEncodedImageSlice(encodedRow) {{
    const range = payload.meta.encodedImageRange;
    if (!range) {{
      return [];
    }}
    return encodedRow.slice(range[0], range[1]);
  }}

  function findCurrentMark() {{
    return marks.find((mark) => mark.frame === state.frame) || null;
  }}

  function formatMark(mark) {{
    const label = mark.detail || mark.event;
    return `${{label}} @ frame=${{mark.frame}} inference=${{mark.inference_index}}`;
  }}

  function drawTimeline() {{
    const canvas = elements.timeline;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.fillStyle = "#0a0d12";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const left = 20;
    const right = canvas.width - 20;
    const top = 18;
    const bottom = canvas.height - 24;
    const width = right - left;
    const height = bottom - top;

    ctx.strokeStyle = "rgba(166, 177, 194, 0.4)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(left, bottom);
    ctx.lineTo(right, bottom);
    ctx.stroke();

    for (let index = 0; index < marks.length; ++index) {{
      const mark = marks[index];
      const x = left + width * (mark.frame / Math.max(1, frameCount - 1));
      ctx.strokeStyle = mark.event === "mark" ? "#ffb347" : "rgba(166,177,194,0.7)";
      ctx.beginPath();
      ctx.moveTo(x, top);
      ctx.lineTo(x, bottom);
      ctx.stroke();
      if (mark.event === "mark") {{
        ctx.fillStyle = "#ffb347";
        ctx.fillText(mark.detail || mark.event, x + 4, top + 12 + (index % 3) * 12);
      }}
    }}

    const currentX = left + width * (state.frame / Math.max(1, frameCount - 1));
    ctx.strokeStyle = "#5ec8ff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(currentX, top);
    ctx.lineTo(currentX, bottom);
    ctx.stroke();

    ctx.fillStyle = "#a6b1c2";
    ctx.fillText(`0`, left, canvas.height - 6);
    ctx.fillText(`${{frameCount - 1}}`, right - 28, canvas.height - 6);
  }}

  function drawContext(canvas, rows, absMax, highlightRow) {{
    if (rows.length === 0) {{
      drawHeatmap(canvas, 1, 1, () => 0.0, () => [0, 0, 0, 255]);
      return;
    }}
    const rowCount = rows.length;
    const colCount = rows[0].length;
    drawHeatmap(
      canvas,
      rowCount,
      colCount,
      (r, c) => rows[r][c],
      (v) => colorSymmetric(v, absMax),
      highlightRow
    );
  }}

  function getContextRows(sourceRows) {{
    const start = Math.max(0, state.frame - contextRadius);
    const end = Math.min(frameCount - 1, state.frame + contextRadius);
    return {{
      rows: sourceRows.slice(start, end + 1),
      highlight: state.frame - start,
    }};
  }}

  function updateSummary(currentMark) {{
    const idx = state.frame;
    const renderFrame = payload.renderFrames ? payload.renderFrames[idx] : null;
    const lines = [
      `record_dir: ${{payload.recordDir}}`,
      `deploy_json: ${{payload.deployJson}}`,
      `focus_mark: ${{payload.focusMark || "(none)"}}`,
      `frame: ${{idx}} / ${{frameCount - 1}}`,
      `inference_index: ${{payload.steps.inferenceIndex[idx]}}`,
      `sim_time: ${{payload.steps.simTime[idx].toFixed(4)}}`,
      `cmd: [${{payload.steps.cmdX[idx].toFixed(3)}}, ${{payload.steps.cmdY[idx].toFixed(3)}}, ${{payload.steps.cmdYaw[idx].toFixed(3)}}]`,
      `obs_dim / encoded_dim / latent_dim / action_dim: ` +
        `${{payload.meta.totalObsDim}} / ${{payload.meta.encodedObsDim}} / ` +
        `${{payload.meta.latentDim}} / ${{payload.meta.actionDim}}`,
      `render_view: ${{renderFrame && renderFrame.uri ? `frame_id=${{renderFrame.renderFrameId}} size=${{renderFrame.width}}x${{renderFrame.height}}` : "missing"}}`,
      `visible_marks: ${{marks.length}}`,
    ];
    if (currentMark) {{
      lines.push(`current_mark: ${{formatMark(currentMark)}}`);
    }}
    elements.summary.textContent = lines.join("\\n");
  }}

  function updateMarkSelect() {{
    elements.markSelect.innerHTML = "";
    const base = document.createElement("option");
    base.value = "";
    base.textContent = "Select mark";
    elements.markSelect.appendChild(base);
    for (const mark of marks) {{
      const option = document.createElement("option");
      option.value = String(mark.frame);
      option.textContent = formatMark(mark);
      elements.markSelect.appendChild(option);
    }}
  }}

  function updateTrajectorySelect() {{
    elements.trajectorySelect.innerHTML = "";
    if (trajectories.length === 0) {{
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "Current only";
      elements.trajectorySelect.appendChild(option);
      elements.trajectorySelect.disabled = true;
      return;
    }}

    for (const trajectory of trajectories) {{
      const option = document.createElement("option");
      option.value = trajectory.viewerUri || "";
      const marksValue =
        trajectory.marks === null || trajectory.marks === undefined ? "?" : trajectory.marks;
      const stepsValue =
        trajectory.steps === null || trajectory.steps === undefined ? "?" : trajectory.steps;
      option.textContent =
        `${{trajectory.label}} | backend=${{trajectory.backend}} | steps=${{stepsValue}} | marks=${{marksValue}}`;
      if (trajectory.isCurrent) {{
        option.selected = true;
      }}
      elements.trajectorySelect.appendChild(option);
    }}
  }}

  function updateMarkList() {{
    elements.markList.innerHTML = "";
    elements.markMeta.textContent =
      `focus=${{payload.focusMark || "(none)"}} | visible=${{marks.length}}`;
    if (marks.length === 0) {{
      elements.markList.textContent = "No marks in selected window.";
      return;
    }}
    for (const mark of marks) {{
      const button = document.createElement("button");
      button.className = "mark-item";
      button.textContent = formatMark(mark);
      button.addEventListener("click", () => setFrame(mark.frame));
      elements.markList.appendChild(button);
    }}
  }}

  function render() {{
    const idx = state.frame;
    const obsRow = payload.obs[idx];
    const encodedRow = payload.encodedObs[idx];
    const latentRow = payload.latent[idx];
    const actionRow = payload.actions[idx];
    const renderFrame = payload.renderFrames ? payload.renderFrames[idx] : null;

    const obs1d = buildObs1d(obsRow);
    const image = buildImageMatrix(obsRow);
    const encodedImage = getEncodedImageSlice(encodedRow);

    if (renderFrame && renderFrame.uri) {{
      elements.renderViewImage.src = renderFrame.uri;
      elements.renderViewImage.style.visibility = "visible";
      elements.renderViewStatus.textContent =
        `render_frame_id=${{renderFrame.renderFrameId}} | size=${{renderFrame.width}}x${{renderFrame.height}}`;
    }} else {{
      elements.renderViewImage.removeAttribute("src");
      elements.renderViewImage.style.visibility = "hidden";
      elements.renderViewStatus.textContent = "No render frame recorded for this step.";
    }}

    drawImageMatrix(
      elements.obsImageCanvas,
      image.matrix,
      image.rows,
      image.cols,
      scales.obsImage.min,
      scales.obsImage.max
    );
    drawStrip(elements.obs1dCanvas, obs1d, Math.max(Math.abs(scales.obs.min), Math.abs(scales.obs.max)));
    drawStrip(elements.encodedCanvas, encodedRow, scales.encodedObs.absMax);
    drawStrip(elements.encodedImageCanvas, encodedImage, scales.encodedImage.absMax);
    drawStrip(elements.latentCanvas, latentRow, scales.latent.absMax);
    drawStrip(elements.actionsCanvas, actionRow, scales.actions.absMax);

    const encodedContext = getContextRows(payload.encodedObs);
    const latentContext = getContextRows(payload.latent);
    const actionsContext = getContextRows(payload.actions);
    drawContext(
      elements.encodedContextCanvas,
      encodedContext.rows,
      scales.encodedObs.absMax,
      encodedContext.highlight
    );
    drawContext(
      elements.latentContextCanvas,
      latentContext.rows,
      scales.latent.absMax,
      latentContext.highlight
    );
    drawContext(
      elements.actionsContextCanvas,
      actionsContext.rows,
      scales.actions.absMax,
      actionsContext.highlight
    );

    elements.frameSlider.value = String(idx);
    drawTimeline();
    updateSummary(findCurrentMark());
    for (const child of elements.markList.children) {{
      child.classList.remove("current");
    }}
    const currentMark = findCurrentMark();
    if (currentMark) {{
      for (const child of elements.markList.children) {{
        if (child.textContent === formatMark(currentMark)) {{
          child.classList.add("current");
        }}
      }}
    }}
  }}

  function setFrame(frame) {{
    state.frame = clamp(frame, 0, frameCount - 1);
    render();
  }}

  function stopPlayback() {{
    if (state.timerId !== null) {{
      clearInterval(state.timerId);
      state.timerId = null;
    }}
    state.playing = false;
    elements.playPause.textContent = "Play";
  }}

  function startPlayback() {{
    stopPlayback();
    state.playing = true;
    elements.playPause.textContent = "Pause";
    state.timerId = setInterval(() => {{
      if (state.frame >= frameCount - 1) {{
        stopPlayback();
        return;
      }}
      setFrame(state.frame + 1);
    }}, 1000.0 / Math.max(1, state.fps));
  }}

  function togglePlayback() {{
    if (state.playing) {{
      stopPlayback();
    }} else {{
      startPlayback();
    }}
  }}

  function jumpToNearestMark(direction) {{
    if (marks.length === 0) {{
      return;
    }}
    const sorted = marks.slice().sort((a, b) => a.frame - b.frame);
    if (direction > 0) {{
      const next = sorted.find((mark) => mark.frame > state.frame) || sorted[0];
      setFrame(next.frame);
      return;
    }}
    const reversed = sorted.slice().reverse();
    const prev = reversed.find((mark) => mark.frame < state.frame) || reversed[0];
    setFrame(prev.frame);
  }}

  elements.playPause.addEventListener("click", togglePlayback);
  elements.prevFrame.addEventListener("click", () => setFrame(state.frame - 1));
  elements.nextFrame.addEventListener("click", () => setFrame(state.frame + 1));
  elements.prevMark.addEventListener("click", () => jumpToNearestMark(-1));
  elements.nextMark.addEventListener("click", () => jumpToNearestMark(1));
  elements.frameSlider.addEventListener("input", (event) => {{
    setFrame(Number(event.target.value));
  }});
  elements.fpsInput.addEventListener("change", (event) => {{
    state.fps = clamp(Number(event.target.value) || payload.initialFps, 1, 60);
    if (state.playing) {{
      startPlayback();
    }}
  }});
  elements.scaleSlider.addEventListener("input", (event) => {{
    applyScale(Number(event.target.value));
  }});
  elements.markSelect.addEventListener("change", (event) => {{
    const value = event.target.value;
    if (value === "") {{
      return;
    }}
    setFrame(Number(value));
  }});
  elements.trajectorySelect.addEventListener("change", (event) => {{
    const value = event.target.value;
    if (!value) {{
      return;
    }}
    window.location.href = value;
  }});

  window.addEventListener("keydown", (event) => {{
    if (event.target.tagName === "INPUT" || event.target.tagName === "SELECT") {{
      return;
    }}
    if (event.code === "Space") {{
      event.preventDefault();
      togglePlayback();
    }} else if (event.code === "ArrowRight") {{
      event.preventDefault();
      setFrame(state.frame + (event.shiftKey ? 10 : 1));
    }} else if (event.code === "ArrowLeft") {{
      event.preventDefault();
      setFrame(state.frame - (event.shiftKey ? 10 : 1));
    }} else if (event.key === "]") {{
      jumpToNearestMark(1);
    }} else if (event.key === "[") {{
      jumpToNearestMark(-1);
    }}
  }});

  applyScale(86);
  updateTrajectorySelect();
  updateMarkSelect();
  updateMarkList();
  if (marks.length > 0 && payload.focusMark) {{
    state.frame = marks[0].frame;
  }}
  render();
  </script>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    if args.stride <= 0:
        print("--stride must be >= 1", file=sys.stderr)
        return 2

    record_dir = args.record_dir.resolve()
    if not record_dir.is_dir():
        print(f"record_dir does not exist: {record_dir}", file=sys.stderr)
        return 2

    required_files = [
        record_dir / "steps.csv",
        record_dir / "obs.csv",
        record_dir / "encoded_obs.csv",
        record_dir / "latent.csv",
        record_dir / "actions.csv",
        record_dir / "events.csv",
    ]
    missing = [path for path in required_files if not path.exists()]
    if missing:
        print("Missing recording files:", file=sys.stderr)
        for path in missing:
            print(f"  - {path}", file=sys.stderr)
        return 2

    deploy_json = args.deploy_json.resolve()
    if not deploy_json.exists():
        print(f"deploy_json does not exist: {deploy_json}", file=sys.stderr)
        return 2

    payload = build_payload(
        record_dir=record_dir,
        deploy_json=deploy_json,
        focus_mark=args.focus_mark,
        window=args.window,
        stride=args.stride,
        context_radius=args.context_radius,
        initial_fps=args.fps,
    )

    output_path = make_output_path(record_dir, args.output, args.focus_mark).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_html(payload), encoding="utf-8")

    print(f"Wrote heatmap viewer: {output_path}")
    print(f"Frames exported: {payload['selectedFrameCount']}")
    if args.focus_mark:
        print(f"Focus mark: {args.focus_mark}")

    if args.open:
        webbrowser.open(output_path.as_uri())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
