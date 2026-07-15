#!/usr/bin/env python3
"""Tk control panel for one MuJoCo collection job and key replay.

The window is intentionally a thin process controller.  Terrain discovery,
trajectory naming, status parsing, and resume semantics come from
``run_data_collection.py``; simulation and rendering remain in the C++
executables.  Importing this module, including the ``--check`` path, never
creates a Tk root window.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import run_data_collection as runner  # noqa: E402


REPLAY_RATES: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)


def repo_root() -> Path:
    return TOOLS_DIR.parent


def _first_existing(candidates: Sequence[Path]) -> Path:
    return next((path for path in candidates if path.is_file()), candidates[0])


def default_binary(root: Path, name: str) -> Path:
    """Return the conventional build output, preferring one that exists."""

    candidates = [
        root / "mujoco/C++/build_onnx" / name,
        root / "mujoco/C++/build" / name,
        root / "mujoco/C++" / name,
    ]
    candidates.extend(sorted((root / "mujoco/C++").glob(f"build*/{name}")))
    return _first_existing(candidates)


@dataclass(frozen=True)
class AppConfig:
    root: Path
    data_root: Path
    runner_script: Path
    collector_binary: Path
    replay_binary: Path
    policy: Path
    python_executable: Path
    max_attempts: int = 5


@dataclass(frozen=True)
class TrajectoryStatus:
    code: str
    label: str
    detail: str = ""
    replayable: bool = False


class TrajectoryCatalog:
    """Read-only task/terrain/job view over the generated data directory."""

    def __init__(self, data_root: Path):
        self.data_root = data_root.expanduser().resolve()
        self._terrains: list[runner.Terrain] = []
        self._by_task: dict[str, list[runner.Terrain]] = {}
        self.refresh()

    def refresh(self) -> None:
        terrains = runner.discover_terrains(self.data_root)
        by_task: dict[str, list[runner.Terrain]] = {}
        for terrain in terrains:
            by_task.setdefault(terrain.task_name, []).append(terrain)
        for values in by_task.values():
            values.sort(key=lambda item: item.terrain_id)
        self._terrains = terrains
        self._by_task = dict(sorted(by_task.items()))

    @property
    def terrain_count(self) -> int:
        return len(self._terrains)

    def task_names(self) -> tuple[str, ...]:
        return tuple(self._by_task)

    def terrains_for_task(self, task_name: str) -> tuple[runner.Terrain, ...]:
        return tuple(self._by_task.get(task_name, ()))

    def terrain_ids(self, task_name: str) -> tuple[str, ...]:
        return tuple(item.terrain_id for item in self.terrains_for_task(task_name))

    def terrain(self, task_name: str, terrain_id: str) -> runner.Terrain:
        for terrain in self._by_task.get(task_name, ()):
            if terrain.terrain_id == terrain_id:
                return terrain
        raise KeyError(f"unknown terrain: {task_name}/{terrain_id}")

    def job(self, task_name: str, terrain_id: str, speed: str | Decimal) -> runner.CollectionJob:
        return runner.CollectionJob(
            terrain=self.terrain(task_name, terrain_id),
            speed=runner.parse_speed(speed),
        )

    @staticmethod
    def terrain_details(terrain: runner.Terrain) -> str:
        description = str(terrain.metadata.get("description") or "(no description)")
        params = terrain.metadata.get("params")
        lines = [description, "", f"terrain_id: {terrain.terrain_id}"]
        if isinstance(params, Mapping) and params:
            lines.append("parameters:")
            lines.extend(f"  {name}: {value}" for name, value in sorted(params.items()))
        pid_config, pid_path = runner.load_pid_config(terrain.data_root)
        source = str(pid_path) if pid_path is not None else "built-in defaults"
        lines.extend(["", f"heading PID: {source}"])
        lines.extend(
            f"  {name}: {value:g}" for name, value in pid_config.as_dict().items()
        )
        return "\n".join(lines)

    @staticmethod
    def status(job: runner.CollectionJob) -> TrajectoryStatus:
        status_value = runner.load_status(job.status_path)
        status_current = runner.status_matches_job(job, status_value)
        status_name = runner.normalized_status(status_value)
        key_valid, key_error = runner.validate_key_xml(job.key_path)

        if key_valid:
            if status_value is not None and not status_current:
                return TrajectoryStatus(
                    "stale_success",
                    "已有 Key，但输入定义已变化",
                    "Key 只引用当前 terrain.xml，无法保证仍是采集时的地形；请重新采集。",
                    False,
                )
            return TrajectoryStatus("success", "采集成功，可回放", replayable=True)

        if job.key_path.exists():
            return TrajectoryStatus("invalid_key", "Key XML 无效", key_error, False)

        if status_value is not None and not status_current:
            return TrajectoryStatus(
                "stale",
                "状态已过期",
                "地形、元数据或采集协议已变化，需要重新采集。",
                False,
            )

        reason = ""
        if status_value:
            reason = str(status_value.get("reason") or status_value.get("failure_reason") or "")
        labels = {
            "running": "正在采集",
            "failed": "采集失败",
            "infrastructure_error": "采集程序错误",
            "success": "状态成功但 Key 缺失",
        }
        if status_name in labels:
            return TrajectoryStatus(status_name, labels[status_name], reason, False)
        return TrajectoryStatus("pending", "尚未采集", reason, False)


def build_collection_command(config: AppConfig, job: runner.CollectionJob) -> list[str]:
    """Build the exact one-job visual collection command used by the UI."""

    return [
        str(config.python_executable),
        str(config.runner_script),
        "--data-root",
        str(config.data_root),
        "--binary",
        str(config.collector_binary),
        "--policy",
        str(config.policy),
        "--task",
        job.terrain.task_name,
        "--terrain-id",
        job.terrain.terrain_id,
        "--speed",
        runner.speed_text(job.speed),
        "--workers",
        "1",
        "--max-attempts",
        str(config.max_attempts),
        "--force",
        "--collector-arg=--visualize",
    ]


def build_replay_command(
    config: AppConfig,
    job: runner.CollectionJob,
    *,
    rate: float = 1.0,
    paused: bool = False,
    loop: bool = False,
) -> list[str]:
    command = [
        str(config.replay_binary),
        "--trajectory",
        str(job.key_path),
        "--metadata",
        str(job.terrain.metadata_path),
        "--rate",
        f"{rate:g}",
    ]
    if paused:
        command.append("--paused")
    if loop:
        command.append("--loop")
    return command


def replay_message(command: str, **values: Any) -> str:
    """Encode one stdin control message for ``mujoco_key_replayer``."""

    payload = {"command": command}
    payload.update(values)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"


@dataclass(frozen=True)
class ProcessEvent:
    kind: str
    process_kind: str
    text: str = ""
    payload: Mapping[str, Any] | None = None
    returncode: int | None = None


class BackgroundProcess:
    """Line-buffered subprocess bridge that never waits on the Tk thread."""

    def __init__(self, *, cwd: Path):
        self.cwd = cwd
        self.events: queue.Queue[ProcessEvent] = queue.Queue()
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._kind = ""

    def running(self) -> bool:
        with self._lock:
            return self._process is not None and self._process.poll() is None

    def start(self, command: Sequence[str], process_kind: str) -> None:
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError("another collection or replay process is already running")
            process = subprocess.Popen(
                list(command),
                cwd=self.cwd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=environment,
                start_new_session=True,
            )
            self._process = process
            self._kind = process_kind
        self.events.put(ProcessEvent("started", process_kind, text=" ".join(command)))
        threading.Thread(
            target=self._read_process,
            args=(process, process_kind),
            name=f"data-{process_kind}-output",
            daemon=True,
        ).start()

    def _read_process(self, process: subprocess.Popen[str], process_kind: str) -> None:
        assert process.stdout is not None
        for raw_line in process.stdout:
            text = raw_line.rstrip("\r\n")
            payload: Mapping[str, Any] | None = None
            try:
                decoded = json.loads(text)
                if isinstance(decoded, dict):
                    payload = decoded
            except json.JSONDecodeError:
                pass
            self.events.put(ProcessEvent("output", process_kind, text, payload))
        returncode = process.wait()
        with self._lock:
            if self._process is process:
                self._process = None
                self._kind = ""
        self.events.put(ProcessEvent("exit", process_kind, returncode=returncode))

    def send_replay(self, command: str, **values: Any) -> bool:
        message = replay_message(command, **values)
        with self._lock:
            process = self._process
            if (
                self._kind != "replay"
                or process is None
                or process.poll() is not None
                or process.stdin is None
            ):
                return False
            try:
                process.stdin.write(message)
                process.stdin.flush()
            except (BrokenPipeError, OSError):
                return False
        return True

    def stop(self) -> None:
        """Stop the process group without blocking Tk or stranding runner state."""

        with self._lock:
            process = self._process
            process_kind = self._kind
        if process is None or process.poll() is not None:
            return
        if process_kind == "replay":
            self.send_replay("quit")
            graceful_timeout = 1.0
        else:
            # The runner handles KeyboardInterrupt and lets its worker merge
            # the collector result, replacing the temporary "running" status.
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                return
            graceful_timeout = 3.0

        def kill_later() -> None:
            try:
                process.wait(timeout=graceful_timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    return
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

        # Keep this short-lived watchdog non-daemon so closing Tk cannot leave
        # a TERM-resistant runner/collector process group behind.
        threading.Thread(target=kill_later, name="data-process-stop", daemon=False).start()


def check_configuration(config: AppConfig) -> tuple[list[str], list[str]]:
    """Validate paths and discovery without importing or creating Tk."""

    messages: list[str] = []
    errors: list[str] = []

    def directory(path: Path, label: str) -> None:
        if path.is_dir():
            messages.append(f"OK {label}: {path}")
        else:
            errors.append(f"{label} directory does not exist: {path}")

    def executable(path: Path, label: str) -> None:
        if not path.is_file():
            errors.append(f"{label} does not exist: {path}")
        elif not os.access(path, os.X_OK):
            errors.append(f"{label} is not executable: {path}")
        else:
            messages.append(f"OK {label}: {path}")

    directory(config.data_root, "data root")
    directory(config.policy, "policy")
    if config.runner_script.is_file():
        messages.append(f"OK runner: {config.runner_script}")
    else:
        errors.append(f"runner script does not exist: {config.runner_script}")
    executable(config.collector_binary, "collector binary")
    executable(config.replay_binary, "replay binary")
    executable(config.python_executable, "Python executable")

    if config.max_attempts < 1 or config.max_attempts > 5:
        errors.append("max attempts must be between 1 and 5")
    if config.data_root.is_dir():
        try:
            catalog = TrajectoryCatalog(config.data_root)
        except (OSError, ValueError) as exc:
            errors.append(f"cannot discover terrains: {exc}")
        else:
            if catalog.terrain_count:
                messages.append(
                    f"OK terrains: {catalog.terrain_count} collectable across "
                    f"{len(catalog.task_names())} tasks"
                )
            else:
                errors.append(f"no collectable terrains found under: {config.data_root}")
    return messages, errors


class DataCollectionUI:
    """Tk widgets and callbacks; constructed only by :func:`launch_ui`."""

    def __init__(self, root: Any, config: AppConfig):
        import tkinter as tk
        from tkinter import filedialog, messagebox, scrolledtext, ttk

        self.tk = tk
        self.ttk = ttk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.scrolledtext = scrolledtext
        self.root = root
        self.config = config
        self.catalog = TrajectoryCatalog(config.data_root)
        self.process = BackgroundProcess(cwd=config.root)
        self._playing = True
        self._seeking = False
        self._replay_frame_count = 0

        root.title("MuJoCo 数据采集与轨迹回放")
        root.minsize(980, 720)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.task_var = tk.StringVar()
        self.terrain_var = tk.StringVar()
        self.speed_var = tk.StringVar(value=runner.speed_text(runner.SPEEDS[0]))
        self.collector_var = tk.StringVar(value=str(config.collector_binary))
        self.replay_binary_var = tk.StringVar(value=str(config.replay_binary))
        self.status_var = tk.StringVar(value="尚未选择轨迹")
        self.rate_var = tk.StringVar(value="1")
        self.loop_var = tk.BooleanVar(value=False)
        self.progress_var = tk.DoubleVar(value=0.0)
        self.frame_var = tk.StringVar(value="frame - / -")

        self._build_widgets()
        self._populate_tasks()
        self.root.after(50, self._poll_process_events)

    def _build_widgets(self) -> None:
        ttk = self.ttk
        main = ttk.Frame(self.root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(6, weight=1)
        main.rowconfigure(8, weight=2)

        ttk.Label(main, text="采集器").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(main, textvariable=self.collector_var).grid(row=0, column=1, sticky="ew")
        ttk.Button(
            main,
            text="浏览…",
            command=lambda: self._browse_binary(self.collector_var),
        ).grid(row=0, column=2, padx=(6, 0))
        ttk.Label(main, text="回放器").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(main, textvariable=self.replay_binary_var).grid(row=1, column=1, sticky="ew")
        ttk.Button(
            main, text="浏览…", command=lambda: self._browse_binary(self.replay_binary_var)
        ).grid(row=1, column=2, padx=(6, 0))

        selector = ttk.LabelFrame(main, text="轨迹选择", padding=8)
        selector.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(8, 4))
        selector.columnconfigure(1, weight=1)
        selector.columnconfigure(3, weight=3)
        ttk.Label(selector, text="任务").grid(row=0, column=0, sticky="w")
        self.task_combo = ttk.Combobox(selector, textvariable=self.task_var, state="readonly")
        self.task_combo.grid(row=0, column=1, sticky="ew", padx=(4, 12))
        self.task_combo.bind("<<ComboboxSelected>>", self._on_task_changed)
        ttk.Label(selector, text="地形").grid(row=0, column=2, sticky="w")
        self.terrain_combo = ttk.Combobox(
            selector, textvariable=self.terrain_var, state="readonly"
        )
        self.terrain_combo.grid(row=0, column=3, sticky="ew", padx=(4, 12))
        self.terrain_combo.bind("<<ComboboxSelected>>", self._on_selection_changed)
        ttk.Label(selector, text="linv_x (m/s)").grid(row=0, column=4, sticky="w")
        self.speed_combo = ttk.Combobox(
            selector,
            textvariable=self.speed_var,
            values=[runner.speed_text(speed) for speed in runner.SPEEDS],
            state="readonly",
            width=7,
        )
        self.speed_combo.grid(row=0, column=5, sticky="w", padx=(4, 0))
        self.speed_combo.bind("<<ComboboxSelected>>", self._on_selection_changed)

        actions = ttk.Frame(main)
        actions.grid(row=3, column=0, columnspan=3, sticky="ew", pady=5)
        self.collect_button = ttk.Button(
            actions, text="可视采集", command=self._start_collection
        )
        self.collect_button.pack(side="left")
        self.replay_button = ttk.Button(actions, text="回放", command=self._start_replay)
        self.replay_button.pack(side="left", padx=6)
        ttk.Button(actions, text="停止", command=self._stop_process).pack(side="left")
        ttk.Button(actions, text="刷新", command=self._refresh).pack(side="left", padx=6)
        ttk.Label(actions, textvariable=self.status_var).pack(side="left", padx=12)

        replay = ttk.LabelFrame(main, text="回放控制（发送 NDJSON 到 C++）", padding=8)
        replay.grid(row=4, column=0, columnspan=3, sticky="ew", pady=4)
        ttk.Button(replay, text="|◀", width=4, command=lambda: self._send_replay("first")).pack(
            side="left"
        )
        ttk.Button(replay, text="◀", width=4, command=lambda: self._send_replay("prev")).pack(
            side="left", padx=2
        )
        self.play_button = ttk.Button(replay, text="暂停", width=6, command=self._toggle_play)
        self.play_button.pack(side="left", padx=2)
        ttk.Button(replay, text="▶", width=4, command=lambda: self._send_replay("next")).pack(
            side="left", padx=2
        )
        ttk.Button(replay, text="▶|", width=4, command=lambda: self._send_replay("last")).pack(
            side="left"
        )
        ttk.Checkbutton(replay, text="循环", variable=self.loop_var, command=self._set_loop).pack(
            side="left", padx=(12, 5)
        )
        ttk.Label(replay, text="倍率").pack(side="left")
        rate_combo = ttk.Combobox(
            replay,
            textvariable=self.rate_var,
            values=[f"{value:g}" for value in REPLAY_RATES],
            state="readonly",
            width=5,
        )
        rate_combo.pack(side="left", padx=4)
        rate_combo.bind("<<ComboboxSelected>>", self._set_rate)
        ttk.Label(replay, textvariable=self.frame_var).pack(side="right")

        self.progress = ttk.Scale(
            main, from_=0.0, to=1.0, variable=self.progress_var, orient="horizontal"
        )
        self.progress.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(2, 5))
        self.progress.bind("<ButtonPress-1>", lambda _event: self._begin_seek())
        self.progress.bind("<ButtonRelease-1>", lambda _event: self._finish_seek())

        details_frame = ttk.LabelFrame(main, text="地形说明与参数", padding=5)
        details_frame.grid(row=6, column=0, columnspan=3, sticky="nsew", pady=4)
        details_frame.rowconfigure(0, weight=1)
        details_frame.columnconfigure(0, weight=1)
        self.details = self.scrolledtext.ScrolledText(details_frame, height=8, wrap="word")
        self.details.grid(row=0, column=0, sticky="nsew")
        self.details.configure(state="disabled")

        ttk.Label(main, text="实时日志").grid(row=7, column=0, columnspan=3, sticky="w")
        self.log = self.scrolledtext.ScrolledText(main, height=13, wrap="word")
        self.log.grid(row=8, column=0, columnspan=3, sticky="nsew")

    def _browse_binary(self, variable: Any) -> None:
        selected = self.filedialog.askopenfilename(initialdir=str(self.config.root))
        if selected:
            variable.set(selected)

    def _populate_tasks(self, preferred: str = "") -> None:
        tasks = self.catalog.task_names()
        self.task_combo.configure(values=tasks)
        selected = preferred if preferred in tasks else (tasks[0] if tasks else "")
        self.task_var.set(selected)
        self._populate_terrains()

    def _populate_terrains(self, preferred: str = "") -> None:
        terrain_ids = self.catalog.terrain_ids(self.task_var.get())
        self.terrain_combo.configure(values=terrain_ids)
        selected = (
            preferred
            if preferred in terrain_ids
            else (terrain_ids[0] if terrain_ids else "")
        )
        self.terrain_var.set(selected)
        self._update_selection()

    def _on_task_changed(self, _event: Any = None) -> None:
        self._populate_terrains()

    def _on_selection_changed(self, _event: Any = None) -> None:
        self._update_selection()

    def _selected_job(self) -> runner.CollectionJob:
        return self.catalog.job(self.task_var.get(), self.terrain_var.get(), self.speed_var.get())

    def _update_selection(self) -> None:
        try:
            job = self._selected_job()
            detail_text = self.catalog.terrain_details(job.terrain)
            status = self.catalog.status(job)
        except (KeyError, OSError, ValueError) as exc:
            self.status_var.set(f"轨迹配置无效：{exc}" if str(exc) else "没有可选轨迹")
            self.replay_button.configure(state="disabled")
            return
        if status.detail:
            detail_text += f"\n\nstatus: {status.label}\n{status.detail}"
        else:
            detail_text += f"\n\nstatus: {status.label}"
        self.details.configure(state="normal")
        self.details.delete("1.0", "end")
        self.details.insert("1.0", detail_text)
        self.details.configure(state="disabled")
        self.status_var.set(status.label)
        self.replay_button.configure(state="normal" if status.replayable else "disabled")

    def _runtime_config(self) -> AppConfig:
        return replace(
            self.config,
            collector_binary=Path(self.collector_var.get()).expanduser().resolve(),
            replay_binary=Path(self.replay_binary_var.get()).expanduser().resolve(),
        )

    def _start_collection(self) -> None:
        try:
            job = self._selected_job()
            status = self.catalog.status(job)
            if status.code != "pending" and not self.messagebox.askyesno(
                "确认重新采集",
                f"当前轨迹状态为“{status.label}”。\n"
                "强制重新采集会删除或覆盖已有 Key 和状态，是否继续？",
            ):
                return
            config = self._runtime_config()
            if not config.collector_binary.is_file():
                raise ValueError(f"采集器不存在：{config.collector_binary}")
            if not config.policy.is_dir():
                raise ValueError(f"策略目录不存在：{config.policy}")
            self.process.start(build_collection_command(config, job), "collection")
        except (KeyError, RuntimeError, ValueError, OSError) as exc:
            self.messagebox.showerror("无法启动采集", str(exc))

    def _start_replay(self) -> None:
        try:
            job = self._selected_job()
            status = self.catalog.status(job)
            if not status.replayable:
                raise ValueError("当前轨迹没有可回放的有效 Key XML")
            config = self._runtime_config()
            if not config.replay_binary.is_file():
                raise ValueError(f"回放器不存在：{config.replay_binary}")
            rate = float(self.rate_var.get())
            command = build_replay_command(
                config, job, rate=rate, loop=self.loop_var.get(), paused=False
            )
            self.process.start(command, "replay")
            self._playing = True
            self.play_button.configure(text="暂停")
        except (KeyError, RuntimeError, ValueError, OSError) as exc:
            self.messagebox.showerror("无法启动回放", str(exc))

    def _stop_process(self) -> None:
        self.process.stop()

    def _send_replay(self, command: str, **values: Any) -> None:
        if not self.process.send_replay(command, **values):
            self.status_var.set("回放进程未运行")

    def _toggle_play(self) -> None:
        self._playing = not self._playing
        command = "play" if self._playing else "pause"
        self._send_replay(command)
        self.play_button.configure(text="暂停" if self._playing else "播放")

    def _set_loop(self) -> None:
        self._send_replay("loop", enabled=bool(self.loop_var.get()))

    def _set_rate(self, _event: Any = None) -> None:
        self._send_replay("rate", rate=float(self.rate_var.get()))

    def _begin_seek(self) -> None:
        self._seeking = True

    def _finish_seek(self) -> None:
        self._seeking = False
        progress = min(1.0, max(0.0, float(self.progress_var.get())))
        self._send_replay("seek", progress=progress)

    def _refresh(self) -> None:
        task = self.task_var.get()
        terrain = self.terrain_var.get()
        try:
            self.catalog.refresh()
        except (OSError, ValueError) as exc:
            self.messagebox.showerror("刷新失败", str(exc))
            return
        self._populate_tasks(task)
        if self.task_var.get() == task:
            self._populate_terrains(terrain)

    def _append_log(self, text: str) -> None:
        self.log.insert("end", text + "\n")
        self.log.see("end")

    def _update_replay_event(self, payload: Mapping[str, Any]) -> None:
        event_name = str(payload.get("event") or payload.get("type") or "")
        if event_name not in {"loaded", "frame", "ended", "error"}:
            return
        frame_value = payload.get("frame", payload.get("frame_index"))
        total_value = payload.get("frames", payload.get("total_frames", payload.get("nkey")))
        if total_value is None:
            total_value = payload.get("frame_count")
        try:
            if total_value is not None:
                self._replay_frame_count = int(total_value)
        except (TypeError, ValueError):
            pass
        if total_value is None and self._replay_frame_count:
            total_value = self._replay_frame_count
        if frame_value is not None:
            try:
                frame_text: Any = int(frame_value) + 1
            except (TypeError, ValueError):
                frame_text = frame_value
            total_text = total_value if total_value is not None else "-"
            self.frame_var.set(f"frame {frame_text} / {total_text}")
        progress = payload.get("progress")
        if progress is None and frame_value is not None and total_value is not None:
            try:
                denominator = max(1, int(total_value) - 1)
                progress = int(frame_value) / denominator
            except (TypeError, ValueError):
                progress = None
        if progress is not None and not self._seeking:
            try:
                self.progress_var.set(min(1.0, max(0.0, float(progress))))
            except (TypeError, ValueError):
                pass
        if isinstance(payload.get("playing"), bool):
            self._playing = bool(payload["playing"])
            self.play_button.configure(text="暂停" if self._playing else "播放")
        if isinstance(payload.get("loop"), bool):
            self.loop_var.set(bool(payload["loop"]))
        if isinstance(payload.get("rate"), (int, float)):
            self.rate_var.set(f"{float(payload['rate']):g}")
        if event_name == "ended":
            self._playing = False
            self.play_button.configure(text="播放")

    def _poll_process_events(self) -> None:
        while True:
            try:
                event = self.process.events.get_nowait()
            except queue.Empty:
                break
            if event.kind == "started":
                self._append_log(f"[{event.process_kind}] START {event.text}")
                status = "正在采集" if event.process_kind == "collection" else "正在回放"
                self.status_var.set(status)
            elif event.kind == "output":
                if event.payload:
                    self._update_replay_event(event.payload)
                event_name = str((event.payload or {}).get("event") or "")
                if event_name != "frame":
                    self._append_log(f"[{event.process_kind}] {event.text}")
            elif event.kind == "exit":
                self._append_log(f"[{event.process_kind}] EXIT {event.returncode}")
                try:
                    self.catalog.refresh()
                except (OSError, ValueError) as exc:
                    self._append_log(f"[refresh] {exc}")
                self._update_selection()
        self.root.after(50, self._poll_process_events)

    def _on_close(self) -> None:
        self.process.stop()
        self.root.destroy()


def launch_ui(config: AppConfig) -> None:
    import tkinter as tk

    try:
        root = tk.Tk()
        DataCollectionUI(root, config)
        root.mainloop()
    except tk.TclError as exc:
        raise RuntimeError(f"cannot initialize Tk: {exc}") from exc


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description="MuJoCo data collection and key replay UI")
    parser.add_argument("--data-root", type=Path, default=root / "data_collection")
    parser.add_argument("--runner-script", type=Path, default=root / "tools/run_data_collection.py")
    parser.add_argument(
        "--collector-binary",
        type=Path,
        default=default_binary(root, "mujoco_data_collector"),
    )
    parser.add_argument(
        "--replay-binary",
        type=Path,
        default=default_binary(root, "mujoco_key_replayer"),
    )
    parser.add_argument("--policy", type=Path, default=root / "policy/vtm_lstm_sru")
    parser.add_argument(
        "--python",
        dest="python_executable",
        type=Path,
        default=Path(sys.executable),
    )
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate paths and terrain discovery without creating a Tk window",
    )
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> AppConfig:
    root = repo_root()
    return AppConfig(
        root=root,
        data_root=args.data_root.expanduser().resolve(),
        runner_script=args.runner_script.expanduser().resolve(),
        collector_binary=args.collector_binary.expanduser().resolve(),
        replay_binary=args.replay_binary.expanduser().resolve(),
        policy=args.policy.expanduser().resolve(),
        python_executable=args.python_executable.expanduser().resolve(),
        max_attempts=args.max_attempts,
    )


def run(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = config_from_args(args)
    if args.check:
        messages, errors = check_configuration(config)
        for message in messages:
            print(message)
        for error in errors:
            print(f"ERROR {error}", file=sys.stderr)
        return 0 if not errors else 2
    try:
        launch_ui(config)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
