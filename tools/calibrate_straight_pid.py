#!/usr/bin/env python3
"""Calibrate one path-heading PID configuration on the generated flat terrain."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Sequence


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import run_data_collection as runner  # noqa: E402


SPEEDS = (0.50, 0.75, 1.00)
CROSS_TRACK_GAINS = (0.75, 1.25, 1.75)
KP_VALUES = (0.8, 1.2, 1.6)
KI_VALUES = (0.0, 0.05)
KD_VALUES = (0.05, 0.10)


@dataclass(frozen=True)
class PidCandidate:
    cross_track_gain: float
    kp: float
    ki: float
    kd: float
    heading_limit_rad: float = runner.DEFAULT_PATH_HEADING_PID.heading_limit_rad
    yaw_cmd_limit_rad_s: float = runner.DEFAULT_PATH_HEADING_PID.yaw_cmd_limit_rad_s
    integral_limit_rad_s: float = runner.DEFAULT_PATH_HEADING_PID.integral_limit_rad_s
    derivative_alpha: float = runner.DEFAULT_PATH_HEADING_PID.derivative_alpha

    @property
    def identifier(self) -> str:
        values = (self.cross_track_gain, self.kp, self.ki, self.kd)
        return "g%s-kp%s-ki%s-kd%s" % tuple(
            f"{value:.2f}".replace(".", "p") for value in values
        )

    def collector_args(self) -> list[str]:
        return runner.PathHeadingPidConfig(**self.as_dict()).collector_args()

    def as_dict(self) -> dict[str, float]:
        return {
            "cross_track_gain": self.cross_track_gain,
            "kp": self.kp,
            "ki": self.ki,
            "kd": self.kd,
            "heading_limit_rad": self.heading_limit_rad,
            "yaw_cmd_limit_rad_s": self.yaw_cmd_limit_rad_s,
            "integral_limit_rad_s": self.integral_limit_rad_s,
            "derivative_alpha": self.derivative_alpha,
        }


@dataclass(frozen=True)
class TrialResult:
    candidate: PidCandidate
    speed: float
    success: bool
    max_abs_cross_track_m: float = math.inf
    final_heading_error_deg: float = math.inf
    reason: str = ""


def candidates() -> Iterable[PidCandidate]:
    for gain, kp, ki, kd in product(
        CROSS_TRACK_GAINS, KP_VALUES, KI_VALUES, KD_VALUES
    ):
        yield PidCandidate(gain, kp, ki, kd)


def wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def quaternion_yaw(values: Sequence[float]) -> float:
    w, x, y, z = values
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def trajectory_metrics(path: Path) -> tuple[float, float]:
    keys = list(ET.parse(path).getroot().iter("key"))
    qposes = [[float(value) for value in key.attrib["qpos"].split()] for key in keys]
    if not qposes or any(len(qpos) < 7 for qpos in qposes):
        raise ValueError("trajectory has no valid free-joint qpos frames")
    x0, y0 = qposes[0][:2]
    yaw0 = quaternion_yaw(qposes[0][3:7])
    cross_track = [
        -math.sin(yaw0) * (qpos[0] - x0) + math.cos(yaw0) * (qpos[1] - y0)
        for qpos in qposes
    ]
    final_yaw = quaternion_yaw(qposes[-1][3:7])
    return max(abs(value) for value in cross_track), math.degrees(
        abs(wrap_angle(final_yaw - yaw0))
    )


def calibration_metadata(
    flat_metadata: Path, destination: Path, distance: float, task_name: str
) -> None:
    value = json.loads(flat_metadata.read_text(encoding="utf-8"))
    value.update(
        {
            "task_name": task_name,
            "terrain_id": "plane-flat-pid-calibration",
            "collect": True,
            "description": "Temporary straight-line PID calibration metadata.",
            "terminal": {
                "target_x": distance,
                "x_tolerance": 0.10,
                "min_base_z": 0.15,
                "max_abs_y": 1.0,
                "stop_duration_s": 1.0,
            },
        }
    )
    destination.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def run_trial(
    candidate: PidCandidate,
    speed: float,
    *,
    binary: Path,
    policy: Path,
    terrain: Path,
    metadata: Path,
    work_root: Path,
    timeout_s: float,
) -> TrialResult:
    trial_dir = work_root / candidate.identifier / f"speed-{speed:.2f}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    token = f"{speed:.2f}".replace(".", "p")
    task_name = f"pidcal_{candidate.identifier}"
    # Generated terrain XML resolves meshes relative to the top-level XML.
    # Keep the transient key beside terrain.xml so MuJoCo resolves the nested
    # asset paths exactly as it does for production keys. Candidate-specific
    # task names avoid collisions between parallel calibration processes.
    output = terrain.parent / f"{task_name}-cmd_linv_x_{token}.xml"
    result = trial_dir / "result.json"
    command = [
        str(binary),
        "--terrain",
        str(terrain),
        "--metadata",
        str(metadata),
        "--output",
        str(output),
        "--result",
        str(result),
        "--policy",
        str(policy),
        "--speed",
        f"{speed:.2f}",
        "--max-attempts",
        "1",
        *candidate.collector_args(),
    ]
    try:
        try:
            completed = subprocess.run(
                command, check=False, capture_output=True, text=True, timeout=timeout_s
            )
        except subprocess.TimeoutExpired:
            return TrialResult(candidate, speed, False, reason="collector_timeout")
        if completed.returncode != 0 or not output.is_file():
            reason = f"collector_exit_{completed.returncode}"
            try:
                reason = str(
                    json.loads(result.read_text(encoding="utf-8")).get("reason") or reason
                )
            except (OSError, json.JSONDecodeError):
                pass
            return TrialResult(candidate, speed, False, reason=reason)
        try:
            lateral, heading = trajectory_metrics(output)
        except (OSError, ValueError, ET.ParseError) as error:
            return TrialResult(candidate, speed, False, reason=f"invalid_key: {error}")
        return TrialResult(candidate, speed, True, lateral, heading)
    finally:
        output.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = runner.repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=root / "data_collection")
    parser.add_argument(
        "--binary",
        type=Path,
        default=runner.resolve_binary(None, root),
    )
    parser.add_argument("--policy", type=Path, default=root / "policy/vtm_lstm_sru")
    parser.add_argument("--output", type=Path, default=root / "data_collection/pid_config.json")
    parser.add_argument("--distance", type=float, default=4.0)
    parser.add_argument("--max-lateral", type=float, default=0.20)
    parser.add_argument("--max-heading-deg", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args(argv)


def run(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.binary is None or not args.binary.is_file():
        raise SystemExit("mujoco_data_collector binary was not found")
    if not args.policy.is_dir():
        raise SystemExit(f"policy directory was not found: {args.policy}")
    if args.workers < 1 or args.distance <= 0 or args.timeout <= 0:
        raise SystemExit("workers, distance, and timeout must be positive")
    flats = [
        terrain
        for terrain in runner.discover_terrains(args.data_root, include_flat=True)
        if terrain.task_name == "flat"
    ]
    if len(flats) != 1:
        raise SystemExit(f"expected exactly one flat terrain, found {len(flats)}")

    all_candidates = tuple(candidates())
    with tempfile.TemporaryDirectory(prefix="go2w-pid-calibration-") as temporary:
        work_root = Path(temporary)
        metadata_by_candidate: dict[PidCandidate, Path] = {}
        for candidate in all_candidates:
            metadata = work_root / candidate.identifier / "terrain.json"
            metadata.parent.mkdir(parents=True, exist_ok=True)
            calibration_metadata(
                flats[0].metadata_path,
                metadata,
                args.distance,
                f"pidcal_{candidate.identifier}",
            )
            metadata_by_candidate[candidate] = metadata
        futures = {}
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for candidate in all_candidates:
                for speed in SPEEDS:
                    future = executor.submit(
                        run_trial,
                        candidate,
                        speed,
                        binary=args.binary.resolve(),
                        policy=args.policy.resolve(),
                        terrain=flats[0].xml_path.resolve(),
                        metadata=metadata_by_candidate[candidate],
                        work_root=work_root,
                        timeout_s=args.timeout,
                    )
                    futures[future] = (candidate, speed)
            results = [future.result() for future in as_completed(futures)]

    grouped: dict[PidCandidate, list[TrialResult]] = {
        candidate: [] for candidate in all_candidates
    }
    for result in results:
        grouped[result.candidate].append(result)

    passing: list[tuple[float, float, PidCandidate, list[TrialResult]]] = []
    for candidate, trials in grouped.items():
        trials.sort(key=lambda trial: trial.speed)
        if len(trials) != len(SPEEDS) or not all(trial.success for trial in trials):
            continue
        if not all(
            trial.max_abs_cross_track_m <= args.max_lateral
            and trial.final_heading_error_deg <= args.max_heading_deg
            for trial in trials
        ):
            continue
        normalized = [
            trial.max_abs_cross_track_m / args.max_lateral
            + trial.final_heading_error_deg / args.max_heading_deg
            for trial in trials
        ]
        passing.append((max(normalized), sum(normalized) / len(normalized), candidate, trials))

    if not passing:
        failures: dict[str, int] = {}
        for result in results:
            if result.reason:
                failures[result.reason] = failures.get(result.reason, 0) + 1
        print(f"Calibration failed: 0/{len(all_candidates)} candidates passed")
        if failures:
            print("Failure reasons:", json.dumps(failures, ensure_ascii=False, sort_keys=True))
        return 2

    _, _, selected, selected_trials = min(passing, key=lambda item: item[:2])
    config: dict[str, Any] = {
        "schema_version": 1,
        "controller": "path_heading_pid",
        "control_period_s": 0.02,
        **selected.as_dict(),
        "calibration": {
            "terrain_id": flats[0].terrain_id,
            "distance_m": args.distance,
            "speeds_m_s": list(SPEEDS),
            "max_cross_track_threshold_m": args.max_lateral,
            "final_heading_threshold_deg": args.max_heading_deg,
            "candidate_count": len(all_candidates),
            "passing_candidate_count": len(passing),
            "trials": [
                {
                    "speed_m_s": trial.speed,
                    "max_abs_cross_track_m": trial.max_abs_cross_track_m,
                    "final_heading_error_deg": trial.final_heading_error_deg,
                }
                for trial in selected_trials
            ],
        },
    }
    # Keep the calibrator output contract identical to what batch/UI loading
    # accepts before committing a selected configuration.
    runner.parse_pid_config(config, source="calibrator output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    runner.atomic_write_json(args.output, config)
    print(
        f"Selected {selected.identifier}; {len(passing)}/{len(all_candidates)} candidates passed"
    )
    print(f"PID config: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
