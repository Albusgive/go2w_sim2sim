#!/usr/bin/env python3
"""Retry missing ditch trajectories with recurrent-policy fallbacks.

The normal LSTM collection result is treated as the baseline.  Jobs that
already have a valid key are never touched.  Remaining jobs run, in order:

1. vtm_lstm_sru with a network-only reset before the ditch;
2. vtm_gru_sru without an approach reset;
3. vtm_gru_sru with the same approach reset.

Each stage owns its normal C++ retry budget and short-circuits on success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import run_data_collection as runner  # noqa: E402


PLAN_ID = "ditch-recurrent-policy-fallback-v1"
POLICY_FINGERPRINT_FILES = (
    "student_deploy.json",
    "student_encoder.onnx",
    "student_memory.onnx",
    "student_actor.onnx",
)


@dataclass(frozen=True)
class FallbackStage:
    stage_id: str
    policy_type: str
    policy_name: str
    policy_path: Path
    reset_before_near_edge_m: float | None

    @property
    def mid_reset(self) -> bool:
        return self.reset_before_near_edge_m is not None


@dataclass(frozen=True)
class FallbackJobResult:
    job: runner.CollectionJob
    status: str
    message: str
    winning_stage: str | None = None


def _policy_digest(digest: Any, label: str, path: Path) -> None:
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    for filename in POLICY_FINGERPRINT_FILES:
        file_path = path / filename
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_path.read_bytes())
        digest.update(b"\0")


def fallback_plan_fingerprint(
    stages: Sequence[FallbackStage], max_attempts: int
) -> str:
    digest = hashlib.sha256()
    digest.update(PLAN_ID.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(max_attempts).encode("ascii"))
    digest.update(b"\0")
    digested_policies: set[tuple[str, Path]] = set()
    for stage in stages:
        digest.update(stage.stage_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(stage.policy_type.encode("utf-8"))
        digest.update(b"\0")
        reset_text = (
            "none"
            if stage.reset_before_near_edge_m is None
            else f"{stage.reset_before_near_edge_m:.12g}"
        )
        digest.update(reset_text.encode("ascii"))
        digest.update(b"\0")
        policy_key = (stage.policy_type, stage.policy_path.resolve())
        if policy_key not in digested_policies:
            _policy_digest(digest, stage.policy_type, stage.policy_path)
            digested_policies.add(policy_key)
    return digest.hexdigest()


def fallback_plan(
    stages: Sequence[FallbackStage], max_attempts: int, fingerprint: str
) -> dict[str, Any]:
    return {
        "id": PLAN_ID,
        "fingerprint": fingerprint,
        "max_attempts_per_stage": max_attempts,
        "max_total_attempts": max_attempts * len(stages),
        "stages": [
            {
                "id": stage.stage_id,
                "policy_type": stage.policy_type,
                "policy_name": stage.policy_name,
                "policy_path": str(stage.policy_path),
                "mid_reset": stage.mid_reset,
                "reset_before_near_edge_m": stage.reset_before_near_edge_m,
            }
            for stage in stages
        ],
    }


def previous_result_summary(status: Mapping[str, Any] | None) -> dict[str, Any]:
    status = status or {}
    attempts = status.get("attempts")
    return {
        "status": runner.normalized_status(status),
        "reason": str(status.get("reason") or ""),
        "attempt_count": len(attempts) if isinstance(attempts, list) else 0,
        "attempts": attempts if isinstance(attempts, list) else [],
        "finished_at": status.get("finished_at"),
    }


def baseline_result_summary(status: Mapping[str, Any] | None) -> dict[str, Any]:
    """Keep the original baseline provenance across plan changes/forced reruns."""

    if status and status.get("collection_mode") == "ditch_fallback":
        previous = status.get("previous_result")
        if isinstance(previous, Mapping):
            return dict(previous)
    return previous_result_summary(status)


def stage_command(
    *,
    binary: Path,
    stage: FallbackStage,
    job: runner.CollectionJob,
    result_path: Path,
    max_attempts: int,
) -> list[str]:
    pid_config, _ = runner.load_pid_config(job.terrain.data_root)
    command = [
        str(binary),
        "--terrain",
        str(job.terrain.xml_path),
        "--metadata",
        str(job.terrain.metadata_path),
        "--output",
        str(job.key_path),
        "--speed",
        runner.speed_text(job.speed),
        "--policy",
        str(stage.policy_path),
        "--policy-type",
        stage.policy_type,
        "--result",
        str(result_path),
        "--max-attempts",
        str(max_attempts),
        *pid_config.collector_args(),
    ]
    if stage.reset_before_near_edge_m is not None:
        command.extend(
            [
                "--reset-before-near-edge",
                f"{stage.reset_before_near_edge_m:g}",
            ]
        )
    return command


def _annotated_attempts(stage_records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for stage_record in stage_records:
        attempts = stage_record.get("attempts")
        if not isinstance(attempts, list):
            continue
        for stage_attempt_index, raw_attempt in enumerate(attempts, start=1):
            attempt = dict(raw_attempt) if isinstance(raw_attempt, Mapping) else {}
            attempt["attempt"] = len(flattened) + 1
            attempt["stage_attempt"] = stage_attempt_index
            attempt["stage"] = stage_record.get("id")
            attempt["policy_type"] = stage_record.get("policy_type")
            attempt["policy_name"] = stage_record.get("policy_name")
            attempt["mid_reset"] = bool(stage_record.get("mid_reset"))
            flattened.append(attempt)
    return flattened


def _progress_status(
    job: runner.CollectionJob,
    *,
    plan: Mapping[str, Any],
    previous_result: Mapping[str, Any],
    stage_records: Sequence[Mapping[str, Any]],
    current_stage: str | None,
    started_at: str,
) -> dict[str, Any]:
    status = runner._base_status(job)
    attempts = _annotated_attempts(stage_records)
    status.update(
        {
            "status": "running",
            "reason": "",
            "collection_mode": "ditch_fallback",
            "fallback_plan": dict(plan),
            "fallback_plan_status": "running",
            "previous_result": dict(previous_result),
            "stages": list(stage_records),
            "attempts": attempts,
            "attempt_count": len(attempts),
            "current_stage": current_stage,
            "started_at": started_at,
        }
    )
    return status


def _stage_record(
    stage: FallbackStage,
    *,
    raw: Mapping[str, Any] | None,
    returncode: int | None,
    status: str,
    reason: str,
    duration_s: float,
    stdout: str = "",
    stderr: str = "",
) -> dict[str, Any]:
    raw = raw or {}
    attempts = raw.get("attempts") if isinstance(raw.get("attempts"), list) else []
    record: dict[str, Any] = {
        "id": stage.stage_id,
        "status": status,
        "reason": reason,
        "policy_type": stage.policy_type,
        "policy_name": stage.policy_name,
        "policy_path": str(stage.policy_path),
        "mid_reset": stage.mid_reset,
        "reset_before_near_edge_m": stage.reset_before_near_edge_m,
        "attempt_count": len(attempts),
        "attempts": attempts,
        "returncode": returncode,
        "duration_s": round(duration_s, 3),
    }
    for field in (
        "frames",
        "sim_time_s",
        "max_abs_cross_track_m",
        "final_heading_error_deg",
        "near_edge_x_m",
        "recurrent_reset_threshold_x_m",
    ):
        if field in raw:
            record[field] = raw[field]
    if stdout.strip():
        record["stdout_tail"] = runner._tail(stdout)
    if stderr.strip():
        record["stderr_tail"] = runner._tail(stderr)
    return record


def _final_status(
    job: runner.CollectionJob,
    *,
    plan: Mapping[str, Any],
    previous_result: Mapping[str, Any],
    stage_records: Sequence[Mapping[str, Any]],
    status_value: str,
    reason: str,
    started_at: str,
    duration_s: float,
    winning_stage: str | None = None,
) -> dict[str, Any]:
    status = runner._base_status(job)
    attempts = _annotated_attempts(stage_records)
    status.update(
        {
            "status": status_value,
            "reason": reason,
            "collection_mode": "ditch_fallback",
            "fallback_plan": dict(plan),
            "fallback_plan_status": (
                "success"
                if status_value == "success"
                else "exhausted"
                if status_value == "failed"
                else "infrastructure_error"
            ),
            "previous_result": dict(previous_result),
            "stages": list(stage_records),
            "attempts": attempts,
            "attempt_count": len(attempts),
            "winning_stage": winning_stage,
            "started_at": started_at,
            "finished_at": runner.utc_now(),
            "duration_s": round(duration_s, 3),
        }
    )
    if winning_stage is not None:
        winner = next(
            record for record in stage_records if record.get("id") == winning_stage
        )
        status["policy_type"] = winner.get("policy_type")
        status["policy_name"] = winner.get("policy_name")
        status["policy_path"] = winner.get("policy_path")
        status["mid_reset"] = winner.get("mid_reset")
        status["reset_before_near_edge_m"] = winner.get(
            "reset_before_near_edge_m"
        )
        for field in (
            "frames",
            "sim_time_s",
            "max_abs_cross_track_m",
            "final_heading_error_deg",
        ):
            if field in winner:
                status[field] = winner[field]
    return status


def _matching_completed_stages(
    status: Mapping[str, Any] | None,
    plan_fingerprint: str,
    stages: Sequence[FallbackStage],
    max_attempts: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not status or status.get("collection_mode") != "ditch_fallback":
        return baseline_result_summary(status), []
    existing_plan = status.get("fallback_plan")
    if not isinstance(existing_plan, Mapping) or existing_plan.get(
        "fingerprint"
    ) != plan_fingerprint:
        return baseline_result_summary(status), []
    previous = status.get("previous_result")
    previous_summary = (
        dict(previous)
        if isinstance(previous, Mapping)
        else previous_result_summary(status)
    )
    expected_ids = [stage.stage_id for stage in stages]
    raw_records = status.get("stages")
    if not isinstance(raw_records, list):
        raw_records = []
    records_by_id = {
        str(record.get("id")): dict(record)
        for record in raw_records
        if isinstance(record, Mapping)
    }
    completed: list[dict[str, Any]] = []
    for stage_id in expected_ids:
        record = records_by_id.get(stage_id)
        if not record:
            break
        try:
            attempt_count = int(record.get("attempt_count", 0))
        except (TypeError, ValueError):
            break
        if record.get("status") != "failed" or attempt_count < max_attempts:
            break
        completed.append(record)
    return previous_summary, completed


def _execute_fallback_job_locked(
    job: runner.CollectionJob,
    *,
    binary: Path,
    stages: Sequence[FallbackStage],
    plan: Mapping[str, Any],
    max_attempts: int,
    timeout_s: float,
    force_restart: bool = False,
) -> FallbackJobResult:
    started_at = runner.utc_now()
    started = time.monotonic()
    existing = runner.load_status(job.status_path)
    plan_fingerprint = str(plan["fingerprint"])
    if force_restart:
        previous_result = baseline_result_summary(existing)
        stage_records: list[dict[str, Any]] = []
    else:
        previous_result, stage_records = _matching_completed_stages(
            existing, plan_fingerprint, stages, max_attempts
        )
    completed_ids = {str(record.get("id")) for record in stage_records}

    for stage in stages:
        if stage.stage_id in completed_ids:
            continue
        progress = _progress_status(
            job,
            plan=plan,
            previous_result=previous_result,
            stage_records=stage_records,
            current_stage=stage.stage_id,
            started_at=started_at,
        )
        runner.atomic_write_json(job.status_path, progress)
        scratch = (
            job.status_path.parent
            / ".fallback"
            / job.stem
            / f"{stage.stage_id}.json"
        )
        scratch.parent.mkdir(parents=True, exist_ok=True)
        try:
            scratch.unlink(missing_ok=True)
        except OSError:
            pass
        command = stage_command(
            binary=binary,
            stage=stage,
            job=job,
            result_path=scratch,
            max_attempts=max_attempts,
        )
        stage_started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            stage_duration = time.monotonic() - stage_started
            raw = runner.load_status(scratch)
            key_valid, key_error = runner.validate_key_xml(job.key_path)
            raw_status = runner.normalized_status(raw)
            attempts = raw.get("attempts") if isinstance(raw, Mapping) else None
            attempt_count = len(attempts) if isinstance(attempts, list) else 0
            if key_valid:
                stage_status = "success"
                stage_reason = "terminal_reached"
            elif (
                completed.returncode == 2
                and raw_status == "failed"
                and attempt_count >= max_attempts
            ):
                stage_status = "failed"
                stage_reason = str(raw.get("reason") or "max_attempts_exhausted")
            else:
                stage_status = "infrastructure_error"
                stage_reason = (
                    key_error
                    if completed.returncode == 0
                    else str(
                        (raw or {}).get("reason")
                        or f"collector exited {completed.returncode}"
                    )
                )
            record = _stage_record(
                stage,
                raw=raw,
                returncode=completed.returncode,
                status=stage_status,
                reason=stage_reason,
                duration_s=stage_duration,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
        except subprocess.TimeoutExpired as exc:
            stage_duration = time.monotonic() - stage_started
            record = _stage_record(
                stage,
                raw=runner.load_status(scratch),
                returncode=None,
                status="infrastructure_error",
                reason=f"collector timed out after {timeout_s:g}s",
                duration_s=stage_duration,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
            )
        except OSError as exc:
            stage_duration = time.monotonic() - stage_started
            record = _stage_record(
                stage,
                raw=None,
                returncode=None,
                status="infrastructure_error",
                reason=f"cannot execute collector: {exc}",
                duration_s=stage_duration,
            )

        stage_records.append(record)
        if record["status"] == "success":
            final = _final_status(
                job,
                plan=plan,
                previous_result=previous_result,
                stage_records=stage_records,
                status_value="success",
                reason="",
                started_at=started_at,
                duration_s=time.monotonic() - started,
                winning_stage=stage.stage_id,
            )
            runner.atomic_write_json(job.status_path, final)
            return FallbackJobResult(
                job, "success", "success", winning_stage=stage.stage_id
            )
        if record["status"] == "infrastructure_error":
            final = _final_status(
                job,
                plan=plan,
                previous_result=previous_result,
                stage_records=stage_records,
                status_value="infrastructure_error",
                reason=str(record["reason"]),
                started_at=started_at,
                duration_s=time.monotonic() - started,
            )
            final["current_stage"] = stage.stage_id
            runner.atomic_write_json(job.status_path, final)
            return FallbackJobResult(job, "infrastructure_error", str(record["reason"]))

        # Persist every exhausted stage immediately. If the process is
        # interrupted before the next stage starts, resume will not repeat it.
        runner.atomic_write_json(
            job.status_path,
            _progress_status(
                job,
                plan=plan,
                previous_result=previous_result,
                stage_records=stage_records,
                current_stage=None,
                started_at=started_at,
            ),
        )

    final = _final_status(
        job,
        plan=plan,
        previous_result=previous_result,
        stage_records=stage_records,
        status_value="failed",
        reason="max_attempts_exhausted",
        started_at=started_at,
        duration_s=time.monotonic() - started,
    )
    runner.atomic_write_json(job.status_path, final)
    return FallbackJobResult(job, "failed", "all fallback stages exhausted")


def execute_fallback_job(
    job: runner.CollectionJob,
    *,
    binary: Path,
    stages: Sequence[FallbackStage],
    plan: Mapping[str, Any],
    max_attempts: int,
    timeout_s: float,
    force_restart: bool = False,
) -> FallbackJobResult:
    """Run the complete fallback chain under the shared per-job lock."""

    try:
        lock = runner.acquire_job_lock(job)
    except OSError as exc:
        return FallbackJobResult(
            job, "infrastructure_error", f"cannot acquire job lock: {exc}"
        )
    if lock is None:
        return FallbackJobResult(
            job,
            "infrastructure_error",
            f"job is already locked by another collector: "
            f"{runner.job_lock_path(job)}",
        )
    try:
        return _execute_fallback_job_locked(
            job,
            binary=binary,
            stages=stages,
            plan=plan,
            max_attempts=max_attempts,
            timeout_s=timeout_s,
            force_restart=force_restart,
        )
    finally:
        runner.release_job_lock(lock)


def _comma_values(values: Sequence[str] | None) -> set[str]:
    return {
        item.strip()
        for value in values or ()
        for item in value.split(",")
        if item.strip()
    }


def _plan_is_terminal(
    status: Mapping[str, Any] | None, plan_fingerprint: str
) -> bool:
    if not status or status.get("collection_mode") != "ditch_fallback":
        return False
    plan = status.get("fallback_plan")
    if not isinstance(plan, Mapping) or plan.get("fingerprint") != plan_fingerprint:
        return False
    # A recorded success is terminal only while its key XML is still valid;
    # select_fallback_jobs checks that first.  If the key was removed or became
    # corrupt, rerun from the winning stage instead of silently skipping it.
    return status.get("fallback_plan_status") == "exhausted"


def select_fallback_jobs(
    jobs: Sequence[runner.CollectionJob],
    *,
    plan_fingerprint: str,
    force: bool = False,
) -> tuple[list[runner.CollectionJob], list[runner.CollectionJob], list[str]]:
    pending: list[runner.CollectionJob] = []
    skipped: list[runner.CollectionJob] = []
    errors: list[str] = []
    for job in jobs:
        status = runner.load_status(job.status_path)
        key_valid, key_error = runner.validate_key_xml(job.key_path)
        if key_valid:
            if status is not None and not runner.status_matches_job(job, status):
                errors.append(
                    f"{job.terrain.terrain_id}/{job.stem}: valid key has stale status"
                )
            else:
                skipped.append(job)
            continue
        if job.key_path.exists() and not force:
            errors.append(
                f"{job.terrain.terrain_id}/{job.stem}: {key_error}; "
                "inspect it or rerun with --force"
            )
            continue
        if status is not None and not runner.status_matches_job(job, status):
            errors.append(
                f"{job.terrain.terrain_id}/{job.stem}: status is stale; run baseline collection first"
            )
            continue
        if not force and _plan_is_terminal(status, plan_fingerprint):
            skipped.append(job)
            continue
        pending.append(job)
    return pending, skipped, errors


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = runner.repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=root / "data_collection")
    parser.add_argument("--binary", type=Path, default=None)
    parser.add_argument(
        "--lstm-policy", type=Path, default=root / "policy/vtm_lstm_sru"
    )
    parser.add_argument(
        "--gru-policy", type=Path, default=root / "policy/vtm_gru_sru"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=210.0)
    parser.add_argument("--reset-before-near-edge", type=float, default=1.0)
    parser.add_argument("--terrain-id", action="append")
    parser.add_argument("--speed", action="append")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def run(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.workers < 1 or not 1 <= args.max_attempts <= 5:
        print("error: workers must be >=1 and max-attempts must be in [1,5]", file=sys.stderr)
        return 2
    if args.timeout <= 0 or args.reset_before_near_edge <= 0:
        print("error: timeout and reset distance must be positive", file=sys.stderr)
        return 2

    data_root = args.data_root.expanduser().resolve()
    lstm_policy = args.lstm_policy.expanduser().resolve()
    gru_policy = args.gru_policy.expanduser().resolve()
    try:
        speeds = runner.selected_speeds(args.speed)
        terrains = [
            terrain
            for terrain in runner.discover_terrains(data_root)
            if terrain.task_name == "ditch"
        ]
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    terrain_filter = _comma_values(args.terrain_id)
    known_ids = {terrain.terrain_id for terrain in terrains}
    unknown_ids = terrain_filter - known_ids
    if unknown_ids:
        print(f"error: unknown ditch terrain(s): {', '.join(sorted(unknown_ids))}", file=sys.stderr)
        return 2
    if terrain_filter:
        terrains = [terrain for terrain in terrains if terrain.terrain_id in terrain_filter]

    binary = runner.resolve_binary(args.binary, runner.repo_root())
    if binary is None or not binary.is_file():
        print("error: mujoco_data_collector binary not found", file=sys.stderr)
        return 2
    for policy_path in (lstm_policy, gru_policy):
        try:
            missing = [name for name in POLICY_FINGERPRINT_FILES if not (policy_path / name).is_file()]
        except OSError:
            missing = list(POLICY_FINGERPRINT_FILES)
        if missing:
            print(
                f"error: policy {policy_path} is missing: {', '.join(missing)}",
                file=sys.stderr,
            )
            return 2

    stages = (
        FallbackStage(
            "lstm_mid_reset",
            "lstm_sru",
            "vtm_lstm_sru",
            lstm_policy,
            args.reset_before_near_edge,
        ),
        FallbackStage(
            "gru_baseline", "gru_sru", "vtm_gru_sru", gru_policy, None
        ),
        FallbackStage(
            "gru_mid_reset",
            "gru_sru",
            "vtm_gru_sru",
            gru_policy,
            args.reset_before_near_edge,
        ),
    )
    try:
        plan_fingerprint = fallback_plan_fingerprint(stages, args.max_attempts)
    except OSError as exc:
        print(f"error: cannot fingerprint fallback policies: {exc}", file=sys.stderr)
        return 2
    plan = fallback_plan(stages, args.max_attempts, plan_fingerprint)
    jobs = runner.build_jobs(terrains, speeds)
    pending, skipped, errors = select_fallback_jobs(
        jobs, plan_fingerprint=plan_fingerprint, force=args.force
    )
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 2

    print(
        f"Ditch fallback {PLAN_ID}: selected={len(jobs)}, run={len(pending)}, "
        f"skip={len(skipped)}, reset_before_near_edge={args.reset_before_near_edge:g}m"
    )
    if args.dry_run:
        for job in pending:
            print(
                f"RUN  {job.terrain.terrain_id} linv_x={runner.speed_text(job.speed)}"
            )
        return 0
    if not pending:
        print("Fallback matrix is already complete; nothing to run.")
        return 0

    counts = {"success": 0, "failed": 0, "infrastructure_error": 0}
    winning_counts: dict[str, int] = {}
    with ThreadPoolExecutor(
        max_workers=args.workers, thread_name_prefix="ditch-fallback"
    ) as executor:
        futures: dict[Future[FallbackJobResult], runner.CollectionJob] = {
            executor.submit(
                execute_fallback_job,
                job,
                binary=binary,
                stages=stages,
                plan=plan,
                max_attempts=args.max_attempts,
                timeout_s=args.timeout,
                force_restart=args.force,
            ): job
            for job in pending
        }
        try:
            for index, future in enumerate(as_completed(futures), start=1):
                job = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = FallbackJobResult(
                        job, "infrastructure_error", f"runner exception: {exc}"
                    )
                    current = runner.load_status(job.status_path) or {}
                    status = dict(current)
                    status.update(runner._base_status(job))
                    status.update(
                        {
                            "status": result.status,
                            "reason": result.message,
                            "collection_mode": "ditch_fallback",
                            "fallback_plan": plan,
                            "fallback_plan_status": "infrastructure_error",
                            "finished_at": runner.utc_now(),
                        }
                    )
                    runner.atomic_write_json(job.status_path, status)
                counts[result.status] = counts.get(result.status, 0) + 1
                if result.winning_stage:
                    winning_counts[result.winning_stage] = (
                        winning_counts.get(result.winning_stage, 0) + 1
                    )
                stage_text = (
                    f" via {result.winning_stage}" if result.winning_stage else ""
                )
                print(
                    f"[{index}/{len(pending)}] {result.status.upper():20s} "
                    f"{job.terrain.terrain_id} linv_x={runner.speed_text(job.speed)}"
                    f"{stage_text} {result.message}"
                )
        except KeyboardInterrupt:
            for future in futures:
                future.cancel()
            print("Interrupted; completed fallback stages are resumable.", file=sys.stderr)
            return 130

    print(
        "Fallback finished: "
        + ", ".join(f"{name}={count}" for name, count in counts.items())
    )
    if winning_counts:
        print(
            "Winning stages: "
            + ", ".join(
                f"{name}={count}" for name, count in sorted(winning_counts.items())
            )
        )
    return 1 if counts["infrastructure_error"] else 0


if __name__ == "__main__":
    raise SystemExit(run())
