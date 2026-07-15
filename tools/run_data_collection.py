#!/usr/bin/env python3
"""Run the MuJoCo terrain data-collection matrix.

The C++ collector owns simulation retries and writes one MJCF key file per
terrain/command pair.  This script owns matrix discovery, bounded parallelism,
resume semantics, and a small atomic status file for every trajectory.

Only the Python standard library is required.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SPEED_STEP = Decimal("0.05")
SPEEDS: tuple[Decimal, ...] = tuple(
    Decimal("0.50") + index * SPEED_STEP for index in range(11)
)
STATUS_DIR_NAME = ".collection_status"
EXPECTED_COLLECTABLE_TERRAINS = 133
EXPECTED_TRAJECTORIES = EXPECTED_COLLECTABLE_TERRAINS * len(SPEEDS)
STATUS_SCHEMA_VERSION = 2
# Bump this whenever collector semantics change in a way that makes prior
# terminal results unsafe to resume.  Terrain/XML contents are hashed as well.
COLLECTION_PROTOCOL_REVISION = "mujoco-key-collection-v4-lstm-near-edge-reset-1m"
COLLECTION_POLICY_TYPE = "lstm_sru"
RESET_BEFORE_NEAR_EDGE_M = 1.0
PID_CONFIG_FILENAME = "pid_config.json"


@dataclass(frozen=True)
class PathHeadingPidConfig:
    """Effective straight-path controller values accepted by the C++ collector."""

    cross_track_gain: float = 1.25
    kp: float = 1.20
    ki: float = 0.05
    kd: float = 0.10
    heading_limit_rad: float = 0.35
    yaw_cmd_limit_rad_s: float = 0.50
    integral_limit_rad_s: float = 0.50
    derivative_alpha: float = 0.20

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

    def collector_args(self) -> list[str]:
        return [
            "--pid-cross-track-gain",
            f"{self.cross_track_gain:g}",
            "--pid-kp",
            f"{self.kp:g}",
            "--pid-ki",
            f"{self.ki:g}",
            "--pid-kd",
            f"{self.kd:g}",
            "--pid-heading-limit",
            f"{self.heading_limit_rad:g}",
            "--pid-yaw-cmd-limit",
            f"{self.yaw_cmd_limit_rad_s:g}",
            "--pid-integral-limit",
            f"{self.integral_limit_rad_s:g}",
            "--pid-derivative-alpha",
            f"{self.derivative_alpha:g}",
        ]


DEFAULT_PATH_HEADING_PID = PathHeadingPidConfig()
_PID_VALUE_FIELDS = frozenset(DEFAULT_PATH_HEADING_PID.as_dict())
_PID_REQUIRED_FIELDS = _PID_VALUE_FIELDS | {
    "schema_version",
    "controller",
    "control_period_s",
}
_PID_OPTIONAL_FIELDS = frozenset({"calibration"})


def _pid_number(document: Mapping[str, Any], field: str, source: str) -> float:
    value = document[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{source}: PID field {field!r} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{source}: PID field {field!r} must be finite")
    return number


def parse_pid_config(
    document: Mapping[str, Any], *, source: str = PID_CONFIG_FILENAME
) -> PathHeadingPidConfig:
    """Strictly parse the schema emitted by ``calibrate_straight_pid.py``."""

    if not isinstance(document, Mapping):
        raise ValueError(f"{source}: PID config root must be a JSON object")
    fields = set(document)
    missing = sorted(_PID_REQUIRED_FIELDS - fields)
    unknown = sorted(fields - _PID_REQUIRED_FIELDS - _PID_OPTIONAL_FIELDS)
    if missing:
        raise ValueError(f"{source}: PID config is missing field(s): {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{source}: PID config has unknown field(s): {', '.join(unknown)}")
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise ValueError(f"{source}: PID schema_version must be integer 1")
    if document["controller"] != "path_heading_pid":
        raise ValueError(f"{source}: PID controller must be 'path_heading_pid'")
    control_period = _pid_number(document, "control_period_s", source)
    if not math.isclose(control_period, 0.02, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{source}: PID control_period_s must be 0.02")
    if "calibration" in document and not isinstance(document["calibration"], Mapping):
        raise ValueError(f"{source}: PID calibration must be a JSON object")

    values = {field: _pid_number(document, field, source) for field in _PID_VALUE_FIELDS}
    for field in ("cross_track_gain", "kp", "ki", "kd"):
        if values[field] < 0.0:
            raise ValueError(f"{source}: PID field {field!r} must be nonnegative")
    for field in ("heading_limit_rad", "yaw_cmd_limit_rad_s", "integral_limit_rad_s"):
        if values[field] <= 0.0:
            raise ValueError(f"{source}: PID field {field!r} must be positive")
    if not 0.0 <= values["derivative_alpha"] <= 1.0:
        raise ValueError(f"{source}: PID field 'derivative_alpha' must be in [0, 1]")
    return PathHeadingPidConfig(**values)


def load_pid_config(data_root: Path) -> tuple[PathHeadingPidConfig, Path | None]:
    """Load ``data_root/pid_config.json`` or return immutable defaults."""

    path = data_root.expanduser().resolve() / PID_CONFIG_FILENAME
    if not path.exists():
        return DEFAULT_PATH_HEADING_PID, None
    if not path.is_file():
        raise ValueError(f"PID config is not a regular file: {path}")
    document = _load_json_object(path)
    return parse_pid_config(document, source=str(path)), path


def pid_config_summary(config: PathHeadingPidConfig, path: Path | None) -> str:
    source = str(path) if path is not None else "built-in defaults"
    values = " ".join(f"{name}={value:g}" for name, value in config.as_dict().items())
    return f"Heading PID: {source}; {values}"


@dataclass(frozen=True)
class Terrain:
    """One generated terrain and the metadata needed by the collector."""

    task_name: str
    terrain_id: str
    directory: Path
    metadata_path: Path
    xml_path: Path
    metadata: Mapping[str, Any]
    data_root: Path


@dataclass(frozen=True)
class CollectionJob:
    terrain: Terrain
    speed: Decimal

    @property
    def stem(self) -> str:
        return f"{self.terrain.task_name}-cmd_linv_x_{speed_token(self.speed)}"

    @property
    def key_path(self) -> Path:
        return self.terrain.directory / f"{self.stem}.xml"

    @property
    def status_path(self) -> Path:
        return self.terrain.directory / STATUS_DIR_NAME / f"{self.stem}.json"


@dataclass(frozen=True)
class JobResult:
    job: CollectionJob
    status: str
    message: str
    duration_s: float = 0.0


@dataclass(frozen=True)
class JobFileLock:
    """An acquired process-wide advisory lock for one collection job."""

    path: Path
    file_descriptor: int


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def speed_text(speed: Decimal) -> str:
    return f"{speed.quantize(Decimal('0.00')):.2f}"


def speed_token(speed: Decimal) -> str:
    return speed_text(speed).replace(".", "p")


def parse_speed(value: str | float | Decimal) -> Decimal:
    try:
        raw_speed = Decimal(str(value))
        speed = raw_speed.quantize(Decimal("0.00"))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Invalid linv_x speed: {value!r}") from exc
    if raw_speed != speed or speed not in SPEEDS:
        allowed = f"{speed_text(SPEEDS[0])}..{speed_text(SPEEDS[-1])} in 0.05 steps"
        raise ValueError(f"Unsupported linv_x speed {speed_text(speed)}; expected {allowed}")
    return speed


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def discover_terrains(data_root: Path, include_flat: bool = False) -> list[Terrain]:
    """Discover generated terrain metadata recursively and deterministically."""

    terrains: list[Terrain] = []
    data_root = data_root.expanduser().resolve()
    if not data_root.is_dir():
        return terrains
    # Validate the optional shared controller configuration at discovery time
    # so batch, report, and UI entry points all reject the same bad schema.
    load_pid_config(data_root)

    for metadata_path in sorted(data_root.rglob("terrain.json")):
        metadata = _load_json_object(metadata_path)
        task_name = str(
            metadata.get("task_name") or metadata.get("task") or metadata_path.parent.parent.name
        ).strip()
        terrain_id = str(
            metadata.get("terrain_id") or metadata.get("name") or metadata_path.parent.name
        ).strip()
        collect = bool(metadata.get("collect", task_name != "flat"))
        if not include_flat and (task_name == "flat" or not collect):
            continue

        xml_value = metadata.get("terrain_xml", "terrain.xml")
        xml_path = Path(str(xml_value))
        if not xml_path.is_absolute():
            xml_path = metadata_path.parent / xml_path
        terrains.append(
            Terrain(
                task_name=task_name,
                terrain_id=terrain_id,
                directory=metadata_path.parent,
                metadata_path=metadata_path,
                xml_path=xml_path,
                metadata=metadata,
                data_root=data_root,
            )
        )

    duplicate_keys: dict[tuple[str, str], list[Path]] = {}
    for terrain in terrains:
        duplicate_keys.setdefault((terrain.task_name, terrain.terrain_id), []).append(
            terrain.metadata_path
        )
    duplicates = {key: paths for key, paths in duplicate_keys.items() if len(paths) > 1}
    if duplicates:
        details = "; ".join(
            f"{task}/{terrain_id}: {', '.join(map(str, paths))}"
            for (task, terrain_id), paths in sorted(duplicates.items())
        )
        raise ValueError(f"Duplicate terrain identifiers: {details}")

    return sorted(terrains, key=lambda item: (item.task_name, item.terrain_id))


def build_jobs(terrains: Iterable[Terrain], speeds: Iterable[Decimal]) -> list[CollectionJob]:
    return [
        CollectionJob(terrain=terrain, speed=speed)
        for terrain in terrains
        for speed in speeds
    ]


def validate_key_xml(path: Path) -> tuple[bool, str]:
    """Perform the inexpensive validation needed for safe resume/reporting."""

    if not path.is_file():
        return False, "key XML is missing"
    try:
        if path.stat().st_size == 0:
            return False, "key XML is empty"
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        return False, f"key XML cannot be parsed: {exc}"

    root_tag = root.tag.rsplit("}", 1)[-1]
    if root_tag != "mujoco":
        return False, f"key XML root is <{root_tag}>, expected <mujoco>"
    keys = [node for node in root.iter() if node.tag.rsplit("}", 1)[-1] == "key"]
    if not keys:
        return False, "key XML contains no <key> elements"
    return True, ""


def load_status(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return _load_json_object(path)
    except ValueError:
        return None


def job_input_fingerprint(job: CollectionJob) -> str:
    """Fingerprint the inputs whose change requires a trajectory rerun."""

    digest = hashlib.sha256()
    digest.update(COLLECTION_PROTOCOL_REVISION.encode("utf-8"))
    digest.update(b"\0")
    digest.update(job.terrain.task_name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(job.terrain.terrain_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(speed_text(job.speed).encode("ascii"))
    for label, path in (
        (b"metadata", job.terrain.metadata_path),
        (b"terrain", job.terrain.xml_path),
    ):
        digest.update(b"\0")
        digest.update(label)
        digest.update(b"\0")
        try:
            digest.update(path.read_bytes())
        except OSError as exc:
            # Missing inputs must never match a previously terminal status.
            digest.update(f"unreadable:{exc}".encode("utf-8", errors="replace"))
    pid_config, pid_path = load_pid_config(job.terrain.data_root)
    digest.update(b"\0heading_pid\0")
    if pid_path is None:
        digest.update(b"built-in-default\0")
        digest.update(
            json.dumps(
                pid_config.as_dict(), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        )
    else:
        try:
            # Include the complete calibrator document, including provenance,
            # so any explicit config edit invalidates terminal resume state.
            digest.update(pid_path.read_bytes())
        except OSError as exc:
            digest.update(f"unreadable:{exc}".encode("utf-8", errors="replace"))
    return digest.hexdigest()


def status_matches_job(job: CollectionJob, status: Mapping[str, Any] | None) -> bool:
    """Return whether a runner-owned status belongs to the current inputs."""

    if not status:
        return False
    return (
        status.get("schema_version") == STATUS_SCHEMA_VERSION
        and status.get("input_fingerprint") == job_input_fingerprint(job)
    )


def normalized_status(status: Mapping[str, Any] | None) -> str:
    if not status:
        return ""
    value = str(status.get("status") or status.get("result") or "").lower()
    aliases = {
        "validated": "success",
        "error": "infrastructure_error",
        "infra": "infrastructure_error",
        "infrastructure": "infrastructure_error",
    }
    return aliases.get(value, value)


def is_completed(job: CollectionJob, required_attempts: int = 5) -> tuple[bool, str]:
    """Return whether a job has a trustworthy terminal result."""

    status_value = load_status(job.status_path)
    if status_value is not None and not status_matches_job(job, status_value):
        return False, ""
    key_valid, _ = validate_key_xml(job.key_path)
    status = normalized_status(status_value)
    if key_valid:
        return True, "existing key XML is valid"
    if job.key_path.exists():
        return False, ""
    attempts = status_value.get("attempts") if status_value else None
    attempt_count = len(attempts) if isinstance(attempts, list) else 0
    if status_value and not attempt_count:
        try:
            attempt_count = int(status_value.get("attempt_count", 0))
        except (TypeError, ValueError):
            attempt_count = 0
    if status == "failed" and attempt_count >= required_attempts:
        return True, "collector already exhausted all attempts"
    return False, ""


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def job_lock_path(job: CollectionJob) -> Path:
    """Return the persistent lock-file path shared by batch and UI runners."""

    return job.status_path.with_suffix(".lock")


def acquire_job_lock(job: CollectionJob) -> JobFileLock | None:
    """Acquire one job's non-blocking exclusive lock.

    The lock file intentionally remains on disk after release.  Removing an
    advisory lock file can create two independently locked inodes during a
    hand-off between processes.
    """

    path = job_lock_path(job)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(file_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(file_descriptor)
        if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK}:
            return None
        raise
    return JobFileLock(path=path, file_descriptor=file_descriptor)


def release_job_lock(lock: JobFileLock) -> None:
    """Release a lock returned by :func:`acquire_job_lock`."""

    try:
        fcntl.flock(lock.file_descriptor, fcntl.LOCK_UN)
    finally:
        os.close(lock.file_descriptor)


def cleanup_stale_key_temporaries(job: CollectionJob) -> int:
    """Remove dead collector temporaries belonging to exactly one job.

    The C++ writer embeds its PID and final output basename in every temporary
    filename.  Checking that PID prevents a second runner from deleting an
    in-flight write for the same terrain/command pair.
    """

    prefix = ".collector-"
    suffix = f"-{job.key_path.name}.tmp.xml"
    removed = 0
    try:
        candidates = tuple(job.terrain.directory.iterdir())
    except OSError:
        return removed

    for candidate in candidates:
        name = candidate.name
        if not name.startswith(prefix) or not name.endswith(suffix):
            continue
        pid_text = name[len(prefix) : -len(suffix)]
        if not pid_text.isdigit():
            continue
        try:
            os.kill(int(pid_text), 0)
        except ProcessLookupError:
            pass
        except (PermissionError, OSError):
            continue
        else:
            continue
        try:
            candidate.unlink()
        except (FileNotFoundError, OSError):
            continue
        removed += 1
    return removed


def _tail(value: str | bytes, limit: int = 4000) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    value = value.strip()
    if len(value) <= limit:
        return value
    return f"…{value[-limit:]}"


def _base_status(job: CollectionJob) -> dict[str, Any]:
    pid_config, pid_path = load_pid_config(job.terrain.data_root)
    return {
        "schema_version": STATUS_SCHEMA_VERSION,
        "input_fingerprint": job_input_fingerprint(job),
        "task_name": job.terrain.task_name,
        "terrain_id": job.terrain.terrain_id,
        "terrain": str(job.terrain.xml_path),
        "metadata": str(job.terrain.metadata_path),
        "output": str(job.key_path),
        "speed": float(job.speed),
        "speed_text": speed_text(job.speed),
        "collection_profile": COLLECTION_PROTOCOL_REVISION,
        "policy_type": COLLECTION_POLICY_TYPE,
        "reset_before_near_edge_m": RESET_BEFORE_NEAR_EDGE_M,
        "heading_pid": pid_config.as_dict(),
        "pid_config": str(pid_path) if pid_path is not None else "built-in-default",
    }


def collector_command(
    binary: Path,
    policy: Path,
    job: CollectionJob,
    max_attempts: int,
    extra_args: Sequence[str] = (),
) -> list[str]:
    pid_config, _ = load_pid_config(job.terrain.data_root)
    return [
        str(binary),
        "--terrain",
        str(job.terrain.xml_path),
        "--metadata",
        str(job.terrain.metadata_path),
        "--output",
        str(job.key_path),
        "--speed",
        speed_text(job.speed),
        "--policy",
        str(policy),
        "--policy-type",
        COLLECTION_POLICY_TYPE,
        "--reset-before-near-edge",
        f"{RESET_BEFORE_NEAR_EDGE_M:g}",
        "--result",
        str(job.status_path),
        "--max-attempts",
        str(max_attempts),
        *pid_config.collector_args(),
        *extra_args,
    ]


def _merge_collector_status(
    job: CollectionJob,
    *,
    returncode: int,
    started_at: str,
    duration_s: float,
    stdout: str,
    stderr: str,
) -> dict[str, Any]:
    raw = load_status(job.status_path) or {}
    raw_status = str(raw.get("status") or raw.get("result") or "")
    key_valid, key_error = validate_key_xml(job.key_path)

    if returncode == 0 and key_valid:
        final_status = "success"
        reason = ""
    elif returncode == 2:
        final_status = "failed"
        reason = str(raw.get("reason") or raw.get("failure_reason") or "attempts exhausted")
    else:
        final_status = "infrastructure_error"
        if returncode == 0:
            reason = key_error
        else:
            reason = str(raw.get("reason") or raw.get("error") or f"collector exited {returncode}")

    merged = dict(raw)
    merged.update(_base_status(job))
    merged.update(
        {
            "status": final_status,
            "collector_status": raw_status,
            "reason": reason,
            "returncode": returncode,
            "started_at": started_at,
            "finished_at": utc_now(),
            "duration_s": round(duration_s, 3),
        }
    )
    if stdout.strip():
        merged["stdout_tail"] = _tail(stdout)
    if stderr.strip():
        merged["stderr_tail"] = _tail(stderr)
    return merged


def _execute_job_locked(
    job: CollectionJob,
    *,
    binary: Path,
    policy: Path,
    max_attempts: int,
    timeout_s: float,
    force: bool,
    extra_args: Sequence[str] = (),
) -> JobResult:
    if not force:
        completed, reason = is_completed(job, max_attempts)
        if completed:
            key_valid, _ = validate_key_xml(job.key_path)
            if key_valid and normalized_status(load_status(job.status_path)) != "success":
                recovered = _base_status(job)
                recovered.update(
                    {
                        "status": "success",
                        "reason": "",
                        "recovered_existing_key": True,
                        "finished_at": utc_now(),
                    }
                )
                atomic_write_json(job.status_path, recovered)
            return JobResult(job, "skipped", reason)

    cleanup_stale_key_temporaries(job)
    started_at = utc_now()
    running = _base_status(job)
    running.update({"status": "running", "started_at": started_at})
    atomic_write_json(job.status_path, running)
    command = collector_command(binary, policy, job, max_attempts, extra_args)
    started = time.monotonic()
    try:
        completed_process = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        duration_s = time.monotonic() - started
        final_status = _merge_collector_status(
            job,
            returncode=completed_process.returncode,
            started_at=started_at,
            duration_s=duration_s,
            stdout=completed_process.stdout,
            stderr=completed_process.stderr,
        )
    except subprocess.TimeoutExpired as exc:
        duration_s = time.monotonic() - started
        final_status = _base_status(job)
        final_status.update(
            {
                "status": "infrastructure_error",
                "reason": f"collector timed out after {timeout_s:g}s",
                "started_at": started_at,
                "finished_at": utc_now(),
                "duration_s": round(duration_s, 3),
                "stdout_tail": _tail(exc.stdout or ""),
                "stderr_tail": _tail(exc.stderr or ""),
            }
        )
    except OSError as exc:
        duration_s = time.monotonic() - started
        final_status = _base_status(job)
        final_status.update(
            {
                "status": "infrastructure_error",
                "reason": f"cannot execute collector: {exc}",
                "started_at": started_at,
                "finished_at": utc_now(),
                "duration_s": round(duration_s, 3),
            }
        )

    cleanup_stale_key_temporaries(job)
    atomic_write_json(job.status_path, final_status)
    status = normalized_status(final_status)
    message = str(final_status.get("reason") or status)
    return JobResult(job, status, message, duration_s)


def execute_job(
    job: CollectionJob,
    *,
    binary: Path,
    policy: Path,
    max_attempts: int,
    timeout_s: float,
    force: bool,
    extra_args: Sequence[str] = (),
) -> JobResult:
    """Execute one job while holding its cross-process non-blocking lock."""

    try:
        lock = acquire_job_lock(job)
    except OSError as exc:
        return JobResult(
            job,
            "infrastructure_error",
            f"cannot acquire job lock: {exc}",
        )
    if lock is None:
        return JobResult(
            job,
            "infrastructure_error",
            f"job is already locked by another collector: {job_lock_path(job)}",
        )
    try:
        return _execute_job_locked(
            job,
            binary=binary,
            policy=policy,
            max_attempts=max_attempts,
            timeout_s=timeout_s,
            force=force,
            extra_args=extra_args,
        )
    finally:
        release_job_lock(lock)


def _comma_values(values: Sequence[str] | None) -> list[str]:
    return [item.strip() for value in values or () for item in value.split(",") if item.strip()]


def selected_speeds(values: Sequence[str] | None) -> tuple[Decimal, ...]:
    raw = _comma_values(values)
    if not raw or raw == ["all"]:
        return SPEEDS
    return tuple(sorted({parse_speed(value) for value in raw}))


def selected_tasks(values: Sequence[str] | None) -> set[str]:
    return set(_comma_values(values))


def resolve_binary(explicit: Path | None, root: Path) -> Path | None:
    if explicit is not None:
        return explicit.expanduser().resolve()
    candidates = (
        root / "mujoco/C++/build_onnx/mujoco_data_collector",
        root / "mujoco/C++/build/mujoco_data_collector",
        root / "mujoco/C++/mujoco_data_collector",
    )
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description="Collect the generated MuJoCo terrain/speed matrix in parallel."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=root / "data_collection",
        help="Generated terrain root (default: %(default)s).",
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=None,
        help="mujoco_data_collector executable; build directories are searched by default.",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=root / "policy/vtm_lstm_sru",
        help="vtm_lstm_sru policy directory (default: %(default)s).",
    )
    parser.add_argument(
        "--task",
        action="append",
        help="Task name to collect; repeat or use commas. Defaults to every collectable task.",
    )
    parser.add_argument(
        "--terrain-id",
        action="append",
        help="Exact terrain_id to collect; repeat or use commas. Defaults to all selected terrains.",
    )
    parser.add_argument(
        "--speed",
        action="append",
        help="linv_x speed to collect; repeat or use commas. Defaults to all 11 speeds.",
    )
    parser.add_argument("--workers", type=int, default=4, help="Parallel collector processes.")
    parser.add_argument("--max-attempts", type=int, default=5, help="Attempts made by C++ per job.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=210.0,
        help="Wall-clock timeout in seconds for one collector invocation.",
    )
    parser.add_argument("--force", action="store_true", help="Rerun terminal successes and failures.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the selected/resumed matrix without executing."
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-trajectory progress; print only matrix and final summaries.",
    )
    parser.add_argument(
        "--collector-arg",
        action="append",
        default=[],
        help="Extra argument passed verbatim to each collector process; repeat as needed.",
    )
    return parser.parse_args(argv)


def _validate_options(args: argparse.Namespace) -> None:
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be at least 1")
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    managed_flags = {"--policy-type", "--reset-before-near-edge"}
    conflicts = [
        value
        for value in args.collector_arg
        if value.split("=", 1)[0] in managed_flags
    ]
    if conflicts:
        raise ValueError(
            "collection profile manages these collector arguments: "
            + ", ".join(conflicts)
        )


def run(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        _validate_options(args)
        speeds = selected_speeds(args.speed)
        data_root = args.data_root.expanduser().resolve()
        pid_config, pid_config_path = load_pid_config(data_root)
        terrains = discover_terrains(data_root)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    task_filter = selected_tasks(args.task)
    available_tasks = {terrain.task_name for terrain in terrains}
    unknown_tasks = task_filter - available_tasks
    if unknown_tasks:
        print(
            f"error: unknown task filter(s): {', '.join(sorted(unknown_tasks))}; "
            f"available: {', '.join(sorted(available_tasks))}",
            file=sys.stderr,
        )
        return 2
    if task_filter:
        terrains = [terrain for terrain in terrains if terrain.task_name in task_filter]
    terrain_filter = set(_comma_values(args.terrain_id))
    available_terrain_ids = {terrain.terrain_id for terrain in terrains}
    unknown_terrain_ids = terrain_filter - available_terrain_ids
    if unknown_terrain_ids:
        print(
            "error: unknown terrain_id filter(s) for the selected task(s): "
            + ", ".join(sorted(unknown_terrain_ids)),
            file=sys.stderr,
        )
        return 2
    if terrain_filter:
        terrains = [terrain for terrain in terrains if terrain.terrain_id in terrain_filter]
    jobs = build_jobs(terrains, speeds)
    if not jobs:
        print(f"No collectable terrain jobs found under {args.data_root}", file=sys.stderr)
        return 2

    pending: list[CollectionJob] = []
    skipped: list[CollectionJob] = []
    for job in jobs:
        completed, _ = is_completed(job, args.max_attempts)
        if completed and not args.force:
            skipped.append(job)
        else:
            pending.append(job)

    print(pid_config_summary(pid_config, pid_config_path))
    print(
        f"Selected {len(terrains)} terrains × {len(speeds)} speeds = {len(jobs)} jobs; "
        f"run {len(pending)}, resume-skip {len(skipped)}."
    )
    if args.dry_run:
        if not args.quiet:
            for job in pending:
                print(
                    f"RUN  {job.terrain.task_name}/{job.terrain.terrain_id} "
                    f"linv_x={speed_text(job.speed)} -> {job.key_path.name}"
                )
            for job in skipped:
                print(
                    f"SKIP {job.terrain.task_name}/{job.terrain.terrain_id} "
                    f"linv_x={speed_text(job.speed)}"
                )
        return 0

    root = repo_root()
    binary = resolve_binary(args.binary, root)
    if binary is None or not binary.is_file():
        requested = args.binary or "mujoco/C++/{build_onnx,build}/mujoco_data_collector"
        print(f"error: collector binary not found: {requested}", file=sys.stderr)
        return 2
    policy = args.policy.expanduser().resolve()
    if not policy.is_dir():
        print(f"error: policy directory not found: {policy}", file=sys.stderr)
        return 2
    if not pending:
        print("Collection matrix is already complete; nothing to run.")
        return 0

    counts = {"success": 0, "failed": 0, "infrastructure_error": 0, "skipped": len(skipped)}
    executor = ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="mj-collector")
    futures: dict[Future[JobResult], CollectionJob] = {}
    try:
        for job in pending:
            future = executor.submit(
                execute_job,
                job,
                binary=binary,
                policy=policy,
                max_attempts=args.max_attempts,
                timeout_s=args.timeout,
                force=args.force,
                extra_args=args.collector_arg,
            )
            futures[future] = job
        for index, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # Keep unrelated jobs running after an unexpected worker bug.
                result = JobResult(job, "infrastructure_error", f"runner exception: {exc}")
                fallback = _base_status(job)
                fallback.update(
                    {"status": result.status, "reason": result.message, "finished_at": utc_now()}
                )
                atomic_write_json(job.status_path, fallback)
            counts[result.status] = counts.get(result.status, 0) + 1
            if not args.quiet:
                print(
                    f"[{index}/{len(pending)}] {result.status.upper():20s} "
                    f"{job.terrain.task_name}/{job.terrain.terrain_id} "
                    f"linv_x={speed_text(job.speed)} {result.message}"
                )
    except KeyboardInterrupt:
        print("Interrupted; completed status files are resumable.", file=sys.stderr)
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        return 130
    else:
        executor.shutdown(wait=True)

    print(
        "Collection finished: "
        + ", ".join(f"{name}={count}" for name, count in counts.items())
    )
    return 1 if counts.get("infrastructure_error", 0) else 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
