#!/usr/bin/env python3
"""Generate the local Markdown/SVG report for MuJoCo data collection."""

from __future__ import annotations

import argparse
import html
import math
import os
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from run_data_collection import (
        EXPECTED_COLLECTABLE_TERRAINS,
        EXPECTED_TRAJECTORIES,
        SPEEDS,
        CollectionJob,
        Terrain,
        build_jobs,
        discover_terrains,
        load_status,
        normalized_status,
        repo_root,
        speed_text,
        status_matches_job,
        validate_key_xml,
    )
except ModuleNotFoundError:  # Allow importing as tools.generate_data_collection_report.
    from tools.run_data_collection import (  # type: ignore[no-redef]
        EXPECTED_COLLECTABLE_TERRAINS,
        EXPECTED_TRAJECTORIES,
        SPEEDS,
        CollectionJob,
        Terrain,
        build_jobs,
        discover_terrains,
        load_status,
        normalized_status,
        repo_root,
        speed_text,
        status_matches_job,
        validate_key_xml,
    )


CATEGORIES = ("success", "failed", "infrastructure_error", "pending")
TASK_ORDER = ("single_platform", "ditch", "double_platform")


@dataclass(frozen=True)
class Outcome:
    job: CollectionJob
    category: str
    attempts: int
    reason: str
    duration_s: float | None
    status: Mapping[str, Any] | None


@dataclass(frozen=True)
class ReportSummary:
    terrain_count: int
    expected: int
    success: int
    failed: int
    infrastructure_error: int
    pending: int

    def as_dict(self) -> dict[str, int]:
        return {
            "success": self.success,
            "failed": self.failed,
            "infrastructure_error": self.infrastructure_error,
            "pending": self.pending,
        }


@dataclass(frozen=True)
class ChartSpec:
    task_name: str
    parameter_keys: tuple[str, ...]
    title: str
    filename: str


CHART_SPECS = (
    ChartSpec(
        "single_platform",
        ("height_m", "platform_height_m", "height"),
        "Single platform — height",
        "single_platform_height.svg",
    ),
    ChartSpec(
        "ditch",
        ("gap_m", "gap_width_m", "longitudinal_gap_m"),
        "Ditch — longitudinal gap",
        "ditch_gap.svg",
    ),
    ChartSpec(
        "double_platform",
        ("first_height_m", "h1_m", "first_height"),
        "Double platform — first height",
        "double_platform_first_height.svg",
    ),
    ChartSpec(
        "double_platform",
        ("height_delta_m", "delta_height_m", "height_delta"),
        "Double platform — height delta",
        "double_platform_height_delta.svg",
    ),
    ChartSpec(
        "double_platform",
        ("gap_m", "gap_width_m"),
        "Double platform — inter-platform gap",
        "double_platform_gap.svg",
    ),
)


def _attempt_count(status: Mapping[str, Any] | None) -> int:
    if not status:
        return 0
    attempts = status.get("attempts")
    if isinstance(attempts, list):
        return len(attempts)
    if isinstance(attempts, int):
        return attempts
    for key in ("attempt_count", "attempts_used"):
        try:
            if status.get(key) is not None:
                return int(status[key])
        except (TypeError, ValueError):
            pass
    return 0


def _duration(status: Mapping[str, Any] | None) -> float | None:
    if not status:
        return None
    try:
        if status.get("duration_s") is not None:
            return float(status["duration_s"])
        if status.get("sim_time_s") is not None:
            return float(status["sim_time_s"])
    except (TypeError, ValueError):
        pass
    attempts = status.get("attempts")
    if isinstance(attempts, list):
        durations: list[float] = []
        for attempt in attempts:
            if not isinstance(attempt, Mapping):
                continue
            try:
                duration = attempt.get("duration_s", attempt.get("sim_time_s", 0.0))
                durations.append(float(duration))
            except (TypeError, ValueError):
                pass
        if durations:
            return sum(durations)
    return None


def _reason(status: Mapping[str, Any] | None, fallback: str = "") -> str:
    if not status:
        return fallback
    generic_reason = ""
    for key in ("reason", "failure_reason", "error", "message"):
        value = status.get(key)
        if value:
            generic_reason = str(value)
            if generic_reason != "max_attempts_exhausted":
                return generic_reason
    attempts = status.get("attempts")
    if isinstance(attempts, list) and attempts:
        attempt_reasons = Counter(
            str(attempt.get("reason"))
            for attempt in attempts
            if isinstance(attempt, Mapping) and attempt.get("reason")
        )
        if generic_reason == "max_attempts_exhausted" and attempt_reasons:
            detail = ", ".join(
                f"{reason} × {count}" for reason, count in attempt_reasons.items()
            )
            return f"{generic_reason} ({detail})"
        last = attempts[-1]
        if isinstance(last, Mapping):
            for key in ("reason", "failure_reason", "error", "message"):
                if last.get(key):
                    return str(last[key])
    return generic_reason or fallback


def classify_job(job: CollectionJob) -> Outcome:
    status_exists = job.status_path.is_file()
    status = load_status(job.status_path)
    if status is not None and not status_matches_job(job, status):
        return Outcome(
            job=job,
            category="pending",
            attempts=0,
            reason="terrain or collector inputs changed; rerun required",
            duration_s=None,
            status=status,
        )
    state = normalized_status(status)
    key_exists = job.key_path.exists()
    key_valid, key_error = validate_key_xml(job.key_path)

    if key_valid:
        category = "success"
        reason = ""
    elif key_exists:
        category = "infrastructure_error"
        reason = key_error
    elif state == "failed" and _attempt_count(status) >= 5:
        category = "failed"
        reason = _reason(status, "attempts exhausted")
    elif state == "failed":
        category = "pending"
        reason = _reason(status, "partial attempts will be resumed")
    elif state in {"success", "infrastructure_error"}:
        category = "infrastructure_error"
        fallback = "successful status has no valid key XML" if state == "success" else "collector error"
        reason = _reason(status, fallback)
    elif status_exists and status is None:
        category = "infrastructure_error"
        reason = "status JSON cannot be parsed"
    else:
        category = "pending"
        reason = _reason(status)

    return Outcome(
        job=job,
        category=category,
        attempts=_attempt_count(status),
        reason=reason,
        duration_s=_duration(status),
        status=status,
    )


def collect_outcomes(terrains: Iterable[Terrain]) -> list[Outcome]:
    return [classify_job(job) for job in build_jobs(terrains, SPEEDS)]


def summarize(terrains: Sequence[Terrain], outcomes: Sequence[Outcome]) -> ReportSummary:
    counts = Counter(outcome.category for outcome in outcomes)
    expected = len(terrains) * len(SPEEDS)
    if sum(counts.values()) != expected:
        raise ValueError(
            f"Internal report mismatch: {sum(counts.values())} outcomes for {expected} jobs"
        )
    return ReportSummary(
        terrain_count=len(terrains),
        expected=expected,
        success=counts["success"],
        failed=counts["failed"],
        infrastructure_error=counts["infrastructure_error"],
        pending=counts["pending"],
    )


def _parameter_value(terrain: Terrain, keys: Sequence[str]) -> float | None:
    params = terrain.metadata.get("params")
    sources = [params, terrain.metadata.get("features"), terrain.metadata]
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        for key in keys:
            try:
                if source.get(key) is not None:
                    return float(source[key])
            except (TypeError, ValueError):
                continue
    return None


def aggregate_chart(
    outcomes: Sequence[Outcome], spec: ChartSpec
) -> tuple[list[str], list[float], list[float]]:
    buckets: dict[float, Counter[str]] = defaultdict(Counter)
    for outcome in outcomes:
        if outcome.job.terrain.task_name != spec.task_name:
            continue
        parameter = _parameter_value(outcome.job.terrain, spec.parameter_keys)
        if parameter is None:
            continue
        buckets[parameter][outcome.category] += 1

    labels: list[str] = []
    success_rates: list[float] = []
    failure_rates: list[float] = []
    for parameter, counts in sorted(buckets.items()):
        total = sum(counts.values())
        labels.append(f"{parameter:.2f} m")
        success_rates.append(counts["success"] / total if total else 0.0)
        failure_rates.append(counts["failed"] / total if total else 0.0)
    return labels, success_rates, failure_rates


def _polar_point(center_x: float, center_y: float, radius: float, angle: float) -> tuple[float, float]:
    return center_x + radius * math.cos(angle), center_y + radius * math.sin(angle)


def _point_text(points: Iterable[tuple[float, float]]) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def render_radar_svg(
    *,
    title: str,
    labels: Sequence[str],
    success_rates: Sequence[float],
    failure_rates: Sequence[float],
) -> str:
    if not labels or len(labels) != len(success_rates) or len(labels) != len(failure_rates):
        raise ValueError("Radar chart requires equally sized non-empty label/value arrays")

    width, height = 760, 700
    center_x, center_y, radius = 380.0, 340.0, 225.0
    angles = [-math.pi / 2 + 2 * math.pi * index / len(labels) for index in range(len(labels))]
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 700" role="img">',
        f"  <title>{html.escape(title)}</title>",
        "  <rect width=\"760\" height=\"700\" fill=\"#ffffff\"/>",
        f'  <text x="{width / 2:.0f}" y="34" text-anchor="middle" '
        f'font-family="sans-serif" font-size="22" font-weight="600">{html.escape(title)}</text>',
    ]

    for fraction in (0.25, 0.50, 0.75, 1.00):
        ring_points = [_polar_point(center_x, center_y, radius * fraction, angle) for angle in angles]
        if len(ring_points) >= 3:
            lines.append(
                f'  <polygon points="{_point_text(ring_points)}" fill="none" '
                'stroke="#d1d5db" stroke-width="1"/>'
            )
        else:
            lines.append(
                f'  <circle cx="{center_x}" cy="{center_y}" r="{radius * fraction:.1f}" '
                'fill="none" stroke="#d1d5db" stroke-width="1"/>'
            )
        lines.append(
            f'  <text x="{center_x + 4:.1f}" y="{center_y - radius * fraction - 3:.1f}" '
            f'font-family="sans-serif" font-size="10" fill="#6b7280">{fraction:.0%}</text>'
        )

    for label, angle in zip(labels, angles):
        end_x, end_y = _polar_point(center_x, center_y, radius, angle)
        label_x, label_y = _polar_point(center_x, center_y, radius + 35, angle)
        anchor = "middle" if abs(math.cos(angle)) < 0.25 else ("start" if math.cos(angle) > 0 else "end")
        lines.append(
            f'  <line x1="{center_x}" y1="{center_y}" x2="{end_x:.1f}" y2="{end_y:.1f}" '
            'stroke="#d1d5db" stroke-width="1"/>'
        )
        lines.append(
            f'  <text x="{label_x:.1f}" y="{label_y + 4:.1f}" text-anchor="{anchor}" '
            f'font-family="sans-serif" font-size="12" fill="#374151">{html.escape(label)}</text>'
        )

    def polygon(values: Sequence[float], color: str) -> str:
        points = [
            _polar_point(center_x, center_y, radius * max(0.0, min(1.0, value)), angle)
            for value, angle in zip(values, angles)
        ]
        return (
            f'  <polygon points="{_point_text(points)}" fill="{color}" fill-opacity="0.18" '
            f'stroke="{color}" stroke-width="3" stroke-linejoin="round"/>'
        )

    lines.append(polygon(success_rates, "#16a34a"))
    lines.append(polygon(failure_rates, "#dc2626"))
    lines.extend(
        [
            '  <rect x="245" y="646" width="16" height="4" fill="#16a34a"/>',
            '  <text x="268" y="653" font-family="sans-serif" font-size="13" fill="#374151">Success / expected</text>',
            '  <rect x="430" y="646" width="16" height="4" fill="#dc2626"/>',
            '  <text x="453" y="653" font-family="sans-serif" font-size="13" fill="#374151">Policy failure / expected</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _percent(value: int, total: int) -> str:
    return f"{100.0 * value / total:.1f}%" if total else "0.0%"


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def _relative_link(target: Path, source_directory: Path) -> str:
    return Path(os.path.relpath(target, source_directory)).as_posix()


def _task_summary(outcomes: Sequence[Outcome], task_name: str) -> Counter[str]:
    return Counter(outcome.category for outcome in outcomes if outcome.job.terrain.task_name == task_name)


def build_markdown(
    *,
    data_root: Path,
    output_path: Path,
    terrains: Sequence[Terrain],
    outcomes: Sequence[Outcome],
    summary: ReportSummary,
    charts: Sequence[tuple[ChartSpec, Path | None]],
) -> str:
    generated = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    lines = [
        "# MuJoCo 数据采集报告",
        "",
        f"生成时间：`{generated}`",
        "",
        "## 总览",
        "",
        "| 指标 | 数量 | 占已发现矩阵比例 |",
        "|---|---:|---:|",
        f"| 规范预期轨迹 | {EXPECTED_TRAJECTORIES:,} | — |",
        f"| 已发现地形 | {summary.terrain_count:,} | — |",
        f"| 已发现轨迹矩阵 | {summary.expected:,} | 100.0% |",
        f"| 成功 | {summary.success:,} | {_percent(summary.success, summary.expected)} |",
        f"| 策略失败（采集方案耗尽） | {summary.failed:,} | {_percent(summary.failed, summary.expected)} |",
        f"| 基础设施错误 | {summary.infrastructure_error:,} | {_percent(summary.infrastructure_error, summary.expected)} |",
        f"| 待采集 | {summary.pending:,} | {_percent(summary.pending, summary.expected)} |",
        "",
    ]
    if summary.terrain_count != EXPECTED_COLLECTABLE_TERRAINS:
        lines.extend(
            [
                f"> ⚠️ 发现 {summary.terrain_count} 个可采集地形；规范要求 "
                f"{EXPECTED_COLLECTABLE_TERRAINS} 个、{EXPECTED_TRAJECTORIES:,} 条轨迹。",
                "",
            ]
        )

    lines.extend(
        [
            "## 分任务结果",
            "",
            "| 任务 | 地形数 | 轨迹数 | 成功 | 策略失败 | 策略失败率 | 基础设施错误 | 待采集 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for task_name in TASK_ORDER:
        task_terrains = [terrain for terrain in terrains if terrain.task_name == task_name]
        counts = _task_summary(outcomes, task_name)
        total = sum(counts.values())
        lines.append(
            f"| `{task_name}` | {len(task_terrains):,} | {total:,} | "
            f"{counts['success']:,} | {counts['failed']:,} | "
            f"{_percent(counts['failed'], total)} | "
            f"{counts['infrastructure_error']:,} | {counts['pending']:,} |"
        )
    lines.append("")

    lines.extend(
        [
            "## 成功 / 失败雷达图",
            "",
            "绿色为成功数占该参数全部预期轨迹的比例；红色仅为策略失败比例。基础设施错误和待采集项保留在分母中，但不画入红线。",
            "",
        ]
    )
    for spec, chart_path in charts:
        lines.append(f"### {spec.title}")
        lines.append("")
        if chart_path is None:
            lines.append("未发现可用于该维度聚合的地形参数。")
        else:
            relative = _relative_link(chart_path, output_path.parent)
            lines.append(f"![{spec.title}]({relative})")
        lines.append("")

    failures = [
        outcome
        for outcome in outcomes
        if outcome.category in {"failed", "infrastructure_error"}
    ]
    lines.extend(
        [
            "## 失败与错误明细",
            "",
        ]
    )
    if not failures:
        lines.extend(["当前没有策略失败或基础设施错误。", ""])
    else:
        lines.extend(
            [
                "| 任务 | 地形 | linv_x (m/s) | 类别 | 尝试次数 | 耗时 (s) | 原因 | 状态 |",
                "|---|---|---:|---|---:|---:|---|---|",
            ]
        )
        for outcome in failures:
            job = outcome.job
            terrain_link = _relative_link(job.terrain.directory, output_path.parent) + "/"
            terrain = f"[{_markdown_cell(job.terrain.terrain_id)}]({terrain_link})"
            duration = "—" if outcome.duration_s is None else f"{outcome.duration_s:.2f}"
            status_link = (
                f"[JSON]({_relative_link(job.status_path, output_path.parent)})"
                if job.status_path.is_file()
                else "—"
            )
            category = "策略失败" if outcome.category == "failed" else "基础设施错误"
            lines.append(
                f"| `{job.terrain.task_name}` | {terrain} | {speed_text(job.speed)} | "
                f"{category} | {outcome.attempts or '—'} | {duration} | "
                f"{_markdown_cell(outcome.reason) or '—'} | {status_link} |"
            )
        lines.append("")

    lines.extend(
        [
            "## 说明",
            "",
            "- 有效 key XML 必须可解析、根节点为 `<mujoco>`，并至少包含一个 `<key>`。",
            "- 策略失败表示 C++ 采集器已耗尽该速度的全部尝试；损坏/缺失输出、加载或进程错误单列为基础设施错误。",
            "- `success + failed + infrastructure_error + pending` 始终等于已发现轨迹矩阵数量。",
            "",
        ]
    )
    return "\n".join(lines)


def generate_report(data_root: Path, output_path: Path, assets_dir: Path) -> ReportSummary:
    terrains = discover_terrains(data_root)
    if not terrains:
        raise ValueError(f"No collectable terrain.json files found under {data_root}")
    outcomes = collect_outcomes(terrains)
    summary = summarize(terrains, outcomes)

    charts: list[tuple[ChartSpec, Path | None]] = []
    for spec in CHART_SPECS:
        labels, success_rates, failure_rates = aggregate_chart(outcomes, spec)
        if not labels:
            charts.append((spec, None))
            continue
        chart_path = assets_dir / spec.filename
        svg = render_radar_svg(
            title=spec.title,
            labels=labels,
            success_rates=success_rates,
            failure_rates=failure_rates,
        )
        atomic_write_text(chart_path, svg)
        charts.append((spec, chart_path))

    markdown = build_markdown(
        data_root=data_root,
        output_path=output_path,
        terrains=terrains,
        outcomes=outcomes,
        summary=summary,
        charts=charts,
    )
    atomic_write_text(output_path, markdown)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Markdown and SVG radar charts from collection status/key files."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=repo_root() / "data_collection",
        help="Generated terrain/result root (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Markdown output (default: <data-root>/collection_report.md).",
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=None,
        help="SVG output directory (default: <data-root>/report_assets).",
    )
    return parser.parse_args(argv)


def run(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = args.data_root.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else data_root / "collection_report.md"
    )
    assets_dir = (
        args.assets_dir.expanduser().resolve()
        if args.assets_dir is not None
        else data_root / "report_assets"
    )
    try:
        summary = generate_report(data_root, output_path, assets_dir)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"Wrote {output_path}: expected={summary.expected}, success={summary.success}, "
        f"failed={summary.failed}, infrastructure_error={summary.infrastructure_error}, "
        f"pending={summary.pending}"
    )
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
