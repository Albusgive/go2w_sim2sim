#!/usr/bin/env python3
"""Recollect the full matrix with a 1 m near-terrain recurrent reset.

The collection protocol revision in ``run_data_collection.py`` makes results
from older protocols stale, so this command is safely resumable without
``--force``. After collection it always regenerates the Markdown report and
radar charts, then prints per-task policy failure rates.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import generate_data_collection_report as report  # noqa: E402
import run_data_collection as runner  # noqa: E402


def validate_near_edge_metadata(
    terrains: Sequence[runner.Terrain], reset_distance_m: float
) -> None:
    errors: list[str] = []
    for terrain in terrains:
        params = terrain.metadata.get("params")
        near_edge: Any = (
            params.get("near_edge_x_m") if isinstance(params, Mapping) else None
        )
        try:
            near_edge_m = float(near_edge)
        except (TypeError, ValueError):
            errors.append(f"{terrain.task_name}/{terrain.terrain_id}: missing near_edge_x_m")
            continue
        if near_edge_m <= reset_distance_m:
            errors.append(
                f"{terrain.task_name}/{terrain.terrain_id}: near_edge_x_m="
                f"{near_edge_m:g} must exceed reset distance {reset_distance_m:g}"
            )
    if errors:
        raise ValueError("invalid near-edge metadata:\n" + "\n".join(errors))


def print_failure_rates(
    terrains: Sequence[runner.Terrain], outcomes: Sequence[report.Outcome]
) -> None:
    print("Final policy failure rates:")
    for task_name in report.TASK_ORDER:
        task_outcomes = [
            outcome
            for outcome in outcomes
            if outcome.job.terrain.task_name == task_name
        ]
        counts = Counter(outcome.category for outcome in task_outcomes)
        total = len(task_outcomes)
        rate = 100.0 * counts["failed"] / total if total else 0.0
        print(
            f"  {task_name}: failed={counts['failed']}/{total} "
            f"({rate:.2f}%), success={counts['success']}, "
            f"infrastructure_error={counts['infrastructure_error']}, "
            f"pending={counts['pending']}"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = runner.repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=root / "data_collection")
    parser.add_argument("--binary", type=Path, default=None)
    parser.add_argument("--policy", type=Path, default=root / "policy/vtm_lstm_sru")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=210.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def run(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = args.data_root.expanduser().resolve()
    try:
        terrains = runner.discover_terrains(data_root)
        validate_near_edge_metadata(
            terrains, runner.RESET_BEFORE_NEAR_EDGE_M
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if len(terrains) != runner.EXPECTED_COLLECTABLE_TERRAINS:
        print(
            f"error: expected {runner.EXPECTED_COLLECTABLE_TERRAINS} collectable "
            f"terrains, found {len(terrains)}",
            file=sys.stderr,
        )
        return 2

    runner_args = [
        "--data-root",
        str(data_root),
        "--policy",
        str(args.policy.expanduser().resolve()),
        "--workers",
        str(args.workers),
        "--max-attempts",
        str(args.max_attempts),
        "--timeout",
        f"{args.timeout:g}",
        "--quiet",
    ]
    if args.binary is not None:
        runner_args.extend(["--binary", str(args.binary.expanduser().resolve())])
    if args.dry_run:
        runner_args.append("--dry-run")

    print(
        f"Collection profile: {runner.COLLECTION_PROTOCOL_REVISION}; "
        f"policy={runner.COLLECTION_POLICY_TYPE}; recurrent reset="
        f"{runner.RESET_BEFORE_NEAR_EDGE_M:g} m before terrain"
    )
    collection_code = runner.run(runner_args)
    if args.dry_run or collection_code == 2:
        return collection_code

    output_path = data_root / "collection_report.md"
    assets_dir = data_root / "report_assets"
    try:
        summary = report.generate_report(
            data_root, output_path, assets_dir
        )
        outcomes = report.collect_outcomes(terrains)
    except ValueError as exc:
        print(f"error: report generation failed: {exc}", file=sys.stderr)
        return 2

    print(
        f"Wrote {output_path}: success={summary.success}, failed={summary.failed}, "
        f"infrastructure_error={summary.infrastructure_error}, "
        f"pending={summary.pending}"
    )
    print_failure_rates(terrains, outcomes)
    return collection_code


if __name__ == "__main__":
    raise SystemExit(run())
