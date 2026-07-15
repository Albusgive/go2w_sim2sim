#!/usr/bin/env python3
"""Simple command-line trajectory replay helper for data_collection.

This is intentionally minimal:
- discover generated trajectories (task / terrain / speed)
- optionally list or pick one
- run ``mujoco_key_replayer`` with a small set of common flags.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import run_data_collection as runner  # noqa: E402


@dataclass(frozen=True)
class ReplayAppConfig:
    repo_root: Path
    data_root: Path
    replay_binary: Path


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_replay_binary(root: Path) -> Path:
    candidates = [
        root / "mujoco/C++/build_onnx/mujoco_key_replayer",
        root / "mujoco/C++/build/mujoco_key_replayer",
        root / "mujoco/C++/mujoco_key_replayer",
    ]
    candidates.extend(sorted(root.glob("mujoco/C++/build*/mujoco_key_replayer")))
    return next((path for path in candidates if path.is_file()), candidates[0])


def discover_replay_jobs(
    data_root: Path,
    task: str | None = None,
    terrain_id: str | None = None,
    speed: str | None = None,
    include_flat: bool = False,
) -> list[runner.CollectionJob]:
    terrains = runner.discover_terrains(data_root, include_flat=include_flat)
    jobs: list[runner.CollectionJob] = []

    for terrain in terrains:
        if task is not None and terrain.task_name != task:
            continue
        if terrain_id is not None and terrain.terrain_id != terrain_id:
            continue

        speeds = runner.SPEEDS
        if speed is not None:
            speeds = (runner.parse_speed(speed),)

        for value in speeds:
            job = runner.CollectionJob(terrain=terrain, speed=value)
            valid, _reason = runner.validate_key_xml(job.key_path)
            if valid:
                jobs.append(job)
            elif job.key_path.exists():
                # keep invalid entries invisible in replay mode
                pass

    return sorted(jobs, key=lambda job: (job.terrain.task_name, job.terrain.terrain_id, job.speed))


def print_jobs(jobs: list[runner.CollectionJob]) -> None:
    if not jobs:
        print("没有可回放的轨迹。")
        return

    print(f"总计 {len(jobs)} 条可回放轨迹：")
    print("  编号 | 任务名            | 地形ID                                  | 速度")
    print("  ---- | ----------------- | --------------------------------------- | -----")
    for index, job in enumerate(jobs, start=1):
        print(
            f"  {index:>4d} | {job.terrain.task_name:<17} | "
            f"{job.terrain.terrain_id:<39} | {runner.speed_text(job.speed)} m/s"
        )


def build_replay_command(
    config: ReplayAppConfig,
    job: runner.CollectionJob,
    *,
    rate: float,
    paused: bool,
    loop: bool,
) -> list[str]:
    command = [
        str(config.replay_binary),
        "--trajectory",
        str(job.key_path),
        "--metadata",
        str(job.terrain.metadata_path),
        "--rate",
        f"{rate}",
    ]
    if paused:
        command.append("--paused")
    if loop:
        command.append("--loop")
    return command


def run_replay(command: list[str], *, dry_run: bool = False) -> int:
    print("$", " ".join(command))
    if dry_run:
        return 0
    process = subprocess.run(command, check=False)
    return process.returncode


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay collected MuJoCo key trajectories")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=repo_root() / "data_collection",
        help="数据集合目录，默认 data_collection",
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=None,
        help="mujoco_key_replayer 路径，默认自动检测",
    )
    parser.add_argument("--task", help="可选：任务名，例如 ditch / double_platform")
    parser.add_argument("--terrain-id", help="可选：地形目录名")
    parser.add_argument("--speed", help="可选：linv_x（0.50-1.00，步进0.05），例如 0.75")
    parser.add_argument("--index", type=int, help="当过滤到多个轨迹时，按列表序号选择回放")
    parser.add_argument("--rate", type=float, default=1.0, help="回放速率（0.05-16），默认 1.0")
    parser.add_argument("--paused", action="store_true", help="启动后暂停")
    parser.add_argument("--loop", action="store_true", help="循环播放")
    parser.add_argument("--include-flat", action="store_true", help="包含 flat 任务的轨迹")
    parser.add_argument("--list", action="store_true", help="仅列出可回放轨迹")
    parser.add_argument(
        "--check",
        action="store_true",
        help="仅验证轨迹（调用 mujoco_key_replayer --check）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印重放命令，不实际执行",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="多个候选时弹交互式选择（未指定 --index 或 speed/task/terrain 时自动启用）",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> tuple[ReplayAppConfig, list[runner.CollectionJob]]:
    root = repo_root()
    binary = args.binary if args.binary is not None else default_replay_binary(root)
    config = ReplayAppConfig(
        repo_root=root,
        data_root=args.data_root.expanduser().resolve(),
        replay_binary=binary,
    )

    if not config.data_root.is_dir():
        raise FileNotFoundError(f"数据目录不存在: {config.data_root}")

    jobs = discover_replay_jobs(
        data_root=config.data_root,
        task=args.task,
        terrain_id=args.terrain_id,
        speed=args.speed,
        include_flat=args.include_flat,
    )

    if args.rate < 0.05 or args.rate > 16.0:
        raise ValueError("rate 需要在 [0.05, 16] 之间")

    if args.speed is not None and not jobs:
        available = ", ".join(runner.speed_text(v) for v in runner.SPEEDS)
        raise ValueError(f"当前筛选条件无可回放轨迹。可用速度: {available}")

    if not args.list and not args.dry_run:
        if not config.replay_binary.is_file():
            raise FileNotFoundError(
                "未找到 mujoco_key_replayer，可用 --binary 显式指定，或先执行 cmake --build --target mujoco_key_replayer"
            )

    return config, jobs


def choose_job(jobs: list[runner.CollectionJob], args: argparse.Namespace) -> runner.CollectionJob:
    if not jobs:
        raise RuntimeError("没有可回放轨迹")

    if args.index is None:
        print_jobs(jobs)
        if not args.interactive:
            raise RuntimeError(
                "请使用 --index 指定轨迹序号，或加 --interactive 进入交互选择。"
            )

        while True:
            value = input("请输入轨迹序号，回车直接退出: ").strip()
            if not value:
                raise SystemExit(0)
            try:
                selected = int(value)
            except ValueError:
                print("非法输入，请输入数字")
                continue
            if selected < 1 or selected > len(jobs):
                print("超出范围，请重试")
                continue
            return jobs[selected - 1]

    selected = args.index
    if selected < 1 or selected > len(jobs):
        raise IndexError(f"--index 超出范围: 1~{len(jobs)}")
    return jobs[selected - 1]


def check_jobs(config: ReplayAppConfig, jobs: list[runner.CollectionJob]) -> int:
    failed: list[tuple[runner.CollectionJob, str]] = []
    for job in jobs:
        command = build_replay_command(
            config,
            job,
            rate=1.0,
            paused=False,
            loop=False,
        )
        command.append("--check")
        result = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if result.returncode != 0:
            failed.append((job, result.stdout.strip().splitlines()[-1] if result.stdout else "unknown"))
            continue

    if not failed:
        print(f"共 {len(jobs)} 条可回放轨迹校验通过")
        return 0

    print(f"校验失败 {len(failed)} 条：")
    for job, error in failed:
        print(f"- {job.terrain.task_name}/{job.terrain.terrain_id} {runner.speed_text(job.speed)}: {error}")
    return 2


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        config, jobs = validate_args(args)
    except Exception as exc:
        print(f"参数错误: {exc}", file=sys.stderr)
        return 2

    if args.list:
        print_jobs(jobs)
        return 0

    if args.check:
        return check_jobs(config, jobs)

    if len(jobs) == 1 and not args.interactive and args.index is None:
        job = jobs[0]
    else:
        job = choose_job(jobs, args)

    print("选择轨迹:")
    print(f"  {job.terrain.task_name}/{job.terrain.terrain_id} @ {runner.speed_text(job.speed)}")
    print(f"  文件: {job.key_path}")

    command = build_replay_command(
        config,
        job,
        rate=args.rate,
        paused=args.paused,
        loop=args.loop,
    )
    return run_replay(command, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
