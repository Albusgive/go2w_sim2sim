from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest import mock


TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import generate_data_collection_report as report  # noqa: E402
import generate_data_collection_terrains as terrain_generator  # noqa: E402
import run_data_collection as runner  # noqa: E402


class DataCollectionToolsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name) / "data_collection"
        self.root.mkdir()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def make_terrain(
        self,
        task_name: str,
        terrain_id: str,
        params: dict[str, float] | None = None,
        *,
        collect: bool = True,
    ) -> runner.Terrain:
        directory = self.root / task_name / terrain_id
        directory.mkdir(parents=True)
        metadata = {
            "schema_version": 1,
            "terrain_id": terrain_id,
            "task_name": task_name,
            "collect": collect,
            "description": f"Fixture for {terrain_id}",
            "params": params or {},
            "terminal": {
                "target_x": 4.0,
                "min_base_z": 0.2,
                "max_abs_y": 0.6,
                "stop_duration_s": 1.0,
            },
        }
        (directory / "terrain.json").write_text(json.dumps(metadata), encoding="utf-8")
        (directory / "terrain.xml").write_text(
            '<mujoco model="fixture"><worldbody/></mujoco>\n', encoding="utf-8"
        )
        terrains = [
            terrain
            for terrain in runner.discover_terrains(self.root, include_flat=True)
            if terrain.task_name == task_name and terrain.terrain_id == terrain_id
        ]
        self.assertEqual(len(terrains), 1)
        return terrains[0]

    @staticmethod
    def write_valid_key(path: Path) -> None:
        path.write_text(
            '<mujoco model="result"><keyframe><key name="frame_000000" '
            'time="0" qpos="0" qvel="0"/></keyframe></mujoco>\n',
            encoding="utf-8",
        )

    @staticmethod
    def write_current_status(
        job: runner.CollectionJob, value: dict[str, object]
    ) -> None:
        status = runner._base_status(job)
        status.update(value)
        runner.atomic_write_json(job.status_path, status)

    def test_discovery_speed_matrix_and_names(self) -> None:
        self.make_terrain("flat", "plane-10x10m", collect=False)
        self.make_terrain("single_platform", "platform-h0p10", {"height_m": 0.1})
        self.make_terrain("ditch", "ditch-gap0p30", {"gap_m": 0.3})
        self.make_terrain(
            "double_platform",
            "double-h10p10-dh0p10-gap0p10",
            {"first_height_m": 0.1, "height_delta_m": 0.1, "gap_m": 0.1},
        )

        terrains = runner.discover_terrains(self.root)
        self.assertEqual([terrain.task_name for terrain in terrains], [
            "ditch",
            "double_platform",
            "single_platform",
        ])
        self.assertEqual(len(runner.SPEEDS), 11)
        self.assertEqual(runner.SPEEDS[0], Decimal("0.50"))
        self.assertEqual(runner.SPEEDS[-1], Decimal("1.00"))
        jobs = runner.build_jobs(terrains, runner.SPEEDS)
        self.assertEqual(len(jobs), 33)
        single_first = next(
            job
            for job in jobs
            if job.terrain.task_name == "single_platform" and job.speed == Decimal("0.50")
        )
        self.assertEqual(single_first.key_path.name, "single_platform-cmd_linv_x_0p50.xml")
        self.assertEqual(
            single_first.status_path.name, "single_platform-cmd_linv_x_0p50.json"
        )

    def test_generated_matrix_contract(self) -> None:
        specs = terrain_generator.all_specs()
        counts = {
            task: sum(spec.task_name == task for spec in specs)
            for task in terrain_generator.EXPECTED_COUNTS
        }
        self.assertEqual(
            counts,
            {"flat": 1, "single_platform": 9, "ditch": 7, "double_platform": 117},
        )
        ditch_gaps = sorted(
            spec.params["gap_m"] for spec in specs if spec.task_name == "ditch"
        )
        self.assertEqual(ditch_gaps, [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60])
        ditch_specs = [spec for spec in specs if spec.task_name == "ditch"]
        self.assertTrue(
            all(
                spec.params["floor_box_depth_m"] == 2.0
                and all(
                    geom.pos[2] == -1.0 and geom.size[2] == 1.0
                    for geom in spec.geoms
                )
                for spec in ditch_specs
            )
        )
        double_platforms = [spec for spec in specs if spec.task_name == "double_platform"]
        self.assertTrue(
            all(
                spec.params["height_delta_m"] + spec.params["gap_m"] <= 0.60 + 1e-9
                for spec in double_platforms
            )
        )
        self.assertEqual(terrain_generator.COMMAND["min"], 0.5)
        self.assertEqual(runner.EXPECTED_COLLECTABLE_TERRAINS, 133)
        self.assertEqual(runner.EXPECTED_TRAJECTORIES, 1463)
        self.assertEqual(
            runner.COLLECTION_PROTOCOL_REVISION,
            "mujoco-key-collection-v4-lstm-near-edge-reset-1m",
        )
        self.assertEqual(runner.COLLECTION_POLICY_TYPE, "lstm_sru")
        self.assertEqual(runner.RESET_BEFORE_NEAR_EDGE_M, 1.0)

    def test_resume_recognizes_valid_keys_and_exhausted_failures(self) -> None:
        terrain = self.make_terrain(
            "single_platform", "platform-h0p10", {"height_m": 0.1}
        )
        success_job = runner.CollectionJob(terrain, Decimal("0.50"))
        self.write_valid_key(success_job.key_path)
        self.assertEqual(runner.is_completed(success_job), (True, "existing key XML is valid"))

        failed_job = runner.CollectionJob(terrain, Decimal("0.55"))
        self.write_current_status(
            failed_job,
            {"status": "failed", "attempts": [{"reason": "stalled"}] * 5},
        )
        self.assertEqual(
            runner.is_completed(failed_job),
            (True, "collector already exhausted all attempts"),
        )

        partial_job = runner.CollectionJob(terrain, Decimal("0.65"))
        self.write_current_status(
            partial_job,
            {"status": "failed", "attempts": [{"reason": "timeout"}]},
        )
        self.assertEqual(runner.is_completed(partial_job), (False, ""))
        self.assertEqual(
            runner.is_completed(partial_job, required_attempts=1),
            (True, "collector already exhausted all attempts"),
        )

        invalid_job = runner.CollectionJob(terrain, Decimal("0.60"))
        invalid_job.key_path.write_text("not xml", encoding="utf-8")
        self.write_current_status(
            invalid_job,
            {"status": "failed", "attempts": [{"reason": "stalled"}] * 5},
        )
        self.assertEqual(runner.is_completed(invalid_job), (False, ""))

        stale_job = runner.CollectionJob(terrain, Decimal("0.70"))
        self.write_current_status(
            stale_job,
            {"status": "failed", "attempts": [{"reason": "timeout"}] * 5},
        )
        terrain.metadata_path.write_text(
            terrain.metadata_path.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
        self.assertEqual(runner.is_completed(stale_job), (False, ""))
        self.assertEqual(report.classify_job(stale_job).category, "pending")

    def test_dry_run_filters_and_does_not_write_status(self) -> None:
        terrain = self.make_terrain("ditch", "ditch-gap0p30", {"gap_m": 0.3})
        self.make_terrain("ditch", "ditch-gap0p35", {"gap_m": 0.35})
        completed_job = runner.CollectionJob(terrain, Decimal("0.50"))
        self.write_valid_key(completed_job.key_path)

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = runner.run(
                [
                    "--data-root",
                    str(self.root),
                    "--task",
                    "ditch",
                    "--terrain-id",
                    "ditch-gap0p30",
                    "--speed",
                    "0.50,0.55",
                    "--dry-run",
                ]
            )
        self.assertEqual(exit_code, 0)
        self.assertIn("run 1, resume-skip 1", output.getvalue())
        self.assertIn("linv_x=0.55", output.getvalue())
        self.assertIn("linv_x=0.50", output.getvalue())
        self.assertIn("Heading PID: built-in defaults", output.getvalue())
        self.assertNotIn("ditch-gap0p35", output.getvalue())
        self.assertFalse(completed_job.status_path.exists())

    def test_collector_command_matches_cpp_interface(self) -> None:
        terrain = self.make_terrain("ditch", "ditch-gap0p30", {"gap_m": 0.3})
        job = runner.CollectionJob(terrain, Decimal("0.50"))
        command = runner.collector_command(
            Path("/tmp/mujoco_data_collector"),
            Path("/tmp/policy"),
            job,
            5,
        )
        self.assertEqual(command[0], "/tmp/mujoco_data_collector")
        self.assertEqual(command[command.index("--speed") + 1], "0.50")
        self.assertEqual(command[command.index("--max-attempts") + 1], "5")
        self.assertEqual(
            command[command.index("--policy-type") + 1], "lstm_sru"
        )
        self.assertEqual(
            command[command.index("--reset-before-near-edge") + 1], "1"
        )
        self.assertEqual(command[command.index("--output") + 1], str(job.key_path))
        self.assertEqual(command[command.index("--result") + 1], str(job.status_path))

    def test_execute_job_merges_collector_result_and_validates_key(self) -> None:
        terrain = self.make_terrain("ditch", "ditch-gap0p30", {"gap_m": 0.3})
        job = runner.CollectionJob(terrain, Decimal("0.50"))

        def fake_run(command: list[str], **_: object) -> runner.subprocess.CompletedProcess[str]:
            output_path = Path(command[command.index("--output") + 1])
            result_path = Path(command[command.index("--result") + 1])
            self.write_valid_key(output_path)
            runner.atomic_write_json(
                result_path,
                {
                    "status": "success",
                    "reason": "target_reached",
                    "attempts": [{"attempt": 1, "status": "success", "sim_time_s": 4.0}],
                },
            )
            return runner.subprocess.CompletedProcess(command, 0, '{"status":"success"}\n', "")

        with mock.patch.object(runner.subprocess, "run", side_effect=fake_run):
            result = runner.execute_job(
                job,
                binary=Path("/tmp/mujoco_data_collector"),
                policy=Path("/tmp/policy"),
                max_attempts=5,
                timeout_s=10.0,
                force=False,
            )

        self.assertEqual(result.status, "success")
        status = runner.load_status(job.status_path)
        self.assertIsNotNone(status)
        assert status is not None
        self.assertEqual(status["status"], "success")
        self.assertEqual(status["collector_status"], "success")
        self.assertEqual(len(status["attempts"]), 1)
        self.assertEqual(report.classify_job(job).category, "success")

        lock = runner.acquire_job_lock(job)
        self.assertIsNotNone(lock, "execute_job must release the job lock")
        assert lock is not None
        runner.release_job_lock(lock)

    def test_job_file_lock_is_nonblocking_and_job_scoped(self) -> None:
        terrain = self.make_terrain("ditch", "ditch-gap0p30", {"gap_m": 0.3})
        job = runner.CollectionJob(terrain, Decimal("0.50"))
        other_job = runner.CollectionJob(terrain, Decimal("0.55"))

        first = runner.acquire_job_lock(job)
        self.assertIsNotNone(first)
        assert first is not None
        try:
            self.assertIsNone(runner.acquire_job_lock(job))
            other = runner.acquire_job_lock(other_job)
            self.assertIsNotNone(other)
            assert other is not None
            runner.release_job_lock(other)
        finally:
            runner.release_job_lock(first)

        reacquired = runner.acquire_job_lock(job)
        self.assertIsNotNone(reacquired)
        assert reacquired is not None
        runner.release_job_lock(reacquired)
        self.assertTrue(runner.job_lock_path(job).is_file())

    def test_execute_job_lock_conflict_does_not_touch_status_or_key(self) -> None:
        terrain = self.make_terrain("ditch", "ditch-gap0p30", {"gap_m": 0.3})
        job = runner.CollectionJob(terrain, Decimal("0.50"))
        self.write_current_status(
            job,
            {"status": "failed", "reason": "sentinel", "attempts": []},
        )
        original_status = job.status_path.read_bytes()

        lock = runner.acquire_job_lock(job)
        self.assertIsNotNone(lock)
        assert lock is not None
        try:
            with mock.patch.object(runner.subprocess, "run") as run_process:
                result = runner.execute_job(
                    job,
                    binary=Path("/tmp/mujoco_data_collector"),
                    policy=Path("/tmp/policy"),
                    max_attempts=5,
                    timeout_s=10.0,
                    force=True,
                )
        finally:
            runner.release_job_lock(lock)

        self.assertEqual(result.status, "infrastructure_error")
        self.assertIn("already locked", result.message)
        run_process.assert_not_called()
        self.assertEqual(job.status_path.read_bytes(), original_status)
        self.assertFalse(job.key_path.exists())

    def test_stale_key_temporary_cleanup_is_job_scoped_and_pid_safe(self) -> None:
        terrain = self.make_terrain("ditch", "ditch-gap0p30", {"gap_m": 0.3})
        job = runner.CollectionJob(terrain, Decimal("0.50"))
        other_job = runner.CollectionJob(terrain, Decimal("0.55"))

        stale = terrain.directory / f".collector-999999999-{job.key_path.name}.tmp.xml"
        active = terrain.directory / f".collector-{runner.os.getpid()}-{job.key_path.name}.tmp.xml"
        other = terrain.directory / f".collector-999999999-{other_job.key_path.name}.tmp.xml"
        malformed = terrain.directory / f".collector-not-a-pid-{job.key_path.name}.tmp.xml"
        for path in (stale, active, other, malformed):
            path.write_text("temporary", encoding="utf-8")

        self.assertEqual(runner.cleanup_stale_key_temporaries(job), 1)
        self.assertFalse(stale.exists())
        self.assertTrue(active.exists())
        self.assertTrue(other.exists())
        self.assertTrue(malformed.exists())

    def test_report_math_failure_details_and_all_radar_dimensions(self) -> None:
        single = self.make_terrain(
            "single_platform", "platform-h0p10", {"height_m": 0.1}
        )
        self.make_terrain("ditch", "ditch-gap0p30", {"gap_m": 0.3})
        self.make_terrain(
            "double_platform",
            "double-h10p10-dh0p10-gap0p10",
            {"first_height_m": 0.1, "height_delta_m": 0.1, "gap_m": 0.1},
        )

        success_job = runner.CollectionJob(single, Decimal("0.50"))
        self.write_valid_key(success_job.key_path)
        failed_job = runner.CollectionJob(single, Decimal("0.55"))
        self.write_current_status(
            failed_job,
            {
                "status": "failed",
                "attempts": [{"reason": "stalled", "duration_s": 1.0}] * 5,
                "reason": "stalled after five attempts",
                "duration_s": 5.0,
            },
        )
        broken_job = runner.CollectionJob(single, Decimal("0.60"))
        broken_job.key_path.write_text("<mujoco>", encoding="utf-8")

        output_path = self.root / "collection_report.md"
        assets_dir = self.root / "report_assets"
        summary = report.generate_report(self.root, output_path, assets_dir)

        self.assertEqual(summary.expected, 3 * 11)
        self.assertEqual(summary.success, 1)
        self.assertEqual(summary.failed, 1)
        self.assertEqual(summary.infrastructure_error, 1)
        self.assertEqual(summary.pending, 30)
        self.assertEqual(sum(summary.as_dict().values()), summary.expected)
        markdown = output_path.read_text(encoding="utf-8")
        self.assertIn("stalled after five attempts", markdown)
        self.assertIn("策略失败", markdown)
        self.assertIn("基础设施错误", markdown)
        for spec in report.CHART_SPECS:
            chart = assets_dir / spec.filename
            self.assertTrue(chart.is_file(), spec.filename)
            self.assertIn("<svg", chart.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
