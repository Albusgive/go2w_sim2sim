from __future__ import annotations

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

import retry_failed_ditch_fallback as fallback  # noqa: E402
import run_data_collection as runner  # noqa: E402


class DitchFallbackTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name) / "data_collection"
        self.root.mkdir()
        self.lstm_policy = self.make_policy("vtm_lstm_sru", b"lstm")
        self.gru_policy = self.make_policy("vtm_gru_sru", b"gru")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def make_policy(self, name: str, marker: bytes) -> Path:
        policy = Path(self.temporary_directory.name) / "policy" / name
        policy.mkdir(parents=True)
        for index, filename in enumerate(fallback.POLICY_FINGERPRINT_FILES):
            (policy / filename).write_bytes(marker + bytes([index]))
        return policy

    def make_terrain(self, terrain_id: str = "ditch-gap0p30") -> runner.Terrain:
        directory = self.root / "ditch" / terrain_id
        directory.mkdir(parents=True)
        metadata = {
            "schema_version": 1,
            "terrain_id": terrain_id,
            "task_name": "ditch",
            "collect": True,
            "params": {"gap_m": 0.3, "near_edge_x_m": 3.0},
            "terminal": {
                "target_x": 4.3,
                "min_base_z": 0.2,
                "max_abs_y": 0.6,
                "stop_duration_s": 1.0,
            },
        }
        (directory / "terrain.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        (directory / "terrain.xml").write_text(
            '<mujoco model="ditch"><worldbody/></mujoco>\n', encoding="utf-8"
        )
        terrains = runner.discover_terrains(self.root)
        match = [terrain for terrain in terrains if terrain.terrain_id == terrain_id]
        self.assertEqual(len(match), 1)
        return match[0]

    @staticmethod
    def write_valid_key(path: Path) -> None:
        path.write_text(
            '<mujoco model="result"><keyframe><key name="frame_000000" '
            'time="0" qpos="0" qvel="0"/></keyframe></mujoco>\n',
            encoding="utf-8",
        )

    @staticmethod
    def write_status(job: runner.CollectionJob, **values: object) -> None:
        status = runner._base_status(job)
        status.update(values)
        runner.atomic_write_json(job.status_path, status)

    def stages(self, reset_distance: float = 1.0) -> tuple[fallback.FallbackStage, ...]:
        return (
            fallback.FallbackStage(
                "lstm_mid_reset",
                "lstm_sru",
                "vtm_lstm_sru",
                self.lstm_policy,
                reset_distance,
            ),
            fallback.FallbackStage(
                "gru_baseline",
                "gru_sru",
                "vtm_gru_sru",
                self.gru_policy,
                None,
            ),
            fallback.FallbackStage(
                "gru_mid_reset",
                "gru_sru",
                "vtm_gru_sru",
                self.gru_policy,
                reset_distance,
            ),
        )

    def test_stage_command_sets_policy_type_and_optional_mid_reset(self) -> None:
        job = runner.CollectionJob(self.make_terrain(), Decimal("0.55"))
        result_path = self.root / "scratch.json"

        reset_command = fallback.stage_command(
            binary=Path("/tmp/mujoco_data_collector"),
            stage=self.stages()[0],
            job=job,
            result_path=result_path,
            max_attempts=5,
        )
        self.assertEqual(reset_command[reset_command.index("--speed") + 1], "0.55")
        self.assertEqual(
            reset_command[reset_command.index("--policy-type") + 1], "lstm_sru"
        )
        self.assertEqual(
            reset_command[reset_command.index("--policy") + 1],
            str(self.lstm_policy),
        )
        self.assertEqual(
            reset_command[reset_command.index("--result") + 1], str(result_path)
        )
        self.assertEqual(
            reset_command[reset_command.index("--reset-before-near-edge") + 1],
            "1",
        )

        baseline_command = fallback.stage_command(
            binary=Path("/tmp/mujoco_data_collector"),
            stage=self.stages()[1],
            job=job,
            result_path=result_path,
            max_attempts=5,
        )
        self.assertEqual(
            baseline_command[baseline_command.index("--policy-type") + 1], "gru_sru"
        )
        self.assertNotIn("--reset-before-near-edge", baseline_command)

    def test_plan_fingerprint_tracks_attempts_reset_and_policy_contents(self) -> None:
        stages = self.stages()
        original = fallback.fallback_plan_fingerprint(stages, 5)

        self.assertNotEqual(
            original, fallback.fallback_plan_fingerprint(stages, 4)
        )
        self.assertNotEqual(
            original,
            fallback.fallback_plan_fingerprint(self.stages(reset_distance=0.9), 5),
        )

        actor = self.gru_policy / "student_actor.onnx"
        actor.write_bytes(actor.read_bytes() + b"-changed")
        self.assertNotEqual(
            original, fallback.fallback_plan_fingerprint(stages, 5)
        )

    def test_selection_never_schedules_a_valid_key(self) -> None:
        terrain = self.make_terrain()
        valid = runner.CollectionJob(terrain, Decimal("0.50"))
        missing = runner.CollectionJob(terrain, Decimal("0.55"))
        exhausted_fallback = runner.CollectionJob(terrain, Decimal("0.60"))
        invalid_key = runner.CollectionJob(terrain, Decimal("0.65"))
        lost_success_key = runner.CollectionJob(terrain, Decimal("0.70"))
        plan_fingerprint = "fixture-plan"

        self.write_valid_key(valid.key_path)
        self.write_status(valid, status="success", attempts=[{"attempt": 1}])
        self.write_status(
            missing,
            status="failed",
            reason="baseline exhausted",
            attempts=[{"attempt": index + 1} for index in range(5)],
        )
        self.write_status(
            exhausted_fallback,
            status="failed",
            collection_mode="ditch_fallback",
            fallback_plan={"fingerprint": plan_fingerprint},
            fallback_plan_status="exhausted",
            attempts=[{"attempt": index + 1} for index in range(15)],
        )
        invalid_key.key_path.write_text("not XML", encoding="utf-8")
        self.write_status(invalid_key, status="failed", attempts=[])
        self.write_status(
            lost_success_key,
            status="success",
            collection_mode="ditch_fallback",
            fallback_plan={"fingerprint": plan_fingerprint},
            fallback_plan_status="success",
            attempts=[{"attempt": 1, "status": "success"}],
        )

        jobs = [valid, missing, exhausted_fallback, invalid_key, lost_success_key]
        pending, skipped, errors = fallback.select_fallback_jobs(
            jobs, plan_fingerprint=plan_fingerprint
        )
        self.assertEqual(pending, [missing, lost_success_key])
        self.assertEqual(skipped, [valid, exhausted_fallback])
        self.assertEqual(len(errors), 1)
        self.assertIn("key XML cannot be parsed", errors[0])

        forced, forced_skipped, errors = fallback.select_fallback_jobs(
            jobs, plan_fingerprint=plan_fingerprint, force=True
        )
        self.assertEqual(
            forced, [missing, exhausted_fallback, invalid_key, lost_success_key]
        )
        self.assertEqual(forced_skipped, [valid])
        self.assertEqual(errors, [])

    def test_execution_short_circuits_after_gru_baseline_success(self) -> None:
        job = runner.CollectionJob(self.make_terrain(), Decimal("0.50"))
        baseline_attempts = [{"attempt": index + 1, "reason": "fell"} for index in range(5)]
        self.write_status(
            job,
            status="failed",
            reason="baseline exhausted",
            attempts=baseline_attempts,
            finished_at="2026-07-14T00:00:00+00:00",
        )
        stages = self.stages()
        fingerprint = fallback.fallback_plan_fingerprint(stages, 2)
        plan = fallback.fallback_plan(stages, 2, fingerprint)
        commands: list[list[str]] = []

        def fake_run(command: list[str], **_: object) -> runner.subprocess.CompletedProcess[str]:
            commands.append(command)
            result_path = Path(command[command.index("--result") + 1])
            if len(commands) == 1:
                runner.atomic_write_json(
                    result_path,
                    {
                        "status": "failed",
                        "reason": "max_attempts_exhausted",
                        "attempts": [
                            {"attempt": 1, "reason": "fell"},
                            {"attempt": 2, "reason": "fell"},
                        ],
                    },
                )
                return runner.subprocess.CompletedProcess(command, 2, "", "")

            self.write_valid_key(job.key_path)
            runner.atomic_write_json(
                result_path,
                {
                    "status": "success",
                    "reason": "terminal_reached",
                    "attempts": [{"attempt": 1, "status": "success"}],
                    "frames": 120,
                },
            )
            return runner.subprocess.CompletedProcess(command, 0, "", "")

        with mock.patch.object(fallback.subprocess, "run", side_effect=fake_run):
            result = fallback.execute_fallback_job(
                job,
                binary=Path("/tmp/mujoco_data_collector"),
                stages=stages,
                plan=plan,
                max_attempts=2,
                timeout_s=10.0,
            )

        self.assertEqual(result.status, "success")
        self.assertEqual(result.winning_stage, "gru_baseline")
        self.assertEqual(len(commands), 2)
        self.assertEqual(commands[0][commands[0].index("--policy-type") + 1], "lstm_sru")
        self.assertIn("--reset-before-near-edge", commands[0])
        self.assertEqual(commands[1][commands[1].index("--policy-type") + 1], "gru_sru")
        self.assertNotIn("--reset-before-near-edge", commands[1])

        status = runner.load_status(job.status_path)
        self.assertIsNotNone(status)
        assert status is not None
        self.assertTrue(runner.status_matches_job(job, status))
        self.assertEqual(status["fallback_plan_status"], "success")
        self.assertEqual(status["winning_stage"], "gru_baseline")
        self.assertEqual(status["previous_result"]["reason"], "baseline exhausted")
        self.assertEqual(status["previous_result"]["attempt_count"], 5)
        self.assertEqual([stage["status"] for stage in status["stages"]], ["failed", "success"])
        self.assertEqual(status["attempt_count"], 3)
        self.assertEqual(
            [attempt["stage"] for attempt in status["attempts"]],
            ["lstm_mid_reset", "lstm_mid_reset", "gru_baseline"],
        )

    def test_resume_keeps_completed_stage_and_starts_at_next_stage(self) -> None:
        job = runner.CollectionJob(self.make_terrain(), Decimal("0.70"))
        stages = self.stages()
        fingerprint = fallback.fallback_plan_fingerprint(stages, 2)
        plan = fallback.fallback_plan(stages, 2, fingerprint)
        completed_stage = {
            "id": "lstm_mid_reset",
            "status": "failed",
            "reason": "max_attempts_exhausted",
            "policy_type": "lstm_sru",
            "policy_name": "vtm_lstm_sru",
            "mid_reset": True,
            "attempt_count": 2,
            "attempts": [{"reason": "fell"}, {"reason": "fell"}],
        }
        self.write_status(
            job,
            status="infrastructure_error",
            collection_mode="ditch_fallback",
            fallback_plan=plan,
            fallback_plan_status="infrastructure_error",
            previous_result={
                "status": "failed",
                "reason": "baseline exhausted",
                "attempt_count": 5,
                "attempts": [],
                "finished_at": None,
            },
            stages=[completed_stage],
            attempts=[],
        )
        commands: list[list[str]] = []

        def fake_run(command: list[str], **_: object) -> runner.subprocess.CompletedProcess[str]:
            commands.append(command)
            self.write_valid_key(job.key_path)
            result_path = Path(command[command.index("--result") + 1])
            runner.atomic_write_json(
                result_path,
                {"status": "success", "attempts": [{"status": "success"}]},
            )
            return runner.subprocess.CompletedProcess(command, 0, "", "")

        with mock.patch.object(fallback.subprocess, "run", side_effect=fake_run):
            result = fallback.execute_fallback_job(
                job,
                binary=Path("/tmp/mujoco_data_collector"),
                stages=stages,
                plan=plan,
                max_attempts=2,
                timeout_s=10.0,
            )

        self.assertEqual(result.winning_stage, "gru_baseline")
        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0][commands[0].index("--policy-type") + 1], "gru_sru")
        self.assertNotIn("--reset-before-near-edge", commands[0])
        status = runner.load_status(job.status_path)
        self.assertIsNotNone(status)
        assert status is not None
        self.assertEqual([stage["id"] for stage in status["stages"]], [
            "lstm_mid_reset",
            "gru_baseline",
        ])
        self.assertEqual(status["previous_result"]["reason"], "baseline exhausted")


if __name__ == "__main__":
    unittest.main()
