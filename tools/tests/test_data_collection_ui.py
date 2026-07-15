from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest import mock


TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import data_collection_ui as ui  # noqa: E402
import run_data_collection as runner  # noqa: E402


class DataCollectionUITest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.temp = Path(self.temporary_directory.name)
        self.data_root = self.temp / "data_collection"
        self.data_root.mkdir()
        self.policy = self.temp / "policy"
        self.policy.mkdir()
        self.runner_script = self._executable("run_data_collection.py")
        self.collector = self._executable("mujoco_data_collector")
        self.replayer = self._executable("mujoco_key_replayer")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _executable(self, name: str) -> Path:
        path = self.temp / name
        path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        path.chmod(0o755)
        return path

    def make_terrain(
        self,
        task_name: str,
        terrain_id: str,
        *,
        description: str = "fixture terrain",
        params: dict[str, float] | None = None,
    ) -> runner.Terrain:
        directory = self.data_root / task_name / terrain_id
        directory.mkdir(parents=True)
        metadata = {
            "schema_version": 1,
            "task_name": task_name,
            "terrain_id": terrain_id,
            "collect": True,
            "description": description,
            "params": params or {},
        }
        (directory / "terrain.json").write_text(json.dumps(metadata), encoding="utf-8")
        (directory / "terrain.xml").write_text(
            '<mujoco model="fixture"><worldbody/></mujoco>\n', encoding="utf-8"
        )
        return next(
            terrain
            for terrain in runner.discover_terrains(self.data_root)
            if terrain.task_name == task_name and terrain.terrain_id == terrain_id
        )

    @staticmethod
    def write_valid_key(path: Path) -> None:
        path.write_text(
            '<mujoco model="key"><keyframe><key name="frame_000000" '
            'time="0" qpos="0" qvel="0"/></keyframe></mujoco>\n',
            encoding="utf-8",
        )

    @staticmethod
    def write_current_status(job: runner.CollectionJob, **values: object) -> None:
        status = runner._base_status(job)
        status.update(values)
        runner.atomic_write_json(job.status_path, status)

    def config(self) -> ui.AppConfig:
        return ui.AppConfig(
            root=self.temp,
            data_root=self.data_root,
            runner_script=self.runner_script,
            collector_binary=self.collector,
            replay_binary=self.replayer,
            policy=self.policy,
            python_executable=Path(sys.executable),
            max_attempts=5,
        )

    def test_catalog_cascade_details_and_directory_status(self) -> None:
        self.make_terrain("single_platform", "platform-h0p20", params={"height_m": 0.2})
        self.make_terrain("single_platform", "platform-h0p10", params={"height_m": 0.1})
        self.make_terrain("ditch", "ditch-gap0p30", params={"gap_m": 0.3})

        catalog = ui.TrajectoryCatalog(self.data_root)
        self.assertEqual(catalog.task_names(), ("ditch", "single_platform"))
        self.assertEqual(
            catalog.terrain_ids("single_platform"),
            ("platform-h0p10", "platform-h0p20"),
        )
        job = catalog.job("single_platform", "platform-h0p10", "0.50")
        self.assertEqual(job.speed, Decimal("0.50"))
        details = catalog.terrain_details(job.terrain)
        self.assertIn("height_m: 0.1", details)
        self.assertIn("heading PID: built-in defaults", details)
        self.assertIn("cross_track_gain: 1.25", details)
        self.assertEqual(catalog.status(job).code, "pending")

        self.write_current_status(
            job,
            status="failed",
            reason="max_attempts_exhausted",
            attempts=[{"reason": "stalled"}] * 5,
        )
        failed = catalog.status(job)
        self.assertEqual(failed.code, "failed")
        self.assertIn("max_attempts_exhausted", failed.detail)
        self.assertFalse(failed.replayable)

        self.write_valid_key(job.key_path)
        success = catalog.status(job)
        self.assertEqual(success.code, "success")
        self.assertTrue(success.replayable)

    def test_invalid_and_stale_statuses_are_distinguished(self) -> None:
        terrain = self.make_terrain("ditch", "ditch-gap0p30")
        invalid_job = runner.CollectionJob(terrain, Decimal("0.50"))
        invalid_job.key_path.write_text("not XML", encoding="utf-8")
        invalid = ui.TrajectoryCatalog.status(invalid_job)
        self.assertEqual(invalid.code, "invalid_key")
        self.assertFalse(invalid.replayable)

        stale_job = runner.CollectionJob(terrain, Decimal("0.55"))
        self.write_current_status(stale_job, status="running")
        terrain.metadata_path.write_text(
            terrain.metadata_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
        )
        stale = ui.TrajectoryCatalog.status(stale_job)
        self.assertEqual(stale.code, "stale")

        self.write_valid_key(stale_job.key_path)
        stale_success = ui.TrajectoryCatalog.status(stale_job)
        self.assertEqual(stale_success.code, "stale_success")
        self.assertFalse(stale_success.replayable)

    def test_collection_replay_and_ndjson_commands(self) -> None:
        terrain = self.make_terrain("double_platform", "double-h10p10-h20p20-gap0p10")
        job = runner.CollectionJob(terrain, Decimal("0.65"))
        config = self.config()

        command = ui.build_collection_command(config, job)
        self.assertEqual(command[:2], [str(Path(sys.executable)), str(self.runner_script)])
        self.assertEqual(command[command.index("--task") + 1], "double_platform")
        self.assertEqual(
            command[command.index("--terrain-id") + 1],
            "double-h10p10-h20p20-gap0p10",
        )
        self.assertEqual(command[command.index("--speed") + 1], "0.65")
        self.assertEqual(command[command.index("--workers") + 1], "1")
        self.assertIn("--force", command)
        self.assertIn("--collector-arg=--visualize", command)

        replay = ui.build_replay_command(config, job, rate=0.5, paused=True, loop=True)
        self.assertEqual(replay[0], str(self.replayer))
        self.assertEqual(replay[replay.index("--trajectory") + 1], str(job.key_path))
        self.assertEqual(replay[replay.index("--metadata") + 1], str(terrain.metadata_path))
        self.assertEqual(replay[replay.index("--rate") + 1], "0.5")
        self.assertIn("--paused", replay)
        self.assertIn("--loop", replay)

        payload = json.loads(ui.replay_message("seek", progress=0.375))
        self.assertEqual(payload, {"command": "seek", "progress": 0.375})

    def test_check_validates_without_launching_tk(self) -> None:
        self.make_terrain("single_platform", "platform-h0p10")
        args = [
            "--data-root",
            str(self.data_root),
            "--runner-script",
            str(self.runner_script),
            "--collector-binary",
            str(self.collector),
            "--replay-binary",
            str(self.replayer),
            "--policy",
            str(self.policy),
            "--python",
            sys.executable,
            "--check",
        ]
        output = io.StringIO()
        with mock.patch.object(ui, "launch_ui", side_effect=AssertionError("Tk was launched")):
            with contextlib.redirect_stdout(output):
                result = ui.run(args)
        self.assertEqual(result, 0)
        self.assertIn("OK terrains: 1 collectable", output.getvalue())

        self.collector.chmod(0o644)
        _messages, errors = ui.check_configuration(self.config())
        self.assertTrue(any("collector binary is not executable" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
