from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import calibrate_straight_pid as calibration  # noqa: E402
import run_data_collection as runner  # noqa: E402


class PidCalibrationTest(unittest.TestCase):
    @staticmethod
    def config_document(**overrides: object) -> dict[str, object]:
        document: dict[str, object] = {
            "schema_version": 1,
            "controller": "path_heading_pid",
            "control_period_s": 0.02,
            **runner.DEFAULT_PATH_HEADING_PID.as_dict(),
            "calibration": {"fixture": True},
        }
        document.update(overrides)
        return document

    def test_candidate_grid_and_cli_are_stable(self) -> None:
        candidates = tuple(calibration.candidates())
        self.assertEqual(len(candidates), 36)
        default = next(
            candidate
            for candidate in candidates
            if candidate.cross_track_gain == 1.25
            and candidate.kp == 1.2
            and candidate.ki == 0.05
            and candidate.kd == 0.10
        )
        arguments = default.collector_args()
        self.assertEqual(arguments[arguments.index("--pid-kp") + 1], "1.2")
        self.assertEqual(
            arguments[arguments.index("--pid-yaw-cmd-limit") + 1], "0.5"
        )
        parsed = runner.parse_pid_config(
            self.config_document(**default.as_dict()), source="test calibrator output"
        )
        self.assertEqual(parsed.as_dict(), default.as_dict())

    def test_runner_loads_defaults_and_strict_calibrator_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary)
            config, source = runner.load_pid_config(data_root)
            self.assertEqual(config, runner.DEFAULT_PATH_HEADING_PID)
            self.assertIsNone(source)

            path = data_root / runner.PID_CONFIG_FILENAME
            path.write_text(
                json.dumps(self.config_document(kp=1.6)), encoding="utf-8"
            )
            config, source = runner.load_pid_config(data_root)
            self.assertEqual(config.kp, 1.6)
            self.assertEqual(source, path)

        valid = self.config_document()
        invalid_documents = []
        missing = dict(valid)
        missing.pop("kd")
        invalid_documents.append(missing)
        invalid_documents.extend(
            [
                {**valid, "unknown": 1},
                {**valid, "schema_version": True},
                {**valid, "controller": "other"},
                {**valid, "control_period_s": 0.01},
                {**valid, "kp": -0.1},
                {**valid, "heading_limit_rad": 0.0},
                {**valid, "derivative_alpha": 1.1},
                {**valid, "ki": math.nan},
                {**valid, "calibration": []},
            ]
        )
        for document in invalid_documents:
            with self.subTest(document=document):
                with self.assertRaises(ValueError):
                    runner.parse_pid_config(document)

    def test_pid_config_changes_fingerprint_and_collector_cli(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary)
            directory = data_root / "ditch" / "ditch-gap0p30"
            directory.mkdir(parents=True)
            (directory / "terrain.json").write_text(
                json.dumps(
                    {
                        "task_name": "ditch",
                        "terrain_id": "ditch-gap0p30",
                        "collect": True,
                    }
                ),
                encoding="utf-8",
            )
            (directory / "terrain.xml").write_text(
                '<mujoco model="fixture"><worldbody/></mujoco>\n', encoding="utf-8"
            )
            terrain = runner.discover_terrains(data_root)[0]
            job = runner.CollectionJob(terrain, runner.SPEEDS[0])
            default_fingerprint = runner.job_input_fingerprint(job)

            config_path = data_root / runner.PID_CONFIG_FILENAME
            document = self.config_document(kp=1.6, calibration={"run": 1})
            config_path.write_text(json.dumps(document), encoding="utf-8")
            configured_fingerprint = runner.job_input_fingerprint(job)
            self.assertNotEqual(default_fingerprint, configured_fingerprint)

            command = runner.collector_command(
                Path("/tmp/collector"), Path("/tmp/policy"), job, 5
            )
            expected_arguments = {
                "--pid-cross-track-gain": "1.25",
                "--pid-kp": "1.6",
                "--pid-ki": "0.05",
                "--pid-kd": "0.1",
                "--pid-heading-limit": "0.35",
                "--pid-yaw-cmd-limit": "0.5",
                "--pid-integral-limit": "0.5",
                "--pid-derivative-alpha": "0.2",
            }
            for flag, expected in expected_arguments.items():
                self.assertEqual(command[command.index(flag) + 1], expected)

            document["calibration"] = {"run": 2}
            config_path.write_text(json.dumps(document), encoding="utf-8")
            self.assertNotEqual(
                configured_fingerprint, runner.job_input_fingerprint(job)
            )

    def test_trajectory_metrics_use_initial_heading_frame(self) -> None:
        yaw = math.pi / 2.0
        quaternion = f"{math.cos(yaw / 2.0)} 0 0 {math.sin(yaw / 2.0)}"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "key.xml"
            path.write_text(
                "<mujoco><keyframe>"
                f'<key qpos="0 0 0 {quaternion}"/>'
                f'<key qpos="0.10 2 0 {quaternion}"/>'
                "</keyframe></mujoco>\n",
                encoding="utf-8",
            )
            lateral, heading = calibration.trajectory_metrics(path)
        self.assertAlmostEqual(lateral, 0.10, places=8)
        self.assertAlmostEqual(heading, 0.0, places=8)


if __name__ == "__main__":
    unittest.main()
