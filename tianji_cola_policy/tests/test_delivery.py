"""Offline checks only: no simulator, hardware, SDK, network or language model."""

import importlib
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from check_delivery import check
from tianji_cola.policy import RGBPolicy
from tianji_cola.run import require_commissioned, run, verify_policy

ROOT = Path(__file__).resolve().parents[1]


def commissioned_record():
    profile = json.loads((ROOT / "config/hardware_profile.template.json").read_text())
    profile.update({k: True for k in profile if k.endswith("_verified")})
    profile.update(status="COMMISSIONED", driver="offline_fake", robot_serial="TEST_ONLY")
    return profile


class DeliveryTests(unittest.TestCase):
    def test_native_evidence_and_hashes(self):
        self.assertEqual(check()["native_simulation_successes"], "12/12")
        verify_policy()

    def test_uncommissioned_template_never_calls_robot(self):
        robot = Mock()
        profile = json.loads((ROOT / "config/hardware_profile.template.json").read_text())
        with self.assertRaisesRegex(ValueError, "incomplete"):
            run(robot, profile)
        self.assertEqual(robot.mock_calls, [])

    def test_missing_readiness_flag_rejected(self):
        profile = commissioned_record()
        del profile["gripper_aperture_mapping_verified"]
        with self.assertRaises(ValueError):
            require_commissioned(profile)

    def test_grip_preserves_all_arm_and_right_channels(self):
        command = np.linspace(-0.2, 0.3, 16)
        command[7] = command[15] = 1
        rows = []

        def send(actions):
            rows.extend(actions)
            return SimpleNamespace(commanded_action=actions[-1])

        controller = RGBPolicy(
            SimpleNamespace(initial_action=lambda: command.copy(), action_sequence=send)
        )
        controller.grip(0)
        controller.grip(1)
        np.testing.assert_allclose(np.asarray(rows)[:, 8:], np.tile(command[8:], (len(rows), 1)))
        np.testing.assert_allclose(np.asarray(rows)[:, :7], np.tile(command[:7], (len(rows), 1)))

    def test_no_visible_red_can_is_rejected(self):
        controller = RGBPolicy(
            SimpleNamespace(
                initial_action=lambda: np.zeros(16),
                image=lambda _: np.zeros((480, 640, 3), np.uint8),
            )
        )
        with self.assertRaisesRegex(ValueError, "No supported red can"):
            controller.red()

    def test_stuck_stops_driver_and_preserves_reason(self):
        robot = Mock()
        module = importlib.import_module("tianji_cola.run")
        with patch.object(module, "policy", return_value={"status": "stuck", "reason": "IK"}):
            result = run(robot, commissioned_record())
        robot.safe_stop.assert_called_once_with()
        self.assertEqual(result["reason"], "IK")

    def test_keyboard_interrupt_stops_driver(self):
        robot = Mock()
        module = importlib.import_module("tianji_cola.run")
        with patch.object(module, "policy", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                run(robot, commissioned_record())
        robot.safe_stop.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
