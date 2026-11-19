"""Explicit integration entry; no connection, reset, or motion on import."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .interface import RobotPort
from .policy import policy

POLICY_SHA256 = "2569babba60a3f8be6afd1db02d6b8048773fd76abcd18453de532376173188f"


def verify_policy() -> None:
    actual = hashlib.sha256(Path(__file__).with_name("policy.py").read_bytes()).hexdigest()
    if actual != POLICY_SHA256:
        raise ValueError("Policy changed: frozen simulation evidence no longer applies")


def require_commissioned(profile: dict) -> None:
    required = (
        "joint_order_verified",
        "joint_sign_zero_and_limits_verified",
        "initial_posture_verified",
        "fk_ik_and_tool_frame_verified",
        "three_cameras_intrinsic_extrinsic_verified",
        "rectified_rgb_and_freshness_verified",
        "table_and_can_dimensions_verified",
        "gripper_aperture_mapping_verified",
        "tracking_and_emergency_stop_verified",
        "low_speed_dry_run_verified",
        "supported_release_and_retreat_verified",
    )
    if (
        profile.get("status") != "COMMISSIONED"
        or profile.get("control_hz") != 20
        or not profile.get("driver")
        or not profile.get("robot_serial")
        or any(profile.get(key) is not True for key in required)
    ):
        raise ValueError("Hardware commissioning record is incomplete; no motion dispatched")


def run(robot: RobotPort, profile: dict) -> dict:
    """Caller owns commissioning/readiness checks and robot lifecycle.

    This wrapper is new handoff code, covered by offline tests only. The 12/12
    simulation certificate covers policy.py with the original PyRUA adapter.
    A stop failure is propagated rather than hidden.
    """
    require_commissioned(profile)
    try:
        verify_policy()
        result = policy(robot)
    except BaseException:
        robot.safe_stop()
        raise
    if result.get("status") != "visually_complete":
        robot.safe_stop()
    return result
