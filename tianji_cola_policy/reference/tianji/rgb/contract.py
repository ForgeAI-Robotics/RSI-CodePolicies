"""Public embodiment constants and the sole 16-dimensional actuator contract."""

from __future__ import annotations

import numpy as np

from tianji.description import Description
from tianji.task_spec import HEAD_CENTER_TASK

CAMERAS = ("head", "left_wrist", "right_wrist")
ACTION_NAMES = tuple(
    [f"Joint{i}_L" for i in range(1, 8)]
    + ["left_gripper"]
    + [f"Joint{i}_R" for i in range(1, 8)]
    + ["right_gripper"]
)
LOWER = np.radians([-170, -120, -170, -145, -170, -60, -90])
UPPER = np.radians([170, 120, 170, 60, 170, 60, 90])


def validate_action(action) -> np.ndarray:
    value = np.asarray(action, dtype=float)
    if value.shape != (16,) or not np.isfinite(value).all():
        raise ValueError("Action must contain exactly 16 finite numbers")
    for offset in (0, 8):
        if np.any(value[offset : offset + 7] < LOWER) or np.any(value[offset : offset + 7] > UPPER):
            raise ValueError("Joint target exceeds manufacturer limits")
        if not 0 <= value[offset + 7] <= 1:
            raise ValueError("Gripper opening must lie in [0, 1]")
    return value


def initial_action(task_variant="two_can") -> np.ndarray:
    if task_variant == HEAD_CENTER_TASK:
        from tianji.sample_calibration import initial_action as sample_initial_action

        return validate_action(sample_initial_action())
    q = Description().horizontal_home()
    return np.array([q[n] if n in q else 1.0 for n in ACTION_NAMES])


def command_angles(action, task_variant="two_can", camera_profile="urdf") -> dict[str, float]:
    q = Description().horizontal_home()
    if task_variant == HEAD_CENTER_TASK:
        from tianji.sample_calibration import joint_angles

        q.update(joint_angles(validate_action(action), camera_profile=camera_profile))
        return q
    q.update(
        {
            n: float(v)
            for n, v in zip(ACTION_NAMES, validate_action(action), strict=True)
            if "gripper" not in n
        }
    )
    return q
