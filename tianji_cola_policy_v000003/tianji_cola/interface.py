"""Policy-facing port. Implement transport and calibrated model outside the policy."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class ActionResult:
    commanded_action: Sequence[float]


class RobotPort(Protocol):
    """Units: metres, joint radians, XYZ Euler degrees, normalized gripper.

    Matrices map optical/tool coordinates into chassis_base. Image arrays are
    rectified RGB uint8, (480, 640, 3). Never expose simulator truth to policy.
    Driver-internal encoder/force monitoring is separate from this port.
    """

    def initial_action(self) -> Sequence[float]: ...

    def image(self, camera: str) -> np.ndarray: ...

    def action_sequence(self, actions: list[list[float]]) -> ActionResult:
        """Validate entire batch; execute every row at 20Hz; block until complete.

        Keep right channels fixed, enforce commissioned limits and tracking,
        abort on emergency stop/tracking fault. ACK is last completed command,
        never echo an unexecuted batch target. Do not teleport to final row.
        """
        ...

    def solve_pose(
        self, arm: str, xyz: list[float], rpy: list[float], reference_action: list[float]
    ) -> dict[str, Any]: ...

    def solve_pose_continuous(
        self, arm: str, xyz: list[float], rpy: list[float], reference_action: list[float]
    ) -> dict[str, Any]: ...

    def fk(self, arm: str, reference_action: list[float]) -> dict[str, Any]: ...

    def camera_model(self, camera: str) -> dict[str, Any]: ...

    def project(self, camera: str, row: float, col: float, height: float) -> dict[str, Any]: ...

    def rim(
        self, camera: str, expected_xyz: list[float], plane_height: float, radius: float
    ) -> dict[str, Any]: ...

    def gripper_geometry(self, arm: str) -> dict[str, Any]: ...

    def safe_stop(self) -> None:
        """Stop/hold via commissioned controller; do not blindly release load."""
        ...
