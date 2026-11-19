"""Native gravity feedforward added to Isaac's unchanged implicit PD drive."""

from __future__ import annotations

import numpy as np


class NativeGravityFeedforward:
    def __init__(self, robot, action_type, configured_max_efforts: np.ndarray) -> None:
        self.robot = robot
        self.action_type = action_type
        self.view = robot._articulation_view
        self.names = list(robot.dof_names)
        self.maximum = np.asarray(configured_max_efforts, dtype=float)
        if (
            self.maximum.shape != (len(self.names),)
            or not np.isfinite(self.maximum).all()
            or not np.all(self.maximum > 0)
        ):
            raise ValueError("Every native DOF requires its configured positive effort limit")
        if not self.view._physics_view.shared_metatype.fixed_base:
            raise ValueError("Gravity feedforward requires the checkpoint's fixed articulation")
        self.max_observed_fraction = 0.0

    def apply(self, position_targets: np.ndarray, enabled: bool = True) -> np.ndarray:
        targets = np.asarray(position_targets, dtype=float)
        if targets.shape != self.maximum.shape or not np.isfinite(targets).all():
            raise ValueError("Position targets must be finite and match native DOF order")
        if enabled:
            if not self.view.is_physics_handle_valid():
                raise RuntimeError("Native articulation handle is invalid")
            values = self.view.get_generalized_gravity_forces(clone=True)
            if values is None or np.asarray(values).shape != (1, len(self.names)):
                raise RuntimeError("Native gravity compensation returned an invalid DOF vector")
            efforts = np.asarray(values, dtype=float)[0]
            if not np.isfinite(efforts).all():
                raise RuntimeError("Native gravity compensation is nonfinite")
            fraction = float(np.max(np.abs(efforts) / self.maximum))
            self.max_observed_fraction = max(self.max_observed_fraction, fraction)
            if fraction > 1.0:
                raise RuntimeError("Gravity feedforward exceeds configured drive effort limits")
        else:
            efforts = np.zeros_like(targets)
        # Tensor API documents actuation forces as independent of the implicit
        # PD controller. Do not switch to effort-only mode or zero drive gains.
        self.robot.apply_action(self.action_type(joint_positions=targets, joint_efforts=efforts))
        return efforts.copy()
