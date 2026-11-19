"""Optional bounded transport references; native PD/FF still owns dynamics.

The trajectory advances only inside execute(). Host observation/RPC work does
not run physics asynchronously. Limits below concern the commanded FK path;
measured tracking errors are checked separately and can stop execution.
The scalar trapezoid/triangle has continuous velocity and piecewise constant
acceleration. It does not impose a jerk limit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .motion import MotionPlan
from .types import CONTINUOUS_TRANSPORT_PHASES


@dataclass(frozen=True)
class TransportLimits:
    speed: float
    acceleration: float
    angular_speed: float
    angular_acceleration: float
    joint_speed: float = 0.4
    joint_acceleration: float = 0.8

    @classmethod
    def from_action(cls, action):
        values = [
            action.max_speed_m_s,
            action.max_acceleration_m_s2,
            action.max_angular_speed_rad_s,
            action.max_angular_acceleration_rad_s2,
        ]
        if any(x is None for x in values) or not np.isfinite(values).all():
            raise ValueError("Continuous transport requires finite explicit limits")
        if any(x <= 0 for x in values) or any(
            x > cap for x, cap in zip(values, (0.10, 0.20, 0.40, 0.80), strict=True)
        ):
            raise ValueError("Continuous transport limits exceed the conservative profile")
        return cls(*values)


def action_key(action) -> tuple:
    if (
        action.kind != "move_eef"
        or action.reason not in CONTINUOUS_TRANSPORT_PHASES
        or not isinstance(action.trajectory_id, str)
        or not action.trajectory_id
        or action.target is None
        or action.rpy is None
        or action.gripper_close is not None
        or not 0 < action.duration_s <= 0.05
    ):
        raise ValueError("Unsupported continuous transport action")
    limits = TransportLimits.from_action(action)
    return (action.trajectory_id, action.reason, tuple(action.target), tuple(action.rpy), limits)


def _pose(motion, q):
    motion._fk(q)
    sid = motion.description.eef_id
    return motion.data.site_xpos[sid].copy(), motion.data.site_xmat[sid].reshape(3, 3).copy()


def _kinematic_peaks(q, positions, rotations, dt):
    def peaks(values):
        velocity = np.diff(values, axis=0) / dt
        acceleration = (
            np.diff(
                np.concatenate(
                    [np.zeros_like(velocity[:1]), velocity, np.zeros_like(velocity[:1])]
                ),
                axis=0,
            )
            / dt
        )
        return float(np.max(np.linalg.norm(velocity, axis=1))), float(
            np.max(np.linalg.norm(acceleration, axis=1))
        )

    speed, acceleration = peaks(positions)
    rotation = Rotation.from_matrix(rotations)
    angular_velocity = (rotation[1:] * rotation[:-1].inv()).as_rotvec() / dt
    angular_acceleration = (
        np.diff(np.concatenate([np.zeros((1, 3)), angular_velocity, np.zeros((1, 3))]), axis=0) / dt
    )
    joint_velocity = np.diff(q, axis=0) / dt
    joint_acceleration = (
        np.diff(
            np.concatenate([np.zeros_like(q[:1]), joint_velocity, np.zeros_like(q[:1])]), axis=0
        )
        / dt
    )
    return {
        "speed_m_s": speed,
        "acceleration_m_s2": acceleration,
        "angular_speed_rad_s": float(np.max(np.linalg.norm(angular_velocity, axis=1))),
        "angular_acceleration_rad_s2": float(np.max(np.linalg.norm(angular_acceleration, axis=1))),
        "joint_speed_rad_s": float(np.max(abs(joint_velocity))),
        "joint_acceleration_rad_s2": float(np.max(abs(joint_acceleration))),
    }


def _scalar_trapezoid(distance, speed, acceleration):
    """Unit-path acceleration, ramp time and duration before time dilation."""
    if distance == 0:
        return 4 / 0.05**2, 0.025, 0.05
    scalar_speed = speed / distance
    scalar_acceleration = acceleration / distance
    ramp = min(scalar_speed / scalar_acceleration, np.sqrt(1 / scalar_acceleration))
    peak_speed = scalar_acceleration * ramp
    cruise = max(0.0, (1 - scalar_acceleration * ramp**2) / peak_speed)
    return scalar_acceleration, ramp, 2 * ramp + cruise


def _trapezoid_progress(times, acceleration, ramp, duration):
    """Integrate the continuous-velocity scalar law, with zero endpoint speed."""
    peak_speed = acceleration * ramp
    return np.where(
        times < ramp,
        0.5 * acceleration * times**2,
        np.where(
            times <= duration - ramp,
            0.5 * acceleration * ramp**2 + peak_speed * (times - ramp),
            1 - 0.5 * acceleration * (duration - times) ** 2,
        ),
    )


@dataclass
class TransportTrajectory:
    key: tuple
    samples: np.ndarray
    positions: np.ndarray
    rotations: np.ndarray
    dt: float
    peaks: dict
    endpoint_plan: MotionPlan
    index: int = 0

    @classmethod
    def build(cls, motion, action, plan, dt):
        key = action_key(action)
        if not np.isfinite(dt) or not 0 < dt <= 0.001:
            raise ValueError("Continuous transport requires physics steps at most 1ms")
        limits = key[-1]
        start = motion.description.data.qpos[motion.qids].copy()
        goal = np.array([plan.joints[name] for name in motion.names])
        if not np.isfinite([start, goal]).all():
            raise ValueError("Nonfinite transport joint path")
        delta = goal - start
        acceleration, ramp, base_duration = _scalar_trapezoid(
            float(np.max(abs(delta))), limits.joint_speed, limits.joint_acceleration
        )
        duration = max(0.05, base_duration)
        # Iterative time dilation; actual FK at every commanded physics point,
        # not a joint-distance claim about Cartesian speed.
        for _ in range(8):
            ticks = max(2, int(np.ceil(duration / dt)))
            if ticks * dt > 30:
                raise ValueError("Transport reference exceeds the bounded 30s construction")
            # Map the whole quantized interval onto the same scalar law.
            # Quantization and later dilation can only slow this initial law.
            times = np.linspace(0, base_duration, ticks + 1)
            blend = _trapezoid_progress(times, acceleration, ramp, base_duration)
            blend[0], blend[-1] = 0.0, 1.0
            q = start + blend[:, None] * delta
            q[0], q[-1] = start, goal
            poses = [_pose(motion, row) for row in q]
            positions = np.array([p for p, _ in poses])
            rotations = np.array([r for _, r in poses])
            peaks = _kinematic_peaks(q, positions, rotations, dt)
            ratios = [
                peaks["speed_m_s"] / limits.speed,
                np.sqrt(peaks["acceleration_m_s2"] / limits.acceleration),
                peaks["angular_speed_rad_s"] / limits.angular_speed,
                np.sqrt(peaks["angular_acceleration_rad_s2"] / limits.angular_acceleration),
                peaks["joint_speed_rad_s"] / limits.joint_speed,
                np.sqrt(peaks["joint_acceleration_rad_s2"] / limits.joint_acceleration),
            ]
            if max(ratios) <= 1:
                return cls(key, q, positions, rotations, dt, peaks, plan)
            duration = ticks * dt * max(ratios) * 1.02
        raise ValueError("Unable to bound the FK transport reference")

    def end_index(self, duration):
        ticks = round(duration / self.dt)
        if ticks < 1 or not np.isclose(ticks * self.dt, duration, atol=1e-10, rtol=0):
            raise ValueError("Transport feedback duration must be a whole number of physics steps")
        return min(self.index + ticks, len(self.samples) - 1)

    def tracking_error(self, motion):
        actual = motion.description.data.qpos[motion.qids]
        p, rotation = _pose(motion, actual)
        return {
            "joint_error_rad": float(np.max(abs(actual - self.samples[self.index]))),
            "position_error_m": float(np.linalg.norm(p - self.positions[self.index])),
            "orientation_error_rad": float(
                np.linalg.norm(
                    (
                        Rotation.from_matrix(rotation)
                        * Rotation.from_matrix(self.rotations[self.index]).inv()
                    ).as_rotvec()
                )
            ),
        }

    def precheck(self, motion, duration):
        """Recheck current state, full endpoint path and executable reference.

        This uses the new observation cloud. No previous collision approval is
        retained even when the IK endpoint and smooth reference are reused.
        """
        current = motion.description.data.qpos[motion.qids].copy()
        for target in (self.samples[-1], self.samples[self.index]):
            count = max(2, int(np.max(abs(target - current)) / 0.015) + 2)
            for alpha in np.linspace(0, 1, count):
                q = current + alpha * (target - current)
                if np.any(q < motion.bounds[:, 0]) or np.any(q > motion.bounds[:, 1]):
                    return "transport joint bound violation"
                motion._fk(q)
                clear, reason = motion._clear()
                if not clear:
                    return reason
        end = self.end_index(duration)
        # Every command in the next <=50ms interval is checked, not only IK end.
        for q in self.samples[self.index : end + 1]:
            motion._fk(q)
            clear, reason = motion._clear()
            if not clear:
                return reason
        return ""
