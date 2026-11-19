"""Sensor-only discrete momentum balance, with explicit window timing."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ForceEstimate:
    # Parent-joint force minus change of distal momentum: reaction on environment.
    # External force applied by the environment to the robot has opposite sign.
    reaction_world_n: tuple[float, float, float]
    window_start: float
    window_end: float
    stamp: float


class MomentumForceObserver:
    """Subtract distal momentum change from integrated compensated wrist force.

    Force at t is the average native force for the just-completed (t-dt,t] step.
    First sample establishes a momentum endpoint; N further steps fill a window.
    Inputs contain no object identity, native contact report, or target pose.
    """

    def __init__(self, dt: float = 0.001, window_s: float = 0.05):
        if not np.isfinite([dt, window_s]).all() or dt <= 0 or window_s < dt:
            raise ValueError("Invalid force observer timestep/window")
        self.dt = float(dt)
        self.samples = round(window_s / dt)
        if not np.isclose(self.samples * dt, window_s, rtol=0, atol=1e-10):
            raise ValueError("Force window must contain an integer number of physics steps")
        self.reset()

    def reset(self) -> None:
        self._history = deque(maxlen=self.samples + 1)
        self._impulse = np.zeros(3)
        self._key = None
        self.estimate = None

    def update(self, stamp: float, compensated_world_force, momentum_world, calibration_key):
        force = np.asarray(compensated_world_force, dtype=float)
        momentum = np.asarray(momentum_world, dtype=float)
        if (
            calibration_key is None
            or not np.isfinite(stamp)
            or force.shape != (3,)
            or momentum.shape != (3,)
            or not np.isfinite([force, momentum]).all()
        ):
            self.reset()
            return None
        if calibration_key != self._key or (
            self._history
            and not np.isclose(stamp - self._history[-1][0], self.dt, rtol=0, atol=1e-9)
        ):
            self.reset()
        self._key = calibration_key
        if self._history:
            self._impulse += force * self.dt
        self._history.append((float(stamp), self._impulse.copy(), momentum.copy()))
        self.estimate = None
        if len(self._history) == self.samples + 1:
            start, impulse, previous_momentum = self._history[0]
            reaction = (self._impulse - impulse - momentum + previous_momentum) / (stamp - start)
            self.estimate = ForceEstimate(
                tuple(map(float, reaction)), start, float(stamp), float((stamp + start) / 2)
            )
        return self.estimate


class DistalMomentumKinematics:
    """Linear momentum from immutable CAD mass and encoder-only Jacobians.

    Optional payload is a calibrated point mass rigidly attached to the wrist.
    Flexible payload deformation is not measured or reconstructed here.
    """

    def __init__(self, model, body_ids, joint_names, wrist_id):
        self.model = model
        self.body_ids = tuple(body_ids)
        self.dof_ids = np.array([int(model.joint(n).dofadr[0]) for n in joint_names])
        self.wrist_id = wrist_id
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))
        self._qdot = np.zeros(model.nv)

    def momentum(self, data, velocities, payload_mass=0.0, payload_com_local=None):
        import mujoco

        velocity = np.asarray(velocities, dtype=float)
        if velocity.shape != self.dof_ids.shape or not np.isfinite(velocity).all():
            raise ValueError("Invalid native joint velocity vector")
        if not np.isfinite(payload_mass) or payload_mass < 0:
            raise ValueError("Invalid calibrated payload mass")
        self._qdot[:] = 0.0
        self._qdot[self.dof_ids] = velocity
        momentum = np.zeros(3)
        for body in self.body_ids:
            mujoco.mj_jacBodyCom(self.model, data, self._jacp, self._jacr, body)
            momentum += float(self.model.body_mass[body]) * (self._jacp @ self._qdot)
        if payload_mass:
            local = np.asarray(payload_com_local, dtype=float)
            if local.shape != (3,) or not np.isfinite(local).all():
                raise ValueError("Payload requires a finite calibrated local COM")
            point = data.xpos[self.wrist_id] + data.xmat[self.wrist_id].reshape(3, 3) @ local
            mujoco.mj_jac(self.model, data, self._jacp, self._jacr, point, self.wrist_id)
            momentum += payload_mass * (self._jacp @ self._qdot)
        return momentum
