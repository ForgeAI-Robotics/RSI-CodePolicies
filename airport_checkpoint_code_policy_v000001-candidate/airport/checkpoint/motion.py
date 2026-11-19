"""Fixed-base Cartesian primitives over native joint drives and measured depth.

Only robot FK and CAD are read from the existing description. No passenger USD
pose, native semantic label or task-success field enters the motion planner.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .types import Action, SurfacePatch


@dataclass(frozen=True)
class MotionPlan:
    reached: bool
    joints: dict[str, float]
    error_m: float
    orientation_error_rad: float
    reason: str = ""


class FixedBaseMotion:
    """A bounded IK/path primitive; dynamics remain in the native runtime."""

    def __init__(self, description, *, collision_profile: str = "legacy"):
        if collision_profile not in {"legacy", "oriented_v1"}:
            raise ValueError("Unknown measured-cloud collision profile")
        self.collision_profile = collision_profile
        self.last_rejection: dict | None = None
        self.description = description
        self.model = description.model
        self.data = mujoco.MjData(self.model)
        self.names = [
            name
            for name in description.joint_names
            if name
            in {"ankle_pitch_joint", "knee_pitch_joint", "hip_pitch_joint", "torso_yaw_joint"}
            or name.startswith("right_")
            and "claw" not in name
            and "wheel" not in name
        ]
        self.qids = np.array([self.model.joint(n).qposadr[0] for n in self.names])
        self.bounds = np.array([self.model.jnt_range[self.model.joint(n).id] for n in self.names])
        self.home = description.data.qpos[self.qids].copy()
        root = self.model.body("base_link").id
        robot_bodies = set()
        for bid in range(1, self.model.nbody):
            ancestor = bid
            while ancestor and ancestor != root:
                ancestor = int(self.model.body_parentid[ancestor])
            if ancestor == root:
                robot_bodies.add(bid)
        self.collision_geoms = [
            i
            for i in range(self.model.ngeom)
            if self.model.geom_bodyid[i] in robot_bodies and self.model.geom_contype[i] != 0
        ]
        self.obstacles = np.empty((0, 3))
        self.human = np.zeros(0, dtype=bool)
        self.graspable = np.zeros(0, dtype=bool)
        self._grasp_phase = False
        self.last_plan: MotionPlan | None = None
        self._robot_geom_set = set(self.collision_geoms)
        self._tool_tip_local: np.ndarray | None = None
        self._tool_radius = 0.02
        self._tool_contact_patch: SurfacePatch | None = None

    def _fk(self, q: np.ndarray) -> None:
        self.data.qpos[:] = self.description.data.qpos
        self.data.qpos[self.qids] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def _boxes(self):
        ids = self.collision_geoms
        rotations = self.data.geom_xmat[ids].reshape(-1, 3, 3)
        centres = self.data.geom_xpos[ids] + np.einsum(
            "ijk,ik->ij", rotations, self.model.geom_aabb[ids, :3]
        )
        half = np.einsum("ijk,ik->ij", abs(rotations), self.model.geom_aabb[ids, 3:])
        return centres, half

    def set_measured_cloud(
        self,
        points: np.ndarray,
        human: np.ndarray,
        graspable: np.ndarray | None = None,
        *,
        self_labels: np.ndarray | None = None,
        supported_human: np.ndarray | None = None,
    ) -> None:
        points = np.asarray(points, float)
        human = np.asarray(human, bool)
        graspable = (
            np.zeros(len(points), dtype=bool) if graspable is None else np.asarray(graspable, bool)
        )
        if graspable.shape != (len(points),):
            raise ValueError("Expected N measured graspable flags")
        if points.ndim != 2 or points.shape[1] != 3 or human.shape != (len(points),):
            raise ValueError("Expected measured Nx3 cloud and N human flags")
        keep = np.isfinite(points).all(1) & (points[:, 2] > 0.08)
        supported = (
            np.zeros(len(points), dtype=bool)
            if supported_human is None
            else np.asarray(supported_human, bool)
        )
        if supported.shape != (len(points),):
            raise ValueError("Expected N instance-supported human flags")
        human = human | supported
        if self_labels is not None:
            labels = np.asarray(self_labels)
            if labels.shape != (len(points),):
                raise ValueError("Expected N registered own-depth labels")
            # Empirical candidate removal, never unconditional OBB deletion.
            # Foreground, unknown, mesh misses and missing registration survive.
            keep &= ~(
                np.isin(labels, ["self_candidate", "surface_candidate"]) & ~graspable & ~supported
            )
        else:
            # Robot CAD is a conventional depth self-filter. It does not delete
            # points based on native actor identity or target class labels.
            self._fk(self.description.data.qpos[self.qids])
            for gid in self.collision_geoms:
                rotation = self.data.geom_xmat[gid].reshape(3, 3)
                local = (points - self.data.geom_xpos[gid]) @ rotation
                center, extent = self.model.geom_aabb[gid, :3], self.model.geom_aabb[gid, 3:]
                # An oriented local box avoids deleting empty corners of a large
                # world AABB. Preserve weak skin/red candidates in the legacy path.
                robot_volume = np.all(np.abs(local - center) <= extent + 0.002, axis=1)
                keep &= ~(robot_volume & ~human & ~graspable)
        self.obstacles, self.human, self.graspable = points[keep], human[keep], graspable[keep]

    def set_grasp_phase(self, enabled: bool) -> None:
        """Permit current measured paper points against finger CAD only.

        The backend enables this only for PICK_GRASP/PICK_CLOSE and refreshes
        graspable flags from the tracked visible paper component every frame.
        Human points always remain protected even if accidentally dual-labelled.
        """
        self._grasp_phase = bool(enabled)

    def set_measured_tool(
        self, eef, tip, width: float = 0.03, contact_patch: SurfacePatch | None = None
    ) -> None:
        """Attach an observed paper envelope to encoder FK, or clear with tip=None.

        Only the distal 20 mm may contact the explicitly measured sampling patch.
        This capsule includes a 5 mm bend allowance, not a calibrated flexible
        material model. Actual flex/contact safety remains native-audited.
        """
        if tip is None:
            self._tool_tip_local = None
            self._tool_contact_patch = None
            return
        eef, tip = np.asarray(eef, float), np.asarray(tip, float)
        if (
            eef.shape != (3,)
            or tip.shape != (3,)
            or not np.isfinite([*eef, *tip, width]).all()
            or not 0 < width <= 0.10
        ):
            raise ValueError("Invalid measured tool envelope")
        if not 0.005 <= np.linalg.norm(tip - eef) <= 0.30:
            raise ValueError("Measured tool endpoint outside supported grasp envelope")
        self._fk(self.description.data.qpos[self.qids])
        rotation = self.data.site_xmat[self.description.eef_id].reshape(3, 3)
        self._tool_tip_local = rotation.T @ (tip - eef)
        self._tool_radius = width / 2 + 0.005
        self._tool_contact_patch = contact_patch

    def _tool_clear(self) -> tuple[bool, str]:
        if self._tool_tip_local is None:
            return True, ""
        eef = self.data.site_xpos[self.description.eef_id]
        rotation = self.data.site_xmat[self.description.eef_id].reshape(3, 3)
        axis = rotation @ self._tool_tip_local
        length = float(np.linalg.norm(axis))
        along = (self.obstacles - eef) @ axis / length**2
        closest = eef + np.clip(along, 0, 1)[:, None] * axis
        collision = np.linalg.norm(self.obstacles - closest, axis=1) <= self._tool_radius
        patch = self._tool_contact_patch
        if patch is not None:
            delta = self.obstacles - patch.center
            plane = np.abs(delta @ np.asarray(patch.normal))
            local_patch = np.linalg.norm(delta, axis=1) <= 0.10
            distal = along >= max(0.0, 1.0 - 0.02 / length)
            permitted = distal & local_patch & (plane <= 0.006)
            collision &= ~permitted
        if np.any(collision):
            return False, "measured obstacle intersects held-paper envelope"
        return True, ""

    def _clear(self) -> tuple[bool, str]:
        self.last_rejection = None
        if not len(self.obstacles):
            self.last_rejection = {"kind": "missing_cloud", "profile": self.collision_profile}
            return False, "no measured collision cloud"
        # Read-only CAD collision; does not advance MuJoCo dynamics. Respect
        # authored masks/exclusions and ignore all old fixture/person contacts.
        if self.model.nflex:
            mujoco.mj_flex(self.model, self.data)
        mujoco.mj_collision(self.model, self.data)
        for contact in self.data.contact:
            a, b = map(int, contact.geom)
            if a in self._robot_geom_set and b in self._robot_geom_set and contact.dist < -0.001:
                self.last_rejection = {
                    "kind": "robot_self_collision",
                    "profile": self.collision_profile,
                    "geom_ids": [a, b],
                    "distance_m": float(contact.dist),
                }
                return False, "robot self-collision in enabled CAD pair"
        centres, half = self._boxes()
        margin = np.where(self.human, 0.025, 0.003)
        for gid, centre, extent in zip(self.collision_geoms, centres, half, strict=True):
            gap = np.abs(self.obstacles - centre) - extent
            rotation = self.data.geom_xmat[gid].reshape(3, 3)
            local_center = self.model.geom_aabb[gid, :3]
            # Fixed numerical enclosure pad for imported float mesh vertices;
            # not a contact allowance or a hardware uncertainty certification.
            numeric_pad = 1e-6 if self.collision_profile == "oriented_v1" else 0.0
            local_half = self.model.geom_aabb[gid, 3:] + numeric_pad
            if self.collision_profile == "oriented_v1":
                # Broad phase must enclose the locally margin-expanded OBB.
                # Adding just m on each world axis can miss rotated corners.
                padding = (margin[:, None] + numeric_pad) * np.sum(abs(rotation), axis=1)
                hits = np.all(gap <= padding, axis=1)
                candidates = np.flatnonzero(hits)
                local = (self.obstacles[candidates] - self.data.geom_xpos[gid]) @ rotation
                local_gap = abs(local - local_center) - local_half
                hits[candidates] = np.all(local_gap <= margin[candidates, None], axis=1)
            else:
                # Preserve the baseline's exact world-axis margin semantics.
                hits = np.all(gap <= margin[:, None], axis=1)
            body = self.model.body(int(self.model.geom_bodyid[gid])).name
            if self._grasp_phase and body in {"right_claw_link", "right_claw_link_g"}:
                hits &= ~(self.graspable & ~self.human)
            if np.any(hits):
                index = int(np.flatnonzero(hits)[0])
                local_point = (self.obstacles[index] - self.data.geom_xpos[gid]) @ rotation
                self.last_rejection = {
                    "kind": "measured_robot_envelope",
                    "profile": self.collision_profile,
                    "geom_id": int(gid),
                    "geom_name": self.model.geom(gid).name,
                    "body_name": body,
                    "point_index": index,
                    "point_work_m": self.obstacles[index].tolist(),
                    "point_local_m": local_point.tolist(),
                    "human": bool(self.human[index]),
                    "graspable": bool(self.graspable[index])
                    if index < len(self.graspable)
                    else False,
                    "margin_m": float(margin[index]),
                    "numeric_padding_m": numeric_pad,
                    "local_center_m": local_center.tolist(),
                    "local_half_m": local_half.tolist(),
                    "local_axis_gap_m": (abs(local_point - local_center) - local_half).tolist(),
                    "world_aabb_axis_gap_m": gap[index].tolist(),
                    "geom_rotation": rotation.tolist(),
                    "joint_positions": dict(
                        zip(self.names, map(float, self.data.qpos[self.qids]), strict=True)
                    ),
                    "hit_count_for_geom": int(np.sum(hits)),
                    "scope": "conservative box rejection; not exact mesh distance",
                }
                return False, "measured obstacle intersects conservative robot envelope"
        clear, reason = self._tool_clear()
        if not clear:
            self.last_rejection = {"kind": "held_paper_envelope", "profile": self.collision_profile}
        return clear, reason

    def plan(self, action: Action) -> MotionPlan:
        self.last_rejection = None
        if action.kind not in {"move_eef", "retreat"} or action.target is None:
            raise ValueError("Only bounded Cartesian actions can be planned")
        target = np.asarray(action.target, float)
        if target.shape != (3,) or not np.isfinite(target).all() or not 0.45 < target[2] < 1.8:
            return MotionPlan(False, {}, float("inf"), float("inf"), "invalid Cartesian target")
        start = self.description.data.qpos[self.qids].copy()
        self._fk(start)
        rpy = (
            action.rpy
            if action.rpy is not None
            else Rotation.from_matrix(
                self.data.site_xmat[self.description.eef_id].reshape(3, 3)
            ).as_euler("xyz")
        )
        rpy = np.asarray(rpy, float)
        if rpy.shape != (3,) or not np.isfinite(rpy).all():
            return MotionPlan(
                False, {}, float("inf"), float("inf"), "invalid Cartesian orientation"
            )
        desired = Rotation.from_euler("xyz", rpy)

        def residual(q):
            self._fk(q)
            actual = Rotation.from_matrix(
                self.data.site_xmat[self.description.eef_id].reshape(3, 3)
            )
            return np.r_[
                3 * (self.data.site_xpos[self.description.eef_id] - target),
                0.30 * (desired * actual.inv()).as_rotvec(),
                0.001 * (q - start),
                0.0005 * (q[:3] - self.home[:3]),
            ]

        low, high = self.bounds[:, 0] + 1e-6, self.bounds[:, 1] - 1e-6
        solution = least_squares(
            residual,
            np.clip(start, low, high),
            bounds=(low, high),
            max_nfev=80,
            ftol=1e-7,
            xtol=1e-7,
            gtol=1e-7,
        )
        residual(solution.x)
        error = float(np.linalg.norm(self.data.site_xpos[self.description.eef_id] - target))
        angular = float(
            np.linalg.norm(
                (
                    desired
                    * Rotation.from_matrix(
                        self.data.site_xmat[self.description.eef_id].reshape(3, 3)
                    ).inv()
                ).as_rotvec()
            )
        )
        reason = ""
        if error > 0.008 or angular > 0.10:
            reason = "fixed-base IK unreachable"
        else:
            count = max(2, int(np.max(np.abs(solution.x - start)) / 0.015) + 2)
            for fraction in np.linspace(0, 1, count):
                self._fk(start + fraction * (solution.x - start))
                valid, reason = self._clear()
                if not valid:
                    break
        plan = MotionPlan(
            not bool(reason),
            dict(zip(self.names, map(float, solution.x), strict=True)),
            error,
            angular,
            reason,
        )
        self.last_plan = plan
        return plan
