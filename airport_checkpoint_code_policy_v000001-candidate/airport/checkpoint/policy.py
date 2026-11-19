"""Deterministic feedback policy. Completion is not a native success judgement."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .perception import is_lateral
from .tool_alignment import (
    bounded_orientation,
    eef_for_measured_tip,
    orientation_distance,
    sampling_orientation,
)
from .types import (
    CONTINUOUS_TRANSPORT_PHASES,
    Action,
    Observation,
    PersonTrack,
    SurfacePatch,
    Vec3,
)


@dataclass(frozen=True)
class PolicyConfig:
    batch_size: int = 3
    dwell_s: float = 3.0
    stage_timeout_s: float = 30.0
    sensor_max_age_s: float = 0.25
    stationary_speed_m_s: float = 0.025
    base_translation_tolerance_m: float = 0.002
    base_yaw_tolerance_rad: float = 0.01
    reach_tolerance_m: float = 0.006
    min_patch_confidence: float = 0.8
    standoff_m: float = 0.04
    extraction_distance_m: float = 0.10
    extraction_clearance_m: float = 0.01
    extracted_visible_length_m: float = 0.11
    stroke_m: float = 0.06
    min_contact_force_n: float = 0.0005
    max_contact_force_n: float = 1.5
    target_contact_force_n: float = 0.001
    min_contact_time_s: float = 0.25
    min_contact_span_m: float = 0.05
    max_force_adjustment_m: float = 0.004
    min_grip_force_n: float = 0.10
    max_grip_aperture_m: float = 0.025
    contact_advance_m: float = 0.002
    max_contact_advance_m: float = 0.045
    transport_speed_m_s: float = 0.10
    sampling_speed_m_s: float = 0.02
    action_duration_s: float = 0.05
    rotation_speed_rad_s: float = 0.40
    orientation_tolerance_rad: float = 0.03
    outward_tilt_rad: float = 0.85
    transport_acceleration_m_s2: float = 0.20
    rotation_acceleration_rad_s2: float = 0.80
    control_profile: Literal["legacy", "reactive_v1"] = "legacy"
    collision_profile: Literal["legacy", "oriented_v1"] = "legacy"
    vision_profile: Literal["legacy", "supply_bounds_v1"] = "legacy"

    def __post_init__(self) -> None:
        if self.control_profile not in {"legacy", "reactive_v1"}:
            raise ValueError("Unsupported control profile")
        if self.collision_profile not in {"legacy", "oriented_v1"}:
            raise ValueError("Unsupported collision profile")
        if self.vision_profile not in {"legacy", "supply_bounds_v1"}:
            raise ValueError("Unsupported vision profile")
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        for key, value in vars(self).items():
            if key in {"batch_size", "control_profile", "collision_profile", "vision_profile"}:
                continue
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if self.control_profile == "reactive_v1" and self.action_duration_s > 0.05:
            raise ValueError("Reactive transport requires feedback windows <=50 ms")
        if not self.min_contact_force_n <= self.target_contact_force_n < self.max_contact_force_n:
            raise ValueError("Invalid force window")
        if self.min_patch_confidence > 1:
            raise ValueError("Invalid confidence threshold")


class CheckpointPolicy:
    """One paper per batch: extract, sample each bag/forearm, deposit in box.

    Dwell and person association require a real tracking pipeline. Grasp checks
    use calibrated gripper feedback plus observed paper motion; the policy does
    not read simulator held/contact/success booleans. A deployment must supply a
    verified tool orientation, payload/force calibration and collision checker.
    """

    def __init__(self, config: PolicyConfig | None = None):
        self.config = config or PolicyConfig()
        self.state = (
            "PREPARE_PAPER" if self.config.control_profile == "reactive_v1" else "WAIT_PERSON"
        )
        self.completed_person_ids: list[str] = []
        self.failure_reason: str | None = None
        self._base: np.ndarray | None = None
        self._last_sequence = -1
        self._last_stamp = -math.inf
        self._stage_started: float | None = None
        self._stable_since: float | None = None
        self._candidate_id: str | None = None
        self._person_id: str | None = None
        self._person_anchor: np.ndarray | None = None
        self._target: Vec3 | None = None
        self._target_rpy: Vec3 | None = None
        self._normal: Vec3 = (0.0, 0.0, 1.0)
        self._tangent: Vec3 = (1.0, 0.0, 0.0)
        self._paper_initial: np.ndarray | None = None
        self._extraction_eef_start: np.ndarray | None = None
        self._has_paper = False
        self._contact_advance = 0.0
        self._stroke_origin: np.ndarray | None = None
        self._observed_stroke = 0.0
        self._contact_started = 0.0
        self._force_adjustment = 0.0
        self._surface_anchor: np.ndarray | None = None
        self._trajectory_generation = 0

    @property
    def finished(self) -> bool:
        return self.state in {"DONE", "STOPPED"}

    @property
    def protocol_complete(self) -> bool:
        return self.state == "DONE"

    def _stop(self, reason: str) -> Action:
        self.failure_reason = reason
        self.state = "STOPPED"
        return Action("stop", reason=reason)

    def _enter(self, state: str, obs: Observation, target: np.ndarray | Vec3 | None = None) -> None:
        self.state = state
        self._stage_started = obs.stamp
        self._target = tuple(float(v) for v in target) if target is not None else None
        self._trajectory_generation += 1

    def _patch_valid(self, patch: SurfacePatch | None, obs: Observation) -> bool:
        if patch is None:
            return False
        values = (*patch.center, *patch.normal, *patch.tangent, patch.confidence, patch.stamp)
        return bool(
            np.isfinite(values).all()
            and self.config.min_patch_confidence <= patch.confidence <= 1
            and 0 <= obs.stamp - patch.stamp <= self.config.sensor_max_age_s
            and abs(np.linalg.norm(patch.normal) - 1) < 0.02
            and abs(np.linalg.norm(patch.tangent) - 1) < 0.02
            and abs(np.dot(patch.normal, patch.tangent)) < 0.02
        )

    def _guard(self, obs: Observation) -> str | None:
        if (
            not math.isfinite(obs.stamp)
            or obs.sequence <= self._last_sequence
            or obs.stamp <= self._last_stamp
        ):
            return "nonmonotonic observation"
        if (
            not obs.joint_positions
            or not np.isfinite(
                [*obs.base_pose, *obs.eef_position, *obs.eef_rpy, *obs.joint_positions.values()]
            ).all()
        ):
            return "invalid robot telemetry"
        for name in ("rgbd", "joints", "gripper", "force"):
            sensor = obs.sensors.get(name)
            if (
                sensor is None
                or not sensor.calibrated
                or not sensor.source
                or sensor.kind not in {"measured", "synthetic"}
            ):
                return f"missing or uncalibrated sensor: {name}"
            if (
                not math.isfinite(sensor.stamp)
                or not 0 <= obs.stamp - sensor.stamp <= self.config.sensor_max_age_s
            ):
                return f"stale sensor: {name}"
        values = (obs.gripper_aperture_m, obs.gripper_force_n, obs.wrist_force_n)
        if any(value is None or not math.isfinite(value) for value in values):
            return "missing force/gripper feedback"
        if obs.gripper_aperture_m < 0 or obs.gripper_force_n < 0:
            return "invalid gripper feedback"
        if self._base is None:
            self._base = np.asarray(obs.base_pose)
        delta = np.asarray(obs.base_pose) - self._base
        yaw_error = abs(math.atan2(math.sin(delta[3]), math.cos(delta[3])))
        if (
            np.linalg.norm(delta[:3]) > self.config.base_translation_tolerance_m
            or yaw_error > self.config.base_yaw_tolerance_rad
        ):
            return "fixed base moved"
        if abs(obs.wrist_force_n) > self.config.max_contact_force_n:
            return "force limit exceeded"
        if (
            self._stage_started is not None
            and obs.stamp - self._stage_started > self.config.stage_timeout_s
        ):
            return "stage timeout: " + self.state
        if obs.paper_tip is not None and not np.isfinite(obs.paper_tip).all():
            return "invalid paper observation"
        return None

    def _move(self, obs: Observation, kind: str = "move_eef") -> Action:
        if (
            self.config.control_profile == "reactive_v1"
            and kind == "move_eef"
            and self.state in CONTINUOUS_TRANSPORT_PHASES
        ):
            # Freeze an otherwise unspecified orientation once for this goal.
            # The backend interpolates; this remains a short feedback window.
            if self._target_rpy is None:
                self._target_rpy = tuple(float(v) for v in obs.eef_rpy)
            return Action(
                "move_eef",
                target=self._target,
                rpy=self._target_rpy,
                duration_s=self.config.action_duration_s,
                reason=self.state,
                trajectory_id=f"checkpoint:{self._trajectory_generation}:{self.state}",
                max_speed_m_s=self.config.transport_speed_m_s,
                max_acceleration_m_s2=self.config.transport_acceleration_m_s2,
                max_angular_speed_rad_s=self.config.rotation_speed_rad_s,
                max_angular_acceleration_rad_s2=self.config.rotation_acceleration_rad_s2,
            )
        delta = np.asarray(self._target) - obs.eef_position
        speed = (
            self.config.sampling_speed_m_s
            if self.state.startswith(("CONTACT_", "SCAN_", "RETRACT_", "RETREAT_"))
            else self.config.transport_speed_m_s
        )
        distance = float(np.linalg.norm(delta))
        max_step = speed * self.config.action_duration_s
        if distance > max_step:
            delta *= max_step / distance
        target = tuple(np.asarray(obs.eef_position) + delta)
        return Action(
            kind,
            target=target,
            rpy=bounded_orientation(
                obs.eef_rpy,
                self._target_rpy or obs.eef_rpy,
                self.config.rotation_speed_rad_s * self.config.action_duration_s,
            ),
            duration_s=self.config.action_duration_s,
            reason=self.state,
        )

    def _reached(self, obs: Observation) -> bool:
        tolerance = (
            min(self.config.reach_tolerance_m, 0.0002)
            if self.state.startswith("CONTACT_")
            else self.config.reach_tolerance_m
        )
        return (
            self._target is not None
            and np.linalg.norm(np.asarray(obs.eef_position) - self._target) <= tolerance
            and (
                self._target_rpy is None
                or orientation_distance(obs.eef_rpy, self._target_rpy)
                <= self.config.orientation_tolerance_rad
            )
        )

    def _gripped(self, obs: Observation) -> bool:
        return bool(
            obs.gripper_force_n >= self.config.min_grip_force_n
            and obs.gripper_aperture_m <= self.config.max_grip_aperture_m
        )

    def _person_valid(self, person: PersonTrack, obs: Observation) -> bool:
        return bool(
            np.isfinite([*person.position, *person.forward, *person.velocity, person.stamp]).all()
            and 0 <= obs.stamp - person.stamp <= self.config.sensor_max_age_s
            and is_lateral(obs.base_pose[:3], person.position, person.forward)
        )

    def _start_surface(self, obs: Observation, person: PersonTrack, region: str) -> Action:
        patch = person.bag if region == "BAG" else person.forearm
        if not self._patch_valid(patch, obs) or obs.paper_tip is None:
            return self._stop("missing measured " + region.lower() + " surface/paper tip")
        if np.linalg.norm(person.velocity) > self.config.stationary_speed_m_s:
            return self._stop("passenger moved before approach")
        self._normal, self._tangent = patch.normal, patch.tangent
        self._surface_anchor = np.asarray(patch.center)
        # Rotate the calibrated paper axes and compensate the measured tip lever arm.
        desired_tip = np.asarray(patch.center) + self.config.standoff_m * np.asarray(patch.normal)
        self._target_rpy = sampling_orientation(patch, self.config.outward_tilt_rad)
        target = eef_for_measured_tip(
            obs.eef_position, obs.eef_rpy, obs.paper_tip, tuple(desired_tip), self._target_rpy
        )
        self._contact_advance = 0.0
        self._enter("APPROACH_" + region, obs, target)
        return self._move(obs)

    def _start_pick(self, obs: Observation) -> Action:
        if (
            not all(self._patch_valid(p, obs) for p in (obs.paper_grasp, obs.supply_exit))
            or obs.paper_tip is None
        ):
            return self._stop("missing measured paper/slot geometry")
        self._normal = obs.supply_exit.normal
        self._target_rpy = (0.0, 0.0, 0.0)  # Calibrated vertical supply slot.
        self._paper_initial = np.asarray(obs.paper_tip)
        self._enter(
            "PICK_APPROACH",
            obs,
            np.asarray(obs.paper_grasp.center) + self.config.standoff_m * np.asarray(self._normal),
        )
        return Action("grip", gripper_close=False, reason="open before extraction")

    def step(self, obs: Observation) -> Action:
        if self.finished:
            return Action("hold" if self.protocol_complete else "stop", reason=self.state)
        problem = self._guard(obs)
        if problem:
            return self._stop(problem)
        self._last_stamp, self._last_sequence = obs.stamp, obs.sequence
        if self._stage_started is None:
            self._stage_started = obs.stamp
        person = next((p for p in obs.people if p.track_id == self._person_id), None)
        active_contact = self.state != "RETREAT_PERSON" and any(
            self.state.startswith(prefix)
            for prefix in ("APPROACH_", "CONTACT_", "SCAN_", "RETRACT_")
        )
        if active_contact:
            if person is None or not self._person_valid(person, obs):
                return self._stop("person tracking/side condition lost")
            moved = np.linalg.norm(person.velocity) > self.config.stationary_speed_m_s
            moved |= (
                self._person_anchor is not None
                and np.linalg.norm(np.asarray(person.position) - self._person_anchor) > 0.025
            )
            patch = person.bag if self.state.endswith("BAG") else person.forearm
            if not self._patch_valid(patch, obs):
                return self._stop("active sampling surface observation lost")
            if self._surface_anchor is not None:
                moved |= np.linalg.norm(np.asarray(patch.center) - self._surface_anchor) > 0.015
                moved |= np.dot(patch.normal, self._normal) < 0.98
            if moved:
                self.failure_reason = "person moved; retreat then stop"
                self._enter(
                    "RETREAT_PERSON",
                    obs,
                    np.asarray(obs.eef_position) + 0.06 * np.asarray(self._normal),
                )
                return self._move(obs, "retreat")
        if (
            self._has_paper
            and self.state not in {"RELEASE", "VERIFY_DEPOSIT", "RETREAT_PERSON"}
            and not self._gripped(obs)
        ):
            return self._stop("grip feedback lost")
        if self.state == "RETREAT_PERSON":
            if self._reached(obs):
                return self._stop(self.failure_reason or "person moved")
            return self._move(obs, "retreat")
        if self.state == "PREPARE_PAPER":
            return self._start_pick(obs)
        if self.state == "WAIT_PERSON":
            if (
                self.config.control_profile == "reactive_v1"
                and self._has_paper
                and obs.paper_tip is None
            ):
                return self._stop("held paper observation lost while waiting")
            eligible = [
                p
                for p in obs.people
                if p.track_id not in self.completed_person_ids
                and p.in_queue
                and self._person_valid(p, obs)
                and np.linalg.norm(p.velocity) <= self.config.stationary_speed_m_s
            ]
            candidate = sorted(eligible, key=lambda p: p.track_id)[0] if eligible else None
            if candidate is None:
                self._candidate_id, self._stable_since = None, None
                return Action("hold", reason="waiting for stationary lateral passenger")
            if candidate.track_id != self._candidate_id:
                self._candidate_id, self._stable_since = candidate.track_id, obs.stamp
            if obs.stamp - self._stable_since < self.config.dwell_s:
                return Action("hold", reason="passenger dwell")
            self._person_id, self._person_anchor = (
                candidate.track_id,
                np.asarray(candidate.position),
            )
            if self._has_paper:
                return self._start_surface(obs, candidate, "BAG")
            return self._start_pick(obs)
        if self.state in {"PICK_APPROACH", "PICK_GRASP"} and not self._patch_valid(
            obs.paper_grasp, obs
        ):
            return self._stop("paper grasp observation lost")
        if self.state == "PICK_APPROACH":
            if not self._reached(obs):
                return self._move(obs)
            self._enter("PICK_GRASP", obs, obs.paper_grasp.center)
            return self._move(obs)
        if self.state == "PICK_GRASP":
            if not self._reached(obs):
                return self._move(obs)
            self._enter("PICK_CLOSE", obs)
            return Action("grip", gripper_close=True, reason="grasp exposed half of paper")
        if self.state == "PICK_CLOSE":
            if not self._gripped(obs):
                return Action("hold", reason="await calibrated grasp feedback")
            self._has_paper = True
            self._extraction_eef_start = np.asarray(obs.eef_position)
            self._enter(
                "EXTRACT",
                obs,
                np.asarray(obs.eef_position)
                + self.config.extraction_distance_m * np.asarray(self._normal),
            )
            return self._move(obs)
        if self.state == "EXTRACT":
            if not self._reached(obs):
                return self._move(obs)
            if obs.paper_tip is None or not self._patch_valid(obs.supply_exit, obs):
                return self._stop("extraction cannot be observed")
            if self._extraction_eef_start is None:
                return self._stop("extraction encoder start missing")
            travel = np.dot(np.asarray(obs.eef_position) - self._extraction_eef_start, self._normal)
            clearance = np.dot(np.asarray(obs.paper_tip) - obs.supply_exit.center, self._normal)
            if (
                travel < self.config.extraction_distance_m * 0.9
                or clearance < self.config.extraction_clearance_m
            ):
                return self._stop("paper not visibly extracted from slot")
            if (
                obs.paper_visible_length_m is None
                or not math.isfinite(obs.paper_visible_length_m)
                or obs.paper_visible_length_m < self.config.extracted_visible_length_m
            ):
                return self._stop("insufficient visible paper span to confirm extraction")
            if self.config.control_profile == "reactive_v1" and self._person_id is None:
                self._enter("WAIT_PERSON", obs)
                return Action("hold", reason="paper prepared; await stationary passenger")
            if person is None or not self._person_valid(person, obs):
                return self._stop("passenger lost during extraction")
            return self._start_surface(obs, person, "BAG")
        for region in ("BAG", "FOREARM"):
            if self.state == "APPROACH_" + region:
                if not self._reached(obs):
                    return self._move(obs)
                self._enter("CONTACT_" + region, obs, obs.eef_position)
            if self.state == "CONTACT_" + region:
                if abs(obs.wrist_force_n) >= self.config.min_contact_force_n:
                    if obs.paper_tip is None:
                        return self._stop("paper tip missing at contact")
                    self._stroke_origin = np.asarray(obs.paper_tip)
                    self._observed_stroke = 0.0
                    self._contact_started = obs.stamp
                    self._force_adjustment = 0.0
                    self._enter(
                        "SCAN_" + region,
                        obs,
                        np.asarray(obs.eef_position)
                        + self.config.stroke_m * np.asarray(self._tangent),
                    )
                    return self._move(obs)
                if not self._reached(obs):
                    return self._move(obs)
                if self._contact_advance >= self.config.max_contact_advance_m:
                    return self._stop("contact search exhausted")
                self._contact_advance += self.config.contact_advance_m
                self._target = tuple(
                    np.asarray(obs.eef_position)
                    - self.config.contact_advance_m * np.asarray(self._normal)
                )
                return self._move(obs)
            if self.state == "SCAN_" + region:
                if (
                    abs(obs.wrist_force_n) < self.config.min_contact_force_n
                    or obs.paper_tip is None
                ):
                    return self._stop("contact or visible paper lost during sampling")
                self._observed_stroke = float(
                    np.dot(np.asarray(obs.paper_tip) - self._stroke_origin, self._tangent)
                )
                # Conservative force relief: never press deeper while stroking.
                if abs(obs.wrist_force_n) > 2 * self.config.target_contact_force_n:
                    if self._force_adjustment >= self.config.max_force_adjustment_m:
                        return self._stop("force regulation exhausted")
                    adjustment = min(
                        0.0002, self.config.max_force_adjustment_m - self._force_adjustment
                    )
                    self._force_adjustment += adjustment
                    self._target = tuple(
                        np.asarray(self._target) + adjustment * np.asarray(self._normal)
                    )
                if not self._reached(obs):
                    return self._move(obs)
                if self._observed_stroke < max(
                    self.config.stroke_m * 0.8, self.config.min_contact_span_m
                ):
                    return self._stop("paper motion did not follow sampling stroke")
                if obs.stamp - self._contact_started < self.config.min_contact_time_s:
                    return Action("hold", reason="minimum sampling contact duration")
                self._enter(
                    "RETRACT_" + region,
                    obs,
                    np.asarray(obs.eef_position)
                    + self.config.standoff_m * np.asarray(self._normal),
                )
                return self._move(obs, "retreat")
            if self.state == "RETRACT_" + region:
                if not self._reached(obs):
                    return self._move(obs, "retreat")
                if region == "BAG":
                    return self._start_surface(obs, person, "FOREARM")
                self._enter("WAIT_EXIT", obs)
                return Action("hold", reason="await passenger in waiting area")
        if self.state == "WAIT_EXIT":
            if (
                person is None
                or not math.isfinite(person.stamp)
                or not 0 <= obs.stamp - person.stamp <= self.config.sensor_max_age_s
            ):
                return self._stop("passenger exit tracking lost")
            if not person.in_waiting_area:
                return Action("hold", reason="await observed waiting-area entry")
            self.completed_person_ids.append(person.track_id)
            self._person_id, self._candidate_id, self._stable_since = None, None, None
            if len(self.completed_person_ids) < self.config.batch_size:
                self._enter("WAIT_PERSON", obs)
                return Action("hold", reason="next passenger")
            if not self._patch_valid(obs.deposit, obs) or obs.paper_tip is None:
                return self._stop("missing measured deposit opening")
            target = np.asarray(obs.eef_position) + np.asarray(obs.deposit.center) - obs.paper_tip
            self._enter("DEPOSIT", obs, target)
            return self._move(obs)
        if self.state == "DEPOSIT":
            if not self._reached(obs):
                return self._move(obs)
            self._enter("RELEASE", obs)
            return Action("grip", gripper_close=False, reason="release paper into box")
        if self.state == "RELEASE":
            if obs.gripper_force_n >= self.config.min_grip_force_n:
                return Action("hold", reason="await release feedback")
            self._has_paper = False
            self._enter("VERIFY_DEPOSIT", obs)
        if self.state == "VERIFY_DEPOSIT":
            if not self._patch_valid(obs.deposit, obs) or obs.paper_tip is None:
                return self._stop("deposit not visually observed; native audit required")
            if np.linalg.norm(np.asarray(obs.paper_tip) - obs.deposit.center) > 0.05:
                return self._stop("paper not observed at deposit")
            self._enter("DONE", obs)
            return Action("hold", reason="protocol complete; native audit still required")
        return self._stop("invalid policy state: " + self.state)
