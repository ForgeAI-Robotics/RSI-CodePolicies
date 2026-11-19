"""Sensor/control adapter for the project's native Isaac checkpoint robot.

Imported inside the Isaac worker, after SimulationApp starts. The agent sees
measured features and bounded commands; environment truth is audit-only.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .motion import FixedBaseMotion, MotionPlan
from .self_depth import RegisteredSelfDepth, instance_supported_human
from .sensors import RawMeasuredObservation, deproject_depth
from .trajectory import TransportTrajectory, action_key
from .types import Action, Observation, SensorProvenance
from .vision import CheckpointVision, VisionConfig, classify_colors


class NativePolicyBackend:
    def __init__(
        self,
        runtime,
        observation_recorder: Callable[[RawMeasuredObservation, dict], object] | None = None,
        *,
        collision_profile: str = "legacy",
        vision_profile: str = "legacy",
    ):
        self.runtime = runtime
        runtime.enable_recording()
        self.motion = (
            FixedBaseMotion(runtime.description)
            if collision_profile == "legacy"
            else FixedBaseMotion(runtime.description, collision_profile=collision_profile)
        )
        self.collision_profile = collision_profile
        profile = getattr(runtime, "depth_filter_profile", None)
        geometry_type = RegisteredSelfDepth
        if (
            profile is not None
            and getattr(getattr(runtime, "config", None), "self_depth_backend", "cpu")
            == "cuda_verified"
        ):
            from .gpu_geometry import VerifiedOwnGeometry

            geometry_type = VerifiedOwnGeometry
        self._self_depth = (
            geometry_type(runtime.description, profile) if profile is not None else None
        )
        self.vision = (
            CheckpointVision()
            if vision_profile == "legacy"
            else CheckpointVision(VisionConfig(vision_profile=vision_profile))
        )
        self.vision_profile = vision_profile
        self.latest: Observation | None = None
        self._plan = None
        self._planned_action: Action | None = None
        self._released = 0
        self._tool_qualified = False
        self._grasp_commanded = False
        self._planned_step: int | None = None
        self._transport: TransportTrajectory | None = None
        self._transport_approved_end: int | None = None
        # Session identity outlives active references and collision approvals.
        # Rejection, hold and cancellation must never permit rebinding an ID.
        self._transport_bindings: dict[str, tuple] = {}
        self._contact_track_id: str | None = None
        self._contact_region: str | None = None
        self._observation_recorder = observation_recorder
        self._latest_raw: RawMeasuredObservation | None = None
        self._capture_context: dict = {}
        self._captured_sequences: set[int] = set()
        self._capture_phase = "initial"
        self._captured_phase = None
        self._previous_paper_visible = False
        self._streams = {
            name: (runtime.output / f"{name}.jsonl").open("x")
            for name in ("observations", "actions", "perception", "primitives")
        }

    def _log(self, name, value):
        self._streams[name].write(json.dumps(value) + "\n")
        self._streams[name].flush()

    def _capture_latest(self) -> None:
        raw = self._latest_raw
        if self._observation_recorder and raw and raw.sequence not in self._captured_sequences:
            self._observation_recorder(raw, self._capture_context)
            self._captured_sequences.add(raw.sequence)

    def observe(self) -> Observation:
        if self.runtime.stopped:
            raise RuntimeError("Native runtime is stopped; cached observation is not current")
        self._plan, self._planned_action, self._planned_step = None, None, None
        self._transport_approved_end = None
        if self.latest is not None and self.runtime.steps <= self.latest.sequence:
            self.runtime.step()
        raw = self.runtime.observe()
        self._latest_raw = raw
        if self.latest is not None and (
            raw.sequence <= self.latest.sequence or raw.stamp <= self.latest.stamp
        ):
            raise RuntimeError("Native sensor timestamps did not advance")
        description = self.runtime.description
        model, data = description.model, description.data
        eef = data.site_xpos[description.eef_id]
        rotation = data.site_xmat[description.eef_id].reshape(3, 3)
        pads = [model.geom(name).id for name in ("right_claw_link_pad", "right_claw_link_g_pad")]
        # Pad surfaces from encoder FK and immutable CAD, never paper world pose.
        separation = abs(
            float(np.dot(data.geom_xpos[pads[0]] - data.geom_xpos[pads[1]], rotation[:, 1]))
        )
        thickness = sum(
            float(
                np.dot(
                    abs(data.geom_xmat[gid].reshape(3, 3).T @ rotation[:, 1]), model.geom_size[gid]
                )
            )
            for gid in pads
        )
        aperture = max(0.0, separation - thickness)
        grip = min(
            raw.contact_force_n.get(name, 0.0) for name in ("right_claw_link", "right_claw_link_g")
        )
        force = None
        if (
            raw.wrist_bias_valid
            and raw.wrist_wrench6 is not None
            and raw.wrist_bias6 is not None
            and raw.wrist_zero_offset6 is not None
            and raw.wrist_gravity_wrench6 is not None
        ):
            wrench, expected = np.asarray(raw.wrist_wrench6), np.asarray(raw.wrist_bias6)
            zero, gravity = (
                np.asarray(raw.wrist_zero_offset6),
                np.asarray(raw.wrist_gravity_wrench6),
            )
            if (
                wrench.shape == (6,)
                and expected.shape == (6,)
                and zero.shape == (6,)
                and gravity.shape == (6,)
                and np.isfinite([wrench, expected, zero, gravity]).all()
                and np.allclose(expected, zero + gravity, atol=1e-6, rtol=1e-6)
            ):
                reaction = np.asarray(raw.wrist_reaction_world_n, dtype=float)
                times = (
                    raw.wrist_force_window_start,
                    raw.wrist_force_window_end,
                    raw.wrist_force_stamp,
                    raw.wrist_bias_stamp,
                )
                if (
                    reaction.shape == (3,)
                    and np.isfinite(reaction).all()
                    and all(value is not None and np.isfinite(value) for value in times)
                ):
                    start, end, midpoint, bias_stamp = times
                    if (
                        0 <= bias_stamp <= start
                        and np.isclose(end, raw.stamp, atol=1e-9, rtol=0)
                        and np.isclose(end - start, 0.05, atol=1e-9, rtol=0)
                        and np.isclose(midpoint, (start + end) / 2, atol=1e-9, rtol=0)
                    ):
                        force = float(np.linalg.norm(reaction))
        if (
            raw.robot_base_pose is None
            or len(raw.robot_base_pose) != 4
            or not np.isfinite(raw.robot_base_pose).all()
        ):
            raise RuntimeError("Native own-robot base pose measurement unavailable")
        sensors = {
            name: SensorProvenance(
                "synthetic",
                valid,
                source,
                raw.wrist_force_stamp if name == "force" and force is not None else raw.stamp,
            )
            for name, valid, source in (
                ("joints", True, "native_articulation_encoder_and_CAD_FK"),
                ("rgbd", raw.rgbd is not None, "robot_head_RTX_registered_RGBD"),
                ("gripper", True, "native_pad_impulses_and_encoder_CAD_aperture"),
                (
                    "force",
                    force is not None,
                    "native_wrench_minus_CAD_gravity_bias_and_distal_momentum_50ms",
                ),
            )
        }
        telemetry = Observation(
            raw.stamp,
            raw.sequence,
            tuple(map(float, raw.robot_base_pose)),
            raw.joint_positions,
            tuple(map(float, eef)),
            tuple(map(float, Rotation.from_matrix(rotation).as_euler("xyz"))),
            gripper_aperture_m=aperture,
            gripper_force_n=grip,
            wrist_force_n=force,
            sensors=sensors,
        )
        if raw.rgbd is None:
            raise RuntimeError("Native RGB-D frame unavailable")
        from PIL import Image

        Image.fromarray(raw.rgbd.rgb).save(self.runtime.output / "overview.png", compress_level=1)
        self._capture_context = {
            "phase": self._capture_phase,
            "telemetry": asdict(telemetry),
            "vision_input_state": self.vision.diagnostic_state(),
        }
        observation = self.vision.observe(raw.rgbd, telemetry)
        # Retain the rejected measurement too: failure diagnosis must not stop
        # at the last successful frame.
        self._log("observations", asdict(observation))
        self._log("perception", self.vision.diagnostics)
        visible = observation.paper_tip is not None
        if self._captured_phase != self._capture_phase or (
            self._previous_paper_visible and not visible
        ):
            self._capture_latest()
            self._captured_phase = self._capture_phase
        self._previous_paper_visible = visible
        if self._tool_qualified and (
            observation.paper_tip is None or not len(self.vision.paper_points)
        ):
            raise RuntimeError("Held-paper visual geometry lost; cannot check tool envelope")
        paper_near_grasp = (
            observation.paper_grasp is not None
            and np.linalg.norm(np.asarray(observation.paper_grasp.center) - eef) <= 0.065
        )
        self._tool_qualified = bool(
            self._grasp_commanded
            and grip >= 0.1
            and aperture < 0.025
            and (self._tool_qualified or paper_near_grasp)
            and observation.paper_tip is not None
        )
        if self._grasp_commanded and grip >= 0.1 and aperture < 0.025 and not self._tool_qualified:
            raise RuntimeError("Grasp pressure without current confirmed paper envelope")
        points, pixels = deproject_depth(raw.rgbd, stride=4)
        colors = classify_colors(raw.rgbd.rgb)
        # Only the actually tracked deformable paper component is graspable /
        # tool self geometry. A broad white mask would erase unrelated fixtures.
        paper_points = self.vision.paper_points
        paper = np.zeros(len(points), dtype=bool)
        if len(paper_points):
            distances, _ = cKDTree(paper_points).query(points, distance_upper_bound=0.002)
            paper = distances <= 0.002
        keep = ~paper if self._tool_qualified else np.ones(len(points), dtype=bool)
        human = (
            colors["person"][pixels[:, 0], pixels[:, 1]]
            | colors["skin"][pixels[:, 0], pixels[:, 1]]
        )
        self_labels = None
        supported_human = instance_supported_human(points, human, observation)
        if self._self_depth is not None:
            self_labels = self._self_depth.labels(raw, pixels, paper, supported_human)
            diagnostic = self._self_depth.diagnostics
            self._capture_context["self_depth"] = diagnostic
            self._log("perception", {"self_depth": diagnostic})
        if self_labels is None:
            self.motion.set_measured_cloud(points[keep], human[keep], paper[keep])
        else:
            self.motion.set_measured_cloud(
                points[keep],
                human[keep],
                paper[keep],
                self_labels=self_labels[keep],
                supported_human=supported_human[keep],
            )
        self.motion.set_measured_tool(
            observation.eef_position,
            observation.paper_tip if self._tool_qualified else None,
        )
        self.latest = observation
        return observation

    def check_reachability(self, action: Action) -> bool:
        if (
            self.latest is None
            or self.runtime.stopped
            or self.runtime.steps != self.latest.sequence
        ):
            return False
        self.motion.last_rejection = None
        self.motion.set_grasp_phase(action.reason in {"PICK_GRASP", "PICK_CLOSE"})
        patch = None
        if action.reason.startswith(("CONTACT_", "SCAN_", "RETRACT_", "RETREAT_")) and not (
            action.reason == "RETREAT_PERSON" and self._contact_region is None
        ):
            region = (
                self._contact_region
                if action.reason == "RETREAT_PERSON"
                else ("bag" if action.reason.endswith("BAG") else "forearm")
            )
            candidates = (
                [(person, getattr(person, region)) for person in self.latest.people]
                if region
                else []
            )
            candidates = [(person, item) for person, item in candidates if item is not None]
            if self._contact_track_id is not None:
                candidates = [
                    (person, item)
                    for person, item in candidates
                    if person.track_id == self._contact_track_id
                ]
            if not candidates:
                return False
            candidates.sort(
                key=lambda pair: np.linalg.norm(
                    np.asarray(pair[1].center)
                    - np.asarray(self.latest.paper_tip or self.latest.eef_position)
                )
            )
            if len(candidates) > 1:
                distances = [
                    np.linalg.norm(
                        np.asarray(item.center)
                        - np.asarray(self.latest.paper_tip or self.latest.eef_position)
                    )
                    for _, item in candidates[:2]
                ]
                if distances[1] - distances[0] < 0.05:
                    return False
            person, patch = candidates[0]
            if not 0 <= self.latest.stamp - patch.stamp <= 0.25:
                return False
            self._contact_track_id, self._contact_region = person.track_id, region
        self.motion.set_measured_tool(
            self.latest.eef_position,
            self.latest.paper_tip if self._tool_qualified else None,
            contact_patch=patch,
        )
        if getattr(action, "trajectory_id", None) is not None:
            self._plan = self._plan_transport(action)
        else:
            self._transport = None
            self._transport_approved_end = None
            self._plan = self.motion.plan(action)
        self._planned_action = action
        self._planned_step = self.runtime.steps
        if not self._plan.reached:
            self._log(
                "primitives",
                {
                    "event": "motion_rejected",
                    "sequence": self.runtime.steps,
                    "action": asdict(action),
                    "plan": asdict(self._plan),
                    "collision_profile": getattr(self, "collision_profile", "legacy"),
                    "collision": getattr(self.motion, "last_rejection", None),
                },
            )
        return self._plan.reached

    def _plan_transport(self, action: Action) -> MotionPlan:
        self._transport_approved_end = None
        try:
            key = action_key(action)
            bound_key = self._transport_bindings.get(key[0])
            if bound_key is not None and bound_key != key:
                raise ValueError("Stable trajectory ID cannot change its goal or limits")
            self._transport_bindings[key[0]] = key
            cached = self._transport
            if cached is not None and cached.key[0] == key[0] and cached.key != key:
                raise ValueError("Stable trajectory ID cannot change its goal or limits")
            if cached is None or cached.key != key:
                plan = self.motion.plan(action)
                if not plan.reached:
                    self._transport = None
                    return plan
                cached = TransportTrajectory.build(self.motion, action, plan, self.runtime.dt)
                self._transport = cached
                self._log(
                    "primitives",
                    {
                        "event": "transport_reference_created",
                        "trajectory_id": action.trajectory_id,
                        "sequence": self.runtime.steps,
                        "duration_s": (len(cached.samples) - 1) * cached.dt,
                        "FK_reference_peaks": cached.peaks,
                        "scope": "commanded reference limits; not measured dynamics limits",
                    },
                )
            error = cached.tracking_error(self.motion)
            if self._transport_tracking_failed(error):
                raise ValueError("Measured robot has left the bounded transport reference")
            reason = cached.precheck(self.motion, action.duration_s)
            plan = cached.endpoint_plan
            if reason:
                return MotionPlan(
                    False, plan.joints, plan.error_m, plan.orientation_error_rad, reason
                )
            self._transport_approved_end = cached.end_index(action.duration_s)
            self._log(
                "primitives",
                {
                    "event": "transport_segment_prechecked",
                    "trajectory_id": action.trajectory_id,
                    "sequence": self.runtime.steps,
                    "start_index": cached.index,
                    "end_index": self._transport_approved_end,
                    "tracking_error": error,
                    "fresh_cloud": True,
                },
            )
            return plan
        except (ValueError, KeyError) as exc:
            self._transport = None
            return MotionPlan(False, {}, float("inf"), float("inf"), str(exc))

    @staticmethod
    def _transport_tracking_failed(error: dict) -> bool:
        return (
            not np.isfinite(list(error.values())).all()
            or error["joint_error_rad"] > 0.03
            or error["position_error_m"] > 0.008
            or error["orientation_error_rad"] > 0.10
        )

    def _execute_transport(self, action: Action) -> None:
        trajectory = self._transport
        end = self._transport_approved_end
        if trajectory is None or end is None or trajectory.key != action_key(action):
            raise RuntimeError("Continuous transport requires its fresh segment approval")
        start = trajectory.index
        # Consume approval before any physical mutation: an exception cannot
        # leave an executable old plan or resume an interrupted segment.
        self._plan, self._planned_action, self._planned_step = None, None, None
        self._transport_approved_end = None
        peak = {"joint_error_rad": 0.0, "position_error_m": 0.0, "orientation_error_rad": 0.0}
        try:
            ticks = round(action.duration_s / trajectory.dt)
            for _ in range(ticks):
                if self.runtime.stopped:
                    raise RuntimeError("Native runtime stopped during transport")
                trajectory.index = min(trajectory.index + 1, end)
                q = trajectory.samples[trajectory.index]
                self.runtime.apply_joint_targets(
                    dict(zip(self.motion.names, map(float, q), strict=True))
                )
                # The unchanged native per-step PD/FF/contact/base interlocks run.
                self.runtime.step(1)
                error = trajectory.tracking_error(self.motion)
                peak = {key: max(peak[key], error[key]) for key in peak}
                if self._transport_tracking_failed(error):
                    raise RuntimeError("Measured transport tracking error exceeded its bound")
        except Exception as exc:
            self._transport = None
            self._log(
                "primitives",
                {
                    "event": "transport_interrupted",
                    "reason": str(exc),
                    "sequence": self.runtime.steps,
                },
            )
            self.stop(str(exc))
            raise
        self._log(
            "primitives",
            {
                "event": "transport_segment_end",
                "trajectory_id": action.trajectory_id,
                "sequence": self.runtime.steps,
                "start_index": start,
                "end_index": trajectory.index,
                "tracking_error_peak": peak,
                "await_new_visual_observation": True,
            },
        )

    def _freeze_approach_targets(self, action: Action) -> dict:
        """Replace planner-controlled targets with same-epoch own encoders.

        This preserves physical velocity and the existing PD/FF/interlocks.
        It does not assert that motion immediately stops after the target reset.
        """
        raw, obs = self._latest_raw, self.latest
        planned = self._planned_action
        if (
            action.duration_s != 0.05
            or action.target is not None
            or action.rpy is not None
            or action.gripper_close is not None
            or action.reason != "bounded approach perception recheck"
            or raw is None
            or obs is None
            or raw.sequence != obs.sequence
            or obs.sequence != self.runtime.steps
            or raw.stamp != obs.stamp
            or self._planned_step != self.runtime.steps
            or planned is None
            or planned.kind != "move_eef"
            or planned.reason != "PICK_APPROACH"
            or self._plan is None
            or self._plan.reached
            or self._tool_qualified
            or self._grasp_commanded
        ):
            raise RuntimeError("Encoder-target freeze requires a fresh rejected unheld approach")
        names = tuple(self.motion.names)
        if not names or any(name.startswith("base_") or "wheel" in name for name in names):
            raise RuntimeError("Encoder-target freeze cannot command base or wheels")
        targets = {name: raw.joint_positions[name] for name in names}
        velocities = {name: raw.joint_velocities[name] for name in names}
        if not np.isfinite([*targets.values(), *velocities.values()]).all() or any(
            obs.joint_positions[name] != targets[name] for name in names
        ):
            raise RuntimeError("Encoder-target freeze requires consistent finite own encoders")
        previous = {
            name: float(self.runtime.control[self.runtime.names.index(name)]) for name in names
        }
        event = {
            "event": "encoder_target_freeze_start",
            "sequence": raw.sequence,
            "stamp": raw.stamp,
            "previous_targets": previous,
            "encoder_targets": targets,
            "joint_velocities": velocities,
            "eef_position": obs.eef_position,
            "rejected_action": asdict(planned),
            "rejected_plan": asdict(self._plan),
        }
        # Existing native target setter validates all joint names/limits before
        # any mutation. Never clip q, teleport state, zero qdot or recalibrate FT.
        self.runtime.apply_joint_targets(targets)
        self._plan, self._planned_action, self._planned_step = None, None, None
        self._log("primitives", event)
        return event

    def execute(self, action: Action) -> None:
        if self.runtime.stopped:
            raise RuntimeError("Native runtime is stopped")
        if getattr(action, "trajectory_id", None) is not None:
            action_key(action)
        self._capture_phase = action.reason or action.kind
        self._log("actions", {"physics_step": self.runtime.steps, **asdict(action)})
        freeze = None
        if getattr(action, "trajectory_id", None) is None:
            self._transport = None
            self._transport_approved_end = None
        if action.kind in {"move_eef", "retreat"}:
            if (
                self._planned_action != action
                or self._plan is None
                or not self._plan.reached
                or self._planned_step != self.runtime.steps
            ):
                raise RuntimeError("Motion requires a fresh successful whole-path precheck")
            if getattr(action, "trajectory_id", None) is not None:
                self._execute_transport(action)
                return
            self.runtime.apply_joint_targets(self._plan.joints)
            self._plan, self._planned_action, self._planned_step = None, None, None
        elif action.kind == "hold_pose":
            freeze = self._freeze_approach_targets(action)
        elif action.kind == "grip":
            if action.gripper_close is None:
                raise ValueError("Missing gripper command")
            self._grasp_commanded = action.gripper_close
            self.runtime.grip(action.gripper_close)
            if not action.gripper_close:
                self._tool_qualified = False
        elif action.kind == "stop":
            self.stop(action.reason)
            return
        elif action.kind != "hold":
            raise ValueError("Unsupported fixed-base action")
        if action.reason == "next passenger":
            self._contact_track_id, self._contact_region = None, None
        if action.reason == "await passenger in waiting area":
            # Queue coordinator is an external environment actor. This request
            # does not imply sampling success and cannot change the evaluator.
            self.runtime.release_passenger(self._released)
            self._released += 1
        self.runtime.step(max(1, round(action.duration_s / self.runtime.dt)))
        if freeze is not None:
            measured = self.runtime.observe()
            description = self.runtime.description
            eef = description.data.site_xpos[description.eef_id]
            self._log(
                "primitives",
                {
                    "event": "encoder_target_freeze_end",
                    "sequence": measured.sequence,
                    "stamp": measured.stamp,
                    "anchor_sequence": freeze["sequence"],
                    "encoder_targets": freeze["encoder_targets"],
                    "joint_positions": measured.joint_positions,
                    "joint_velocities": measured.joint_velocities,
                    "eef_position": tuple(map(float, eef)),
                    "eef_displacement_m": float(
                        np.linalg.norm(eef - np.asarray(freeze["eef_position"]))
                    ),
                    "contact_force_n": measured.contact_force_n,
                    "wrist_wrench6": measured.wrist_wrench6,
                    "wrist_reaction_world_n": measured.wrist_reaction_world_n,
                    "scope": "measured transient after target reset; not proof of zero velocity",
                },
            )

    def stop(self, reason: str) -> None:
        self._transport = None
        self._transport_approved_end = None
        if not self.runtime.stopped:
            self.runtime.stop(reason)
        if (
            self._observation_recorder
            and self._latest_raw is not None
            and self.runtime.steps != self._latest_raw.sequence
        ):
            self._latest_raw = self.runtime.observe()
            self._capture_context = {
                "phase": "terminal_native_safety",
                "vision_input_state": self.vision.diagnostic_state(),
                "last_policy_telemetry": self._capture_context.get("telemetry"),
                "scope": "Shutdown measurement; not an executed policy observation",
            }
        self._capture_latest()

    def native_success(self) -> bool:
        return self.runtime.diagnostics()["native_success"] is True

    def close(self):
        for stream in self._streams.values():
            stream.close()
