"""Main-thread native PhysX execution and separate privileged audit channels."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .appearance import configure_lighting
from .force_estimation import DistalMomentumKinematics, MomentumForceObserver
from .native_gravity import NativeGravityFeedforward
from .native_io import LosslessContactWriter, RigidPoseSnapshot
from .scene import (
    ZERO_DELAY_QUALIFICATION_SHA256,
    CheckpointSceneConfig,
    build_checkpoint_scene,
    load_robot_description,
)
from .self_depth import SelfDepthProfile, measurement_digest
from .sensors import (
    CameraCalibration,
    RawMeasuredObservation,
    RenderSynchrony,
    RGBDFrame,
    optical_transform,
)


class NativeCheckpointRuntime:
    """Explicitly live execution. No recorded qpos or replay input is accepted."""

    def __init__(
        self,
        output: Path,
        asset_root: Path,
        app,
        config: CheckpointSceneConfig | None = None,
        dt: float = 0.001,
        frame_callback=None,
    ):
        import carb.settings
        import omni.replicator.core as rep
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.nodes.bindings import _isaacsim_core_nodes
        from isaacsim.core.prims import RigidPrim, SingleArticulation, SingleRigidPrim
        from isaacsim.core.utils.types import ArticulationAction
        from omni.physx import get_physx_simulation_interface
        from pxr import Gf, PhysicsSchemaTools, PhysxSchema, UsdGeom, UsdPhysics

        if not 0.0001 <= dt <= 0.005:
            raise ValueError("Unsupported physics timestep")
        self.output, self.app, self.dt = Path(output), app, dt
        self.config = config or CheckpointSceneConfig()
        self._render_settings = carb.settings.get_settings()
        self._render_epoch = None
        self._previous_render_frame = None
        self.depth_filter_profile = None
        if self.config.render_profile == "zero_delay_v1":
            if not self._zero_delay_settings_verified():
                raise RuntimeError("Native zero-delay experience settings were not applied")
            self.depth_filter_profile = SelfDepthProfile(
                profile="zero_delay_v1",
                evidence_sha256=ZERO_DELAY_QUALIFICATION_SHA256,
                epsilon_m=self.config.self_depth_epsilon_m,
            )
        self._core_nodes = _isaacsim_core_nodes.acquire_interface()
        self.offset = np.asarray(self.config.offset)
        self.description = load_robot_description(self.output, self.config)
        self.steps = 0
        self.frame_callback = frame_callback
        self._frame_stride = max(1, round(0.04 / dt))
        self._frame_log = None
        self.held = False
        self.two_finger_qualified = False
        self.extracted = False
        self.release_after_batch = False
        self.deposited_stable_s = 0.0
        self._previous_paper_poses = None
        self._paper_max_speed = 0.0
        self.stopped = False
        self.violations = []
        self.contacts = []
        self._contact_writer = None
        self._last_render_step = None
        self._rgbd_cache_step = None
        self._rgbd_cache = None
        self.events = []
        self.action_type = ArticulationAction
        self._force_observer = MomentumForceObserver(dt, window_s=0.05)
        self._wrist_bias = None
        self._wrist_bias_stamp = None
        self._wrist_bias_rotation = None
        self._wrist_bias_held = False
        self._payload_com_wrist = None
        omni.usd.get_context().new_stage()
        self.stage = omni.usd.get_context().get_stage()
        self.native_manifest = build_checkpoint_scene(
            self.stage, asset_root, self.description, self.config
        )
        robot_path = self.native_manifest["robot_path"]
        joint = UsdPhysics.Joint(self.stage.GetPrimAtPath(robot_path + "/right_wrist_pitch_joint"))
        joint_q = joint.GetLocalRot1Attr().Get()
        self._wrist_joint_rotation = Rotation.from_quat(
            [*joint_q.GetImaginary(), joint_q.GetReal()]
        ).as_matrix()
        self._wrist_joint_position = np.asarray(joint.GetLocalPos1Attr().Get(), float)
        self._wrist_distal_ids = []
        for bid in range(self.description.model.nbody):
            ancestor = bid
            while ancestor and ancestor != self.description.wrist_id:
                ancestor = int(self.description.model.body_parentid[ancestor])
            if ancestor == self.description.wrist_id:
                self._wrist_distal_ids.append(bid)
        for name, value in zip(self.description.joint_names, self.description.control, strict=True):
            if name in {"base_x", "base_y"}:
                continue
            prim = self.stage.GetPrimAtPath(robot_path + "/" + name)
            state = PhysxSchema.JointStateAPI.Apply(prim, "angular")
            state.CreatePositionAttr(float(np.degrees(value)))
            state.CreateVelocityAttr(0.0)
            UsdPhysics.DriveAPI(prim, "angular").GetTargetPositionAttr().Set(
                float(np.degrees(value))
            )
        self.native_manifest["lighting"] = configure_lighting(
            self.stage, self.offset, self.config.lighting_profile
        )
        self.world = World(
            physics_dt=dt,
            rendering_dt=0.04,
            stage_units_in_meters=1.0,
            physics_prim_path="/World/NativePhysics",
        )
        physics = PhysxSchema.PhysxSceneAPI.Apply(self.stage.GetPrimAtPath("/World/NativePhysics"))
        physics.CreateEnableGPUDynamicsAttr(False)
        physics.CreateBroadphaseTypeAttr("MBP")
        self.robot = self.world.scene.add(
            SingleArticulation(prim_path=robot_path, name="checkpoint_robot")
        )
        self.paper_views = [
            self.world.scene.add(SingleRigidPrim(prim_path=p, name=f"paper_{i}"))
            for i, p in enumerate(self.native_manifest["paper_bodies"])
        ]
        self.wrist_view = self.world.scene.add(
            SingleRigidPrim(
                prim_path=self.native_manifest["body_paths"][self.description.wrist_id],
                name="checkpoint_wrist",
            )
        )
        self.base_view = self.world.scene.add(
            SingleRigidPrim(prim_path=robot_path + "/base_link", name="checkpoint_base")
        )
        # Stage-authored kinematic actors need no SingleRigidPrim reset wrapper,
        # which would incorrectly try to set their linear/angular velocities.
        self.person_positions = [
            np.asarray(p["initial_position"], float) for p in self.native_manifest["people"]
        ]
        self.person_states = ["queue"] * self.config.batch_size
        self.inspection_entered_s = [None] * self.config.batch_size
        self.active_person = 0
        self.sampling_metrics = [
            {
                region: {
                    "contact_s": 0.0,
                    "travel_m": 0.0,
                    "material_sliding_m": 0.0,
                    "span_m": 0.0,
                    "peak_force_n": 0.0,
                }
                for region in ("forearm", "bag")
            }
            for _ in self.person_states
        ]
        self._contact_positions = {}
        self._last_contact = {}

        def on_contact(headers, points):
            for header in headers:
                actors = [
                    str(PhysicsSchemaTools.intToSdfPath(v)) for v in (header.actor0, header.actor1)
                ]
                colliders = [
                    str(PhysicsSchemaTools.intToSdfPath(getattr(header, key, actor)))
                    for key, actor in (("collider0", header.actor0), ("collider1", header.actor1))
                ]
                start = header.contact_data_offset
                for point in points[start : start + header.num_contact_data]:
                    self.contacts.append(
                        {
                            "actors": actors,
                            "colliders": colliders,
                            "position": (np.asarray(point.position, float) - self.offset).tolist(),
                            "impulse": np.asarray(point.impulse, float).tolist(),
                            "normal": np.asarray(point.normal, float).tolist(),
                            "separation": float(point.separation),
                        }
                    )

        self.contact_subscription = (
            get_physx_simulation_interface().subscribe_contact_report_events(on_contact)
        )
        self._rep = rep
        self._render_products = []
        self._annotators = {}
        width, height = 640, 480
        focal, aperture = 14.0, 20.955
        self.camera_body_id = self.description.model.body("sens_link").id
        sensor_frame = np.eye(4)
        sensor_frame[:3, :3] = self.description.data.xmat[self.camera_body_id].reshape(3, 3)
        sensor_frame[:3, 3] = self.description.data.xpos[self.camera_body_id]
        if self.config.camera_mount_mode == "urdf_sensor":
            self.camera_mount = np.eye(4)
            self.camera_mount[2, 3] = self.config.sensor_forward_offset_m
            transform = sensor_frame @ self.camera_mount
        else:
            eye, target = np.array([-0.1, -0.45, 1.65]), np.array([0.65, -0.05, 1.0])
            transform = optical_transform(eye, target)
            self.camera_mount = np.linalg.inv(sensor_frame) @ transform
        camera_path = self.native_manifest["body_paths"][self.camera_body_id] + "/checkpoint_rgbd"
        camera = UsdGeom.Camera.Define(self.stage, camera_path)
        camera.CreateFocalLengthAttr(focal)
        camera.CreateHorizontalApertureAttr(aperture)
        camera.CreateVerticalApertureAttr(aperture * height / width)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.02, 30.0))
        camera_xf = UsdGeom.Xformable(camera)
        camera_xf.AddTranslateOp().Set(Gf.Vec3d(*self.camera_mount[:3, 3]))
        usd_rotation = self.camera_mount[:3, :3] @ np.diag([1, -1, -1])
        quat = Rotation.from_matrix(usd_rotation).as_quat()
        camera_xf.AddOrientOp().Set(Gf.Quatf(float(quat[3]), Gf.Vec3f(*quat[:3])))
        product = rep.create.render_product(camera_path, (width, height))
        self._render_products.append(product)
        for channel in ("rgb", "distance_to_image_plane"):
            annotator = rep.AnnotatorRegistry.get_annotator(channel)
            annotator.attach([product])
            self._annotators[channel] = annotator
        self._reference_time = rep.AnnotatorRegistry.get_annotator("ReferenceTime")
        self._reference_time.attach([product])
        self.camera_calibration = CameraCalibration(
            width,
            height,
            focal / aperture * width,
            focal / aperture * width,
            width / 2,
            height / 2,
            tuple(tuple(float(x) for x in row) for row in transform),
            source=(
                f"encoder_FK_and_{self.config.camera_mount_mode}; "
                "experimental_optical_mount_not_hardware_calibrated"
            ),
            pixel_offset_xy=(0.5, 0.5)
            if self.config.render_profile == "zero_delay_v1"
            else (0.0, 0.0),
        )
        self.world.reset()
        self.person_views = [
            SingleRigidPrim(prim_path=p["path"], name=f"person_audit_{i}")
            for i, p in enumerate(self.native_manifest["people"])
        ]
        for view in self.person_views:
            view.initialize()
        pose_paths = [
            *self.native_manifest["paper_bodies"],
            *(spec["path"] for spec in self.native_manifest["people"]),
            robot_path + "/base_link",
            self.native_manifest["body_paths"][self.description.wrist_id],
        ]
        # Retain existing SingleRigidPrim initialization and its physics setup;
        # this extra batch view reads only and never changes contact settings.
        self._pose_view = RigidPrim(
            prim_paths_expr=pose_paths,
            name="checkpoint_audit_pose_batch",
            reset_xform_properties=False,
            prepare_contact_sensors=False,
            track_contact_forces=False,
        )
        self._pose_view.initialize()
        self._pose_snapshot = RigidPoseSnapshot(self._pose_view, pose_paths)
        self._base_path = robot_path + "/base_link"
        self._person_paths = [spec["path"] for spec in self.native_manifest["people"]]
        self.names = list(self.robot.dof_names)
        expected = set(self.description.joint_names) - {"base_x", "base_y"}
        if set(self.names) != expected:
            raise RuntimeError(f"Native joint mismatch: {self.names}; expected {sorted(expected)}")
        self.native_indexes = np.array([self.description.joint_names.index(n) for n in self.names])
        # Isaac Sim 4.5 source single_articulation.py documents row = joint_index+1.
        self._wrist_force_row = (
            1 + self.robot._articulation_view._metadata.joint_indices["right_wrist_pitch_joint"]
        )
        self._gravity_feedforward = NativeGravityFeedforward(
            self.robot,
            self.action_type,
            np.asarray(
                [
                    UsdPhysics.DriveAPI(
                        self.stage.GetPrimAtPath(robot_path + "/" + name), "angular"
                    )
                    .GetMaxForceAttr()
                    .Get()
                    for name in self.names
                ],
                dtype=float,
            ),
        )
        self._gravity_feedforward_enabled = bool(
            self.config.robot_gravity_enabled and self.config.gravity_feedforward_enabled
        )
        self._momentum_kinematics = DistalMomentumKinematics(
            self.description.model, self._wrist_distal_ids, self.names, self.description.wrist_id
        )
        self.control = self.description.control[self.native_indexes].copy()
        self.robot.set_joint_positions(self.control)
        self.robot.set_joint_velocities(np.zeros_like(self.control))
        self.robot.apply_action(self.action_type(joint_positions=self.control))
        base_positions, base_orientations = self._native_poses([self._base_path])
        self._base_initial = base_positions[0]
        self._base_rotation_initial = base_orientations[0]
        self._sync_kinematics()
        for _ in range(6):
            self.world.render()
        self._world_clock_origin = float(self.world.current_time) - self.steps * self.dt
        self.native_manifest.update(
            engine="isaac_physx",
            engine_version="4.5.0",
            timestep_s=dt,
            mujoco_dynamics_steps=0,
            primitive_set="no-vla",
            camera_calibration=asdict(self.camera_calibration),
            camera_png_compress_level=1,
            render_profile=self.config.render_profile,
            render_settings={
                "waitIdle": self._render_settings.get("/app/hydraEngine/waitIdle"),
                "checkForHydraRenderComplete": self._render_settings.get(
                    "/app/updateOrder/checkForHydraRenderComplete"
                ),
            },
            world_clock_origin_s=self._world_clock_origin,
            depth_filter_profile=None
            if self.depth_filter_profile is None
            else asdict(self.depth_filter_profile),
            camera_mount={
                "parent": "sens_link",
                "optical_to_sensor": self.camera_mount.tolist(),
                "mode": self.config.camera_mount_mode,
                "asset_frame": "authored URDF sens_link via sens_joint",
                "assumption": (
                    "optical axes equal sens_link; virtual optical center along local Z"
                    if self.config.camera_mount_mode == "urdf_sensor"
                    else "legacy virtual bracket from zero-neck world eye/look"
                ),
                "hardware_calibrated": False,
                "extrinsics": "encoder FK times fixed optical_to_sensor",
                "effective_initial_head_angles_rad": (
                    list(self.config.initial_head_angles)
                    if self.config.camera_mount_mode == "urdf_sensor"
                    else [0.0, 0.0]
                ),
                "focal_length_mm": focal,
                "horizontal_aperture_mm": aperture,
                "resolution": [width, height],
            },
            gravity_feedforward={
                "requested": self.config.gravity_feedforward_enabled,
                "effective": self._gravity_feedforward_enabled,
                "physical_robot_gravity": self.config.robot_gravity_enabled,
                "source": "Isaac 4.5 Articulation.get_generalized_gravity_forces",
                "actuation": "native gravity compensated torque plus unchanged position drive",
                "update_period_s": dt,
                "joint_order": self.names,
                "configured_drive_effort_limits_nm": self._gravity_feedforward.maximum.tolist(),
                "disabled_behavior": "explicit zero direct effort, including gravity disabled",
                "limit_behavior": "stop before stepping on invalid or excessive feedforward",
                "limits_note": "feedforward bound; not a proof of total PD plus FF torque bound",
            },
            wrist_force_model={
                "source": "native incoming joint wrench; CAD gravity + constant calibrated bias",
                "frame": "right_wrist_pitch_joint child frame from authored localRot1",
                "distal_mass_kg": float(
                    np.sum(self.description.model.body_mass[self._wrist_distal_ids])
                ),
                "qualified_paper_payload_mass_kg": 0.0006,
                "paper_torque_model": "nominal strip COM; force compensation is shape independent",
                "force_estimator": (
                    "50ms integrated compensated wrist force minus CAD momentum change"
                ),
                "force_frame": "world reaction on environment; opposite external force on robot",
                "force_timestamp": "window midpoint; no estimate during reset or warmup",
                "momentum": "encoder velocities and CAD mj_jacBodyCom; no dynamics calls",
                "payload_momentum": (
                    "0.0006kg at calibrated nominal wrist COM; flex motion unmeasured"
                ),
            },
            grasp_model="native dual-finger contact qualified assisted fixed joint",
            scenario_scope="batch checkpoint, uncalibrated segmented paper",
            body_poses_privileged=True,
        )
        self.stage.GetRootLayer().Export(str(self.output / "native_scene.usdc"))
        (self.output / "native_manifest.json").write_text(
            json.dumps(self.native_manifest, indent=2) + "\n"
        )
        self._contact_writer = LosslessContactWriter(self.output / "native_contacts.jsonl.gz")
        self._closed = False

    def _sync_kinematics(self) -> None:
        import mujoco

        d = self.description
        q = np.asarray(self.robot.get_joint_positions())
        d.data.qpos[d.qpos_ids[self.native_indexes]] = q
        d.control[self.native_indexes] = self.control
        mujoco.mj_kinematics(d.model, d.data)
        mujoco.mj_comPos(d.model, d.data)

    def apply_joint_targets(self, targets: dict[str, float]) -> None:
        if self.stopped:
            raise RuntimeError("Runtime is stopped")
        if {"base_x", "base_y"} & set(targets):
            raise ValueError("Base translation is physically fixed and forbidden")
        for name, value in targets.items():
            if name not in self.names or not np.isfinite(value):
                raise ValueError(f"Unknown joint or nonfinite target: {name}")
            jid = self.description.model.joint(name).id
            low, high = self.description.model.jnt_range[jid]
            if self.description.model.jnt_limited[jid] and not low <= value <= high:
                raise ValueError(f"Joint target outside limits: {name}={value}")
        for name, value in targets.items():
            self.control[self.names.index(name)] = value
        self.robot.apply_action(self.action_type(joint_positions=self.control.copy()))

    def _advance_people(self) -> None:
        from pxr import Gf

        for i, spec in enumerate(self.native_manifest["people"]):
            state = self.person_states[i]
            if state in {"inspection", "waiting"} or (state == "queue" and i != self.active_person):
                continue
            if state == "queue" and self.steps * self.dt < spec["parameters"]["arrival_delay_s"]:
                continue
            target = np.asarray(
                spec["waiting_position"] if state == "leaving" else spec["inspection_position"]
            )
            delta = target - self.person_positions[i]
            distance = np.linalg.norm(delta)
            if distance < self.config.walking_speed_m_s * self.dt:
                self.person_positions[i] = target
                self.person_states[i] = "waiting" if state == "leaving" else "inspection"
                if self.person_states[i] == "inspection":
                    self.inspection_entered_s[i] = self.steps * self.dt
                self.events.append(
                    {"step": self.steps, "event": self.person_states[i], "person": i}
                )
            else:
                self.person_positions[i] += (
                    delta / distance * self.config.walking_speed_m_s * self.dt
                )
            # Native kinematic actors are environmental inputs, generated online
            # from queue state. This does not replay any recorded robot state.
            self.stage.GetPrimAtPath(spec["path"]).GetAttribute("xformOp:translate").Set(
                Gf.Vec3d(*self.person_positions[i])
            )

    def release_passenger(self, index: int) -> None:
        if not 0 <= index < len(self.person_states) or self.person_states[index] != "inspection":
            raise ValueError("Passenger is not at the inspection stop")
        minimum_dwell = self.native_manifest["people"][index]["parameters"]["dwell_s"]
        if self.steps * self.dt - self.inspection_entered_s[index] < minimum_dwell:
            raise ValueError("Passenger minimum inspection dwell has not elapsed")
        self.person_states[index] = "leaving"
        self.active_person = index + 1
        self.events.append({"step": self.steps, "event": "release_passenger", "person": index})

    def step(self, count: int = 1, render: bool = False) -> None:
        if self.stopped:
            raise RuntimeError("Runtime is stopped")
        if not 1 <= count <= 10000:
            raise ValueError("Step count must be 1..10000")
        for _ in range(count):
            try:
                self._gravity_feedforward.apply(
                    self.control, enabled=self._gravity_feedforward_enabled
                )
            except (ValueError, RuntimeError) as exc:
                reason = "native gravity feedforward: " + str(exc)
                self.violations.append(
                    {"step": self.steps, "kind": "gravity_feedforward", "reason": str(exc)}
                )
                self.stop(reason)
                if self.frame_callback is not None:
                    frame = self._rgbd_for_current_step()
                    if frame is not None:
                        self.frame_callback(frame)
                raise RuntimeError(reason) from exc
            self.contacts = []
            self._advance_people()
            self.world.step(render=False)
            self.steps += 1
            self._sync_kinematics()
            self._update_force_observer()
            self._audit_step()
            if self.violations:
                reason = "native safety violation: " + self.violations[-1]["kind"]
                self.stop(reason)
                if self.frame_callback is not None:
                    frame = self._rgbd_for_current_step()
                    if frame is not None:
                        self.frame_callback(frame)
                raise RuntimeError(reason)
            if self.steps % max(1, round(0.04 / self.dt)) == 0:
                self._render_current_step()
            if self.frame_callback is not None and self.steps % self._frame_stride == 0:
                frame = self._rgbd_for_current_step()
                if frame is not None:
                    self.frame_callback(frame)
        if render:
            self._render_current_step()

    def _native_poses(self, paths: list[str]) -> tuple[np.ndarray, np.ndarray]:
        return self._pose_snapshot.read(self.steps, paths)

    def _paper_poses(self) -> np.ndarray:
        positions, orientations = self._native_poses(self.native_manifest["paper_bodies"])
        return np.column_stack([positions - self.offset, orientations])

    def enable_recording(self, sampling_period_s: float = 0.04) -> None:
        """Persist camera frames during every bounded primitive, including grip."""
        from PIL import Image

        if not 0.02 <= sampling_period_s <= 0.2:
            raise ValueError("Camera recording period must be 0.02..0.2 seconds")
        frames = self.output / "frames"
        frames.mkdir(exist_ok=False)
        self._frame_stride = max(1, round(sampling_period_s / self.dt))
        self._frame_log = (frames / "timestamps.jsonl").open("w")

        def record(frame: RGBDFrame) -> None:
            path = frames / f"frame_{self.steps:09d}.png"
            Image.fromarray(frame.rgb).save(path, compress_level=1)
            self._frame_log.write(
                json.dumps(
                    {
                        "step": self.steps,
                        "stamp": frame.stamp,
                        "path": str(path.relative_to(self.output)),
                    }
                )
                + "\n"
            )
            self._frame_log.flush()

        self.frame_callback = record
        first = self.observe().rgbd
        if first is not None:
            record(first)

    def _audit_step(self) -> None:
        from airport.isaac_paper import material_vertices

        positions, orientations = self._native_poses([self._base_path, *self._person_paths])
        base_p, base_q = positions[0], orientations[0]
        native_people = positions[1:] - self.offset
        for i, position in enumerate(native_people):
            tracking_error = float(np.linalg.norm(position - self.person_positions[i]))
            if tracking_error > 0.02:
                self._violation("kinematic_actor_tracking_error", person=i, error_m=tracking_error)
        drift = float(np.linalg.norm(np.asarray(base_p) - self._base_initial))
        angular = 1 - abs(float(np.dot(base_q, self._base_rotation_initial)))
        if drift > 0.001 or angular > 1e-5:
            self._violation("fixed_base_drift", translation_m=drift, quaternion_error=angular)
        poses = self._paper_poses()
        vertices = material_vertices(poses)
        if self._previous_paper_poses is not None:
            previous_vertices = material_vertices(self._previous_paper_poses)
            self._paper_max_speed = float(
                np.max(np.linalg.norm(vertices - previous_vertices, axis=1)) / self.dt
            )
        mouth = self.native_manifest["paper_supply"]["mouth_z"]
        if self.held and vertices[:, 2].min() > mouth + 0.005:
            if not self.extracted:
                self.events.append({"step": self.steps, "event": "paper_extracted"})
            self.extracted = True
        contacts_by_region = {}
        for contact in self.contacts:
            actors, shapes = contact["actors"], contact["colliders"]
            paper = any("/Paper/" in actor for actor in actors)
            robot = any("/Robot/" in actor for actor in actors)
            person = next(
                (
                    i
                    for i, spec in enumerate(self.native_manifest["people"])
                    if spec["path"] in actors or spec["bag_path"] in actors
                ),
                None,
            )
            if robot and person is not None:
                self._violation("robot_passenger_contact", actors=actors)
            if not paper or person is None:
                continue
            region = (
                "bag"
                if any("/Bags/" in x for x in actors)
                else ("forearm" if any("forearm" in x for x in shapes) else "protected_body")
            )
            force = float(np.linalg.norm(contact["impulse"]) / self.dt)
            if region == "protected_body":
                self._violation("paper_protected_body_contact", actors=actors)
                continue
            if self.person_states[person] != "inspection":
                self._violation("sampling_person_not_stationary", person=person)
            forward = np.asarray(self.native_manifest["people"][person]["forward"][:2])
            relative = (np.asarray(base_p) - self.offset - native_people[person])[:2]
            lateral_cosine = abs(float(np.dot(forward, relative))) / max(
                np.linalg.norm(relative), 1e-8
            )
            if lateral_cosine > 0.5:
                self._violation(
                    "sampling_not_lateral", person=person, forward_cosine=lateral_cosine
                )
            if force > (1.5 if region == "forearm" else 3.0):
                self._violation("paper_force_limit", force_n=force)
            if self.held and self.extracted and force > 0.0005:
                contacts_by_region.setdefault((person, region), []).append(contact)
        for key, contacts in contacts_by_region.items():
            i, region = key
            metric = self.sampling_metrics[i][region]
            position = np.mean([c["position"] for c in contacts], axis=0)
            force = float(
                np.linalg.norm(np.sum([c["impulse"] for c in contacts], axis=0)) / self.dt
            )
            metric["contact_s"] += self.dt
            metric["peak_force_n"] = max(metric["peak_force_n"], force)
            previous = self._last_contact.get(key)
            if previous is not None and previous[0] == self.steps - 1:
                metric["travel_m"] += float(np.linalg.norm(position - previous[1]))
            self._last_contact[key] = (self.steps, position)
            if self._previous_paper_poses is not None:
                slips = []
                for contact in contacts:
                    body = next(
                        (a for a in contact["actors"] if a in self.native_manifest["paper_bodies"]),
                        None,
                    )
                    if body is None:
                        continue
                    segment = self.native_manifest["paper_bodies"].index(body)
                    current = poses[segment]
                    previous = self._previous_paper_poses[segment]
                    current_rotation = Rotation.from_quat(current[[4, 5, 6, 3]])
                    previous_rotation = Rotation.from_quat(previous[[4, 5, 6, 3]])
                    # Follow the same material point across adjacent physics
                    # frames, rather than accumulating contact-point jitter.
                    point = np.asarray(contact["position"])
                    local = current_rotation.inv().apply(point - current[:3])
                    previous_point = previous[:3] + previous_rotation.apply(local)
                    displacement = point - previous_point
                    normal = np.asarray(contact["normal"])
                    normal /= max(np.linalg.norm(normal), 1e-9)
                    tangent_displacement = displacement - np.dot(displacement, normal) * normal
                    slips.append(float(np.linalg.norm(tangent_displacement)))
                if slips:
                    metric["material_sliding_m"] += float(np.mean(slips))
            extrema = self._contact_positions.setdefault(key, [position.copy(), position.copy()])
            extrema[0] = np.minimum(extrema[0], position)
            extrema[1] = np.maximum(extrema[1], position)
            metric["span_m"] = float(np.linalg.norm(extrema[1] - extrema[0]))
        if self.contacts:
            self._contact_writer.append({"step": self.steps, "contacts": self.contacts})
        self._previous_paper_poses = poses
        if self._is_deposited(vertices) and self._paper_max_speed < 0.05:
            self.deposited_stable_s += self.dt
        else:
            self.deposited_stable_s = 0.0

    def _violation(self, kind: str, **kwargs) -> None:
        if not any(v["kind"] == kind for v in self.violations):
            self.violations.append({"step": self.steps, "kind": kind, **kwargs})

    def _zero_delay_settings_verified(self) -> bool:
        return bool(
            self._render_settings.get("/app/hydraEngine/waitIdle") is True
            and self._render_settings.get("/app/updateOrder/checkForHydraRenderComplete") == 1000
        )

    def _encoder_positions(self) -> dict[str, float]:
        return dict(zip(self.names, map(float, self.robot.get_joint_positions()), strict=True))

    def _render_current_step(self) -> None:
        before_time = float(self.world.current_time)
        before_encoders = measurement_digest(self._encoder_positions())
        self.world.render()
        self._render_epoch = None
        reference = self._reference_time.get_data()
        if isinstance(reference, dict) and "referenceTimeNumerator" in reference:
            number = int(reference["referenceTimeNumerator"])
            render_time = float(self._core_nodes.get_sim_time_at_swh_frame(number))
            after_time = float(self.world.current_time)
            after_encoders = measurement_digest(self._encoder_positions())
            advanced_frame = (
                self._previous_render_frame is None or number > self._previous_render_frame
            )
            self._previous_render_frame = number
            if (
                self.depth_filter_profile is not None
                and self._zero_delay_settings_verified()
                and advanced_frame
                and before_time == after_time
                and before_encoders == after_encoders
                and np.isfinite(render_time)
                and abs(render_time - after_time) <= 1e-6
            ):
                self._render_epoch = {
                    "render_frame": number,
                    "render_time": render_time,
                    "physics_time": after_time,
                    "encoder_sha256": after_encoders,
                }
        self._last_render_step = self.steps
        self._rgbd_cache_step = None

    def _rgbd_for_current_step(self) -> RGBDFrame | None:
        if self._last_render_step != self.steps:
            self._render_current_step()
        if self._rgbd_cache_step == self.steps:
            return self._rgbd_cache
        d = self.description.data
        sensor_frame = np.eye(4)
        sensor_frame[:3, :3] = d.xmat[self.camera_body_id].reshape(3, 3)
        sensor_frame[:3, 3] = d.xpos[self.camera_body_id]
        transform = sensor_frame @ self.camera_mount
        c = self.camera_calibration
        self.camera_calibration = CameraCalibration(
            c.width,
            c.height,
            c.fx,
            c.fy,
            c.cx,
            c.cy,
            tuple(tuple(float(x) for x in row) for row in transform),
            source=c.source,
            pixel_offset_xy=c.pixel_offset_xy,
        )
        rgb = np.asarray(self._annotators["rgb"].get_data()).copy()
        depth = np.asarray(self._annotators["distance_to_image_plane"].get_data()).copy()
        frame = None
        c = self.camera_calibration
        if rgb.shape[:2] == (c.height, c.width) and depth.shape == (c.height, c.width):
            synchrony = None
            if self._render_epoch is not None and self.depth_filter_profile is not None:
                synchrony = RenderSynchrony(
                    profile=self.depth_filter_profile.profile,
                    evidence_sha256=self.depth_filter_profile.evidence_sha256,
                    **self._render_epoch,
                    encoder_stamp=self.steps * self.dt,
                    encoder_sequence=self.steps,
                    calibration_sha256=measurement_digest(asdict(c)),
                    settings_verified=self._zero_delay_settings_verified(),
                )
            frame = RGBDFrame(self.steps * self.dt, rgb[:, :, :3], depth, c, synchrony=synchrony)
        self._rgbd_cache = frame
        self._rgbd_cache_step = self.steps
        return frame

    def observe(self) -> RawMeasuredObservation:
        self._sync_kinematics()
        frame = self._rgbd_for_current_step()
        forces = self._robot_contact_forces()
        wrench = self.read_wrist_wrench()
        gravity = self.predict_wrist_gravity()
        payload_known = self._payload_known(forces)
        valid_bias = self._wrist_bias is not None and payload_known
        expected_unloaded = None if self._wrist_bias is None else self._wrist_bias + gravity
        base_positions, base_orientations = self._native_poses([self._base_path])
        base_position = base_positions[0] - self.offset
        base_quaternion = base_orientations[0]
        base_yaw = Rotation.from_quat(np.asarray(base_quaternion)[[1, 2, 3, 0]]).as_euler("xyz")[2]
        estimate = self._force_observer.estimate if valid_bias else None
        return RawMeasuredObservation(
            self.steps * self.dt,
            self.steps,
            dict(zip(self.names, map(float, self.robot.get_joint_positions()), strict=True)),
            dict(zip(self.names, map(float, self.robot.get_joint_velocities()), strict=True)),
            frame,
            forces,
            wrist_wrench6=None if wrench is None else tuple(map(float, wrench)),
            wrist_bias6=None if expected_unloaded is None else tuple(map(float, expected_unloaded)),
            wrist_bias_stamp=self._wrist_bias_stamp,
            wrist_bias_valid=bool(valid_bias),
            robot_base_pose=(*map(float, base_position), float(base_yaw)),
            wrist_zero_offset6=None
            if self._wrist_bias is None
            else tuple(map(float, self._wrist_bias)),
            wrist_gravity_wrench6=tuple(map(float, gravity)),
            wrist_reaction_world_n=None if estimate is None else estimate.reaction_world_n,
            wrist_force_window_start=None if estimate is None else estimate.window_start,
            wrist_force_window_end=None if estimate is None else estimate.window_end,
            wrist_force_stamp=None if estimate is None else estimate.stamp,
        )

    def _robot_contact_forces(self) -> dict[str, float]:
        forces = {
            name: 0.0 for name in ("right_claw_link", "right_claw_link_g", "right_wrist_pitch_link")
        }
        for contact in self.contacts:
            for name in forces:
                if any(actor.endswith("/" + name) for actor in contact["actors"]):
                    forces[name] += float(np.linalg.norm(contact["impulse"]) / self.dt)
        return forces

    def _payload_known(self, forces) -> bool:
        return not self.held or (
            self.two_finger_qualified
            and self._payload_com_wrist is not None
            and min(forces["right_claw_link"], forces["right_claw_link_g"]) > 0.05
        )

    def _update_force_observer(self) -> None:
        if self._wrist_bias is None or not self._payload_known(self._robot_contact_forces()):
            self._force_observer.reset()
            return
        wrench = self.read_wrist_wrench()
        if wrench is None:
            self._force_observer.reset()
            return
        data = self.description.data
        rotation = data.xmat[self.description.wrist_id].reshape(3, 3) @ self._wrist_joint_rotation
        residual = rotation @ (wrench[:3] - self.predict_wrist_gravity()[:3] - self._wrist_bias[:3])
        payload_mass = 0.0006 if self.held else 0.0
        try:
            momentum = self._momentum_kinematics.momentum(
                data, self.robot.get_joint_velocities(), payload_mass, self._payload_com_wrist
            )
        except ValueError:
            self._force_observer.reset()
            return
        key = (
            self._wrist_bias_stamp,
            tuple(self._wrist_bias),
            payload_mass,
            None if self._payload_com_wrist is None else tuple(self._payload_com_wrist),
        )
        self._force_observer.update(self.steps * self.dt, residual, momentum, key)

    def _warm_force_observer(self) -> None:
        # Explicit bounded sensor warmup; never re-zero while touching an object.
        for _ in range(2 * self._force_observer.samples + 2):
            self.step()
            if self._force_observer.estimate is not None:
                break

    def predict_wrist_gravity(self) -> np.ndarray:
        """CAD-only distal load prediction in the native joint child frame."""
        model, data = self.description.model, self.description.data
        wrist = self.description.wrist_id
        rotation = data.xmat[wrist].reshape(3, 3)
        joint_rotation = rotation @ self._wrist_joint_rotation
        joint_position = data.xpos[wrist] + rotation @ self._wrist_joint_position
        force, torque = np.zeros(3), np.zeros(3)
        if self.config.robot_gravity_enabled:
            for bid in self._wrist_distal_ids:
                support = np.array([0.0, 0.0, float(model.body_mass[bid]) * 9.81])
                com = data.xpos[bid] + data.xmat[bid].reshape(3, 3) @ model.body_ipos[bid]
                force += support
                torque += np.cross(com - joint_position, support)
        if self.held and self.two_finger_qualified and self._payload_com_wrist is not None:
            support = np.array([0.0, 0.0, 0.0006 * 9.81])
            com = data.xpos[wrist] + rotation @ self._payload_com_wrist
            force += support
            torque += np.cross(com - joint_position, support)
        return np.r_[joint_rotation.T @ force, joint_rotation.T @ torque]

    def read_wrist_wrench(self) -> np.ndarray | None:
        values = self.robot.get_measured_joint_forces(
            joint_indices=np.array([self._wrist_force_row])
        )
        if values is None:
            return None
        wrench = np.asarray(values, float).reshape(-1, 6)[0]
        return wrench if np.isfinite(wrench).all() else None

    def calibrate_wrist_bias(self, samples: int = 100) -> dict:
        """Explicit unloaded, stationary calibration; caller must select free space.

        This consumes only joint encoders and a native reaction wrench, never
        object identities. CAD gravity is removed before fitting a constant
        sensor offset; no automatic recalibration occurs during contact.
        """
        if not 20 <= samples <= 1000:
            raise ValueError("Wrench calibration requires 20..1000 physics samples")
        readings = []
        for _ in range(samples):
            self.step()
            if np.max(np.abs(self.robot.get_joint_velocities())) > 0.03:
                raise ValueError("Wrist bias calibration requires stationary joints")
            wrench = self.read_wrist_wrench()
            if wrench is None:
                raise RuntimeError("Native joint wrench sensor unavailable")
            readings.append(wrench - self.predict_wrist_gravity())
        readings = np.asarray(readings)
        if np.max(np.std(readings[:, :3], axis=0)) > 0.02:
            raise ValueError("Unstable unloaded force prevents bias calibration")
        self._wrist_bias = np.mean(readings, axis=0)
        self._wrist_bias_stamp = self.steps * self.dt
        self._wrist_bias_rotation = (
            self.description.data.xmat[self.description.wrist_id].reshape(3, 3).copy()
        )
        self._wrist_bias_held = self.held
        self._force_observer.reset()
        self._warm_force_observer()
        return {
            "bias6": self._wrist_bias.tolist(),
            "stamp": self._wrist_bias_stamp,
            "samples": samples,
            "held_payload": self.held,
            "source": "native_joint_wrench_minus_CAD_gravity_stationary_free_space_bias",
        }

    def grip(self, close: bool, opening_rad: float = 0.5) -> dict:
        from pxr import Gf, Sdf, UsdPhysics

        joint_path = "/World/Task/checkpoint_assisted_grasp"
        payload_before = self.held
        if not close and self.held:
            self.stage.RemovePrim(joint_path)
            self.held = False
            self._payload_com_wrist = None
            self.release_after_batch = self._all_sampled() and all(
                s == "waiting" for s in self.person_states
            )
            self.events.append({"step": self.steps, "event": "paper_released"})
        fingers = ("right_claw_joint", "right_claw_joint_g")
        start = np.array([self.robot.get_joint_positions()[self.names.index(n)] for n in fingers])
        target = np.zeros(2) if close else np.array([opening_rad, -opening_rad])
        duration = max(1, round(0.65 / self.dt))
        touched = set()
        for i in range(duration + (round(1.2 / self.dt) if close else 0)):
            alpha = min(1.0, i / max(1, duration - 1))
            alpha = alpha * alpha * (3 - 2 * alpha)
            q = start + alpha * (target - start)
            self.apply_joint_targets(dict(zip(fingers, map(float, q), strict=True)))
            self.step()
            touched = set()
            for contact in self.contacts:
                if self.native_manifest["paper_bodies"][0] in contact["actors"]:
                    for name in ("right_claw_link", "right_claw_link_g"):
                        if any(p.endswith("/" + name) for p in contact["actors"]):
                            touched.add(name)
            if close and not self.held and len(touched) == 2:
                positions, orientations = self._native_poses(
                    [
                        self.native_manifest["body_paths"][self.description.wrist_id],
                        self.native_manifest["paper_bodies"][0],
                    ]
                )
                wp, pp = positions
                wq, pq = orientations
                wr = Rotation.from_quat(np.asarray(wq)[[1, 2, 3, 0]])
                pr = Rotation.from_quat(np.asarray(pq)[[1, 2, 3, 0]])
                position = wr.inv().apply(np.asarray(pp) - np.asarray(wp))
                rotation = (wr.inv() * pr).as_quat()
                joint = UsdPhysics.FixedJoint.Define(self.stage, joint_path)
                joint.CreateBody0Rel().SetTargets(
                    [Sdf.Path(self.native_manifest["body_paths"][self.description.wrist_id])]
                )
                joint.CreateBody1Rel().SetTargets(
                    [Sdf.Path(self.native_manifest["paper_bodies"][0])]
                )
                joint.CreateLocalPos0Attr(Gf.Vec3f(*position))
                joint.CreateLocalRot0Attr(Gf.Quatf(float(rotation[3]), Gf.Vec3f(*rotation[:3])))
                joint.CreateLocalPos1Attr(Gf.Vec3f(0.0))
                joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
                joint.CreateCollisionEnabledAttr(False)
                self.held = self.two_finger_qualified = True
                # Declared tool calibration, independent of hidden paper pose.
                # Only the force (mass*g) is exact; nominal COM torque is an
                # explicit approximation for this flexible strip.
                self._payload_com_wrist = self.description.model.site_pos[
                    self.description.eef_id
                ].copy() + np.array([0.0, 0.0, -0.085])
                self.events.append({"step": self.steps, "event": "two_finger_assisted_grasp"})
                measured = self.robot.get_joint_positions()
                # Keep a small physical inward preload after qualifying contact;
                # the assisted weld itself is never used as pressure feedback.
                self.apply_joint_targets(
                    {
                        name: float(
                            np.sign(measured[self.names.index(name)])
                            * max(0.0, abs(measured[self.names.index(name)]) - 0.01)
                        )
                        for name in fingers
                    }
                )
                break
        if self.held != payload_before:
            self._force_observer.reset()
            self._warm_force_observer()
        return {
            "held": self.held,
            "two_finger_contact": len(touched) == 2,
            "assisted_weld": self.held,
            "friction_only_validation": False,
        }

    def _all_sampled(self) -> bool:
        return all(
            m["contact_s"] >= 0.25 and m["material_sliding_m"] >= 0.06 and m["span_m"] >= 0.05
            for person in self.sampling_metrics
            for m in person.values()
        )

    def _is_deposited(self, vertices: np.ndarray) -> bool:
        return bool(
            not self.held
            and self.extracted
            and self.release_after_batch
            and np.all(np.abs(vertices[:, 0] - 0.38) < 0.17)
            and np.all(np.abs(vertices[:, 1] + 0.95) < 0.11)
            and np.all((vertices[:, 2] > 0.76) & (vertices[:, 2] < 0.82))
        )

    def diagnostics(self) -> dict:
        from airport.isaac_paper import material_vertices

        vertices = material_vertices(self._paper_poses())
        # Existing return tray is centred (.38,-.95,.76), top .775.
        deposited = self._is_deposited(vertices) and self.deposited_stable_s >= 0.25
        all_sampled = self._all_sampled()
        native_people = self._native_poses(self._person_paths)[0] - self.offset
        all_waiting = all(
            state == "waiting"
            and np.linalg.norm(position - np.asarray(spec["waiting_position"])) < 0.02
            for state, position, spec in zip(
                self.person_states, native_people, self.native_manifest["people"], strict=True
            )
        )
        success = bool(
            self.two_finger_qualified
            and self.extracted
            and all_sampled
            and all_waiting
            and deposited
            and not self.violations
        )
        return {
            "privileged": True,
            "engine": "isaac_physx",
            "execution_mode": "native_live",
            "replay": False,
            "steps": self.steps,
            "stamp": self.steps * self.dt,
            "native_success": success,
            "success_source": "native_environment",
            "held": self.held,
            "two_finger_qualified": self.two_finger_qualified,
            "extracted": self.extracted,
            "deposited": bool(deposited),
            "release_after_batch": self.release_after_batch,
            "deposited_stable_s": self.deposited_stable_s,
            "paper_max_speed_m_s": self._paper_max_speed,
            "person_states": self.person_states,
            "sampling_metrics": self.sampling_metrics,
            "native_person_positions": [p.tolist() for p in native_people],
            "paper_poses": self._paper_poses().tolist(),
            "violations": self.violations,
            "events": self.events,
            "gravity_feedforward_max_limit_fraction": (
                self._gravity_feedforward.max_observed_fraction
            ),
            "limitations": [
                "assisted weld after native two-finger contact",
                "uncalibrated segmented paper",
                "procedural kinematic passengers, no gait dynamics",
            ],
        }

    def stop(self, reason: str) -> None:
        self.control = np.asarray(self.robot.get_joint_positions()).copy()
        self.robot.apply_action(
            self.action_type(
                joint_positions=self.control, joint_efforts=np.zeros_like(self.control)
            )
        )
        self.events.append({"step": self.steps, "event": "stop", "reason": reason})
        self.stopped = True

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        (self.output / "checkpoint_native.json").write_text(
            json.dumps(self.diagnostics(), indent=2) + "\n"
        )
        contact_manifest = self._contact_writer.close()
        (self.output / "native_contacts_manifest.json").write_text(
            json.dumps(contact_manifest, indent=2) + "\n"
        )
        self.contact_subscription = None
        if self._frame_log is not None:
            self._frame_log.close()
