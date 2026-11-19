"""Gate-six checkpoint layout built around the existing native robot converter.

MJCF is used only for immutable geometry and forward kinematics. All dynamic
steps, paper bodies, bag joints and robot articulation belong to PhysX.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .appearance import LIGHTING_PROFILES, PAPER_PROFILES, configure_paper

ZERO_DELAY_EXPERIENCE = "/isaac-sim/apps/isaacsim.exp.base.zero_delay.kit"
ZERO_DELAY_EXPERIENCE_SHA256 = "c57651bd0f54b51881736bb2dde125eb8ee0929bd379bf76bfc614c4bca88683"
ZERO_DELAY_QUALIFICATION_SHA256 = "397324c45ab259338197219d49d155075207daf031f4d6a70e5e589013944d94"


@dataclass(frozen=True)
class CheckpointSceneConfig:
    batch_size: int = 3
    seed: int = 1
    offset: tuple[float, float, float] = (-9.83, 3.61, 0.0)
    # The robot is beside the queue: people travel along y, robot stays at x=0.
    inspection_xy: tuple[float, float] = (1.03, 0.04)
    queue_spacing_m: float = 0.85
    walking_speed_m_s: float = 0.35
    passenger_yaw_deg: float = -90.0
    elbow_deg: float = 0.0
    bag_relative: tuple[float, float, float] = (-0.27, -0.28, 0.58)
    robot_gravity_enabled: bool = True
    gravity_feedforward_enabled: bool = True
    camera_mount_mode: str = "urdf_sensor"
    # Qualified on current fixed Isaac 4.5 image, not a hardware timing claim.
    render_profile: str = "zero_delay_v1"
    # Experimental simulation band, not a certified hardware uncertainty bound.
    self_depth_epsilon_m: float = 20e-6
    self_depth_backend: str = "cpu"
    # Experimental optical center outside sens_link CAD shell (zmax 3.58 mm).
    sensor_forward_offset_m: float = 0.005
    initial_head_angles: tuple[float, float] = (-0.45, 0.2)
    passengers: tuple[dict, ...] = ()
    paper_length_m: float = 0.17
    paper_exposed_fraction: float = 0.5
    paper_material_profile: str = "neutral_white"
    lighting_profile: str = "indoor_soft_neutral"

    def __post_init__(self) -> None:
        if self.self_depth_backend not in {"cpu", "cuda_verified"}:
            raise ValueError("Unknown own-geometry compute backend")
        if self.paper_material_profile not in PAPER_PROFILES:
            raise ValueError("Unknown paper material profile")
        if self.lighting_profile not in LIGHTING_PROFILES:
            raise ValueError("Unknown lighting profile")
        if self.render_profile not in {"default", "zero_delay_v1"}:
            raise ValueError("Unknown native render profile")
        if not np.isfinite(self.self_depth_epsilon_m) or not (
            0 < self.self_depth_epsilon_m <= 100e-6
        ):
            raise ValueError("Experimental own-depth band must be within 0..100 micrometres")
        if self.camera_mount_mode not in {"legacy_bracket", "urdf_sensor"}:
            raise ValueError("Unknown camera mount mode")
        if not np.isfinite(self.sensor_forward_offset_m) or not (
            0.004 <= self.sensor_forward_offset_m <= 0.010
        ):
            raise ValueError("Experimental sensor forward offset must be 4..10 mm")
        angles = np.asarray(self.initial_head_angles, dtype=float)
        if angles.shape != (2,) or not np.isfinite(angles).all():
            raise ValueError("Initial head angles must be finite yaw/pitch radians")
        if not 1 <= self.batch_size <= 12:
            raise ValueError("Batch size must be between one and twelve")
        if self.queue_spacing_m < 0.65 or not 0 < self.walking_speed_m_s <= 0.6:
            raise ValueError("Unsafe queue spacing or walking speed")
        if self.paper_length_m != 0.17 or self.paper_exposed_fraction != 0.5:
            raise ValueError("Native paper geometry requires a 170 mm half-exposed strip")
        if self.passengers and len(self.passengers) != self.batch_size:
            raise ValueError("Passenger specifications must match batch size")
        for spec in self.passengers:
            if spec["bag_side"] not in {"left", "right"} or not 1.4 <= spec["height_m"] <= 2.1:
                raise ValueError("Unsupported passenger anthropometry")
            if spec["arrival_delay_s"] < 0 or spec["dwell_s"] < 0:
                raise ValueError("Passenger timing must be nonnegative")


def load_robot_description(output: Path, config: CheckpointSceneConfig) -> SimpleNamespace:
    """Create a read-only kinematic description without any old simulation object."""
    import mujoco

    from airport.cases import Case
    from airport.safe_scene import build_safe_scene

    path = build_safe_scene(
        output,
        Case(
            "checkpoint",
            x=config.inspection_xy[0],
            y=config.inspection_xy[1],
            elbow_deg=config.elbow_deg,
        ),
        paper_orientation="vertical",
        timestep=0.0001,
    )
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    names = [model.joint(int(j)).name for j in model.actuator_trnid[:, 0]]
    qpos_ids = np.array([model.joint(n).qposadr[0] for n in names])
    home = {
        "base_x": -0.15,
        "base_y": 0.0,
        "ankle_pitch_joint": -0.70,
        "knee_pitch_joint": 1.30,
        "hip_pitch_joint": -0.60,
        "left_elbow_joint": -0.15,
        "right_shoulder_pitch_joint": -0.23584522,
        "right_shoulder_roll_joint": -0.90905186,
        "right_shoulder_yaw_joint": 0.57068643,
        "right_elbow_joint": -1.91924152,
        "right_wrist_yaw_joint": -0.88219635,
        "right_wrist_roll_joint": -0.62036652,
        "right_wrist_pitch_joint": 1.56978553,
        "right_claw_joint": 0.45,
        "right_claw_joint_g": -0.45,
    }
    head_names = ("neck_yaw_joint", "neck_pitch_joint")
    for name, value in zip(head_names, config.initial_head_angles, strict=True):
        joint = model.joint(name)
        low, high = model.jnt_range[joint.id]
        if not model.jnt_limited[joint.id] or not low <= value <= high:
            raise ValueError(f"Initial head angle outside authored joint limits: {name}={value}")
    effective_head_angles = (
        config.initial_head_angles if config.camera_mount_mode == "urdf_sensor" else (0.0, 0.0)
    )
    home.update(zip(head_names, effective_head_angles, strict=True))
    control = np.array([home.get(n, 0.0) for n in names])
    data.qpos[qpos_ids] = control
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    return SimpleNamespace(
        output=output,
        model=model,
        data=data,
        joint_names=names,
        qpos_ids=qpos_ids,
        control=control,
        sheet_id=model.body("sampling_sheet").id,
        wrist_id=model.body("right_wrist_pitch_link").id,
        eef_id=model.site("eef").id,
    )


def _fix_carriage(stage, robot_path: str) -> None:
    from pxr import Sdf, UsdPhysics

    for name in ("base_x", "base_y"):
        path = robot_path + "/" + name
        old = UsdPhysics.Joint(stage.GetPrimAtPath(path))
        bodies = (old.GetBody0Rel().GetTargets(), old.GetBody1Rel().GetTargets())
        pos = (old.GetLocalPos0Attr().Get(), old.GetLocalPos1Attr().Get())
        rot = (old.GetLocalRot0Attr().Get(), old.GetLocalRot1Attr().Get())
        # Initial base_x=-0.15 must be encoded in the fixed joint frame.
        body1 = stage.GetPrimAtPath(bodies[1][0])
        body0 = stage.GetPrimAtPath(bodies[0][0])
        p1 = body1.GetAttribute("xformOp:translate").Get()
        p0 = body0.GetAttribute("xformOp:translate").Get()
        from pxr import Gf

        pos0 = Gf.Vec3f(*(np.asarray(p1) - np.asarray(p0) + np.asarray(pos[1])))
        stage.RemovePrim(Sdf.Path(path))
        joint = UsdPhysics.FixedJoint.Define(stage, path)
        joint.CreateBody0Rel().SetTargets(bodies[0])
        joint.CreateBody1Rel().SetTargets(bodies[1])
        joint.CreateLocalPos0Attr(pos0)
        joint.CreateLocalPos1Attr(pos[1])
        joint.CreateLocalRot0Attr(rot[0])
        joint.CreateLocalRot1Attr(rot[1])
        joint.CreateCollisionEnabledAttr(False)


def build_checkpoint_scene(
    stage, asset_root: Path, description, config: CheckpointSceneConfig
) -> dict:
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade

    from airport.isaac_scene import build_native_scene

    manifest = build_native_scene(description, stage, asset_root, np.asarray(config.offset))
    _fix_carriage(stage, manifest["robot_path"])
    for body_path in manifest["body_paths"].values():
        PhysxSchema.PhysxRigidBodyAPI(stage.GetPrimAtPath(body_path)).GetDisableGravityAttr().Set(
            not config.robot_gravity_enabled
        )
    task = "/World/Task"
    root = task + "/Checkpoint"
    UsdGeom.Xform.Define(stage, root)

    def box(path, position, half_size, color, collision=True):
        geom = UsdGeom.Cube.Define(stage, path)
        geom.CreateSizeAttr(2.0)
        xf = UsdGeom.Xformable(geom)
        xf.AddTranslateOp().Set(Gf.Vec3d(*position))
        xf.AddScaleOp().Set(Gf.Vec3f(*half_size))
        geom.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        if collision:
            UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
            api = PhysxSchema.PhysxCollisionAPI.Apply(geom.GetPrim())
            api.CreateContactOffsetAttr(0.0002)
            api.CreateRestOffsetAttr(0.0)
        return geom.GetPrim()

    # Remove the old supply clamp and backpack/standalone static passenger.
    model, data = description.model, description.data
    passenger_id = model.body("passenger").id
    passenger_geoms = []
    supply_geoms = []
    for path, name in list(manifest["geom_paths"].items()):
        if "/Fixtures/" not in path:
            continue
        # The converter assigns geom_<id> to unnamed MJCF geoms; resolve from
        # its stable g_#### prefix rather than looking up a nonexistent name.
        gid = int(path.rsplit("/", 1)[-1].split("_", 2)[1])
        bid = int(model.geom_bodyid[gid])
        ancestor = bid
        while ancestor and ancestor != passenger_id:
            ancestor = int(model.body_parentid[ancestor])
        if ancestor == passenger_id:
            passenger_geoms.append((path, name))
        if bid == model.body("supply_tray").id:
            supply_geoms.append(path)

    people = []
    source_origin = data.xpos[passenger_id].copy()
    from scipy.spatial.transform import Rotation

    visible_z = []
    for path, name in passenger_geoms:
        if name not in {"m011_body", "m011_head", "forearm_visual"}:
            continue
        gid = int(path.rsplit("/", 1)[-1].split("_", 2)[1])
        mid = model.geom_dataid[gid]
        a, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        points = (
            model.mesh_vert[a : a + count] @ data.geom_xmat[gid].reshape(3, 3).T
            + data.geom_xpos[gid]
        )
        visible_z.extend(points[:, 2].tolist())
    template_height = max(visible_z) - min(visible_z)
    for number in range(config.batch_size):
        parameters = (
            dict(config.passengers[number])
            if config.passengers
            else {
                "person_id": f"person-{number + 1}",
                "queue_offset_m": number * config.queue_spacing_m,
                "stop_offset_xy_m": [0.0, 0.0],
                "yaw_rad": 0.0,
                "height_m": template_height,
                "bag_side": "left",
                "bag_offset_xyz_m": [0.0, 0.0, 0.0],
                "arrival_delay_s": 0.0,
                "dwell_s": 0.0,
            }
        )
        height_scale = parameters["height_m"] / template_height
        person_rotation = Rotation.from_euler(
            "z", np.deg2rad(config.passenger_yaw_deg) + parameters["yaw_rad"]
        )
        person_quat = person_rotation.as_quat()
        inspection_xy = np.asarray(config.inspection_xy) + np.asarray(
            parameters["stop_offset_xy_m"]
        )
        path = root + f"/People/person_{number:02d}"
        start = np.array(
            [
                inspection_xy[0],
                inspection_xy[1] + config.queue_spacing_m + parameters["queue_offset_m"],
                0.0,
            ]
        )
        prim = UsdGeom.Xform.Define(stage, path).GetPrim()
        UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*start))
        UsdGeom.Xformable(prim).AddOrientOp().Set(
            Gf.Quatf(float(person_quat[3]), Gf.Vec3f(*person_quat[:3]))
        )
        rb = UsdPhysics.RigidBodyAPI.Apply(prim)
        rb.CreateKinematicEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(70.0 * height_scale**3)
        PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
        for original, name in passenger_geoms:
            # Existing backpack is replaced by a hand-carried dynamic bag.
            if "backpack" in name or "bag" in name or "shoulder_strap" in name:
                continue
            dest = path + "/" + stage.GetPrimAtPath(original).GetName()
            Sdf.CopySpec(stage.GetRootLayer(), original, stage.GetRootLayer(), dest)
            clone = stage.GetPrimAtPath(dest)
            position = np.asarray(clone.GetAttribute("xformOp:translate").Get())
            clone.GetAttribute("xformOp:translate").Set(
                Gf.Vec3d(*((position - source_origin) * height_scale))
            )
            if clone.IsA(UsdGeom.Mesh):
                from pxr import Vt

                mesh = UsdGeom.Mesh(clone)
                points = np.asarray(mesh.GetPointsAttr().Get()) * height_scale
                mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
            elif clone.IsA(UsdGeom.Sphere):
                sphere = UsdGeom.Sphere(clone)
                sphere.GetRadiusAttr().Set(sphere.GetRadiusAttr().Get() * height_scale)
            if clone.HasAPI(UsdPhysics.CollisionAPI) and clone.IsA(UsdGeom.Mesh):
                UsdPhysics.MeshCollisionAPI(clone).GetApproximationAttr().Set("convexHull")
        bag_path = root + f"/Bags/bag_{number:02d}"
        bag_relative = np.asarray(config.bag_relative, float).copy() * height_scale
        bag_relative[1] = abs(bag_relative[1]) * (-1 if parameters["bag_side"] == "left" else 1)
        bag_relative += np.asarray(parameters["bag_offset_xyz_m"])
        bag = UsdGeom.Xform.Define(stage, bag_path).GetPrim()
        UsdGeom.Xformable(bag).AddTranslateOp().Set(
            Gf.Vec3d(*(start + person_rotation.apply(bag_relative)))
        )
        UsdGeom.Xformable(bag).AddOrientOp().Set(
            Gf.Quatf(float(person_quat[3]), Gf.Vec3f(*person_quat[:3]))
        )
        UsdPhysics.RigidBodyAPI.Apply(bag)
        UsdPhysics.MassAPI.Apply(bag).CreateMassAttr(1.2)
        PhysxSchema.PhysxContactReportAPI.Apply(bag).CreateThresholdAttr(0.0)
        # Reuse the already converted, textured, pinned GSO bag mesh. The newer
        # checkout's USD is an LFS pointer on some hosts, so it is not a usable
        # fallback. This original mesh/texture is covered by airport-assets.lock.
        original_bag = next(p for p, n in passenger_geoms if n == "backpack_collision")
        dest = bag_path + "/scanned_bag"
        Sdf.CopySpec(stage.GetRootLayer(), original_bag, stage.GetRootLayer(), dest)
        visual = UsdGeom.Mesh(stage.GetPrimAtPath(dest))
        points = np.asarray(visual.GetPointsAttr().Get())
        center = (points.min(axis=0) + points.max(axis=0)) / 2
        orientation = visual.GetPrim().GetAttribute("xformOp:orient").Get()
        rotation = Rotation.from_quat([*orientation.GetImaginary(), orientation.GetReal()])
        visual.GetPrim().GetAttribute("xformOp:translate").Set(Gf.Vec3d(*(-rotation.apply(center))))
        UsdPhysics.MeshCollisionAPI(visual).GetApproximationAttr().Set("convexHull")
        carry = UsdPhysics.FixedJoint.Define(stage, root + f"/carry_{number:02d}")
        carry.CreateBody0Rel().SetTargets([Sdf.Path(path)])
        carry.CreateBody1Rel().SetTargets([Sdf.Path(bag_path)])
        carry.CreateLocalPos0Attr(Gf.Vec3f(*bag_relative))
        carry.CreateLocalPos1Attr(Gf.Vec3f(0.0))
        carry.CreateCollisionEnabledAttr(False)
        people.append(
            {
                "path": path,
                "bag_path": bag_path,
                "initial_position": start.tolist(),
                "inspection_position": [*inspection_xy, 0.0],
                "waiting_position": [inspection_xy[0] + 0.7, -1.0 - number * 0.7, 0.0],
                "forward": person_rotation.apply([1.0, 0.0, 0.0]).tolist(),
                "parameters": parameters,
                "height_scale": height_scale,
                "bag_relative": bag_relative.tolist(),
                "native_yaw_deg": float(
                    np.rad2deg(np.deg2rad(config.passenger_yaw_deg) + parameters["yaw_rad"])
                ),
            }
        )

    for path, _ in passenger_geoms:
        stage.RemovePrim(path)
        manifest["geom_paths"].pop(path, None)
    for path in supply_geoms:
        stage.RemovePrim(path)
        manifest["geom_paths"].pop(path, None)
    stage.RemovePrim(manifest["supply_joint"])
    manifest["supply_joint"] = None

    # A physical slit contains exactly the bottom half. There is no supply weld
    # and no teleport on extraction; contact and gravity retain the strip.
    paper_top = data.xpos[description.sheet_id].copy()
    mouth = float(paper_top[2] - config.paper_length_m * config.paper_exposed_fraction)
    bottom = float(paper_top[2] - config.paper_length_m)
    x, y = paper_top[:2]
    for sign in (-1, 1):
        box(
            root + f"/Supply/face_{sign:+d}".replace("+", "p").replace("-", "m"),
            [x, y + sign * 0.0045, (mouth + bottom) / 2],
            [0.026, 0.004, (mouth - bottom) / 2],
            [0.2, 0.55, 0.7],
        )
        box(
            root + f"/Supply/side_{sign:+d}".replace("+", "p").replace("-", "m"),
            [x + sign * 0.023, y, (mouth + bottom) / 2],
            [0.005, 0.0085, (mouth - bottom) / 2],
            [0.2, 0.55, 0.7],
        )
    box(root + "/Supply/bottom", [x, y, bottom - 0.004], [0.028, 0.0085, 0.004], [0.2, 0.55, 0.7])
    box(
        root + "/Supply/pedestal",
        [x, y, (bottom - 0.008) / 2],
        [0.04, 0.04, (bottom - 0.008) / 2],
        [0.35, 0.4, 0.45],
    )

    # Existing photo-scene fences are retained. These floor markings are visual
    # task cues, not invisible collision or navigation helpers.
    box(
        root + "/inspection_mark",
        [*config.inspection_xy, 0.003],
        [0.3, 0.32, 0.003],
        [0.75, 0.65, 0.10],
        collision=False,
    )
    box(
        root + "/waiting_mark",
        [1.73, -1.6, 0.003],
        [0.4, 1.0, 0.003],
        [0.15, 0.6, 0.35],
        collision=False,
    )

    material = UsdShade.Material.Define(stage, root + "/PaperContactMaterial")
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr(0.8)
    physics_material.CreateDynamicFrictionAttr(0.6)
    physics_material.CreateRestitutionAttr(0.0)
    for path in manifest["paper_bodies"]:
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(path)).Bind(
            material, materialPurpose="physics"
        )

    manifest.update(
        checkpoint=asdict(config),
        paper_appearance=configure_paper(
            stage, manifest["paper_bodies"], config.paper_material_profile
        ),
        people=people,
        fixed_base=True,
        forbidden_joints=["base_x", "base_y"],
        robot_gravity_enabled=config.robot_gravity_enabled,
        robot_gravity_model="native gravity"
        if config.robot_gravity_enabled
        else "idealized gravity compensated robot",
        paper_supply={
            "mouth_z": mouth,
            "bottom_z": bottom,
            "top_z": float(paper_top[2]),
            "exposed_fraction": 0.5,
            "initial_hold_constraint": False,
        },
        passenger_motion="native kinematic rigid actors; procedural queue; no gait model",
        bag_model=(
            "native dynamic scanned Jansport bag fixed to passenger hand-side frame; rigid straps"
        ),
        bag_asset={
            "source": "existing converted backpack_collision textured mesh",
            "lock": "configs/airport-assets.lock.json",
            "model_obj_sha256": "d1c209c1f66b540864dfca25d269752e39bbc5a4be8cbfc2da5579bff337e531",
        },
        execution_mode="native_live",
        replay=False,
        measured_sensor_boundary="RGB, axial depth, encoders, robot contact channels only",
    )
    return manifest
