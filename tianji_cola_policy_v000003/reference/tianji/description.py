"""Read-only URDF geometry and the delivered manufacturer's FK/IK library."""

from __future__ import annotations

import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "assets/airport/fq-assets-isaac/assets/user_supplied/tianji_robot"
URDF = ASSET / "model.urdf"
BUNDLE = ROOT / "assets/tianji/m6_v4/vendor_bundle"
BASE_JOINTS = {"ZQ", "ZQL", "YQ", "YQL", "ZH", "ZHL", "YH", "YHL"}
HOME_DEG = (-2, -2.7, 20, -85.69, 73.064, -1.9476, 15.978)


class Description:
    def __init__(self):
        self.tree = ET.parse(URDF)
        self.joints = self.tree.findall("joint")
        self.by_name = {j.get("name"): j for j in self.joints}
        self.home = {
            **dict(zip((f"Joint{i}_L" for i in range(1, 8)), np.radians(HOME_DEG), strict=True)),
            **dict(
                zip(
                    (f"Joint{i}_R" for i in range(1, 8)),
                    np.radians(np.asarray(HOME_DEG) * [-1, 1, -1, 1, -1, 1, -1]),
                    strict=True,
                )
            ),
            "head_pitch_joint": -0.30,
        }
        from .rhino import joint_targets

        self.home.update(joint_targets(1.0, "left"))
        self.home.update(joint_targets(1.0, "right"))
        self.mimic = {
            j.get("name"): (
                j.find("mimic").get("joint"),
                float(j.find("mimic").get("multiplier", "1")),
                float(j.find("mimic").get("offset", "0")),
            )
            for j in self.joints
            if j.find("mimic") is not None
        }

        # The asset has no transmission. Add an explicit simulation-only,
        # single-DOF approximation; constrain mirrored and parallel jaw motion.
        for hand in ("left", "right"):
            opened = joint_targets(1.0, hand)
            master = hand + "_rhino_v6_driver_L"
            for name, value in opened.items():
                if name != master:
                    self.mimic[name] = (master, value / opened[master], 0.0)

    def horizontal_home(self) -> dict[str, float]:
        """Straight, forward-facing arms; keep the factory IK reference separate."""
        home = dict(self.home)
        for side in ("L", "R"):
            home.update({f"Joint{i}_{side}": 0.0 for i in range(1, 8)})
            home[f"Joint2_{side}"] = -np.pi / 2
        home["head_pitch_joint"] = 0.05
        return home

    def forward(self, angles: dict[str, float]) -> dict[str, np.ndarray]:
        q = dict(angles)
        for name, (source, scale, offset) in self.mimic.items():
            q.setdefault(name, scale * q.get(source, 0.0) + offset)
        poses = {"chassis_base": np.eye(4)}
        pending = self.joints.copy()
        while pending:
            count = len(pending)
            for joint in pending.copy():
                parent = joint.find("parent").get("link")
                if parent not in poses:
                    continue
                origin = joint.find("origin")
                attrs = {} if origin is None else origin.attrib
                transform = np.eye(4)
                transform[:3, 3] = np.fromstring(attrs.get("xyz", "0 0 0"), sep=" ")
                transform[:3, :3] = Rotation.from_euler(
                    "xyz", np.fromstring(attrs.get("rpy", "0 0 0"), sep=" ")
                ).as_matrix()
                motion = np.eye(4)
                if joint.get("type") in {"revolute", "continuous"}:
                    axis = joint.find("axis")
                    vector = np.fromstring(
                        "1 0 0" if axis is None else axis.get("xyz", "1 0 0"), sep=" "
                    )
                    motion[:3, :3] = Rotation.from_rotvec(
                        vector * q.get(joint.get("name"), 0.0)
                    ).as_matrix()
                poses[joint.find("child").get("link")] = poses[parent] @ transform @ motion
                pending.remove(joint)
            if len(pending) == count:
                raise ValueError("URDF graph is disconnected")
        return poses

    def prepare_import(self, output: Path) -> Path:
        """Create a derived import copy; preserve all original geometry/frames."""
        tree = ET.parse(URDF)
        for mesh in tree.findall(".//mesh"):
            name = mesh.get("filename")
            if name.startswith("package://"):
                raise ValueError("Unresolved asset-library mesh URI: " + name)
            mesh.set("filename", str((ASSET / name).resolve()))
        for joint in tree.findall("joint"):
            if "_rhino_v6_" in joint.get("name", "") and joint.get("type") == "revolute":
                limit = joint.find("limit")
                limit.attrib.update(lower="-3.0", upper="3.0", effort="2.0", velocity="1.0")
            if joint.get("name") in BASE_JOINTS:
                joint.set("type", "fixed")
                for child in list(joint):
                    if child.tag in {"limit", "dynamics", "axis"}:
                        joint.remove(child)
        for joint in tree.findall("joint"):
            relation = self.mimic.get(joint.get("name"))
            if relation is not None and joint.find("mimic") is None:
                source, scale, offset = relation
                ET.SubElement(
                    joint, "mimic", joint=source, multiplier=str(scale), offset=str(offset)
                )
        path = output / "fixed_base_import.urdf"
        tree.write(path, encoding="utf-8", xml_declaration=True)
        return path


class VendorIK:
    """Delegate IK to the existing M6 SDK, then verify with imported URDF FK."""

    def __init__(self, description: Description):
        sys.path.insert(0, str(BUNDLE))
        from tianji_recording.kinematics import TianjiDualArmForwardKinematics

        self.sdk = TianjiDualArmForwardKinematics(
            BUNDLE / "TJ_FX_ROBOT_CONTRL_SDK-master/CommonConfig/ccs_m6_40.MvKDCfg"
        )
        self.description = description

    def right(self, target: np.ndarray, current: dict[str, float]) -> dict[str, float]:
        return self.solve(target, current, "right")

    def left(self, target: np.ndarray, current: dict[str, float]) -> dict[str, float]:
        return self.solve(target, current, "left")

    def solve(self, target, current, side, *, diagnostics=None):
        """Retry SDK reference configurations without relaxing pose or joint limits."""
        try:
            return self._solve_once(target, current, side, diagnostics, "caller_reference")
        except ValueError:
            pass
        try:
            return self._solve_once(
                target, self.description.home, side, diagnostics, "factory_home"
            )
        except ValueError:
            pass
        # The analytic solver's reference selects its null-space branch. A
        # nearby pose can supply a valid reference when a straight-arm seed
        # selects an inadmissible branch. These are IK seeds, never motions.
        for dz in (-0.16, -0.12, -0.06, 0.06, 0.12):
            nearby = target.copy()
            nearby[2, 3] += dz
            try:
                seed = self._solve_once(
                    nearby, self.description.home, side, diagnostics, f"nearby_z_{dz:+.2f}"
                )
                return self._solve_once(
                    target, {**current, **seed}, side, diagnostics, f"nearby_seed_z_{dz:+.2f}"
                )
            except ValueError:
                continue
        raise ValueError("Manufacturer SDK rejected target across bounded reference retries")

    def diagnose(self, target, current, side):
        """Explain the same bounded model IK search, without executing any action.

        A failed bounded search is not a proof that the pose is unreachable.
        Accepted solutions certify only SDK/URDF consistency, not collision clearance.
        """
        attempts = []
        report = {
            "ok": False,
            "attempts": attempts,
            "source": "caller target + reference commands + static URDF/manufacturer SDK",
            "no_collision_guarantee": True,
            "failure_is_not_unreachable_proof": True,
            "side": side,
        }
        try:
            report["joint_target_rad"] = self.solve(
                target, current, side, diagnostics=attempts
            )
            report["ok"] = True
        except ValueError as exc:
            report["error_type"] = type(exc).__name__
            report["error_message"] = str(exc)
        return report

    def _solve_once(self, target, current, side, diagnostics=None, reference_source=None):
        suffix = "R" if side == "right" else "L"
        arm_index = 1 if side == "right" else 0
        from SDK_PYTHON.fx_kine import FX_InvKineSolvePara

        names = [f"Joint{i}_{suffix}" for i in range(1, 8)]
        limits_deg = np.array([170, 120, 170, 145, 170, 60, 90], dtype=float)
        lower_deg = -limits_deg
        upper_deg = limits_deg.copy()
        upper_deg[3] = 60
        reference_deg = [float(np.degrees(current[n])) for n in names]
        source_bounds = {
            name: {
                key: float(np.degrees(float(
                    self.description.by_name[name].find("limit").get(key)
                ))) for key in ("lower", "upper")
            } for name in names
        }
        source_lower = np.array([source_bounds[name]["lower"] for name in names])
        source_upper = np.array([source_bounds[name]["upper"] for name in names])
        row = {
            "reference_source": reference_source,
            "target_world_matrix": np.asarray(target).tolist(),
            "reference_joint_deg": reference_deg,
            "joint_names": names,
            "adapter_joint_bounds_deg": {
                "lower": lower_deg.tolist(), "upper": upper_deg.tolist(),
                "source": "existing adapter manufacturer-limit check; unchanged",
            },
            "source_urdf_joint_bounds_deg": source_bounds,
            "reference_adapter_margin_deg": np.minimum(
                np.asarray(reference_deg) - lower_deg, upper_deg - reference_deg
            ).tolist(),
            "reference_source_urdf_margin_deg": np.minimum(
                np.asarray(reference_deg) - source_lower, source_upper - reference_deg
            ).tolist(),
            "accepted": False,
        }
        if diagnostics is not None:
            diagnostics.append(row)
        local = np.linalg.inv(self.description.forward(current)["Base_" + suffix]) @ target
        local[:3, 3] *= 1000  # Delivered SDK accepts millimetres and degrees.
        row["target_arm_base_matrix_mm"] = local.tolist()
        request = FX_InvKineSolvePara()
        request.set_input_ik_target_tcp(local.ravel().tolist())
        request.set_input_ik_ref_joint(reference_deg)
        request.set_input_ik_zsp_type(0)
        answer = self.sdk._kinematics[arm_index].ik(request)
        row["sdk_returned_false"] = answer is False
        row["sdk_is_out_range"] = None if answer is False else bool(answer.m_Output_IsOutRange)
        row["sdk_is_joint_exceeded"] = None if answer is False else bool(answer.m_Output_IsJntExd)
        if answer is False or answer.m_Output_IsOutRange or answer.m_Output_IsJntExd:
            row["failure"] = "sdk_rejected"
            raise ValueError("Manufacturer SDK rejected the target")
        q = np.radians(answer.m_Output_RetJoint.to_list())
        lower = np.radians(lower_deg)
        upper = np.radians(upper_deg)
        row["returned_joint_deg"] = np.degrees(q).tolist()
        row["returned_adapter_margin_deg"] = np.minimum(
            np.degrees(q) - lower_deg, upper_deg - np.degrees(q)
        ).tolist()
        row["returned_source_urdf_margin_deg"] = np.minimum(
            np.degrees(q) - source_lower, source_upper - np.degrees(q)
        ).tolist()
        if not np.isfinite(q).all() or np.any(q < lower) or np.any(q > upper):
            row["failure"] = "adapter_joint_limits"
            raise ValueError("Manufacturer joint limits exceeded")
        result = dict(zip((f"Joint{i}_{suffix}" for i in range(1, 8)), map(float, q), strict=True))
        actual = self.description.forward({**current, **result})[side + "_tool0"]
        error = np.linalg.norm(actual[:3, 3] - target[:3, 3])
        angle = Rotation.from_matrix(actual[:3, :3] @ target[:3, :3].T).magnitude()
        row["fk_position_error_m"] = float(error)
        row["fk_rotation_error_rad"] = float(angle)
        if error > 0.0002 or angle > 0.002:
            row["failure"] = "sdk_urdf_fk_mismatch"
            raise ValueError(f"SDK/URDF target mismatch: {error=} {angle=}")
        row["accepted"] = True
        return result


class IsolatedVendorIK:
    """Keep the vendor's native library outside the simulator process."""

    def __init__(self, description: Description):
        self.description = description

    def right(self, target: np.ndarray, current: dict[str, float]) -> dict[str, float]:
        return self.solve(target, current, "right")

    def left(self, target: np.ndarray, current: dict[str, float]) -> dict[str, float]:
        return self.solve(target, current, "left")

    def solve(self, target, current, side):
        reply = self._request(target, current, side, "ik")
        return {name: float(value) for name, value in reply["joints"].items()}

    def linear(self, target, current, side="left"):
        return self._request(target, current, side, "linear")["plan"]

    def _request(self, target, current, side, operation):
        process = subprocess.run(
            [sys.executable, "-m", "tianji.ik_worker"],
            input=json.dumps(
                {
                    "target": target.tolist(),
                    "current": current,
                    "side": side,
                    "operation": operation,
                }
            ),
            capture_output=True,
            text=True,
            timeout=20,
        )
        if process.returncode:
            raise RuntimeError(
                f"Isolated manufacturer IK failed ({process.returncode}): {process.stderr[-1000:]}"
            )
        payload = process.stdout.rsplit("TIANJI_IK_RESULT ", 1)[-1].splitlines()[0]
        reply = json.loads(payload)
        if not reply["ok"]:
            raise ValueError(reply["error"])
        return reply
