"""Local model IK with fixed-body clearance; never reads simulator state or images."""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from tianji.description import ASSET, ROOT, URDF, Description

from .collision import _transformed, envelopes_overlap
from .contract import LOWER, UPPER


@lru_cache(maxsize=1)
def native_body_shapes():
    path = ROOT / "configs/tianji-rgb/native_body_geometry.json"
    data = json.loads(path.read_text())
    if hashlib.sha256(URDF.read_bytes()).hexdigest() != data["source_urdf_sha256"]:
        raise ValueError("Static body geometry does not match current URDF")
    for relative, expected in data["source_mesh_sha256"].items():
        if hashlib.sha256((ASSET / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("Static body geometry does not match source mesh:" + relative)
    return Description(), data["shapes"]


def body_clearance(angles, side, *, margin=0.03):
    """Imported static robot meshes, conservatively convexified per mesh.

    This is a sampled model check, not measured clearance or a path certificate.
    """
    description, shapes = native_body_shapes()
    poses = description.forward(angles)
    suffix = "L" if side == "left" else "R"
    active = {f"Link{i}_{suffix}" for i in (2, 3, 4)}
    bodies = [_transformed(np.asarray(row["vertices"]), np.asarray(row["equations"]),
                           poses["chassis_base"])
              for row in shapes if row["body"] == "chassis_base"]
    for row in shapes:
        name = row["body"]
        if name not in active:
            continue
        low, high, normals, bounds = _transformed(
            np.asarray(row["vertices"]), np.asarray(row["equations"]), poses[name],
        )
        envelope = (low-margin, high+margin, normals, bounds+margin)
        for index, other in enumerate(bodies):
            if envelopes_overlap(envelope, other):
                return {"safe": False, "pair": [name, "chassis_base"],
                        "body_shape_index": index, "margin_m": margin}
    return {"safe": True, "pair": None, "margin_m": margin}


def solve_continuous(inverse, target, current, side, *, max_step=0.25):
    """Try the current SDK branch, then bounded numerical IK using its existing FK.

    Bounds stay within hardware limits and max_step of the caller command. Three
    sampled configurations screen torso contacts; native guards remain authoritative.
    No world objects, goals, measured encoders, collision planner or scene queries.
    """
    suffix = "L" if side == "left" else "R"
    names = [f"Joint{i}_{suffix}" for i in range(1, 8)]
    reference = np.array([current[name] for name in names])
    lower, upper = np.maximum(LOWER, reference-max_step), np.minimum(UPPER, reference+max_step)
    if np.any(upper-lower < 1e-8):
        raise ValueError("Continuous IK reference has no valid local joint interval")
    base = inverse.description.forward(current)["Base_"+suffix]
    sdk = inverse.sdk._kinematics[0 if side == "left" else 1]
    target_rotation = Rotation.from_matrix(target[:3, :3])
    rejected = []

    def pose(q):
        local = np.asarray(sdk.fk(joints=np.degrees(q).tolist()), dtype=float)
        local[:3, 3] *= .001
        return base @ local

    def residual(q):
        actual = pose(q)
        rotation = (target_rotation * Rotation.from_matrix(actual[:3, :3]).inv()).as_rotvec()
        return np.r_[actual[:3, 3]-target[:3, 3], .15*rotation, 1e-5*(q-reference)]

    def accept(q):
        if not np.isfinite(q).all() or np.any(q < lower-1e-8) or np.any(q > upper+1e-8):
            return None
        solved = dict(zip(names, map(float, q), strict=True))
        actual = inverse.description.forward({**current, **solved})[side+"_tool0"]
        position_error = float(np.linalg.norm(actual[:3, 3]-target[:3, 3]))
        angle_error = float(Rotation.from_matrix(actual[:3, :3]@target[:3, :3].T).magnitude())
        if position_error > .0002 or angle_error > .002:
            rejected.append("pose residual")
            return None
        for fraction in (.0, .5, 1.):
            angles = {**current, **dict(zip(names, reference+fraction*(q-reference), strict=True))}
            check = body_clearance(angles, side)
            if not check["safe"]:
                rejected.append("body clearance:"+str(check["pair"]))
                return None
        return {"joint_target_rad": solved, "position_error_m": position_error,
                "rotation_error_rad": angle_error,
                "source": "bounded local model IK and sampled fixed-body geometry",
                "max_joint_change_rad": float(np.max(abs(q-reference))),
                "no_collision_guarantee": True}

    try:
        direct = inverse._solve_once(target, current, side)
        result = accept(np.array([direct[name] for name in names]))
        if result is not None:
            return result
    except ValueError:
        pass
    seeds = [reference]
    for index in (2, 0):
        for sign in (-1, 1):
            seed = reference.copy()
            seed[index] += sign*.12
            seeds.append(np.clip(seed, lower+1e-8, upper-1e-8))
    for seed in seeds:
        fitted = least_squares(residual, seed, bounds=(lower, upper), max_nfev=70,
                               ftol=1e-9, xtol=1e-9, gtol=1e-9)
        result = accept(fitted.x)
        if result is not None:
            return result
    raise ValueError("Continuous local IK rejected target: "+str(sorted(set(rejected))))
