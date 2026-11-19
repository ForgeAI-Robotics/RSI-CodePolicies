"""Read-only motion preflight from caller RGB estimates and static URDF geometry.

Convex link envelopes and inflated cylinder envelopes are conservative at sampled
configurations. Joint interpolation sampling is finite: this is not continuous
collision detection and does not certify execution or perception accuracy.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import product

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from tianji.description import ASSET, Description
from tianji.rgb.contract import command_angles, validate_action
from tianji.rhino import joint_targets


def _side(name):
    for side, suffix in (("left", "L"), ("right", "R")):
        if name.startswith(side + "_") or name.startswith("cam_" + side + "_wrist"):
            return side
        if name.startswith("Link") and name.endswith("_" + suffix):
            return side
    return None


@lru_cache(maxsize=1)
def link_envelopes():
    description = Description()
    envelopes = {}
    for link in description.tree.findall("link"):
        name = link.get("name")
        if _side(name) is None:
            continue
        clouds = []
        for collision in link.findall("collision"):
            mesh = collision.find("geometry/mesh")
            if mesh is None:
                box = collision.find("geometry/box")
                cylinder = collision.find("geometry/cylinder")
                sphere = collision.find("geometry/sphere")
                if box is not None:
                    extent = np.fromstring(box.get("size"), sep=" ") / 2
                    vertices = np.asarray(list(product((-1, 1), repeat=3))) * extent
                elif cylinder is not None:
                    radius = float(cylinder.get("radius")) / np.cos(np.pi / 32)
                    theta = np.linspace(0, 2 * np.pi, 32, endpoint=False)
                    vertices = np.array(
                        [
                            [radius * np.cos(t), radius * np.sin(t), z]
                            for t in theta
                            for z in (
                                -float(cylinder.get("length")) / 2,
                                float(cylinder.get("length")) / 2,
                            )
                        ]
                    )
                elif sphere is not None:
                    vertices = np.asarray(list(product((-1, 1), repeat=3))) * float(
                        sphere.get("radius")
                    )
                else:
                    raise ValueError("Unsupported collision geometry: " + name)
            else:
                data = (ASSET / mesh.get("filename")).resolve().read_bytes()
                count = int.from_bytes(data[80:84], "little")
                if len(data) != 84 + 50 * count:
                    raise ValueError("Expected binary STL")
                dtype = np.dtype(
                    [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]
                )
                vertices = np.frombuffer(data, dtype=dtype, count=count, offset=84)["vertices"]
                vertices = vertices.reshape(-1, 3).astype(float)
                vertices *= np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
            origin = collision.find("origin")
            if origin is not None:
                rotation = Rotation.from_euler(
                    "xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
                ).as_matrix()
                vertices = vertices @ rotation.T + np.fromstring(
                    origin.get("xyz", "0 0 0"), sep=" "
                )
            clouds.append(vertices)
        if clouds:
            vertices = np.unique(np.concatenate(clouds), axis=0)
            hull = ConvexHull(vertices)
            # Hull equations are local unit-normal halfspaces n.x + d <= 0.
            envelopes[name] = (vertices[hull.vertices], hull.equations)
    return description, envelopes


def _transformed(vertices, equations, pose):
    points = vertices @ pose[:3, :3].T + pose[:3, 3]
    normals = equations[:, :3] @ pose[:3, :3].T
    bounds = -equations[:, 3] + normals @ pose[:3, 3]
    return points.min(axis=0), points.max(axis=0), normals, bounds


def cylinder_envelope(center, radius, height, margin, rotation=None):
    center = np.asarray(center, dtype=float)
    theta = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    normals = np.r_[np.c_[np.cos(theta), np.sin(theta), np.zeros(16)], [[0, 0, 1], [0, 0, -1]]]
    rotation = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
    if (
        rotation.shape != (3, 3)
        or not np.isfinite(rotation).all()
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        or np.linalg.det(rotation) < 0.99
    ):
        raise ValueError("Invalid caller-estimated cylinder orientation")
    normals = normals @ rotation.T
    bounds = normals @ center + np.r_[np.full(16, radius + margin), np.full(2, height / 2 + margin)]
    extent = np.abs(rotation) @ (np.array([radius, radius, height / 2]) + margin)
    return center - extent, center + extent, normals, bounds


def envelopes_overlap(a, b):
    low, high = np.maximum(a[0], b[0]), np.minimum(a[1], b[1])
    if np.any(high < low):
        return False
    result = linprog(
        np.zeros(3),
        A_ub=np.r_[a[2], b[2]],
        b_ub=np.r_[a[3], b[3]],
        bounds=list(zip(low, high, strict=True)),
        method="highs",
    )
    if result.status not in (0, 2):
        raise ValueError("Collision feasibility solver was inconclusive")
    return result.status == 0


def interpolate_actions(actions, max_joint_step=0.03):
    """Check endpoints and adaptively fill joint/opening intervals; no physics."""
    points = np.asarray([validate_action(row) for row in actions])
    if not len(points):
        raise ValueError("Preflight requires at least one action")
    yield points[0]
    for start, end in zip(points[:-1], points[1:], strict=True):
        # Opening is dimensionless; 0.03 corresponds to 0.018 driver radians.
        count = max(1, int(np.ceil(np.max(np.abs(end - start)) / max_joint_step)))
        for fraction in np.linspace(0, 1, count + 1)[1:]:
            yield start + fraction * (end - start)


def check_motion(
    actions,
    obstacles,
    *,
    arm,
    selected=None,
    allow_grasp_contact=False,
    carried=None,
    margin=0.012,
    keepout_boxes=None,
):
    if arm not in ("left", "right") or not 0.005 <= margin <= 0.04:
        raise ValueError("Invalid arm or uncertainty margin")
    if len(obstacles) > 16:
        raise ValueError("Too many caller-supplied obstacles")
    centers = np.asarray(obstacles, dtype=float).reshape((-1, 3))
    if centers.ndim != 2 or centers.shape[1:] != (3,) or not np.isfinite(centers).all():
        raise ValueError("Obstacles must be finite caller-estimated XYZ centers")
    if selected is not None and not 0 <= int(selected) < len(centers):
        raise ValueError("Invalid selected object")
    keepouts = []
    for box in keepout_boxes or []:
        low, high = np.asarray(box["min_xyz"]), np.asarray(box["max_xyz"])
        if (
            low.shape != (3,)
            or high.shape != (3,)
            or not np.isfinite([low, high]).all()
            or np.any(high <= low)
        ):
            raise ValueError("Invalid caller-declared keepout box")
        keepouts.append((low, high, np.r_[np.eye(3), -np.eye(3)], np.r_[high, -low]))
    if not len(centers) and not keepouts:
        raise ValueError("Unknown objects require an explicit conservative occupancy prior")
    description, envelopes = link_envelopes()
    checked = 0
    for action in interpolate_actions(actions):
        q = command_angles(action)
        q.update(joint_targets(float(action[7]), "left"))
        q.update(joint_targets(float(action[15]), "right"))
        poses = description.forward(q)
        world = {
            name: _transformed(vertices, planes, poses[name])
            for name, (vertices, planes) in envelopes.items()
        }
        objects = centers.copy()
        carried_rotation = None
        if carried is not None:
            if selected is None:
                raise ValueError("Carried object requires a selected index")
            offset = np.asarray(carried["offset_tool"], dtype=float)
            if offset.shape != (3,) or not np.isfinite(offset).all():
                raise ValueError("Invalid caller-estimated carried offset")
            tool = poses[arm + "_tool0"]
            objects[selected] = tool[:3, :3] @ offset + tool[:3, 3]
            if "orientation_tool" not in carried:
                raise ValueError("Carried object requires a caller-estimated relative orientation")
            carried_rotation = tool[:3, :3] @ np.asarray(carried["orientation_tool"], dtype=float)
        object_shapes = [
            cylinder_envelope(
                center, 0.029, 0.1454, margin, carried_rotation if index == selected else None
            )
            for index, center in enumerate(objects)
        ]
        for box_index, keepout in enumerate(keepouts):
            for name, body in world.items():
                if envelopes_overlap(body, keepout):
                    return {
                        "clear": False,
                        "sample": checked,
                        "kind": "unknown_keepout",
                        "link": name,
                        "box_index": box_index,
                        "checked_samples": checked + 1,
                    }
        # Reject robot contact with any unselected can, including the idle arm.
        for index, shape in enumerate(object_shapes):
            for name, body in world.items():
                if (
                    index == selected
                    and allow_grasp_contact
                    and name.startswith(arm + "_rhino_v6_follower_link_")
                ):
                    continue
                if envelopes_overlap(body, shape):
                    return {
                        "clear": False,
                        "sample": checked,
                        "kind": "robot_object",
                        "link": name,
                        "object_index": index,
                        "checked_samples": checked + 1,
                    }
        for left, right in product(
            (n for n in world if _side(n) == "left"), (n for n in world if _side(n) == "right")
        ):
            if envelopes_overlap(world[left], world[right]):
                return {
                    "clear": False,
                    "sample": checked,
                    "kind": "arm_arm",
                    "links": [left, right],
                    "checked_samples": checked + 1,
                }
        if carried is not None:
            for index, shape in enumerate(object_shapes):
                if index != selected and envelopes_overlap(object_shapes[selected], shape):
                    return {
                        "clear": False,
                        "sample": checked,
                        "kind": "carried_object",
                        "object_index": index,
                        "checked_samples": checked + 1,
                    }
        checked += 1
        if checked > 4000:
            raise ValueError("Motion preflight sample budget exceeded")
    return {
        "clear": True,
        "checked_samples": checked,
        "margin_m": margin,
        "source": "caller RGB estimates + static URDF + own commands",
        "limits": "finite joint samples; convex envelopes; caller-estimated rigid carried offset",
    }
