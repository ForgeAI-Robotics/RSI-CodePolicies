"""Measured-surface tool alignment using an explicit rigid grasp calibration."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .types import SurfacePatch, Vec3


def sampling_orientation(patch: SurfacePatch, outward_tilt_rad: float = 0.85) -> Vec3:
    """EEF Y is the paper face normal; -EEF Z is its long sampling end.

    The paper/EEF axes are an installation/grasp calibration assumption, never
    estimated from a hidden simulator pose. Tilting +Z outward keeps the wrist
    farther from the surface than the free paper end.
    """
    normal = np.asarray(patch.normal, float)
    tangent = np.asarray(patch.tangent, float)
    if not np.isfinite([*normal, *tangent, outward_tilt_rad]).all():
        raise ValueError("Invalid measured surface orientation")
    normal /= np.linalg.norm(normal)
    up = np.array([0.0, 0.0, 1.0])
    up -= normal * np.dot(up, normal)
    if np.linalg.norm(up) < 1e-5:
        up = tangent - normal * np.dot(tangent, normal)
    if np.linalg.norm(up) < 1e-5:
        raise ValueError("Degenerate surface axes")
    up /= np.linalg.norm(up)
    z_axis = np.cos(outward_tilt_rad) * up + np.sin(outward_tilt_rad) * normal
    y_axis = np.cos(outward_tilt_rad) * normal - np.sin(outward_tilt_rad) * up
    x_axis = np.cross(y_axis, z_axis)
    return tuple(Rotation.from_matrix(np.column_stack([x_axis, y_axis, z_axis])).as_euler("xyz"))


def eef_for_measured_tip(
    eef: Vec3, current_rpy: Vec3, tip: Vec3, desired_tip: Vec3, target_rpy: Vec3
) -> Vec3:
    local_tip = Rotation.from_euler("xyz", current_rpy).inv().apply(np.asarray(tip) - eef)
    position = np.asarray(desired_tip) - Rotation.from_euler("xyz", target_rpy).apply(local_tip)
    return tuple(position)


def orientation_distance(current_rpy: Vec3, target_rpy: Vec3) -> float:
    return float(
        (
            Rotation.from_euler("xyz", target_rpy) * Rotation.from_euler("xyz", current_rpy).inv()
        ).magnitude()
    )


def bounded_orientation(current_rpy: Vec3, target_rpy: Vec3, max_angle_rad: float) -> Vec3:
    current = Rotation.from_euler("xyz", current_rpy)
    delta = Rotation.from_euler("xyz", target_rpy) * current.inv()
    angle = float(delta.magnitude())
    if angle <= max_angle_rad:
        return target_rpy
    result = Rotation.from_rotvec(delta.as_rotvec() * max_angle_rad / angle) * current
    return tuple(result.as_euler("xyz"))
