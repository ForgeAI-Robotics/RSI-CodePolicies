"""Classical geometry for externally measured ROIs, without semantic ground truth.

These routines do not identify bags, forearms or people. An ROI provider and
validated tracking/association remain required before autonomous deployment.
"""

from __future__ import annotations

import math

import numpy as np

from .types import SurfacePatch, Vec3


def depth_to_points(
    depth: np.ndarray, intrinsics: np.ndarray, camera_to_workspace: np.ndarray
) -> np.ndarray:
    """Deproject finite positive optical-axis depth; use calibrated K/extrinsics."""
    depth = np.asarray(depth, dtype=float)
    k = np.asarray(intrinsics, dtype=float)
    transform = np.asarray(camera_to_workspace, dtype=float)
    if depth.ndim != 2 or k.shape != (3, 3) or transform.shape != (4, 4):
        raise ValueError("Depth, K and extrinsics must have shapes HxW, 3x3, 4x4")
    if not np.isfinite(k).all() or not np.isfinite(transform).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError("Invalid calibration")
    if not np.allclose(transform[3], [0, 0, 0, 1]) or not np.allclose(
        transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5
    ):
        raise ValueError("Extrinsics must be a rigid transform")
    rows, cols = np.nonzero(np.isfinite(depth) & (depth > 0))
    pixels = np.column_stack((cols, rows, np.ones(len(rows))))
    points = (pixels @ np.linalg.inv(k).T) * depth[rows, cols, None]
    return points @ transform[:3, :3].T + transform[:3, 3]


def estimate_patch(
    points: np.ndarray,
    viewpoint: Vec3,
    stamp: float,
    *,
    max_residual_m: float = 0.004,
    min_points: int = 20,
) -> SurfacePatch:
    """Robust SVD plane fit to a measured ROI; reject non-planar/line-like data."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected Nx3 point cloud")
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < min_points or not math.isfinite(stamp):
        raise ValueError("Insufficient finite measurements")
    original_count = len(points)
    for _ in range(4):
        center = np.median(points, axis=0)
        _, singular, vh = np.linalg.svd(points - center, full_matrices=False)
        if singular[1] < 0.003 * math.sqrt(len(points)):
            raise ValueError("ROI has insufficient two-dimensional extent")
        distance = np.abs((points - center) @ vh[-1])
        keep = distance <= max_residual_m
        if keep.all():
            break
        points = points[keep]
        if len(points) < min_points or len(points) / original_count < 0.7:
            raise ValueError("ROI is not a supported planar patch")
    center = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    normal, tangent = vh[-1], vh[0]
    if np.dot(normal, np.asarray(viewpoint) - center) < 0:
        normal = -normal
    # Deterministic sign prevents stroke reversal across equivalent SVD solutions.
    if tangent[np.argmax(np.abs(tangent))] < 0:
        tangent = -tangent
    residual = float(np.sqrt(np.mean(((points - center) @ normal) ** 2)))
    confidence = len(points) / original_count * max(0.0, 1.0 - residual / max_residual_m)
    return SurfacePatch(tuple(center), tuple(normal), tuple(tangent), confidence, stamp)


def measured_velocity(previous: Vec3, current: Vec3, previous_stamp: float, stamp: float) -> Vec3:
    delta = stamp - previous_stamp
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError("Tracking requires increasing timestamps")
    velocity = (np.asarray(current) - np.asarray(previous)) / delta
    if not np.isfinite(velocity).all():
        raise ValueError("Invalid tracked position")
    return tuple(velocity)


def is_lateral(
    base_position: Vec3,
    person_position: Vec3,
    person_forward: Vec3,
    max_forward_cosine: float = 0.5,
) -> bool:
    relative = np.asarray(base_position[:2]) - np.asarray(person_position[:2])
    forward = np.asarray(person_forward[:2])
    denominator = np.linalg.norm(relative) * np.linalg.norm(forward)
    return bool(
        denominator > 1e-6 and abs(np.dot(relative, forward)) / denominator <= max_forward_cosine
    )
