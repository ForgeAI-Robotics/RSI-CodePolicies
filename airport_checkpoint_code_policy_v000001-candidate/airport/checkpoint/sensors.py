"""Measured RGB-D/proprioception boundary, with no semantic or world-pose inputs.

Camera extrinsics are a declared calibration assumption. Simulator truth belongs
in a separate audit stream and must never be used to fill a missing measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CameraCalibration:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    # Optical coordinates: +x right, +y down, +z forward, robot work frame.
    optical_to_work: tuple[tuple[float, ...], ...]
    source: str = "simulation_camera_configuration; requires real-world calibration"

    # Raster sampling in pixels, x then y. Old recordings retain integer rays.
    pixel_offset_xy: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        if min(self.width, self.height, self.fx, self.fy) <= 0:
            raise ValueError("Invalid camera calibration")
        offset = np.asarray(self.pixel_offset_xy, dtype=float)
        if offset.shape != (2,) or not np.isfinite(offset).all() or np.any(abs(offset) > 0.5):
            raise ValueError("Invalid raster pixel offset")
        matrix = np.asarray(self.optical_to_work, dtype=float)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError("Camera transform must be a finite 4x4 matrix")
        if not np.allclose(matrix[3], [0, 0, 0, 1]):
            raise ValueError("Camera transform must be homogeneous")
        if not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6):
            raise ValueError("Camera transform must have an orthonormal rotation")


@dataclass(frozen=True)
class RenderSynchrony:
    """Producer evidence under a separately tested renderer profile.

    ReferenceTime alone does not establish geometry time. Profile/evidence and
    pinned settings must match the explicitly enabled backend configuration.
    Physics/render clocks may have an origin offset from encoder/frame stamps.
    """

    profile: str
    evidence_sha256: str
    render_frame: int
    render_time: float
    physics_time: float
    encoder_stamp: float
    encoder_sequence: int
    encoder_sha256: str
    calibration_sha256: str
    settings_verified: bool = False


@dataclass(frozen=True)
class RGBDFrame:
    stamp: float
    rgb: np.ndarray
    depth_m: np.ndarray
    calibration: CameraCalibration
    source: str = "isaac_rtx_rgb_distance_to_image_plane"
    measured: bool = True
    synchrony: RenderSynchrony | None = None


@dataclass(frozen=True)
class RawMeasuredObservation:
    stamp: float
    sequence: int
    joint_positions: dict[str, float]
    joint_velocities: dict[str, float]
    rgbd: RGBDFrame | None
    # Instrument channels identify robot hardware only, not contacted objects.
    contact_force_n: dict[str, float] = field(default_factory=dict)
    provenance: dict[str, str] = field(
        default_factory=lambda: {
            "joints": "native_articulation_encoder",
            "contact_force": "native_contact_impulse_over_physics_dt; idealized_sensor",
            "rgbd": "RTX RGB and distance_to_image_plane; no semantic annotations",
            "robot_base_pose": "native own-robot root pose; synthetic ideal position/IMU tracker",
        }
    )
    wrist_wrench6: tuple[float, ...] | None = None
    wrist_bias6: tuple[float, ...] | None = None
    wrist_bias_stamp: float | None = None
    wrist_bias_valid: bool = False
    robot_base_pose: tuple[float, float, float, float] | None = None
    wrist_zero_offset6: tuple[float, ...] | None = None
    wrist_gravity_wrench6: tuple[float, ...] | None = None
    # World-frame reaction on environment, averaged over this explicit window.
    # None while the window warms or calibration/payload is unavailable.
    wrist_reaction_world_n: tuple[float, float, float] | None = None
    wrist_force_window_start: float | None = None
    wrist_force_window_end: float | None = None
    wrist_force_stamp: float | None = None


@dataclass(frozen=True)
class PrivilegedTruth:
    """Audit-only type; explicitly incompatible with policy observations."""

    stamp: float
    payload: dict[str, Any]
    privileged: bool = True


def optical_transform(eye: np.ndarray, look_at: np.ndarray) -> np.ndarray:
    """Declared camera calibration in the stationary robot work frame."""
    eye, look_at = np.asarray(eye, float), np.asarray(look_at, float)
    forward = look_at - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(right) < 1e-8:
        raise ValueError("Camera viewing direction is parallel to world up")
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    result = np.eye(4)
    result[:3, :3] = np.column_stack([right, down, forward])
    result[:3, 3] = eye
    return result


def deproject_depth(
    frame: RGBDFrame, mask: np.ndarray | None = None, stride: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Convert measured axial depth into work-frame points and [row,col] pixels."""
    if not frame.measured:
        raise ValueError("Synthetic frame cannot enter measured perception")
    if stride < 1:
        raise ValueError("Stride must be positive")
    c = frame.calibration
    depth = np.asarray(frame.depth_m)
    if depth.shape != (c.height, c.width):
        raise ValueError("Depth shape disagrees with calibration")
    valid = np.isfinite(depth) & (depth > 0)
    if mask is not None:
        if mask.shape != depth.shape:
            raise ValueError("ROI mask shape disagrees with depth")
        valid &= mask.astype(bool)
    sampling = np.zeros(depth.shape, dtype=bool)
    sampling[::stride, ::stride] = True
    rows, cols = np.nonzero(valid & sampling)
    z = depth[rows, cols]
    dx, dy = c.pixel_offset_xy
    optical = np.column_stack([(cols + dx - c.cx) * z / c.fx, (rows + dy - c.cy) * z / c.fy, z])
    transform = np.asarray(c.optical_to_work)
    points = optical @ transform[:3, :3].T + transform[:3, 3]
    return points, np.column_stack([rows, cols])


def estimate_surface_patch(
    frame: RGBDFrame, mask: np.ndarray, min_points: int = 30, maximum_residual_m: float = 0.012
) -> dict:
    """Fit a local plane to a caller's image ROI, without scene geometry access.

    This function cannot identify a person, paper, or bag. That decision must
    come from an image detector or an explicitly configured, visible fixture ROI.
    """
    points, _ = deproject_depth(frame, mask)
    if len(points) < min_points:
        raise ValueError("Insufficient measured depth for a surface patch")
    center = np.median(points, axis=0)
    distance = np.linalg.norm(points - center, axis=1)
    points = points[distance <= np.quantile(distance, 0.90)]
    center = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    normal = vh[-1]
    eye = np.asarray(frame.calibration.optical_to_work)[:3, 3]
    if np.dot(normal, eye - center) < 0:
        normal = -normal
    residual = float(np.sqrt(np.mean(((points - center) @ normal) ** 2)))
    if residual > maximum_residual_m:
        raise ValueError("Measured ROI is not a sufficiently planar surface")
    return {
        "center": center.tolist(),
        "normal": normal.tolist(),
        "tangent": vh[0].tolist(),
        "stamp": frame.stamp,
        "confidence": max(0.0, 1.0 - residual / maximum_residual_m),
        "residual_m": residual,
        "measured_points": len(points),
        "source": frame.source,
    }
