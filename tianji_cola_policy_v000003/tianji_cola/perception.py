"""RGB-only top-rim circle fitting with explicit camera/height/size priors."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import least_squares


def circle_rim(rgb, intrinsic, optical_pose, expected, plane_height, radius=0.029):
    """Return an XY estimate, never observed depth, from projected Canny edges.

    The caller supplies a cylinder-centre estimate, its top-plane assumption,
    camera calibration, and radius prior. No simulator/scene imports occur.
    """
    rgb = np.asarray(rgb, dtype=np.uint8)
    K = np.asarray(intrinsic, dtype=float)
    T = np.asarray(optical_pose, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if expected.shape != (3,) or not np.isfinite(expected).all():
        raise ValueError("Expected finite XYZ prior")
    if not np.isfinite(plane_height) or not 0.005 < radius < 0.10:
        raise ValueError("Invalid plane/radius prior")
    height, width = rgb.shape[:2]
    inverse = np.linalg.inv(K)

    def project_to_plane(uv, z):
        rays = np.c_[uv, np.ones(len(uv))] @ inverse.T @ T[:3, :3].T
        scale = (z - T[2, 3]) / rays[:, 2]
        return T[:3, 3] + rays * scale[:, None]

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 150)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    red = ((hsv[:, :, 0] < 12) | (hsv[:, :, 0] > 170)) & (hsv[:, :, 1] > 85) & (hsv[:, :, 2] > 40)
    saturation = hsv[:, :, 1].astype(float) / 255
    theta = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    unit_ring = np.c_[np.cos(theta), np.sin(theta)]

    def ring_contrast(centre, r):
        values = []
        masks = []
        for dr in (-0.002, 0.002):
            world = np.c_[centre + (r + dr) * unit_ring, np.full(len(theta), plane_height)]
            cam = (world - T[:3, 3]) @ T[:3, :3]
            pix = cam @ K.T
            pix = np.rint(pix[:, :2] / pix[:, 2:3]).astype(int)
            valid = (
                (pix[:, 0] >= 0)
                & (pix[:, 0] < width)
                & (pix[:, 1] >= 0)
                & (pix[:, 1] < height)
                & (cam[:, 2] > 0)
            )
            values.append(
                saturation[np.clip(pix[:, 1], 0, height - 1), np.clip(pix[:, 0], 0, width - 1)]
            )
            masks.append(valid)
        valid = masks[0] & masks[1]
        return float(np.mean(values[1][valid] - values[0][valid])) if valid.sum() > 15 else -1.0

    extent = radius + 0.045
    square = np.array(
        [
            [expected[0] + dx, expected[1] + dy, plane_height]
            for dx in (-extent, extent)
            for dy in (-extent, extent)
        ]
    )
    cam_square = (square - T[:3, 3]) @ T[:3, :3]
    if np.any(cam_square[:, 2] <= 0.01):
        return []
    uv = cam_square @ K.T
    uv = uv[:, :2] / uv[:, 2:3]
    lower = np.floor(uv.min(axis=0)).astype(int)
    upper = np.ceil(uv.max(axis=0)).astype(int)
    roi = np.zeros((height, width), np.uint8)
    roi[
        max(0, lower[1]) : min(height, upper[1] + 1), max(0, lower[0]) : min(width, upper[0] + 1)
    ] = 1
    rows, cols = np.nonzero(edges * roi)
    if len(cols) < 15:
        return []
    points = project_to_plane(np.c_[cols, rows], plane_height)[:, :2]
    points = points[np.isfinite(points).all(axis=1)]
    points = points[np.linalg.norm(points - expected[:2], axis=1) < extent]
    fits = []
    if len(points) < 15:
        return []
    points = points[:: max(1, len(points) // 1000)]
    rng = np.random.default_rng(0)
    pairs = rng.integers(len(points), size=(700, 2))
    a, b = points[pairs[:, 0]], points[pairs[:, 1]]
    delta = b - a
    distance = np.linalg.norm(delta, axis=1)
    midpoint = (a + b) / 2
    best = None
    for r in np.linspace(radius - 0.003, radius + 0.001, 5):
        valid = (distance > 0.006) & (distance < 2 * r)
        unit = delta[valid] / distance[valid, None]
        perpendicular = np.c_[-unit[:, 1], unit[:, 0]]
        offset = np.sqrt(r * r - (distance[valid] / 2) ** 2)[:, None] * perpendicular
        centres = np.r_[midpoint[valid] + offset, midpoint[valid] - offset]
        centres = centres[np.linalg.norm(centres - expected[:2], axis=1) < 0.045]
        for centre in centres:
            error = abs(np.linalg.norm(points - centre, axis=1) - r)
            inside = error < 0.0015
            if inside.sum() < 15:
                continue
            angles = np.arctan2(points[inside, 1] - centre[1], points[inside, 0] - centre[0])
            coverage = len(np.unique(((angles + np.pi) / (2 * np.pi) * 24).astype(int)))
            if coverage < 13:
                continue
            contrast = ring_contrast(centre, r)
            score = float(inside.sum() + 3 * coverage) * (0.5 + max(0.0, contrast))
            if best is None or score > best[0]:
                best = score, centre, r, inside, coverage
    if best is None:
        return []
    score, centre, r, inside, coverage = best
    fit = least_squares(
        lambda z, samples=points[inside]: np.linalg.norm(samples - z[:2], axis=1) - z[2],
        [*centre, r],
        bounds=(
            [centre[0] - 0.015, centre[1] - 0.015, r - 0.001],
            [centre[0] + 0.015, centre[1] + 0.015, r + 0.001],
        ),
        loss="soft_l1",
        f_scale=0.001,
    )
    rms = float(np.sqrt(np.mean(fit.fun**2)))
    if rms > 0.0018 or np.linalg.norm(fit.x[:2] - expected[:2]) > 0.045:
        return []
    angles = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    ring = np.c_[
        fit.x[0] + (fit.x[2] + 0.002) * np.cos(angles),
        fit.x[1] + (fit.x[2] + 0.002) * np.sin(angles),
        np.full(72, plane_height),
    ]
    cam = (ring - T[:3, 3]) @ T[:3, :3]
    pixel = cam @ K.T
    pixel = np.rint(pixel[:, :2] / pixel[:, 2:3]).astype(int)
    valid = (
        (pixel[:, 0] >= 0)
        & (pixel[:, 0] < width)
        & (pixel[:, 1] >= 0)
        & (pixel[:, 1] < height)
        & (cam[:, 2] > 0)
    )
    border_red = float(red[pixel[valid, 1], pixel[valid, 0]].mean()) if valid.any() else 0.0
    fits.append(
        {
            "estimated_xy": fit.x[:2].tolist(),
            "fit_radius_m": float(fit.x[2]),
            "radial_rms_m": rms,
            "angular_bins_of_24": coverage,
            "edge_support": int(inside.sum()),
            "red_border_fraction": border_red,
            "score": score * (0.5 + border_red),
            "plane_height_assumption_m": plane_height,
        }
    )
    return sorted(fits, key=lambda item: -item["score"])
