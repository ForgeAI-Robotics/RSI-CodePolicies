"""Finite own-mesh surface witnesses; no scene identity or filtering policy."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SurfaceMatches:
    """Unmatched rows contain inf distance, -1 face, and NaN coordinates."""

    distance_m: np.ndarray
    triangle_index: np.ndarray
    closest: np.ndarray
    barycentric: np.ndarray


def closest_triangle_surface(
    points: np.ndarray, triangles: np.ndarray, *, max_distance_m: float
) -> SurfaceMatches:
    """Return exact nondegenerate triangle witnesses inside the supplied radius.

    Both inputs must be in the same Cartesian metric frame. Face indices refer
    to the input triangle rows. AABBs only discard impossible candidates; each
    accepted distance comes from a triangle interior or a clamped edge segment.
    Near-degenerate faces are conservatively excluded. No point is classified
    or deleted here. Callers retain responsibility for qualified frame timing,
    own opaque geometry, measurement uncertainty, and protected observations.
    Work memory is O(M + N), with per-point candidate triangles vectorized.
    """
    points = np.asarray(points, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError("Expected Nx3 finite metric points")
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError("Expected Mx3x3 finite metric triangles")
    if not np.isfinite(points).all() or not np.isfinite(triangles).all():
        raise ValueError("Surface geometry must be finite")
    if not np.isfinite(max_distance_m) or max_distance_m <= 0:
        raise ValueError("A positive finite maximum distance is required")
    count = len(points)
    result = SurfaceMatches(
        np.full(count, np.inf),
        np.full(count, -1, dtype=np.int64),
        np.full((count, 3), np.nan),
        np.full((count, 3), np.nan),
    )
    if not count or not len(triangles):
        return result
    lower, upper = triangles.min(axis=1), triangles.max(axis=1)
    a = triangles[:, 0]
    u, v = triangles[:, 1] - a, triangles[:, 2] - a
    uu, vv = np.sum(u * u, axis=1), np.sum(v * v, axis=1)
    uv = np.sum(u * v, axis=1)
    normal = np.cross(u, v)
    area_squared = np.sum(normal * normal, axis=1)
    # Reject numerically ill-conditioned faces, never certify their line hull.
    valid = area_squared > 64 * np.finfo(float).eps * uu * vv
    valid &= np.isfinite(area_squared) & np.isfinite(uu * vv)
    radius_squared = max_distance_m**2
    global_low, global_high = lower.min(axis=0), upper.max(axis=0)
    for row, point in enumerate(points):
        if np.any(point < global_low - max_distance_m) or np.any(
            point > global_high + max_distance_m
        ):
            continue
        gap = np.maximum(np.maximum(lower - point, point - upper), 0)
        ids = np.flatnonzero(valid & (np.sum(gap * gap, axis=1) <= radius_squared))
        if not len(ids):
            continue
        faces = triangles[ids]
        w = point - a[ids]
        wu, wv = np.sum(w * u[ids], axis=1), np.sum(w * v[ids], axis=1)
        s = (wu * vv[ids] - wv * uv[ids]) / area_squared[ids]
        t = (wv * uu[ids] - wu * uv[ids]) / area_squared[ids]
        weights = np.column_stack((1 - s - t, s, t))
        nearest = a[ids] + s[:, None] * u[ids] + t[:, None] * v[ids]
        distance_squared = np.sum((nearest - point) ** 2, axis=1)
        distance_squared[weights.min(axis=1) < 0] = np.inf
        for first, second in ((0, 1), (1, 2), (2, 0)):
            start = faces[:, first]
            delta = faces[:, second] - start
            length_squared = np.sum(delta * delta, axis=1)
            fraction = np.clip(np.sum((point - start) * delta, axis=1) / length_squared, 0, 1)
            candidate = start + fraction[:, None] * delta
            candidate_squared = np.sum((candidate - point) ** 2, axis=1)
            replace = candidate_squared < distance_squared
            distance_squared[replace] = candidate_squared[replace]
            nearest[replace] = candidate[replace]
            edge_weights = np.zeros((len(ids), 3))
            edge_weights[:, first] = 1 - fraction
            edge_weights[:, second] = fraction
            weights[replace] = edge_weights[replace]
        best = int(np.argmin(distance_squared))
        distance = float(np.sqrt(distance_squared[best]))
        if np.isfinite(distance) and distance <= max_distance_m:
            result.distance_m[row] = distance
            result.triangle_index[row] = ids[best]
            result.closest[row] = nearest[best]
            result.barycentric[row] = weights[best]
    return result
