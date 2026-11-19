"""Shared own visible geometry conversion; no simulation or control imports."""

import numpy as np
from scipy.spatial import ConvexHull


def ellipsoid_surface(size: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The existing renderer's exact 13 x 24 sampled convex hull.

    Hull is computed in float64; renderer and FK ray meshes consume float32
    vertices. This is a tessellated surface, not an analytic ellipsoid.
    """
    directions = np.array(
        [
            [np.sin(a) * np.cos(b), np.sin(a) * np.sin(b), np.cos(a)]
            for a in np.linspace(0, np.pi, 13)
            for b in np.linspace(0, 2 * np.pi, 24, endpoint=False)
        ]
    )
    vertices = np.unique(np.round(directions * size, 12), axis=0)
    hull = ConvexHull(vertices)
    return vertices.astype(np.float32), hull.simplices
