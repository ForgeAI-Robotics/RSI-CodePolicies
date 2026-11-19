"""Explicit, uncalibrated RhinoV6 parallel-jaw simulation drive profile.

The publisher supplies zero limits and no actuators. This geometric profile
closes the planar linkage in FK and keeps the jaw orientation parallel. It is
an engineering approximation, not a measured hardware transmission model.
"""

from __future__ import annotations

import numpy as np


def finger_angles(opening: float) -> tuple[float, float, float, float]:
    """Return left finger driver/coupler/spring/follower angles in radians."""
    q = -float(np.clip(opening, 0, 1)) * -0.60
    a = np.array([-0.0130855328263, 0.0781563625797])
    b = np.array([-0.030, 0.072])
    c0 = b + [0.032, 0]
    d0 = a + [0.032, 0.0507543101618]
    rot = np.array([[np.cos(q), -np.sin(q)], [np.sin(q), np.cos(q)]])
    c = b + rot @ (c0 - b)
    r, s = np.linalg.norm(d0 - a), np.linalg.norm(d0 - c0)
    ac = c - a
    dist = np.linalg.norm(ac)
    axis = ac / dist
    along = (r * r - s * s + dist * dist) / (2 * dist)
    height = np.sqrt(max(0, r * r - along * along))
    perp = np.array([-axis[1], axis[0]])
    branch = np.sign(np.cross(c0 - a, d0 - a))
    d = a + along * axis + branch * height * perp

    def angle(v):
        return np.arctan2(v[1], v[0])

    spring = angle(d0 - a) - angle(d - a)
    coupler = angle(d - c) - angle(d0 - c0) - q
    return q, coupler, spring, -spring


def joint_targets(opening: float, hand: str) -> dict[str, float]:
    values = finger_angles(opening)
    return {
        f"{hand}_rhino_v6_{part}_{side}": float(sign * angle)
        for side, sign in [("L", 1), ("R", -1)]
        for part, angle in zip(("driver", "coupler", "spring", "follower"), values, strict=True)
    }
