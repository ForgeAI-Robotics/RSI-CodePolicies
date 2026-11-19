"""Sensor-only checkpoint contract; native truth belongs to a separate auditor.

Positions are metres in one calibrated, fixed robot workspace frame. Timestamps
are seconds from one monotonic clock. No success labels or scene prim IDs enter
this interface. Calibration claims must be backed by the backend's manifest.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from typing import Literal, Protocol

Vec3 = tuple[float, float, float]

CONTINUOUS_TRANSPORT_PHASES = frozenset(
    {"PICK_APPROACH", "EXTRACT", "APPROACH_BAG", "APPROACH_FOREARM", "DEPOSIT"}
)


@dataclass(frozen=True)
class SensorProvenance:
    kind: Literal["measured", "synthetic"]
    calibrated: bool
    source: str
    stamp: float


@dataclass(frozen=True)
class SurfacePatch:
    center: Vec3
    normal: Vec3
    tangent: Vec3
    confidence: float
    stamp: float


@dataclass(frozen=True)
class PersonTrack:
    track_id: str
    position: Vec3
    forward: Vec3
    velocity: Vec3
    stamp: float
    bag: SurfacePatch | None = None
    forearm: SurfacePatch | None = None
    in_queue: bool = False
    in_waiting_area: bool = False


@dataclass(frozen=True)
class Observation:
    stamp: float
    sequence: int
    base_pose: tuple[float, float, float, float]
    joint_positions: Mapping[str, float]
    eef_position: Vec3
    eef_rpy: Vec3
    people: tuple[PersonTrack, ...] = ()
    paper_grasp: SurfacePatch | None = None
    paper_tip: Vec3 | None = None
    # Measured visible point-cloud major-axis span; never full hidden CAD length.
    paper_visible_length_m: float | None = None
    supply_exit: SurfacePatch | None = None
    deposit: SurfacePatch | None = None
    gripper_aperture_m: float | None = None
    gripper_force_n: float | None = None
    # Gravity/payload/bias-compensated force magnitude from native wrench/pad
    # measurement. Never infer this from actor-labelled target contact reports.
    wrist_force_n: float | None = None
    sensors: Mapping[str, SensorProvenance] = field(default_factory=dict)


@dataclass(frozen=True)
class Action:
    kind: Literal["hold", "hold_pose", "stop", "move_eef", "grip", "retreat"]
    target: Vec3 | None = None
    rpy: Vec3 | None = None
    gripper_close: bool | None = None
    duration_s: float = 0.05
    reason: str = ""
    # An explicit transport contract, not permission to jump to this endpoint.
    # The adapter must preserve this goal across fresh observations, generate
    # bounded motion, and recheck the remaining path before each <=50 ms window.
    trajectory_id: str | None = None
    max_speed_m_s: float | None = None
    max_acceleration_m_s2: float | None = None
    max_angular_speed_rad_s: float | None = None
    max_angular_acceleration_rad_s2: float | None = None

    def __post_init__(self) -> None:
        limits = (
            self.max_speed_m_s,
            self.max_acceleration_m_s2,
            self.max_angular_speed_rad_s,
            self.max_angular_acceleration_rad_s2,
        )
        if self.trajectory_id is None:
            if any(value is not None for value in limits):
                raise ValueError("Transport limits require an explicit trajectory_id")
            return
        if (
            not isinstance(self.trajectory_id, str)
            or not self.trajectory_id.strip()
            or self.kind != "move_eef"
            or self.reason not in CONTINUOUS_TRANSPORT_PHASES
            or self.gripper_close is not None
        ):
            raise ValueError("Continuous transport requires an eligible move_eef phase")
        if (
            self.target is None
            or self.rpy is None
            or len(self.target) != 3
            or len(self.rpy) != 3
            or not all(isfinite(value) for value in (*self.target, *self.rpy))
        ):
            raise ValueError("Continuous transport requires a finite full pose goal")
        if not isfinite(self.duration_s) or not 0 < self.duration_s <= 0.05:
            raise ValueError("Continuous transport feedback window must be <=50 ms")
        if any(
            value is None or isinstance(value, bool) or not isfinite(value) or value <= 0
            for value in limits
        ):
            raise ValueError("Continuous transport requires four finite positive motion limits")


class Backend(Protocol):
    """Adapter over existing control, never a newly invented vendor controller.

    hold_pose is a simulation-only explicit encoder-target freeze for rejected
    unheld PICK_APPROACH checks. It is not a zero-velocity command or a verified
    physical-robot stopping primitive. Generic hold retains its old semantics.
    execute must be bounded/preemptible; check_reachability must check the whole
    path with locked base, joint limits, human/fixture collisions and tool shape.
    A point IK result alone does not satisfy this contract. observe must produce
    sensor-derived features and cannot substitute USD poses or native contacts
    labelled by actor identity for perception. Native auditing stays separate.
    trajectory_id actions require explicit support: a legacy adapter must reject
    them instead of treating the stable endpoint as an immediate drive target.
    Stable IDs bind the full pose and limits. Hold, grip, stop, retreat or another
    phase cancels the old transport reference; native interlocks remain active.
    """

    def observe(self) -> Observation: ...
    def execute(self, action: Action) -> None: ...
    def check_reachability(self, action: Action) -> bool: ...
    def stop(self, reason: str) -> None: ...
