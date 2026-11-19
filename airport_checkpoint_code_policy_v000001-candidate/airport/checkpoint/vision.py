"""Classical rendered RGB-D perception with explicit installation priors.

No USD prim path, semantic annotation, actor pose or native task label is accepted.
Colors are a controlled-scene visual convention, not a general human detector.
Workspace boxes are installation ROIs: every reported position still comes from
finite rendered depth. Missing/ambiguous observations are omitted, never filled
with CAD targets. Forward direction requires observed motion and is retained at
rest. The visible paper endpoint is not an estimate of an occluded endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from scipy.ndimage import label as connected_components
from scipy.optimize import linear_sum_assignment

from .perception import estimate_patch
from .policy import PolicyConfig
from .sensors import RGBDFrame, deproject_depth
from .types import Observation, PersonTrack, SensorProvenance, SurfacePatch

Bounds = tuple[tuple[float, float, float], tuple[float, float, float]]


@dataclass(frozen=True)
class VisionConfig:
    workspace: Bounds = ((-0.5, -4.0, 0.05), (2.6, 4.5, 2.1))
    person_roi: Bounds = ((0.5, -4.0, 0.85), (2.5, 4.5, 1.8))
    inspection_roi: Bounds = ((0.60, -0.30, 0.0), (1.45, 0.45, 2.0))
    waiting_roi: Bounds = ((1.30, -3.0, 0.0), (2.25, -0.55, 2.0))
    supply_roi: Bounds = ((0.24, -0.43, 0.90), (0.42, -0.25, 1.17))
    deposit_roi: Bounds = ((0.24, -1.12, 0.70), (0.52, -0.78, 0.93))
    forearm_z: tuple[float, float] = (0.75, 1.20)
    min_color_points: int = 25
    min_patch_points: int = 20
    cluster_voxel_m: float = 0.06
    track_gate_m: float = 0.35
    track_expiry_s: float = 2.0
    heading_displacement_m: float = 0.06
    object_association_m: float = 0.65
    paper_association_m: float = 0.20
    patch_radius_m: float = 0.075
    patch_residual_m: float = 0.006
    pixel_stride: int = 1
    # Ratios tolerate brightness/exposure shifts; thresholds are configurable.
    red_ratio: float = 1.3
    blue_ratio: float = 1.25
    minimum_brightness: float = 0.06
    paper_brightness: float = 0.45
    paper_max_chroma: float = 0.24
    paper_length_m: float = 0.17
    paper_width_m: float = 0.03
    supplied_exposed_fraction: float = 0.5
    # Append to preserve existing positional configuration arguments.
    vision_profile: Literal["legacy", "supply_bounds_v1"] = "legacy"

    def __post_init__(self) -> None:
        if self.vision_profile not in {"legacy", "supply_bounds_v1"}:
            raise ValueError("Invalid vision profile")
        for name in (
            "workspace",
            "person_roi",
            "inspection_roi",
            "waiting_roi",
            "supply_roi",
            "deposit_roi",
        ):
            bounds = np.asarray(getattr(self, name), float)
            if (
                bounds.shape != (2, 3)
                or not np.isfinite(bounds).all()
                or not np.all(bounds[0] < bounds[1])
            ):
                raise ValueError(f"Invalid installation ROI: {name}")
        for name in ("min_color_points", "min_patch_points", "pixel_stride"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"Invalid positive count: {name}")
        for name in (
            "cluster_voxel_m",
            "track_gate_m",
            "track_expiry_s",
            "heading_displacement_m",
            "object_association_m",
            "paper_association_m",
            "patch_radius_m",
            "patch_residual_m",
            "paper_length_m",
            "paper_width_m",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"Invalid positive threshold: {name}")
        if not 0 < self.supplied_exposed_fraction <= 1:
            raise ValueError("Invalid supplied exposed fraction")


def inside(points: np.ndarray, bounds: Bounds) -> np.ndarray:
    return ((points >= np.asarray(bounds[0])) & (points <= np.asarray(bounds[1]))).all(axis=-1)


def classify_colors(rgb: np.ndarray, config: VisionConfig | None = None) -> dict[str, np.ndarray]:
    """Mutually exclusive RGB ratio masks, accepting uint8 or normalized floats."""
    config = config or VisionConfig()
    array = np.asarray(rgb)
    if array.ndim < 2 or array.shape[-1] not in (3, 4):
        raise ValueError("Expected RGB or RGBA array")
    colors = array[..., :3].astype(float)
    if np.issubdtype(array.dtype, np.integer):
        colors /= 255.0
    if not np.isfinite(colors).all() or colors.min(initial=0) < 0 or colors.max(initial=0) > 1:
        raise ValueError("RGB values must be finite uint8 or normalized floats")
    red, green, blue = np.moveaxis(colors, -1, 0)
    brightness = colors.max(axis=-1)
    chroma = brightness - colors.min(axis=-1)
    lit = brightness >= config.minimum_brightness
    warm_hue = (green - blue) / np.maximum(red - blue, 1e-8)
    red_mask = lit & (red > config.red_ratio * green) & (red > 1.4 * blue) & (warm_hue < 0.30)
    blue_mask = (
        lit
        & (blue > config.blue_ratio * red)
        & (blue > 1.10 * green)
        & ((blue - red) > 0.07)
        & (brightness >= 0.12)
    )
    # Skin is orange/pink; a strong red garment is explicitly excluded.
    skin = (
        lit
        & (red > green)
        & (green > 1.03 * blue)
        & (red < 2.5 * green)
        & (blue > 0.04)
        & (warm_hue >= 0.30)
        & (warm_hue <= 0.75)
    )
    skin &= ~red_mask & ~blue_mask
    paper = (brightness >= config.paper_brightness) & (chroma <= config.paper_max_chroma)
    paper &= ~red_mask & ~blue_mask & ~skin
    cyan = lit & (blue > 1.2 * red) & (green > 1.2 * red) & (blue < 1.7 * green)
    orange = lit & (red > 1.2 * green) & (green > 1.2 * blue)
    return {
        "person": red_mask,
        "bag": blue_mask,
        "skin": skin,
        "paper": paper,
        "supply": cyan,
        "deposit": orange,
    }


def _spatial_clusters_sparse(
    points: np.ndarray, voxel_m: float, min_points: int
) -> list[np.ndarray]:
    """26-neighbour voxel components, retaining original depth points."""
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Expected finite Nx3 points")
    if len(points) == 0:
        return []
    keys = np.floor(points / voxel_m).astype(np.int64)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    lookup = {tuple(key): i for i, key in enumerate(unique)}
    unseen = set(lookup)
    groups = []
    while unseen:
        seed = min(unseen)
        unseen.remove(seed)
        stack, indices = [seed], []
        while stack:
            key = stack.pop()
            indices.append(lookup[key])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        neighbour = (key[0] + dx, key[1] + dy, key[2] + dz)
                        if neighbour in unseen:
                            unseen.remove(neighbour)
                            stack.append(neighbour)
        mask = np.isin(inverse, indices)
        if mask.sum() >= min_points:
            groups.append(points[mask])
    return sorted(groups, key=lambda group: tuple(np.median(group, axis=0)))


# Bound dense temporary storage: bool occupancy + int32 labels use <=10 MB.
# Sparse distant observations retain the original traversal without a huge grid.
_MAX_DENSE_VOXELS = 2_000_000


def spatial_clusters(points: np.ndarray, voxel_m: float, min_points: int) -> list[np.ndarray]:
    """Same 26-neighbour components and stable original-point ordering.

    C-order label seeds follow lexicographic voxel order, matching the sparse
    traversal's minimum seed. Stable point grouping preserves within-component
    input order and tied-median component ordering exactly.
    """
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Expected finite Nx3 points")
    if len(points) == 0:
        return []
    keys = np.floor(points / voxel_m).astype(np.int64)
    lower, upper = keys.min(axis=0), keys.max(axis=0)
    # Python integers avoid overflow when distant int64 voxel coordinates are
    # subtracted or multiplied. Never allocate before this bound is checked.
    shape = tuple(int(hi) - int(lo) + 1 for lo, hi in zip(lower, upper, strict=True))
    cells = shape[0] * shape[1] * shape[2]
    if cells > _MAX_DENSE_VOXELS:
        return _spatial_clusters_sparse(points, voxel_m, min_points)
    offsets = keys - lower
    occupied = np.zeros(shape, dtype=bool)
    occupied[tuple(offsets.T)] = True
    labels, count = connected_components(occupied, structure=np.ones((3, 3, 3), dtype=bool))
    point_labels = labels[tuple(offsets.T)]
    counts = np.bincount(point_labels, minlength=count + 1)
    order = np.argsort(point_labels, kind="stable")
    ends = np.cumsum(counts)
    groups = [
        points[order[ends[index - 1] : ends[index]]]
        for index in range(1, count + 1)
        if counts[index] >= min_points
    ]
    return sorted(groups, key=lambda group: tuple(np.median(group, axis=0)))


@dataclass
class _Track:
    track_id: str
    center: np.ndarray
    stamp: float
    heading_origin: np.ndarray
    forward: np.ndarray
    velocity: np.ndarray


class CheckpointVision:
    """Stateful detector/tracker; call once per new rendered camera frame.

    telemetry supplies encoder/FK/gripper/bias-compensated force channels only.
    All of its visual fields are discarded. diagnostics describes each missing
    channel and pixel count for calibration from actual renderer snapshots.
    """

    def __init__(self, config: VisionConfig | None = None):
        self.config = config or VisionConfig()
        self.tracks: dict[str, _Track] = {}
        self._next_id = 0
        self._last_stamp = -np.inf
        self._paper_center: np.ndarray | None = None
        self._supply_paper_center: np.ndarray | None = None
        self._paper_stamp = -np.inf
        self._grip_limits = PolicyConfig()
        self._paper_free_tracking = False
        self.diagnostics: dict = {}
        self.paper_points = np.empty((0, 3))  # Current measured tracked component only.

    def diagnostic_state(self) -> dict:
        """Serializable measurement-derived belief for reproducing a detector call."""
        return {
            "paper_center": None if self._paper_center is None else self._paper_center.tolist(),
            "paper_stamp": float(self._paper_stamp) if np.isfinite(self._paper_stamp) else None,
            "paper_free_tracking": self._paper_free_tracking,
            "supply_paper_center": (
                None if self._supply_paper_center is None else self._supply_paper_center.tolist()
            ),
            "scope": "measurement-derived tracker belief, not environment truth",
        }

    def _track_people(self, components: list[np.ndarray], stamp: float) -> list[_Track]:
        centers = [np.median(component, axis=0) for component in components]
        self.tracks = {
            key: track
            for key, track in self.tracks.items()
            if stamp - track.stamp <= self.config.track_expiry_s
        }
        old = list(self.tracks.values())
        matches = {}
        if old and centers:
            distances = np.array(
                [
                    [np.linalg.norm((center - track.center)[:2]) for center in centers]
                    for track in old
                ]
            )
            rows, cols = linear_sum_assignment(distances)
            matches = {
                int(col): old[row]
                for row, col in zip(rows, cols, strict=True)
                if distances[row, col] <= self.config.track_gate_m
            }
        visible = []
        for index, center in enumerate(centers):
            track = matches.get(index)
            if track is None:
                track = _Track(
                    f"vision-person-{self._next_id:04d}",
                    center,
                    stamp,
                    center.copy(),
                    np.zeros(3),
                    np.zeros(3),
                )
                self._next_id += 1
            else:
                dt = stamp - track.stamp
                measured = (center - track.center) / dt
                track.velocity = 0.5 * measured + 0.5 * track.velocity
                delta = center - track.heading_origin
                delta[2] = 0
                if np.linalg.norm(delta) >= self.config.heading_displacement_m:
                    track.forward = delta / np.linalg.norm(delta)
                    track.heading_origin = center.copy()
                track.center, track.stamp = center, stamp
            self.tracks[track.track_id] = track
            visible.append(track)
        return visible

    def _surface(
        self, points: np.ndarray, eye: np.ndarray, stamp: float, name: str
    ) -> SurfacePatch | None:
        if len(points) < self.config.min_patch_points:
            self.diagnostics[name] = "insufficient measured depth points"
            return None
        # Prefer the robot-facing x surface, then a bounded local patch. This is
        # a workspace convention, not a dynamic object pose from the simulator.
        cutoff = np.quantile(points[:, 0], 0.30)
        facing = points[points[:, 0] <= cutoff + 0.006]
        center = np.median(facing, axis=0)
        best = None
        last_error = "no local plane support"
        for radius in sorted(
            {self.config.patch_radius_m, 0.050, 0.035, 0.025, 0.018, 0.015}, reverse=True
        ):
            if radius > self.config.patch_radius_m:
                continue
            local = facing[np.linalg.norm(facing - center, axis=1) <= radius]
            try:
                patch = estimate_patch(
                    local,
                    tuple(eye),
                    stamp,
                    max_residual_m=self.config.patch_residual_m,
                    min_points=self.config.min_patch_points,
                )
                if abs(patch.normal[0]) < 0.30:
                    last_error = "local patch is not robot-facing"
                    continue
                if patch.normal[0] > 0:
                    patch = replace(patch, normal=tuple(-np.asarray(patch.normal)))
                if best is None or patch.confidence > best.confidence:
                    best = patch
                    self.diagnostics[name] = {
                        "points": len(local),
                        "confidence": patch.confidence,
                        "fit_radius_m": radius,
                    }
                if patch.confidence >= 0.8:
                    break
            except ValueError as exc:
                last_error = str(exc)
        if best is None:
            self.diagnostics[name] = last_error
        return best

    def _fixture(
        self, points: np.ndarray, bounds: Bounds, stamp: float, name: str
    ) -> SurfacePatch | None:
        points = points[inside(points, bounds)]
        if len(points) < self.config.min_patch_points:
            self.diagnostics[name] = "fixture color/depth missing within installation ROI"
            return None
        # Measured upper rim; pulling/entry direction +Z is a fixed installation
        # prior. Position, height and confidence are obtained from visible depth.
        top = np.quantile(points[:, 2], 0.95)
        rim = points[points[:, 2] >= top - 0.008]
        if len(rim) < max(4, self.config.min_patch_points // 4):
            self.diagnostics[name] = "insufficient visible rim"
            return None
        center = np.median(rim, axis=0)
        center[2] = np.median(rim[:, 2])
        self.diagnostics[name] = {"points": len(rim), "direction_prior": "fixed installation +Z"}
        return SurfacePatch(tuple(center), (0, 0, 1), (1, 0, 0), 0.9, stamp)

    def _supply_paper_parts(
        self, points: np.ndarray, supply: SurfacePatch | None, eye: np.ndarray, stamp: float
    ) -> list[np.ndarray]:
        """Require a unique measured sheet matching the declared supply geometry."""
        if (
            supply is None
            or supply.confidence < self._grip_limits.min_patch_confidence
            or not 0 <= stamp - supply.stamp <= self._grip_limits.sensor_max_age_s
        ):
            self.diagnostics["paper_selection"] = "missing current measured supply rim"
            return []
        tolerance = self.config.patch_residual_m
        exposed_length = self.config.paper_length_m * self.config.supplied_exposed_fraction
        direction = np.asarray(supply.normal)
        rim = np.asarray(supply.center)
        points = points[inside(points, self.config.supply_roi)]
        points = points[(points - rim) @ direction > 0.004]
        parts = spatial_clusters(
            points, min(0.02, self.config.cluster_voxel_m), self.config.min_color_points
        )
        accepted = []
        identity_bounds = []
        rejected = []
        for part in parts:
            try:
                patch = estimate_patch(
                    part,
                    tuple(eye),
                    stamp,
                    min_points=self.config.min_patch_points,
                    max_residual_m=tolerance,
                )
                normal = np.asarray(patch.normal)
                if (
                    patch.confidence < self._grip_limits.min_patch_confidence
                    or abs(normal @ direction) * exposed_length > tolerance
                    or np.max(np.abs((part - patch.center) @ normal)) > tolerance
                ):
                    raise ValueError("unsupported upright sheet plane")
                width_axis = np.cross(direction, normal)
                width_norm = np.linalg.norm(width_axis)
                if width_norm < 1e-8:
                    raise ValueError("sheet width axis is undefined")
                width_axis /= width_norm
                length_axis = np.cross(normal, width_axis)
                width = float(np.ptp(part @ width_axis))
                length = float(np.ptp(part @ length_axis))
                heights = (part - rim) @ direction
                center = np.median(part, axis=0)
                lateral = center - rim - ((center - rim) @ direction) * direction
                if (
                    abs(width - self.config.paper_width_m) > 2 * tolerance
                    or abs(length - exposed_length) > 2 * tolerance
                    or heights.min() > 2 * tolerance
                    or abs(heights.max() - exposed_length) > 2 * tolerance
                    or np.linalg.norm(lateral) > self.config.paper_width_m + tolerance
                ):
                    raise ValueError("sheet dimensions or measured rim relation disagree")
                accepted.append(part)
                if self.config.vision_profile == "supply_bounds_v1":
                    # Measured visible bounds, not an inferred hidden sheet edge.
                    # The rim and fitted plane fix the basis independently of
                    # the point-density-dependent dominant in-plane SVD axis.
                    basis = np.column_stack((width_axis, length_axis, normal))
                    projected = (part - rim) @ basis
                    low, high = projected.min(axis=0), projected.max(axis=0)
                    identity_bounds.append((rim + basis @ ((low + high) / 2), basis, low, high))
            except ValueError as exc:
                rejected.append(str(exc))
        self.diagnostics["paper_selection"] = {
            "source": "current depth, measured supply rim and declared paper dimensions",
            "depth_points": len(points),
            "components": len(parts),
            "geometry_candidates": len(accepted),
            "rejected": rejected,
        }
        if len(accepted) != 1:
            return []
        center = np.median(accepted[0], axis=0)
        if self.config.vision_profile == "supply_bounds_v1":
            center, basis, low, high = identity_bounds[0]
            anchor = center if self._supply_paper_center is None else self._supply_paper_center
            self.diagnostics["paper_selection"].update(
                {
                    "vision_profile": self.config.vision_profile,
                    "identity_source": "current visible projected min/max in rim/plane basis",
                    "identity_center_m": center.tolist(),
                    "identity_anchor_m": anchor.tolist(),
                    "identity_distance_m": float(np.linalg.norm(center - anchor)),
                    "identity_threshold_m": 2 * tolerance,
                    "identity_bounds_min_m": low.tolist(),
                    "identity_bounds_max_m": high.tolist(),
                    "identity_basis_columns": basis.tolist(),
                    "identity_origin_m": rim.tolist(),
                }
            )
        if (
            self._supply_paper_center is not None
            and np.linalg.norm(center - self._supply_paper_center) > 2 * tolerance
        ):
            self.diagnostics["paper_selection"]["identity"] = "supply sheet moved or changed"
            return []
        if self._supply_paper_center is None:
            self._supply_paper_center = center.copy()
        return accepted

    def observe(self, frame: RGBDFrame, telemetry: Observation) -> Observation:
        if not frame.measured or not np.isfinite(frame.stamp) or frame.stamp <= self._last_stamp:
            raise ValueError("Vision requires a new finite measured camera timestamp")
        if frame.rgb.shape[:2] != frame.depth_m.shape:
            raise ValueError("RGB/depth registration shapes disagree")
        self._last_stamp = frame.stamp
        masks = classify_colors(frame.rgb, self.config)
        points, pixels = deproject_depth(frame, stride=self.config.pixel_stride)
        keep = inside(points, self.config.workspace)
        points, pixels = points[keep], pixels[keep]
        clouds = {name: points[mask[pixels[:, 0], pixels[:, 1]]] for name, mask in masks.items()}
        self.paper_points = np.empty((0, 3))
        self.diagnostics = {
            "stamp": frame.stamp,
            "source": frame.source,
            "color_depth_counts": {name: len(cloud) for name, cloud in clouds.items()},
            "paper_endpoint": "visible lower endpoint only; occluded endpoint unknown",
        }
        red = clouds["person"]
        red = red[inside(red, self.config.person_roi)]
        components = spatial_clusters(
            red, self.config.cluster_voxel_m, self.config.min_color_points
        )
        # Reject tiny fragments and room-scale merged garments/background.
        components = [
            c
            for c in components
            if 0.06 <= np.ptp(c[:, 2]) <= 1.4 and np.ptp(c[:, 0]) <= 0.8 and np.ptp(c[:, 1]) <= 0.8
        ]
        # A red sleeve and torso may be separated by a black strap. Merge
        # nearby garment fragments within a single-person footprint; queue
        # installation spacing is larger than this 0.5 m association bound.
        merged = []
        for component in components:
            centre = np.median(component, axis=0)
            match = next(
                (
                    i
                    for i, group in enumerate(merged)
                    if np.linalg.norm((np.median(group, axis=0) - centre)[:2]) < 0.50
                ),
                None,
            )
            if match is None:
                merged.append(component)
            else:
                merged[match] = np.vstack([merged[match], component])
        tracks = self._track_people(merged, frame.stamp)
        eye = np.asarray(frame.calibration.optical_to_work)[:3, 3]
        people = []
        for track in tracks:
            bag_points = clouds["bag"]
            nearby = (
                np.linalg.norm(bag_points[:, :2] - track.center[:2], axis=1)
                <= self.config.object_association_m
            )
            bag_points = bag_points[nearby & ~inside(bag_points, self.config.supply_roi)]
            bag_parts = spatial_clusters(
                bag_points, self.config.cluster_voxel_m, self.config.min_color_points
            )
            if len(bag_parts) == 1:
                bag = self._surface(bag_parts[0], eye, frame.stamp, track.track_id + "/bag")
            else:
                bag = None
                self.diagnostics[track.track_id + "/bag"] = (
                    f"ambiguous/missing bag clusters: {len(bag_parts)}"
                )
            skin = clouds["skin"]
            nearby = (
                np.linalg.norm(skin[:, :2] - track.center[:2], axis=1)
                <= self.config.object_association_m
            )
            skin = skin[
                nearby
                & (skin[:, 0] < track.center[0] + 0.05)
                & (skin[:, 2] >= self.config.forearm_z[0])
                & (skin[:, 2] <= self.config.forearm_z[1])
            ]
            arm = self._surface(skin, eye, frame.stamp, track.track_id + "/forearm")
            people.append(
                PersonTrack(
                    track.track_id,
                    tuple(track.center),
                    tuple(track.forward),
                    tuple(track.velocity),
                    frame.stamp,
                    bag,
                    arm,
                    bool(inside(track.center, self.config.inspection_roi)),
                    bool(inside(track.center, self.config.waiting_roi)),
                )
            )
        supply = self._fixture(clouds["supply"], self.config.supply_roi, frame.stamp, "supply_exit")
        deposit = self._fixture(clouds["deposit"], self.config.deposit_roi, frame.stamp, "deposit")
        paper_points = clouds["paper"]
        grip_values = (telemetry.gripper_aperture_m, telemetry.gripper_force_n)
        grip_sensors = [telemetry.sensors.get(name) for name in ("gripper", "force")]
        if (
            not self._paper_free_tracking
            and self._paper_center is not None
            and 0 <= frame.stamp - self._paper_stamp <= self.config.track_expiry_s
            and 0 <= telemetry.stamp - frame.stamp <= self._grip_limits.sensor_max_age_s
            and all(value is not None and np.isfinite(value) for value in grip_values)
            and all(
                sensor is not None
                and sensor.calibrated
                and sensor.source
                and sensor.kind in {"measured", "synthetic"}
                and np.isfinite(sensor.stamp)
                and 0 <= telemetry.stamp - sensor.stamp <= self._grip_limits.sensor_max_age_s
                for sensor in grip_sensors
            )
            and 0 <= telemetry.gripper_aperture_m <= self._grip_limits.max_grip_aperture_m
            and telemetry.gripper_force_n >= self._grip_limits.min_grip_force_n
        ):
            self._paper_free_tracking = True
        if self._paper_center is not None and (
            frame.stamp - self._paper_stamp > self.config.track_expiry_s
        ):
            paper_parts = []
            self.diagnostics["paper_selection"] = (
                "paper identity expired; no automatic reacquisition"
            )
        elif not self._paper_free_tracking:
            paper_parts = self._supply_paper_parts(points, supply, eye, frame.stamp)
        elif self._paper_center is None:
            paper_parts = []
            self.diagnostics["paper_selection"] = (
                "free tracking requires an existing paper identity"
            )
        else:
            paper_points = paper_points[
                np.linalg.norm(paper_points - self._paper_center, axis=1)
                <= self.config.paper_association_m
            ]
            paper_parts = spatial_clusters(
                paper_points, min(0.02, self.config.cluster_voxel_m), self.config.min_color_points
            )
        grasp, tip, visible_length = None, None, None
        if len(paper_parts) == 1:
            paper = paper_parts[0]
            try:
                grasp = estimate_patch(
                    paper,
                    tuple(eye),
                    frame.stamp,
                    min_points=self.config.min_patch_points,
                    max_residual_m=self.config.patch_residual_m,
                )
                # Grasp the exposed upper region, away from the occluding slot.
                upper = paper[paper[:, 2] >= np.quantile(paper[:, 2], 0.75)]
                grasp = replace(grasp, center=tuple(np.median(upper, axis=0)))
                lower = paper[paper[:, 2] <= np.quantile(paper[:, 2], 0.10)]
                tip = tuple(np.median(lower, axis=0))
                projection = paper @ np.asarray(grasp.tangent)
                visible_length = float(
                    np.quantile(projection, 0.98) - np.quantile(projection, 0.02)
                )
                self._paper_center, self._paper_stamp = np.median(paper, axis=0), frame.stamp
                self.paper_points = paper.copy()
                self.diagnostics["paper"] = {"points": len(paper), "confidence": grasp.confidence}
            except ValueError as exc:
                self.diagnostics["paper"] = str(exc)
        else:
            self.diagnostics["paper"] = f"ambiguous/missing paper clusters: {len(paper_parts)}"
        sensors = dict(telemetry.sensors)
        sensors["rgbd"] = SensorProvenance(
            "synthetic",
            True,
            frame.source + "; classical RGB ratios/depth geometry/tracking",
            frame.stamp,
        )
        return replace(
            telemetry,
            people=tuple(people),
            paper_grasp=grasp,
            paper_tip=tip,
            paper_visible_length_m=visible_length,
            supply_exit=supply,
            deposit=deposit,
            sensors=sensors,
        )
