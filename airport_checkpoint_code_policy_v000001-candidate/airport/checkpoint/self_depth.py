"""Empirical own-depth candidates for explicitly qualified simulation profiles.

A candidate within a stated uncertainty band is not definite material identity.
In particular, a real occluder closer by less than epsilon can be a candidate.
"""

from __future__ import annotations

import copy
import hashlib
import json
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import mujoco
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter

from .own_surface import closest_triangle_surface
from .render_geometry import ellipsoid_surface
from .sensors import RawMeasuredObservation, deproject_depth


@dataclass(frozen=True)
class FrameSupport:
    calibration_reference: str = ""
    epoch_reference: str = ""
    epoch_kind: Literal["missing", "measured", "offline_assumed"] = "missing"
    interior_samples: int = 0
    interior_median_error_m: float = float("inf")
    interior_p95_error_m: float = float("inf")


@dataclass(frozen=True)
class ExploratoryGate:
    # Experimental health envelope, not a certified measurement uncertainty.
    minimum_samples: int = 20
    maximum_median_error_m: float = 20e-6
    maximum_p95_error_m: float = 100e-6


def candidate_labels(
    measured: np.ndarray,
    predicted: np.ndarray,
    *,
    epsilon_m: float,
    support: FrameSupport,
    graspable: np.ndarray | None = None,
    confirmed_human: np.ndarray | None = None,
    allow_offline_assumptions: bool = False,
    gate: ExploratoryGate | None = None,
) -> np.ndarray:
    """Label evidence; never delete points or mutate human collision margins.

    Default missing epoch/calibration yields no candidates. Offline assumed
    alignment must be opted into explicitly and must not enter a live adapter.
    RGB threshold flags are not confirmed-human instance measurements.
    """
    gate = gate or ExploratoryGate()
    measured, predicted = np.asarray(measured, float), np.asarray(predicted, float)
    if measured.ndim != 1 or measured.shape != predicted.shape:
        raise ValueError("Expected equal 1D depth arrays")
    if not np.isfinite(epsilon_m) or not 0 < epsilon_m <= 100e-6:
        raise ValueError("Prototype exploration is limited to 0 < epsilon <= 100 micrometres")
    paper = np.zeros(len(measured), bool) if graspable is None else np.asarray(graspable, bool)
    human = (
        np.zeros(len(measured), bool)
        if confirmed_human is None
        else np.asarray(confirmed_human, bool)
    )
    if paper.shape != measured.shape or human.shape != measured.shape:
        raise ValueError("Protection flags must match depth arrays")
    labels = np.full(len(measured), "unknown", dtype="<U32")
    labels[paper] = "protected_paper"
    labels[human] = "protected_human"
    free = ~paper & ~human
    blocker = None
    if not support.calibration_reference:
        blocker = "missing_calibration"
    elif not support.epoch_reference or support.epoch_kind == "missing":
        blocker = "missing_epoch"
    elif support.epoch_kind == "offline_assumed" and not allow_offline_assumptions:
        blocker = "unqualified_offline_epoch"
    elif support.epoch_kind not in {"measured", "offline_assumed"}:
        blocker = "missing_epoch"
    elif (
        support.interior_samples < gate.minimum_samples
        or not np.isfinite([support.interior_median_error_m, support.interior_p95_error_m]).all()
        or not 0 <= support.interior_median_error_m <= support.interior_p95_error_m
        or support.interior_median_error_m > gate.maximum_median_error_m
        or support.interior_p95_error_m > gate.maximum_p95_error_m
    ):
        blocker = "frame_health_failed"
    if blocker:
        labels[free] = blocker
        return labels
    valid = np.isfinite(measured) & (measured > 0)
    hit = np.isfinite(predicted) & (predicted > 0)
    labels[free & ~valid] = "invalid_depth"
    labels[free & valid & ~hit] = "mesh_miss"
    both = free & valid & hit
    residual = measured - predicted
    labels[both & (residual < -epsilon_m)] = "foreground"
    labels[both & (residual > epsilon_m)] = "depth_mismatch"
    labels[both & (abs(residual) <= epsilon_m)] = "self_candidate"
    return labels


class OwnOpaqueDepth:
    """Own rendered-mesh depth using a private, offline MuJoCo FK model.

    No MuJoCo dynamics or external-object geometry participates. The caller must
    supply a dedicated model/data, never a concurrently used live model.
    """

    def __init__(self, description):
        description = private_own_description(description)
        self.model = description.model
        self.data = mujoco.MjData(self.model)
        self.home = description.data.qpos.copy()
        self._encoders_ready = False
        self.required_joints = tuple(
            n for n in description.joint_names if n not in {"base_x", "base_y"}
        )
        root = self.model.body("base_link").id
        self.mesh_ids = []
        for gid in range(self.model.ngeom):
            body = int(self.model.geom_bodyid[gid])
            while body and body != root:
                body = int(self.model.body_parentid[body])
            if (
                body == root
                and self.model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH
                and self.model.geom_rgba[gid, 3] >= 0.999
            ):
                self.mesh_ids.append(gid)
        if not self.mesh_ids:
            raise ValueError("No opaque own robot meshes")
        # The private CAD mesh is immutable; only its measured rigid pose moves.
        self._surface_geometry = {}
        self._surface_triangles = {}
        for gid in self.mesh_ids:
            mid = int(self.model.geom_dataid[gid])
            if mid not in self._surface_geometry:
                va, vn = self.model.mesh_vertadr[mid], self.model.mesh_vertnum[mid]
                fa, fn = self.model.mesh_faceadr[mid], self.model.mesh_facenum[mid]
                vertices = self.model.mesh_vert[va : va + vn].astype(float)
                faces = self.model.mesh_face[fa : fa + fn]
                self._surface_geometry[mid] = (vertices, faces, vertices.min(0), vertices.max(0))

    def set_encoders(self, positions: dict[str, float]) -> None:
        self._encoders_ready = False
        if set(positions) != set(self.required_joints):
            raise ValueError("Complete own articulation encoder set required")
        values = np.array([positions[n] for n in self.required_joints], float)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite own encoders")
        self.data.qpos[:] = self.home
        self.encoder_limit_deviations = []
        for name, value in zip(self.required_joints, values, strict=True):
            joint = self.model.joint(name)
            low, high = self.model.jnt_range[joint.id]
            if self.model.jnt_limited[joint.id] and not low <= value <= high:
                # A measurement can include a numerical limit overshoot. Keep
                # the exact measured FK for diagnosis, never clip or issue it
                # as a control target; expose deviations to the caller.
                self.encoder_limit_deviations.append(
                    {
                        "joint": name,
                        "measured": float(value),
                        "low": float(low),
                        "high": float(high),
                    }
                )
            self.data.qpos[joint.qposadr[0]] = value
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        self._encoders_ready = True

    def predict(self, pixels_rc: np.ndarray, calibration) -> tuple[np.ndarray, np.ndarray]:
        if not self._encoders_ready:
            raise ValueError("Fresh complete own encoders required before predicting")
        pixels = np.asarray(pixels_rc, float)
        if pixels.ndim != 2 or pixels.shape[1] != 2 or not np.isfinite(pixels).all():
            raise ValueError("Finite Nx2 pixels required")
        matrix = np.asarray(calibration.optical_to_work, float)
        if (
            matrix.shape != (4, 4)
            or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1])
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1.0, atol=1e-6)
        ):
            raise ValueError("Valid camera pose required")
        if (
            not np.isfinite([calibration.fx, calibration.fy, calibration.cx, calibration.cy]).all()
            or min(calibration.fx, calibration.fy) <= 0
        ):
            raise ValueError("Valid intrinsics required")
        rows, cols = pixels.T
        if np.any(
            (rows < 0) | (rows >= calibration.height) | (cols < 0) | (cols >= calibration.width)
        ):
            raise ValueError("Pixel outside calibration resolution")
        dx, dy = calibration.pixel_offset_xy
        optical = np.column_stack(
            [
                (cols + dx - calibration.cx) / calibration.fx,
                (rows + dy - calibration.cy) / calibration.fy,
                np.ones(len(rows)),
            ]
        )
        norms = np.linalg.norm(optical, axis=1)
        rays = np.ascontiguousarray((optical / norms[:, None]) @ matrix[:3, :3].T)
        ids = np.full(len(rows), -1, np.int32)
        distances = np.full(len(rows), -1.0)
        previous_groups = self.model.geom_group.copy()
        try:
            self.model.geom_group[:] = 5
            self.model.geom_group[self.mesh_ids] = 0
            mujoco.mj_multiRay(
                self.model,
                self.data,
                np.ascontiguousarray(matrix[:3, 3]),
                rays.ravel(),
                np.array([1, 0, 0, 0, 0, 0], np.uint8),
                1,
                -1,
                ids,
                distances,
                len(rows),
                10.0,
            )
        finally:
            self.model.geom_group[:] = previous_groups
        return np.where(ids >= 0, distances / norms, np.nan), ids

    def surface_support(self, points: np.ndarray, epsilon_m: float) -> dict:
        """Exact triangle evidence after local AABB culling; no box ownership.

        Uses the same opaque own rendered triangle meshes as central rays.
        The Euclidean uncertainty band is distinct from axial-depth error.
        """
        if not self._encoders_ready:
            raise ValueError("Own measured FK is required")
        points = np.asarray(points, float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("Expected finite measured Nx3 points")
        distances = np.full(len(points), np.inf)
        geom_ids = np.full(len(points), -1, dtype=int)
        face_ids = np.full(len(points), -1, dtype=int)
        closest = np.full_like(points, np.nan)
        barycentric = np.full_like(points, np.nan)
        tested_points = 0
        for gid in self.mesh_ids:
            mid = int(self.model.geom_dataid[gid])
            vertices, faces, low, high = self._surface_geometry[mid]
            rotation = self.data.geom_xmat[gid].reshape(3, 3)
            origin = self.data.geom_xpos[gid]
            local = (points - origin) @ rotation
            gap = np.maximum(np.maximum(low - local, local - high), 0)
            indexes = np.flatnonzero(np.linalg.norm(gap, axis=1) <= epsilon_m)
            if not len(indexes):
                continue
            tested_points += len(indexes)
            if mid not in self._surface_triangles:
                self._surface_triangles[mid] = vertices[faces]
            triangles = self._surface_triangles[mid]
            match = closest_triangle_surface(local[indexes], triangles, max_distance_m=epsilon_m)
            better = match.distance_m < distances[indexes]
            chosen = indexes[better]
            distances[chosen] = match.distance_m[better]
            geom_ids[chosen] = gid
            face_ids[chosen] = match.triangle_index[better]
            closest[chosen] = match.closest[better] @ rotation.T + origin
            barycentric[chosen] = match.barycentric[better]
        return {
            "distance_m": distances,
            "geom_id": geom_ids,
            "face_id": face_ids,
            "closest_work": closest,
            "barycentric": barycentric,
            "point_geom_pairs_after_broadphase": tested_points,
        }


def private_own_description(description):
    """Compile a separate FK/ray model containing only the own robot subtree.

    No external body, flex, contact, constraint or actuator is copied. This
    model is never stepped. Its exact initial q is copied by own joint name.
    """
    original = ET.parse(Path(description.output) / "scene.xml").getroot()
    body = original.find("./worldbody/body[@name='base_link']")
    if body is None:
        raise ValueError("Own base must be a direct child of the fixed work frame")
    root = ET.Element("mujoco", {"model": "own_depth_fk_only"})
    for tag in ("compiler", "default"):
        element = original.find(tag)
        if element is not None:
            root.append(copy.deepcopy(element))
    own = copy.deepcopy(body)
    assets = ET.SubElement(root, "asset")
    mesh_names = {geom.get("mesh") for geom in own.iter("geom") if geom.get("mesh")}
    material_names = {geom.get("material") for geom in own.iter("geom") if geom.get("material")}
    source_assets = original.find("asset")
    if source_assets is None:
        raise ValueError("Own CAD description has no assets")
    texture_names = {
        item.get("texture")
        for item in source_assets.findall("material")
        if item.get("name") in material_names and item.get("texture")
    }
    for item in source_assets:
        wanted = {"mesh": mesh_names, "material": material_names, "texture": texture_names}
        if item.get("name") in wanted.get(item.tag, set()):
            assets.append(copy.deepcopy(item))
    for index, geom in enumerate(own.iter("geom")):
        name = geom.get("name")
        if geom.get("type", "sphere") == "ellipsoid":
            if name is None:
                raise ValueError("Rendered ellipsoid must have a stable own CAD name")
            gid = description.model.geom(name).id
            vertices, faces = ellipsoid_surface(description.model.geom_size[gid])
            mesh_name = f"own_rendered_ellipsoid_{index}"
            ET.SubElement(
                assets,
                "mesh",
                {
                    "name": mesh_name,
                    "vertex": " ".join(format(float(v), ".17g") for v in vertices.ravel()),
                    "face": " ".join(str(int(v)) for v in faces.ravel()),
                },
            )
            geom.set("type", "mesh")
            geom.set("mesh", mesh_name)
            geom.attrib.pop("size", None)
    ET.SubElement(root, "worldbody").append(own)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    for jid in range(model.njnt):
        joint = model.joint(jid)
        if model.jnt_type[jid] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            raise ValueError("Only fixed-base scalar own joints are supported")
        data.qpos[joint.qposadr[0]] = description.data.qpos[
            description.model.joint(joint.name).qposadr[0]
        ]
    return SimpleNamespace(model=model, data=data, joint_names=list(description.joint_names))


def measurement_digest(value) -> str:
    """Canonical producer/consumer binding; not a provenance assertion by itself."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class SelfDepthProfile:
    # Explicit empirical simulation configuration, no default deployment epsilon.
    profile: str
    evidence_sha256: str
    epsilon_m: float

    def __post_init__(self):
        if not self.profile or len(self.evidence_sha256) != 64:
            raise ValueError("A pinned profile and evidence SHA256 are required")
        if any(c not in "0123456789abcdef" for c in self.evidence_sha256):
            raise ValueError("Invalid evidence SHA256")
        if not np.isfinite(self.epsilon_m) or not 0 < self.epsilon_m <= 100e-6:
            raise ValueError("Empirical candidate epsilon must be within 0..100 micrometres")


def registration_failure(raw: RawMeasuredObservation, profile: SelfDepthProfile) -> str | None:
    frame = raw.rgbd
    sync = None if frame is None else frame.synchrony
    if frame is None or sync is None:
        return "missing_geometry_registration"
    if not frame.measured:
        return "unmeasured_frame"
    if sync.profile != profile.profile or sync.evidence_sha256 != profile.evidence_sha256:
        return "unapproved_renderer_profile"
    if sync.settings_verified is not True:
        return "unverified_renderer_settings"
    if tuple(frame.calibration.pixel_offset_xy) != (0.5, 0.5):
        return "unqualified_sampling_profile"
    if (
        not isinstance(sync.render_frame, int)
        or isinstance(sync.render_frame, bool)
        or sync.render_frame < 0
        or sync.encoder_sequence != raw.sequence
    ):
        return "invalid_frame_sequence"
    times = [sync.render_time, sync.physics_time, sync.encoder_stamp, raw.stamp, frame.stamp]
    if not np.isfinite(times).all() or min(times) < 0:
        return "invalid_frame_times"
    if (
        abs(sync.render_time - sync.physics_time) > 1e-9
        or abs(sync.encoder_stamp - raw.stamp) > 1e-9
        or abs(frame.stamp - raw.stamp) > 1e-9
    ):
        return "frame_epoch_mismatch"
    if sync.encoder_sha256 != measurement_digest(raw.joint_positions):
        return "encoder_binding_mismatch"
    if sync.calibration_sha256 != measurement_digest(asdict(frame.calibration)):
        return "calibration_binding_mismatch"
    return None


class RegisteredSelfDepth:
    """No point removal until profile registration and whole-frame health pass."""

    def __init__(self, description, profile: SelfDepthProfile):
        self.predictor = OwnOpaqueDepth(description)
        self.profile = profile
        self.last_render_frame = -1
        self.diagnostics = {}

    def labels(self, raw, pixels, paper, supported_human):
        # All unqualified points remain obstacles. No fallback OBB deletion.
        result = np.full(len(pixels), "unknown", dtype="<U40")
        result[paper] = "protected_paper"
        result[supported_human] = "protected_human"
        free = ~paper & ~supported_human
        reason = registration_failure(raw, self.profile)
        sync = None if raw.rgbd is None else raw.rgbd.synchrony
        if reason is None and sync.render_frame <= self.last_render_frame:
            reason = "nonadvancing_render_frame"
        if reason is not None:
            result[free] = reason
            self.diagnostics = {
                "sequence": raw.sequence,
                "stamp": raw.stamp,
                "status": reason,
                "removed_candidates": 0,
            }
            return result
        self.last_render_frame = sync.render_frame
        self.predictor.set_encoders(raw.joint_positions)
        frame = raw.rgbd
        predicted, _ = self.predictor.predict(pixels, frame.calibration)
        rows, cols = pixels.T
        measured = frame.depth_m[rows, cols].astype(float)
        residual = measured - predicted
        span = maximum_filter(frame.depth_m, size=5) - minimum_filter(frame.depth_m, size=5)
        interior = (
            np.isfinite(predicted)
            & (measured > 0.1)
            & (measured < 2.0)
            & (span[rows, cols] < 0.01)
            & (abs(residual) < 0.02)
            & (rows % 8 == 0)
            & (cols % 8 == 0)
            & ~paper
            & ~supported_human
        )
        errors = abs(residual[interior])
        support = FrameSupport(
            calibration_reference=sync.calibration_sha256,
            epoch_reference=sync.evidence_sha256,
            epoch_kind="measured",
            interior_samples=len(errors),
            interior_median_error_m=float(np.median(errors)) if len(errors) else float("inf"),
            interior_p95_error_m=float(np.quantile(errors, 0.95)) if len(errors) else float("inf"),
        )
        result = candidate_labels(
            measured,
            predicted,
            epsilon_m=self.profile.epsilon_m,
            support=support,
            graspable=paper,
            confirmed_human=supported_human,
        )
        # Central rays can see a deeper layer when the measured pixel is at
        # an own-mesh silhouette. Only exact surface evidence inside the same
        # empirical metric band can qualify; large residual alone never can.
        unresolved = np.isin(result, ["mesh_miss", "foreground", "depth_mismatch"])
        indexes = np.flatnonzero(unresolved)
        witnesses = []
        surface_pairs = 0
        if len(indexes):
            mask = np.zeros(frame.depth_m.shape, dtype=bool)
            mask[pixels[indexes, 0], pixels[indexes, 1]] = True
            points, sampled = deproject_depth(frame, mask=mask)
            if not np.array_equal(sampled, pixels[indexes]):
                raise ValueError("Surface evidence pixels must be ordered and unique")
            surface = self.predictor.surface_support(points, self.profile.epsilon_m)
            surface_pairs = surface["point_geom_pairs_after_broadphase"]
            for i in np.flatnonzero(np.isfinite(surface["distance_m"])):
                index = indexes[i]
                gid = int(surface["geom_id"][i])
                witnesses.append(
                    {
                        "pixel_rc": pixels[index].tolist(),
                        "previous_label": str(result[index]),
                        "axial_residual_m": float(residual[index])
                        if np.isfinite(residual[index])
                        else None,
                        "surface_distance_m": float(surface["distance_m"][i]),
                        "geom_id": gid,
                        "geom_name": self.predictor.model.geom(gid).name,
                        "body_name": self.predictor.model.body(
                            int(self.predictor.model.geom_bodyid[gid])
                        ).name,
                        "face_id": int(surface["face_id"][i]),
                        "closest_work": surface["closest_work"][i].tolist(),
                        "barycentric": surface["barycentric"][i].tolist(),
                    }
                )
                result[index] = "surface_candidate"
        names, counts = np.unique(result, return_counts=True)
        self.diagnostics = {
            "sequence": raw.sequence,
            "stamp": raw.stamp,
            "render_minus_physics_time_s": sync.render_time - sync.physics_time,
            "status": "evaluated",
            "empirical_epsilon_m": self.profile.epsilon_m,
            "profile": self.profile.profile,
            "render_frame": sync.render_frame,
            "support": asdict(support),
            "labels": dict(zip(names.tolist(), counts.tolist(), strict=True)),
            "candidate_pixels_rc": pixels[result == "self_candidate"].tolist(),
            "surface_candidates": witnesses,
            "surface_broadphase_point_geom_pairs": surface_pairs,
            "candidate_closer_within_band": int(
                np.sum((result == "self_candidate") & (residual < 0))
            ),
            "encoder_limit_deviations": self.predictor.encoder_limit_deviations,
        }
        return result


def instance_supported_human(points, weak_human, observation):
    """Conservative protection using measured tracks/patches, never CAD actor IDs.

    Association is supporting evidence, not a certified semantic classifier.
    Weak skin/red flags alone are intentionally not called confirmed human.
    """
    protected = np.zeros(len(points), bool)
    for person in observation.people:
        near_track = np.linalg.norm(points[:, :2] - np.asarray(person.position)[:2], axis=1) < 0.5
        protected |= weak_human & near_track
        if person.forearm is not None:
            protected |= np.linalg.norm(points - person.forearm.center, axis=1) < 0.15
    return protected
