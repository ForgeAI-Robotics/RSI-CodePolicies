"""Opt-in pure geometry proposals; CPU20um acceptance, no model/controller changes."""

import importlib
import sys
import time
from dataclasses import asdict

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter

from airport.checkpoint.self_depth import (
    FrameSupport,
    OwnOpaqueDepth,
    candidate_labels,
    registration_failure,
)


def _load_warp():
    sys.path.insert(0, "/isaac-sim/extscache/omni.warp.core-1.5.0+lx64")
    module = importlib.import_module("warp")
    if module.__version__ != "1.5.0":
        raise RuntimeError("GPU geometry requires the pinned Isaac Warp 1.5.0 bundle")
    return module


wp = _load_warp()


@wp.kernel
def propose(
    mesh: wp.uint64,
    origin: wp.vec3,
    rays: wp.array(dtype=wp.vec3),
    points: wp.array(dtype=wp.vec3),
    out_t: wp.array(dtype=float),
    out_distance: wp.array(dtype=float),
):
    i = wp.tid()
    r = wp.mesh_query_ray(mesh, origin, rays[i], 10.0)
    out_t[i] = -1.0
    if r.result:
        out_t[i] = r.t
    q = wp.mesh_query_point_no_sign(mesh, points[i], 0.0001)
    out_distance[i] = -1.0
    if q.result:
        out_distance[i] = wp.length(wp.mesh_eval_position(mesh, q.face, q.u, q.v) - points[i])


class VerifiedOwnGeometry:
    def __init__(self, description, profile, device="cuda:0"):
        start = time.perf_counter()
        wp.init()
        if not wp.get_device(device).is_cuda:
            raise RuntimeError("Explicit CUDA device required; no pretend GPU fallback")
        self.cpu = OwnOpaqueDepth(description)
        self.profile = profile
        self.device = device
        self.last_render_frame = -1
        self.diagnostics = {}
        self.local = []
        faces = []
        start_index = 0
        for gid in self.cpu.mesh_ids:
            m = self.cpu.model
            mid = m.geom_dataid[gid]
            va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
            self.local.append(m.mesh_vert[va : va + vn].astype(float))
            faces.append(m.mesh_face[fa : fa + fn] + start_index)
            start_index += vn
        self.vertices = wp.zeros(int(start_index), dtype=wp.vec3, device=device)
        self.indices = wp.array(
            np.concatenate(faces).astype(np.int32).ravel(), dtype=wp.int32, device=device
        )
        self.mesh = None
        wp.synchronize()
        self.startup_s = time.perf_counter() - start

    def labels(self, raw, pixels, paper, supported_human):
        wp.synchronize()
        start = time.perf_counter()
        result = np.full(len(pixels), "unknown", dtype="<U40")
        result[paper] = "protected_paper"
        result[supported_human] = "protected_human"
        free = ~paper & ~supported_human
        reason = registration_failure(raw, self.profile)
        sync = raw.rgbd.synchrony if raw.rgbd else None
        if reason is None and sync.render_frame <= self.last_render_frame:
            reason = "nonadvancing_render_frame"
        if reason:
            result[free] = reason
            self.diagnostics = {
                "gpu_enabled": True,
                "gpu_queries_executed": False,
                "status": reason,
                "sequence": raw.sequence,
            }
            return result
        self.last_render_frame = sync.render_frame
        t = time.perf_counter()
        self.cpu.set_encoders(raw.joint_positions)
        fk = time.perf_counter() - t
        c = raw.rgbd.calibration
        T = np.array(c.optical_to_work)
        rows, cols = pixels.T
        dx, dy = c.pixel_offset_xy
        measured = raw.rgbd.depth_m[rows, cols].astype(float)
        optical = np.column_stack(
            [(cols + dx - c.cx) / c.fx, (rows + dy - c.cy) / c.fy, np.ones(len(rows))]
        )
        norms = np.linalg.norm(optical, axis=1)
        points = (optical * measured[:, None]) @ T[:3, :3].T + T[:3, 3]
        rays = (optical / norms[:, None]) @ T[:3, :3].T
        t = time.perf_counter()
        world = np.concatenate(
            [
                v @ self.cpu.data.geom_xmat[g].reshape(3, 3).T + self.cpu.data.geom_xpos[g]
                for v, g in zip(self.local, self.cpu.mesh_ids, strict=True)
            ]
        ).astype(np.float32)
        transform = time.perf_counter() - t
        t = time.perf_counter()
        self.vertices.assign(world)
        rp = wp.array(rays.astype(np.float32), dtype=wp.vec3, device=self.device)
        pp = wp.array(points.astype(np.float32), dtype=wp.vec3, device=self.device)
        out_t = wp.empty(len(pixels), dtype=float, device=self.device)
        out_d = wp.empty(len(pixels), dtype=float, device=self.device)
        wp.synchronize()
        upload = time.perf_counter() - t
        t = time.perf_counter()
        if self.mesh is None:
            self.mesh = wp.Mesh(points=self.vertices, indices=self.indices)
        else:
            self.mesh.refit()
        wp.synchronize()
        refit = time.perf_counter() - t
        t = time.perf_counter()
        wp.launch(
            propose,
            len(pixels),
            inputs=[self.mesh.id, wp.vec3(*T[:3, 3].astype(np.float32)), rp, pp, out_t, out_d],
            device=self.device,
        )
        wp.synchronize()
        query = time.perf_counter() - t
        t = time.perf_counter()
        rt = out_t.numpy()
        sd = out_d.numpy()
        wp.synchronize()
        download = time.perf_counter() - t
        proposal = ((rt >= 0) & (abs(rt / norms - measured) <= 100e-6)) | (
            (sd >= 0) & (sd <= 100e-6)
        )
        # Original health reference rays remain exact CPU queries, independently of GPU recall.
        reference = (rows % 8 == 0) & (cols % 8 == 0)
        check = proposal | reference
        predicted = np.full(len(pixels), np.nan)
        t = time.perf_counter()
        predicted[check], _ = self.cpu.predict(pixels[check], c)
        cpu_ray = time.perf_counter() - t
        residual = measured - predicted
        span = maximum_filter(raw.rgbd.depth_m, 5) - minimum_filter(raw.rgbd.depth_m, 5)
        interior = (
            np.isfinite(predicted)
            & (measured > 0.1)
            & (measured < 2)
            & (span[rows, cols] < 0.01)
            & (abs(residual) < 0.02)
            & reference
            & free
        )
        errors = abs(residual[interior])
        support = FrameSupport(
            sync.calibration_sha256,
            sync.evidence_sha256,
            "measured",
            len(errors),
            float(np.median(errors)) if len(errors) else float("inf"),
            float(np.quantile(errors, 0.95)) if len(errors) else float("inf"),
        )
        result = candidate_labels(
            measured,
            predicted,
            epsilon_m=self.profile.epsilon_m,
            support=support,
            graspable=paper,
            confirmed_human=supported_human,
        )
        check_surface = proposal & np.isin(result, ["mesh_miss", "foreground", "depth_mismatch"])
        t = time.perf_counter()
        idx = np.flatnonzero(check_surface)
        if len(idx):
            witness = self.cpu.surface_support(points[idx], self.profile.epsilon_m)
            result[idx[np.isfinite(witness["distance_m"])]] = "surface_candidate"
        cpu_surface = time.perf_counter() - t
        names, counts = np.unique(result, return_counts=True)
        self.diagnostics = {
            "status": "evaluated",
            "stamp": raw.stamp,
            "render_frame": sync.render_frame,
            "labels": dict(zip(names.tolist(), counts.tolist(), strict=True)),
            "candidate_pixels_rc": pixels[result == "self_candidate"].tolist(),
            "gpu_enabled": True,
            "gpu_queries_executed": True,
            "device": self.device,
            "warp_version": wp.__version__,
            "sequence": raw.sequence,
            "proposal_radius_m": 100e-6,
            "acceptance_epsilon_m": self.profile.epsilon_m,
            "CPU_reference_rays": int(reference.sum()),
            "CPU_verified_rays": int(check.sum()),
            "CPU_surface_points": len(idx),
            "support": asdict(support),
            "timing_s": {
                "fk": fk,
                "world_transform": transform,
                "upload": upload,
                "refit": refit,
                "query": query,
                "download": download,
                "CPU_ray": cpu_ray,
                "CPU_surface": cpu_surface,
                "total": time.perf_counter() - start,
            },
        }
        return result
