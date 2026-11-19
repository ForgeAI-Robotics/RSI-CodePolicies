"""Exact-state pose caching and lossless native contact persistence."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


class RigidPoseSnapshot:
    """One PhysX tensor read per physics epoch, retaining native row identity."""

    def __init__(self, view, expected_paths: list[str]):
        self.view = view
        # This is the documented Tensor RigidBodyView.prim_paths property in
        # Isaac 4.5, not the USD regex traversal order of the wrapper.
        actual_paths = list(view._physics_view.prim_paths)
        if len(actual_paths) != len(expected_paths) or set(actual_paths) != set(expected_paths):
            raise RuntimeError("Native pose view does not contain the exact expected bodies")
        self.index = {path: i for i, path in enumerate(actual_paths)}
        self.epoch = None
        self.positions = None
        self.orientations = None

    def read(self, epoch: int, paths: list[str]) -> tuple[np.ndarray, np.ndarray]:
        if not self.view.is_physics_handle_valid():
            raise RuntimeError("Native pose view is invalid; USD fallback is forbidden")
        if self.epoch != epoch:
            positions, orientations = self.view.get_world_poses(clone=True)
            self.positions = np.asarray(positions)
            self.orientations = np.asarray(orientations)
            self.epoch = epoch
        indices = np.asarray([self.index[path] for path in paths], dtype=np.int64)
        # Advanced indexing gives owned arrays; consumers cannot mutate the
        # cached state or retain an alias into a mutable PhysX tensor buffer.
        return self.positions[indices], self.orientations[indices]


class LosslessContactWriter:
    """Write every original record with bounded RAM and two independently useful hashes."""

    def __init__(self, path: Path, flush_every_records: int = 1000):
        if flush_every_records < 1:
            raise ValueError("Contact flush interval must be positive")
        self.path = path
        self._raw = path.open("xb", buffering=1024 * 1024)
        self._gzip = gzip.GzipFile(
            filename="", fileobj=self._raw, mode="wb", compresslevel=1, mtime=0
        )
        self._content_hash = hashlib.sha256()
        self.records = 0
        self.contacts = 0
        self.uncompressed_bytes = 0
        self.flush_every_records = flush_every_records
        self._manifest = None

    def append(self, record: dict) -> None:
        if self._manifest is not None:
            raise RuntimeError("Contact writer is closed")
        # Match the original JSONL encoder exactly; neither float precision,
        # contact order nor timestep sampling is changed.
        payload = (json.dumps(record) + "\n").encode("utf-8")
        self._gzip.write(payload)
        self._content_hash.update(payload)
        self.records += 1
        self.contacts += len(record["contacts"])
        self.uncompressed_bytes += len(payload)
        if self.records % self.flush_every_records == 0:
            self._gzip.flush()

    def close(self) -> dict:
        if self._manifest is not None:
            return dict(self._manifest)
        self._gzip.close()
        self._raw.close()
        file_hash = hashlib.sha256()
        with self.path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                file_hash.update(chunk)
        self._manifest = {
            "schema_version": 1,
            "file": self.path.name,
            "encoding": "gzip JSONL; original contact records without downsampling",
            "lossless": True,
            "physics_step_downsampling": False,
            "record_policy": "every original contactful physics step; all contacts retained",
            "compression_level": 1,
            "records": self.records,
            "contacts": self.contacts,
            "uncompressed_bytes": self.uncompressed_bytes,
            "compressed_bytes": self.path.stat().st_size,
            "sha256": file_hash.hexdigest(),
            "uncompressed_sha256": self._content_hash.hexdigest(),
        }
        return dict(self._manifest)
