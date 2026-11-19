"""Fixed checkpoint scenario protocol; heldout cases never become training input."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "configs/airport-checkpoint-protocol.json"


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def load_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    protocol = json.loads(path.read_text())
    if protocol["schema_version"] != 1 or protocol["primitives"] != "no-vla":
        raise ValueError("Unsupported protocol or primitive set")
    train, heldout = protocol["train_seeds"], protocol["heldout_seeds"]
    if not train or not heldout or set(train) & set(heldout):
        raise ValueError("Train and heldout seeds must be nonempty and disjoint")
    if len(set(train + heldout)) != len(train + heldout):
        raise ValueError("Repeated scenario seeds")
    if protocol["batch_size"] != 3 or protocol["base_motion_allowed"]:
        raise ValueError("This protocol fixes a three-person batch and locked base")
    return protocol


@dataclass(frozen=True)
class PassengerSpec:
    person_id: str
    queue_offset_m: float
    stop_offset_xy_m: tuple[float, float]
    yaw_rad: float
    height_m: float
    bag_side: str
    bag_offset_xyz_m: tuple[float, float, float]
    arrival_delay_s: float
    dwell_s: float


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    split: str
    seed: int
    batch_size: int
    passengers: tuple[PassengerSpec, ...]
    paper_exposed_fraction: float
    protocol_sha256: str

    def to_dict(self) -> dict[str, Any]:
        # Round-trip normalizes tuples to JSON arrays for hash equality on disk.
        return json.loads(canonical_json(asdict(self)))

    @property
    def sha256(self) -> str:
        return digest(self.to_dict())


def scenario(seed: int, protocol: dict[str, Any] | None = None) -> Scenario:
    protocol = load_protocol() if protocol is None else protocol
    if seed in protocol["train_seeds"]:
        split = "train"
    elif seed in protocol["heldout_seeds"]:
        split = "heldout"
    else:
        raise ValueError(f"Seed {seed} is outside the frozen evaluation protocol")
    rng = random.Random(seed)
    people = []
    for index in range(protocol["batch_size"]):
        people.append(
            PassengerSpec(
                person_id=f"person-{index + 1}",
                queue_offset_m=round(index * 0.85, 6),
                stop_offset_xy_m=(
                    round(rng.uniform(-0.06, 0.06), 6),
                    round(rng.uniform(-0.08, 0.08), 6),
                ),
                yaw_rad=round(rng.uniform(-0.25, 0.25), 6),
                height_m=round(rng.uniform(1.55, 1.88), 6),
                bag_side=rng.choice(["left", "right"]),
                bag_offset_xyz_m=(
                    round(rng.uniform(-0.05, 0.05), 6),
                    round(rng.uniform(-0.05, 0.05), 6),
                    round(rng.uniform(-0.08, 0.08), 6),
                ),
                arrival_delay_s=round(index * 2.0 + rng.uniform(0, 0.5), 6),
                dwell_s=protocol["passenger_dwell_s"],
            )
        )
    return Scenario(
        scenario_id=f"checkpoint-v1-{split}-{seed}",
        split=split,
        seed=seed,
        batch_size=protocol["batch_size"],
        passengers=tuple(people),
        paper_exposed_fraction=0.5,
        protocol_sha256=digest(protocol),
    )


def training_scenarios(protocol: dict[str, Any] | None = None) -> tuple[Scenario, ...]:
    protocol = load_protocol() if protocol is None else protocol
    return tuple(scenario(seed, protocol) for seed in protocol["train_seeds"])
