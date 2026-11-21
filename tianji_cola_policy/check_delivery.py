"""Read-only integrity/evidence check; standard library, no robot connection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def check() -> dict:
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Hash mismatch: {name}")
    expected_hash = manifest["policy_sha256"]
    if hashlib.sha256((ROOT / "tianji_cola/policy.py").read_bytes()).hexdigest() != expected_hash:
        raise ValueError("Frozen policy mismatch")
    rows = json.loads((ROOT / "evidence/full_matrix.json").read_text())
    expected_cases = {
        (s, c, r) for s in (101, 102, 103) for c in ("pick_goal", "combined_goal") for r in (0, 1)
    }
    actual_cases = {(r["seed"], r["scenario"], r["repeat"]) for r in rows}
    if len(rows) != 12 or actual_cases != expected_cases:
        raise ValueError("Incomplete native regression matrix")
    for row in rows:
        if (
            row["policy_sha256"] != expected_hash
            or row["environment_success"] is not True
            or row["llm_calls"] != 0
            or row["vla_calls"] != 0
            or row["from_reset"] is not True
        ):
            raise ValueError("Native regression does not certify frozen policy")
    return {
        "integrity": "passed",
        "native_simulation_successes": "12/12",
        "real_robot_tested": False,
        "files_checked": len(manifest["sha256"]),
    }


if __name__ == "__main__":
    print(json.dumps(check(), ensure_ascii=False, indent=2))
