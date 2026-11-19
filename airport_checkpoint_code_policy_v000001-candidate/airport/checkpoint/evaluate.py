"""Run a frozen simulation policy without Codex, model calls or an LLM service."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .deployment import run_policy
from .learning import SKILL_PATH, CandidateStore, audit_run, bind_run, file_hash, write_new_json
from .policy import PolicyConfig
from .remote_backend import make_backend
from .scenarios import ROOT, digest, load_protocol, scenario


def evaluate(store: CandidateStore, candidate_id: str, seed: int, output: Path) -> dict:
    protocol = load_protocol()
    candidate = store.load(candidate_id)
    case = scenario(seed, protocol)
    if case.split == "heldout":
        seal_path = store.root / "heldout" / f"{digest(protocol)}.json"
        seal = json.loads(seal_path.read_text())
        if seal["candidate_id"] != candidate_id or store.current != candidate_id:
            raise ValueError("Heldout evaluation requires the sealed promoted candidate")
    for name, expected in candidate["files"].items():
        source = SKILL_PATH if name == "SKILL.md" else ROOT / "airport/checkpoint" / name
        if file_hash(source) != expected:
            raise ValueError(f"Working source differs from frozen candidate: {name}")
    binding = bind_run(output, candidate, seed, allow_heldout=case.split == "heldout")
    if case.split == "heldout":
        attempts = store.root / "heldout/attempts"
        attempts.mkdir(exist_ok=True)
        # Atomic claim survives failures: reserved cases are not a debug loop.
        write_new_json(attempts / f"{digest(protocol)}-seed-{seed}.json", binding)
    write_new_json(output / "candidate.json", candidate)
    backend = make_backend(
        out_dir=output, scenario=case.to_dict(), parameters=candidate["parameters"], binding=binding
    )
    try:
        report = run_policy(
            backend,
            PolicyConfig(**candidate["parameters"]),
            max_ticks=protocol["max_policy_ticks"],
            deployment_mode="simulation",
        )
        write_new_json(output / "checkpoint_policy_report.json", asdict(report))
    finally:
        try:
            backend.stop("deterministic evaluation ended")
        finally:
            backend.close()
    result = {
        "mode": "deterministic_simulation",
        "llm_calls": 0,
        "seed": seed,
        "split": case.split,
        "candidate_id": candidate_id,
        "audit": audit_run(output, candidate, protocol),
    }
    write_new_json(output / "evaluation_result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate(CandidateStore(args.store), args.candidate, args.seed, args.out), indent=2
        )
    )


if __name__ == "__main__":
    main()
