"""Versioned offline proposals and single-maintainer, evidence-bound promotion.

This module makes no model calls and never executes proposed Python. Model-assisted
exploration uses ``agent.py`` on train scenarios; deployment imports neither module.
The audit checks recorded provenance, not authenticity against a malicious writer.
Keep native worker outputs outside the proposal agents' writable workspaces.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .scenarios import ROOT, canonical_json, digest, load_protocol, scenario

SKILL_PATH = ROOT / "airport/workflows/airport-checkpoint-workflow/SKILL.md"
TRAINABLE_PARAMETERS = {
    "dwell_s": (3.0, 5.0),
    "stage_timeout_s": (15.0, 45.0),
    "standoff_m": (0.03, 0.06),
    "extraction_distance_m": (0.09, 0.13),
    "extraction_clearance_m": (0.005, 0.02),
    "contact_advance_m": (0.0005, 0.002),
}


def validate_parameters(parameters: dict[str, Any]) -> None:
    """Bound ordinary exploration; safety/acceptance thresholds require code review."""
    for name, value in parameters.items():
        if name == "vision_profile":
            if not isinstance(value, str) or value not in {"legacy", "supply_bounds_v1"}:
                raise ValueError("Unknown reviewed vision profile")
            continue
        if name == "collision_profile":
            if not isinstance(value, str) or value not in {"legacy", "oriented_v1"}:
                raise ValueError("Unknown reviewed collision profile")
            continue
        if name == "control_profile":
            # Named, reviewed implementations only. Numeric safety limits still
            # remain outside the trainable parameter surface below.
            if not isinstance(value, str) or value not in {"legacy", "reactive_v1"}:
                raise ValueError("Unknown reviewed control profile")
            continue
        if name not in TRAINABLE_PARAMETERS:
            raise ValueError(f"Parameter is not a trainable control parameter: {name}")
        lower, upper = TRAINABLE_PARAMETERS[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"Parameter must be finite numeric: {name}")
        if not lower <= value <= upper:
            raise ValueError(f"Parameter {name} is outside fixed bounds [{lower}, {upper}]")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new_json(path: Path, value: Any) -> None:
    with path.open("xb") as stream:
        stream.write(canonical_json(value))


def _identifier(value: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", value):
        raise ValueError("Identifier must contain only letters, digits, hyphens or underscores")
    return value


def audit_run(run: Path, candidate: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    """Evaluate host binding plus worker-native evidence, never agent finish text.

    Missing/partial outputs remain recorded failures. An eligible result requires
    explicit live native full-task evaluation, advancing physics and no violations.
    Hashes bind evidence to artifacts; they do not certify a trusted simulator.
    """
    errors: list[str] = []
    try:
        binding = json.loads((run / "checkpoint_binding.json").read_text())
        native = json.loads((run / "checkpoint_native.json").read_text())
        if not isinstance(binding, dict) or not isinstance(native, dict):
            raise ValueError("Binding and native result must be JSON objects")
        case = scenario(binding["seed"], protocol)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"eligible": False, "native_success": False, "errors": [str(exc)]}
    expected = {
        "candidate_id": candidate["candidate_id"],
        "protocol_sha256": digest(protocol),
        "scenario_id": case.scenario_id,
        "scenario_sha256": case.sha256,
        "split": case.split,
        "policy_sha256": candidate["policy_sha256"],
    }
    for key, value in expected.items():
        if binding.get(key) != value:
            errors.append(f"binding mismatch: {key}")
    for key in ("run_id", "scenario_id", "seed", "protocol_sha256", "policy_sha256"):
        if native.get(key) != binding.get(key):
            errors.append(f"native binding mismatch: {key}")
    if native.get("engine") != "isaac_physx":
        errors.append("not an Isaac/PhysX native run")
    if native.get("execution_mode") != "native_live" or native.get("replay") is not False:
        errors.append("replay, construction smoke, or unknown execution mode")
    if native.get("success_source") != "native_environment":
        errors.append("native full-task predicate was not evaluated")
    if native.get("evaluation_scope") != "full_batch":
        errors.append("not a full-batch evaluation")
    if native.get("native_success") is not True:
        errors.append("native success is false or missing")
    if native.get("violations") != []:
        errors.append("safety violations are present or unreported")
    if binding.get("primitives") != "no-vla" or native.get("vla_calls") != 0:
        errors.append("no-vla execution is not established")
    if native.get("base_locked") is not True:
        errors.append("base lock is unverified")
    if type(native.get("steps")) is not int or native["steps"] <= 0:
        errors.append("physics did not advance")
    elif native["steps"] > protocol["max_physics_steps"]:
        errors.append("physics budget exceeded")
    if not isinstance(native.get("wall_s"), (int, float)) or not (
        0 < native["wall_s"] <= protocol["wall_budget_s"]
    ):
        errors.append("wall budget missing or exceeded")
    evidence = native.get("evidence_sha256", {})
    if not isinstance(evidence, dict):
        evidence = {}
    required = {"observations.jsonl", "actions.jsonl", "native_events.jsonl", "scene_manifest.json"}
    if not required.issubset(evidence):
        errors.append("missing native evidence artifact hashes")
    for name, expected_hash in evidence.items():
        path = run / name
        if (
            Path(name).is_absolute()
            or ".." in Path(name).parts
            or path.is_symlink()
            or not path.resolve().is_relative_to(run.resolve())
        ):
            errors.append(f"unsafe evidence path: {name}")
        elif not path.is_file() or file_hash(path) != expected_hash:
            errors.append(f"evidence hash mismatch: {name}")
    # Require a native event ledger with advancing physics and a native terminal
    # verdict; a policy state named 'complete' never supplies this evidence.
    try:
        events = [
            json.loads(line) for line in (run / "native_events.jsonl").read_text().splitlines()
        ]
        steps = [event["physics_step"] for event in events]
        if any(type(step) is not int for step in steps):
            raise ValueError("Physics step counters must be integers")
        if len(steps) < 2 or steps != sorted(steps) or steps[-1] <= steps[0]:
            errors.append("native event physics did not advance monotonically")
        if not events or events[-1].get("native_success") is not True:
            errors.append("no native terminal success event")
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        errors.append("invalid native event ledger")
    return {
        "eligible": not errors,
        "native_success": native.get("native_success") is True,
        "errors": errors,
        "run_id": binding.get("run_id"),
        "seed": case.seed,
        "scenario_id": case.scenario_id,
        "split": case.split,
        "binding_sha256": file_hash(run / "checkpoint_binding.json"),
        "native_sha256": file_hash(run / "checkpoint_native.json"),
        "run_path": str(run.resolve()),
    }


class CandidateStore:
    """Append-only proposals; only the named maintainer updates CURRENT under flock.

    The maintainer name is an accidental-write guard, not OS authentication. Use
    separate filesystem permissions when independent writers are untrusted.
    """

    def __init__(self, root: Path, maintainer: str = "checkpoint-maintainer"):
        self.root = root
        self.maintainer = _identifier(maintainer)
        root.mkdir(parents=True, exist_ok=True)
        for name in ("candidates", "promotions", "heldout"):
            (root / name).mkdir(exist_ok=True)
        with self._lock():
            config_path = root / "store.json"
            if config_path.exists():
                if json.loads(config_path.read_text())["maintainer"] != self.maintainer:
                    raise PermissionError("Store has a different designated maintainer")
            else:
                write_new_json(config_path, {"maintainer": self.maintainer, "schema_version": 1})

    @contextmanager
    def _lock(self) -> Iterator[None]:
        with (self.root / ".maintainer.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    @property
    def current(self) -> str | None:
        path = self.root / "CURRENT"
        return path.read_text().strip() if path.exists() else None

    def load(self, candidate_id: str) -> dict[str, Any]:
        directory = self.root / "candidates" / _identifier(candidate_id)
        data = json.loads((directory / "candidate.json").read_text())
        candidate_body = {k: v for k, v in data.items() if k != "candidate_id"}
        if digest(candidate_body) != candidate_id:
            raise ValueError("Candidate manifest was modified")
        for name, sha256 in data["files"].items():
            if file_hash(directory / "snapshot" / name) != sha256:
                raise ValueError(f"Candidate source changed: {name}")
        return data

    def propose(
        self,
        *,
        author: str,
        parameters: dict[str, Any],
        rationale: str,
        sources: dict[str, Path],
        base_revision: str | None = None,
        evidence_runs: list[Path] | None = None,
    ) -> dict[str, Any]:
        """Snapshot reviewed code/skill and bounded parameters without executing code.

        Evidence may contain failed train attempts; heldout attempts are forbidden.
        A proposal cannot modify active source, shared skill, protocol or safety limits.
        """
        protocol = load_protocol()
        validate_parameters(parameters)
        evidence = []
        for run in evidence_runs or []:
            binding = json.loads((run / "checkpoint_binding.json").read_text())
            if scenario(binding["seed"], protocol).split != "train":
                raise ValueError("Heldout evidence cannot inform proposals")
            evidence.append({"run": str(run.resolve()), "binding_sha256": digest(binding)})
        for name, source in sources.items():
            path = Path(name)
            if path.is_absolute() or ".." in path.parts or source.is_symlink():
                raise ValueError("Sources must be regular files with relative snapshot paths")
        if "SKILL.md" not in sources or not any(name.endswith(".py") for name in sources):
            raise ValueError("Every candidate must snapshot code and SKILL.md")
        with self._lock():
            if base_revision != self.current:
                raise ValueError("Stale base revision; rebase explicitly before proposing")
            files = {name: file_hash(path) for name, path in sorted(sources.items())}
            body = {
                "schema_version": 1,
                "author": _identifier(author),
                "base_revision": base_revision,
                "parameters": parameters,
                "rationale": rationale,
                "files": files,
                "policy_sha256": digest({"files": files, "parameters": parameters}),
                "protocol_sha256": digest(protocol),
                "train_evidence": evidence,
                "training_model": "gpt-6-astra",
                "reasoning_effort": "high",
                "training_claim": "proposal_only; native evidence is recorded separately",
            }
            ident = digest(body)
            directory = self.root / "candidates" / ident
            directory.mkdir(exist_ok=False)
            for name, source in sources.items():
                target = directory / "snapshot" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            body["candidate_id"] = ident
            write_new_json(directory / "candidate.json", body)
            return self.load(ident)

    def promote(self, candidate_id: str, runs: list[Path], *, actor: str) -> dict[str, Any]:
        if actor != self.maintainer:
            raise PermissionError("Only the designated maintainer can promote")
        protocol = load_protocol()
        with self._lock():
            candidate = self.load(candidate_id)
            if candidate["base_revision"] != self.current:
                raise ValueError("Concurrent promotion changed CURRENT; rebase and retest")
            if candidate["protocol_sha256"] != digest(protocol):
                raise ValueError("Protocol changed; start a separate versioned study")
            audits = [audit_run(run, candidate, protocol) for run in runs]
            if any(a.get("split") != "train" or not a["eligible"] for a in audits):
                raise ValueError("Promotion requires eligible native train evidence only")
            if {a["seed"] for a in audits} != set(protocol["train_seeds"]):
                raise ValueError("All frozen training scenarios are required")
            if len({a["run_id"] for a in audits}) != len(audits):
                raise ValueError("Repeated native run identity")
            consumed = set()
            for path in (self.root / "promotions").glob("*.json"):
                consumed.update(a["run_id"] for a in json.loads(path.read_text())["audits"])
            if consumed & {a["run_id"] for a in audits}:
                raise ValueError("Native evidence was already used by an earlier promotion")
            record = {"candidate_id": candidate_id, "previous": self.current, "audits": audits}
            write_new_json(self.root / "promotions" / f"{candidate_id}.json", record)
            temporary = self.root / f".CURRENT-{uuid.uuid4().hex}"
            temporary.write_text(candidate_id + "\n")
            os.replace(temporary, self.root / "CURRENT")
            return record

    def seal_heldout(self, candidate_id: str, *, actor: str) -> Path:
        """Allocate the one heldout evaluation set before looking at its outcomes.

        Failed cases stay in that set. Do not tune/retry a candidate against the same
        heldout set; a later study must preregister new seeds under a new protocol.
        """
        if actor != self.maintainer:
            raise PermissionError("Only the designated maintainer can seal evaluation")
        with self._lock():
            self.load(candidate_id)
            if self.current != candidate_id:
                raise ValueError("Heldout evaluation requires the promoted frozen candidate")
            path = self.root / "heldout" / f"{digest(load_protocol())}.json"
            write_new_json(path, {"candidate_id": candidate_id, "protocol": load_protocol()})
            return path


def bind_run(
    output: Path, candidate: dict[str, Any], seed: int, *, allow_heldout: bool = False
) -> dict:
    """Create a fresh host-owned episode directory before starting any worker."""
    protocol = load_protocol()
    case = scenario(seed, protocol)
    if case.split != "train" and not allow_heldout:
        raise ValueError("Model-assisted exploration is restricted to train scenarios")
    if candidate["protocol_sha256"] != digest(protocol):
        raise ValueError("Candidate uses a different fixed protocol")
    output.mkdir(parents=True, exist_ok=False)
    binding = {
        "schema_version": 1,
        "run_id": uuid.uuid4().hex,
        "seed": seed,
        "split": case.split,
        "scenario_id": case.scenario_id,
        "scenario_sha256": case.sha256,
        "candidate_id": candidate["candidate_id"],
        "policy_sha256": candidate["policy_sha256"],
        "protocol_sha256": digest(protocol),
        "primitives": "no-vla",
    }
    write_new_json(output / "checkpoint_binding.json", binding)
    write_new_json(output / "scenario.json", case.to_dict())
    write_new_json(output / "protocol.json", protocol)
    return binding


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--store", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    propose = sub.add_parser("propose")
    propose.add_argument("--author", required=True)
    propose.add_argument("--parameters", type=Path)
    propose.add_argument("--rationale", required=True)
    propose.add_argument("--base")
    propose.add_argument("--evidence", type=Path, action="append", default=[])
    audit = sub.add_parser("audit")
    audit.add_argument("--candidate", required=True)
    audit.add_argument("--run", type=Path, required=True)
    promote = sub.add_parser("promote")
    promote.add_argument("--candidate", required=True)
    promote.add_argument("--actor", required=True)
    promote.add_argument("--run", type=Path, action="append", required=True)
    heldout = sub.add_parser("seal-heldout")
    heldout.add_argument("--candidate", required=True)
    heldout.add_argument("--actor", required=True)
    args = parser.parse_args()
    store = CandidateStore(args.store)
    if args.command == "propose":
        sources = {p.name: p for p in (ROOT / "airport/checkpoint").glob("*.py")}
        sources["SKILL.md"] = SKILL_PATH
        parameters = json.loads(args.parameters.read_text()) if args.parameters else {}
        result = store.propose(
            author=args.author,
            parameters=parameters,
            rationale=args.rationale,
            sources=sources,
            base_revision=args.base,
            evidence_runs=args.evidence,
        )
    elif args.command == "audit":
        result = audit_run(args.run, store.load(args.candidate), load_protocol())
    elif args.command == "promote":
        result = store.promote(args.candidate, args.run, actor=args.actor)
    else:
        result = {"sealed": str(store.seal_heldout(args.candidate, actor=args.actor))}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
