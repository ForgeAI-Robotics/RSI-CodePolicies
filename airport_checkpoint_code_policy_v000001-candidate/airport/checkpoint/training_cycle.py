"""Three isolated Astra roles review live train evidence and stage a new candidate.

No role promotes itself or operates the robot. The reviewed candidate still needs
fresh native training runs before the maintainer can update CURRENT.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from .learning import (
    TRAINABLE_PARAMETERS,
    CandidateStore,
    file_hash,
    validate_parameters,
    write_new_json,
)
from .scenarios import ROOT, digest, load_protocol, scenario

ROLES = ("proposer", "reviewer", "maintainer")
NATIVE_AUDIT_FIELDS = frozenset(
    {
        "engine",
        "execution_mode",
        "replay",
        "steps",
        "stamp",
        "native_success",
        "success_source",
        "run_id",
        "scenario_id",
        "seed",
        "scene_sha256",
        "policy_sha256",
        "protocol_sha256",
        "candidate_id",
        "primitives",
        "scenario_sha256",
        "schema_version",
        "split",
        "evaluation_scope",
        "base_locked",
        "vla_calls",
        "wall_s",
    }
)
NATIVE_VIOLATION_KINDS = frozenset(
    {
        "gravity_feedforward",
        "kinematic_actor_tracking_error",
        "fixed_base_drift",
        "robot_passenger_contact",
        "paper_protected_body_contact",
        "sampling_person_not_stationary",
        "paper_force_limit",
        "sampling_not_lateral",
    }
)


def native_audit_for_review(native: dict) -> dict:
    """Expose outcome metadata only; simulator geometry is not training observation."""
    result = {
        key: value
        for key, value in native.items()
        if key in NATIVE_AUDIT_FIELDS and isinstance(value, (str, int, float, bool, type(None)))
    }
    counts: dict[str, int] = {}
    for violation in native.get("violations", []):
        kind = violation.get("kind") if isinstance(violation, dict) else None
        kind = kind if isinstance(kind, str) and kind in NATIVE_VIOLATION_KINDS else "unknown"
        counts[kind] = counts.get(kind, 0) + 1
    result["violation_counts"] = counts
    return result


SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["propose", "accept", "reject", "needs_measurement"],
        },
        "rationale": {"type": "string"},
        "parameters": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string", "enum": list(TRAINABLE_PARAMETERS)},
                    "value": {"type": "number"},
                },
                "required": ["name", "value"],
            },
        },
        "skill_addendum": {"type": "string"},
    },
    "required": ["decision", "rationale", "parameters", "skill_addendum"],
}


def collect_train_evidence(runs: list[Path], candidate: dict) -> list[dict]:
    """Accept actual advancing native attempts, including failed task outcomes."""
    protocol = load_protocol()
    evidence = []
    identities = set()
    for run in runs:
        binding = json.loads((run / "checkpoint_binding.json").read_text())
        native = json.loads((run / "checkpoint_native.json").read_text())
        if scenario(binding["seed"]).split != "train":
            raise ValueError("Heldout outputs cannot enter an Astra training cycle")
        if binding["candidate_id"] != candidate["candidate_id"]:
            raise ValueError("Evidence belongs to a different candidate")
        if binding["protocol_sha256"] != digest(protocol):
            raise ValueError("Evidence protocol differs from the fixed study")
        if native.get("engine") != "isaac_physx" or native.get("execution_mode") != "native_live":
            raise ValueError("Training cycle requires a live native attempt, not smoke/replay")
        if native.get("replay") is not False or native.get("steps", 0) <= 0:
            raise ValueError("Native physics did not advance in a fresh live run")
        if native.get("run_id") != binding["run_id"] or binding["run_id"] in identities:
            raise ValueError("Native run identity mismatch or repeated evidence")
        identities.add(binding["run_id"])
        record: dict[str, Any] = {
            "binding": binding,
            "native": native_audit_for_review(native),
            "native_sha256": file_hash(run / "checkpoint_native.json"),
            "native_input_scope": "Outcome/audit allowlist; no privileged geometry or raw events",
        }
        for name in ("checkpoint_policy_report.json", "result.json"):
            path = run / name
            if path.exists():
                record[name] = json.loads(path.read_text())
        proposals = run / "parameter_proposals"
        record["parameter_proposals"] = [
            json.loads(p.read_text()) for p in sorted(proposals.glob("*.json"))
        ]
        evidence.append(record)
    if not evidence:
        raise ValueError("At least one recorded native training attempt is required")
    return evidence


def run_role(
    role: str,
    workspace: Path,
    *,
    codex_home: Path,
    codex_bin: str = "codex",
    budget_s: int = 600,
    editable_files: tuple[str, ...] = ("policy.py",),
) -> dict:
    """Reuse PyRUA's Codex catalog/environment; retain stdout, stderr and usage events."""
    from pyrualean.generate import codex_catalog_flags
    from pyrualean.play import _agent_environment

    write_new_json(workspace / "response_schema.json", SCHEMA)
    instructions = (
        f"You are the {role} Astra in a three-role checkpoint training review. "
        "Read context.json and proposed/SKILL.md. This is simulation-only PyRUA-Lean "
        "Code as Policy, no VLA, no base motion and no online LLM in deployment. "
        "Only live train-case evidence is provided. Do not read outside this workspace, "
        "inspect heldout runs, reset episodes, start simulator/GPU/network services, "
        "or treat agent completion as native success. Never add privileged scene poses "
        "to the policy or weaken guards/native acceptance thresholds. "
        "Use only existing local files; no installation or model download. "
        "Do not claim improvements were tested; a new candidate requires fresh native runs. "
        "Return JSON matching response_schema.json as your final answer. "
    )
    if role == "proposer":
        instructions += (
            "Diagnose the failed/successful train evidence and propose one narrowly justified "
            "bounded parameter change and/or algorithm correction. You may edit only "
            + ", ".join("proposed/" + name for name in editable_files)
            + ". Do not edit any other file. Put proposed parameter overrides "
            "in parameters, evidence-grounded skill notes in skill_addendum; use "
            "needs_measurement when evidence cannot support a change."
        )
    elif role == "reviewer":
        instructions += (
            "Review the proposer output and original/proposed policy difference independently. "
            "Do not edit files. Check sensor-only feedback, limits, batch order and causal "
            "support from native evidence. Return accept only for a change worth testing; "
            "otherwise reject or needs_measurement. Acceptance is not proven improvement."
        )
    else:
        instructions += (
            "Act as the sole candidate maintainer. Do not edit files. Reconcile proposal and "
            "independent review. Return accept only if reviewer accepted and you agree to "
            "stage that exact proposed code for new native tests. Select final bounded "
            "parameters and a short factual skill_addendum. Never promote CURRENT here."
        )
    (workspace / "prompt.txt").write_text(instructions)
    argv = [
        codex_bin,
        "exec",
        "-m",
        "gpt-6-astra",
        "-c",
        'model_reasoning_effort="high"',
        "-c",
        "project_doc_max_bytes=0",
        *codex_catalog_flags("gpt-6-astra"),
        "--disable",
        "memories",
        "--disable",
        "shell_snapshot",
        "-s",
        "workspace-write",
        "-C",
        str(workspace.resolve()),
        "--skip-git-repo-check",
        "--json",
        "--output-schema",
        str((workspace / "response_schema.json").resolve()),
        "-o",
        str((workspace / "answer.json").resolve()),
        "-",
    ]
    write_new_json(
        workspace / "invocation.json",
        {
            "role": role,
            "model": "gpt-6-astra",
            "reasoning_effort": "high",
            "budget_s": budget_s,
            "argv": argv,
            "prompt_sha256": file_hash(workspace / "prompt.txt"),
        },
    )
    started = time.monotonic()
    timed_out = False
    with (
        (workspace / "events.jsonl").open("wb") as events,
        (workspace / "stderr.log").open("wb") as errors,
    ):
        process = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=events,
            stderr=errors,
            start_new_session=True,
            env=_agent_environment(codex_home=str(codex_home.resolve()), api_key_env=None),
        )
        try:
            process.communicate(instructions.encode(), timeout=budget_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    status = {
        "role": role,
        "returncode": process.returncode,
        "timed_out": timed_out,
        "wall_s": time.monotonic() - started,
    }
    write_new_json(workspace / "status.json", status)
    if process.returncode != 0 or timed_out:
        raise RuntimeError(f"{role} runtime failed; inspect {workspace / 'stderr.log'}")
    answer = json.loads((workspace / "answer.json").read_text())
    if answer.get("decision") not in {"propose", "accept", "reject", "needs_measurement"}:
        raise ValueError("Invalid role decision")
    values = answer["parameters"]
    if len({entry["name"] for entry in values}) != len(values):
        raise ValueError("Duplicate parameter override")
    answer["parameters"] = {entry["name"]: entry["value"] for entry in values}
    validate_parameters(answer["parameters"])
    return answer


def run_cycle(
    store: CandidateStore,
    candidate_id: str,
    runs: list[Path],
    output: Path,
    *,
    codex_home: Path,
    codex_bin: str = "codex",
    budget_s: int = 600,
    editable_files: tuple[str, ...] = ("policy.py",),
) -> dict:
    if not editable_files or set(editable_files) - {"policy.py", "vision.py"}:
        raise ValueError("Review authority is restricted to policy.py and vision.py")
    candidate = store.load(candidate_id)
    evidence = collect_train_evidence(runs, candidate)
    output.mkdir(parents=True, exist_ok=False)
    snapshot = store.root / "candidates" / candidate_id / "snapshot"
    context: dict[str, Any] = {
        "candidate": candidate,
        "evidence": evidence,
        "roles": {},
        "parameter_bounds": TRAINABLE_PARAMETERS,
        "scope": "native train evidence review; new changes remain untested",
        "review_authority": {
            "proposer_editable_files": editable_files,
            "reviewer_can_edit": False,
            "maintainer_can_promote": False,
        },
        "review_infrastructure": {"training_cycle.py": file_hash(Path(__file__))},
    }
    write_new_json(
        output / "review_manifest.json",
        {
            "native_candidate_id": candidate_id,
            "review_authority": context["review_authority"],
            "review_infrastructure": context["review_infrastructure"],
            "note": "Review infrastructure is separate from the executed native candidate.",
        },
    )
    previous = None
    for role in ROLES:
        workspace = output / role
        workspace.mkdir()
        shutil.copytree(snapshot, workspace / "original")
        shutil.copytree(previous / "proposed" if previous else snapshot, workspace / "proposed")
        # Preserve the old native snapshot while carrying this later review
        # helper version into a proposed next candidate, with explicit provenance.
        if (snapshot / "training_cycle.py").exists():
            for directory in ("original", "proposed"):
                shutil.copyfile(Path(__file__), workspace / directory / "training_cycle.py")
        write_new_json(workspace / "context.json", context)
        before = {
            p.relative_to(workspace / "proposed").as_posix(): file_hash(p)
            for p in (workspace / "proposed").rglob("*")
            if p.is_file()
        }
        original = {
            p.relative_to(workspace / "original").as_posix(): file_hash(p)
            for p in (workspace / "original").rglob("*")
            if p.is_file()
        }
        answer = run_role(
            role,
            workspace,
            codex_home=codex_home,
            codex_bin=codex_bin,
            budget_s=budget_s,
            editable_files=editable_files,
        )
        after = {
            p.relative_to(workspace / "proposed").as_posix(): file_hash(p)
            for p in (workspace / "proposed").rglob("*")
            if p.is_file()
        }
        original_after = {
            p.relative_to(workspace / "original").as_posix(): file_hash(p)
            for p in (workspace / "original").rglob("*")
            if p.is_file()
        }
        changed = {
            name for name in before.keys() | after.keys() if before.get(name) != after.get(name)
        }
        if (
            changed - (set(editable_files) if role == "proposer" else set())
            or original != original_after
        ):
            raise ValueError(f"{role} changed files outside its candidate authority")
        if file_hash(workspace / "context.json") != digest(context):
            raise ValueError(f"{role} modified its evidence context")
        context["roles"][role] = answer
        previous = workspace
    decisions = context["roles"]
    result = {
        "roles": decisions,
        "candidate_id": None,
        "promoted": False,
        "native_tests_required": True,
    }
    if (
        decisions["reviewer"]["decision"] == "accept"
        and decisions["maintainer"]["decision"] == "accept"
    ):
        merged = output / "merged"
        shutil.copytree(previous / "proposed", merged)
        addendum = decisions["maintainer"]["skill_addendum"].strip()
        if addendum:
            with (merged / "SKILL.md").open("a") as stream:
                stream.write(
                    "\n\n## Candidate review note (not yet validated)\n\n" + addendum + "\n"
                )
        sources = {p.relative_to(merged).as_posix(): p for p in merged.rglob("*") if p.is_file()}
        proposed = store.propose(
            author=store.maintainer,
            parameters={**candidate["parameters"], **decisions["maintainer"]["parameters"]},
            rationale=decisions["maintainer"]["rationale"],
            sources=sources,
            base_revision=candidate["base_revision"],
            evidence_runs=runs,
        )
        result["candidate_id"] = proposed["candidate_id"]
    write_new_json(output / "cycle_result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--codex-home", type=Path, default=ROOT / "environment/airport-codex")
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--role-budget-s", type=int, default=600)
    args = parser.parse_args()
    result = run_cycle(
        CandidateStore(args.store),
        args.candidate,
        args.run,
        args.out,
        codex_home=args.codex_home,
        codex_bin=args.codex_bin,
        budget_s=args.role_budget_s,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
