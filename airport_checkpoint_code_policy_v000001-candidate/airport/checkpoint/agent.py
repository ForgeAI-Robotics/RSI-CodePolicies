"""Train-only PyRUA-Lean extension over the deterministic checkpoint runner.

No vendor files or old airport adapters are changed. The supplied backend factory
must connect to a fresh native episode; this module never starts a remote GPU.
"""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

from pyrualean._robot import RobotBase
from pyrualean.prompt import TaskCard
from pyrualean.robots import RobotAdapter

from .deployment import run_policy
from .learning import (
    SKILL_PATH,
    CandidateStore,
    bind_run,
    file_hash,
    validate_parameters,
    write_new_json,
)
from .policy import PolicyConfig
from .scenarios import ROOT, load_protocol
from .types import Action

TASK = (
    "Keep the base locked beside the queue. With measured sensor feedback, extract a "
    "half-exposed sampling paper, sample the bag and forearm of each of three stationary "
    "passengers, wait for each to enter the waiting area, then deposit the paper in the box. "
    "Stop on missing measurements or failed safety checks."
)


@dataclass(frozen=True)
class CheckpointState:
    task: str
    observation: dict[str, Any] | None
    protocol_complete: bool
    failure_reason: str | None
    terminated: bool
    truncated: bool


class CheckpointRobot(RobotBase):
    NAME = "airport_checkpoint"
    CAMERAS = ("overview",)
    IMAGE_ARTIFACTS = {(name, "high"): name + ".png" for name in CAMERAS}
    ON_MOTION_IMAGES = ()
    ABSENT_MEMBERS = frozenset({"world_map"})
    STATEFUL_TOOLS = frozenset({"execute_workflow"})
    RESULT_TYPES = (CheckpointState,)

    def _state_from_envelope(self, envelope: dict[str, Any]) -> CheckpointState:
        state = envelope["state"]
        return CheckpointState(
            TASK,
            state.get("observation"),
            state.get("protocol_complete", False),
            state.get("failure_reason"),
            envelope["terminated"],
            envelope["truncated"],
        )

    def _summary_dict(self) -> dict[str, Any]:
        return asdict(self._current_state())

    def execute_workflow(self) -> dict[str, Any]:
        """Execute the frozen candidate once with measured feedback and bounded ticks.

        Uses project Code-as-Policy primitives and stops on missing measurements,
        unreachable paths, stale inputs or contact limits. No base-motion, VLA,
        episode reset or success override exists. protocol_complete is not native
        success: use the separately reported environment verdict. A failed episode
        is retained; propose a new candidate and start a fresh training episode.
        """
        return self._stateful("execute_workflow", {}, lambda result: result)

    def measure(self) -> dict[str, Any]:
        """Read current measured RGBD features, joints, grip and force, with provenance.

        Returns unavailable measurements explicitly. No hidden scene pose, actor
        labels or native success details are exposed to the policy.
        """
        return self._readonly("measure", {})

    def check_reachability(self, xyz: list[float], rpy: list[float]) -> dict[str, Any]:
        """Query fixed-base whole-path feasibility without moving or resetting.

        The backend must use its existing collision and IK checks. A true answer
        alone is not contact/task success and does not permit bypassing guards.
        """
        return self._readonly("check_reachability", {"xyz": xyz, "rpy": rpy})

    def propose_parameters(self, parameters: dict[str, Any], rationale: str) -> dict[str, Any]:
        """Save a bounded next-trial parameter proposal for independent review.

        Does not alter this frozen episode or run another trial. A maintainer
        reviews proposals and creates a new candidate with separate native runs.
        Safety limits, perception provenance and acceptance thresholds are fixed.
        """
        return self._readonly(
            "propose_parameters", {"parameters": parameters, "rationale": rationale}
        )


class CheckpointToolkit:
    """Thin PyRUA envelope adapter; all control remains in deployment/backend."""

    def __init__(self, backend: Any, output: Path, parameters: dict[str, Any]):
        self.backend, self.output = backend, output
        self.config = PolicyConfig(**parameters)
        self.protocol = load_protocol()
        self.report = None
        self.observation = None
        self.observation_error = None
        self.closed = False
        self.task_language = TASK
        self.records: list[dict[str, Any]] = []

    def _observe(self) -> None:
        try:
            self.observation = asdict(self.backend.observe())
            self.observation_error = None
        except Exception as exc:
            self.observation = None
            self.observation_error = f"{type(exc).__name__}: {exc}"

    def _envelope(self, result: dict | None = None) -> dict:
        failed = bool(self.report and self.report.failure_reason)
        envelope = {
            "task_language": TASK,
            "step": len(self.records),
            "state": {
                "observation": self.observation,
                "protocol_complete": bool(self.report and self.report.protocol_complete),
                "failure_reason": self.report.failure_reason
                if self.report
                else self.observation_error,
            },
            "terminated": self.solved(),
            "truncated": failed,
            "log": {"result": result or {}},
        }
        return envelope

    def execute_tool(self, name: str, input_dict: dict) -> dict:
        if name == "measure":
            self._observe()
            return {"observation": self.observation, "unavailable": self.observation_error}
        if name == "check_reachability":
            import math

            xyz, rpy = input_dict["xyz"], input_dict["rpy"]
            if len(xyz) != 3 or len(rpy) != 3 or not all(math.isfinite(v) for v in xyz + rpy):
                raise ValueError("Finite xyz/rpy triples required")
            action = Action("move_eef", target=tuple(xyz), rpy=tuple(rpy), duration_s=0.2)
            return {"reachable": bool(self.backend.check_reachability(action)), "executed": False}
        if name == "propose_parameters":
            validate_parameters(input_dict["parameters"])
            directory = self.output / "parameter_proposals"
            directory.mkdir(exist_ok=True)
            index = len(list(directory.glob("*.json")))
            write_new_json(directory / f"{index:04d}.json", input_dict)
            return {"saved": True, "index": index, "active_parameters_changed": False}
        if name == "view_env_state":
            step = input_dict.get("step", -1)
            if step >= 0:
                return self.records[step]
            self._observe()
            return self._envelope()
        if name != "execute_workflow":
            raise ValueError(f"Unsupported checkpoint primitive: {name}")
        if self.report is not None:
            raise RuntimeError(
                "This episode has already executed; resets and repeated runs are forbidden"
            )
        self.report = run_policy(
            self.backend,
            self.config,
            max_ticks=self.protocol["max_policy_ticks"],
            deployment_mode="simulation",
        )
        write_new_json(self.output / "checkpoint_policy_report.json", asdict(self.report))
        self._observe()
        record = self._envelope(asdict(self.report))
        self.records.append(record)
        return record

    def solved(self) -> bool:
        # Do not fall back to self.report.protocol_complete or an agent finish claim.
        verdict = getattr(self.backend, "native_success", None)
        return verdict() is True if callable(verdict) else False

    def env_steps(self) -> int | None:
        counter = getattr(self.backend, "env_steps", None)
        return counter() if callable(counter) else None

    def artifact(self, name: str, step: int = -1) -> bytes:
        if name != "overview.png":
            raise ValueError("Only sensor camera images are exposed")
        return self.backend.artifact(name, step=step)

    def cancel(self) -> None:
        self.backend.stop("PyRUA episode cancelled")

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.backend.stop("PyRUA episode closed")
            closer = getattr(self.backend, "close", None)
            if callable(closer):
                closer()


@dataclass(frozen=True)
class Cell:
    suite: str = "airport_checkpoint_native_v1"
    task: str = "ThreePassengerCheckpoint"
    seed: int = 1001
    max_episode_steps: int = 400000
    cuda_device: int | None = None
    backend_factory: str = ""
    candidate_store: str = ""
    candidate_id: str = ""


ADAPTER = RobotAdapter(
    name="airport_checkpoint",
    robot_cls=CheckpointRobot,
    host="airport.checkpoint.agent",
    knowledge="airport_checkpoint",
    blurb="a fixed-base native Isaac checkpoint robot",
    show_examples='`robo.show("overview")`',
)


def boot(cell: Cell, out_dir: Path, **kwargs: Any) -> tuple[CheckpointToolkit, list]:
    candidate = CandidateStore(Path(cell.candidate_store)).load(cell.candidate_id)
    write_new_json(out_dir / "candidate.json", candidate)
    # Recorded source must be the code actually imported by the worker/runner.
    for name, expected_hash in candidate["files"].items():
        source = SKILL_PATH if name == "SKILL.md" else ROOT / "airport/checkpoint" / name
        if file_hash(source) != expected_hash:
            raise ValueError(f"Working source differs from frozen candidate: {name}")
    module, separator, function = cell.backend_factory.partition(":")
    if not separator:
        raise ValueError("backend_factory must be importable.module:function")
    factory = getattr(importlib.import_module(module), function)
    backend = factory(
        out_dir=out_dir,
        scenario=json.loads((out_dir / "scenario.json").read_text()),
        parameters=candidate["parameters"],
        binding=json.loads((out_dir / "checkpoint_binding.json").read_text()),
    )
    return CheckpointToolkit(backend, out_dir, candidate["parameters"]), []


def make_backend(toolkit: CheckpointToolkit) -> CheckpointToolkit:
    return toolkit


def dump_card(toolkit: CheckpointToolkit, cell: Cell, out_dir: Path) -> TaskCard:
    card = TaskCard(
        suite=cell.suite,
        task=cell.task,
        seed=cell.seed,
        task_language=TASK,
        object_names=["half-exposed paper", "passengers", "carried bags", "deposit box"],
        max_env_steps=cell.max_episode_steps,
        robot="airport_checkpoint",
        facts={
            "split": "train",
            "base_locked": True,
            "candidate_id": cell.candidate_id,
            "deployment_mode": "simulation",
            "success_authority": "independent native environment predicate",
            "instruction": "Inspect state, run execute_workflow once, report failures honestly.",
        },
    )
    (out_dir / "card.json").write_text(card.to_json())
    return card


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--backend-factory", required=True)
    parser.add_argument("--codex-home", default=str(ROOT / "environment/airport-codex"))
    args = parser.parse_args()
    protocol = load_protocol()
    candidate = CandidateStore(args.store).load(args.candidate)
    bind_run(args.out, candidate, args.seed)
    from pyrualean import play

    original_adapter, original_knowledge, original_argv = (
        play.get_adapter,
        play.knowledge_text,
        play._codex_argv,
    )

    def adapter(name):
        return ADAPTER if name == "airport_checkpoint" else original_adapter(name)

    def knowledge(name, primitives="full"):
        if name != "airport_checkpoint":
            return original_knowledge(name, primitives=primitives)
        return (args.store / "candidates" / args.candidate / "snapshot/SKILL.md").read_text()

    def codex_argv(**options):
        argv = original_argv(**options)
        if argv[-1] != "-":
            raise ValueError("Unexpected Codex stdin invocation")
        return [
            *argv[:-1],
            "-c",
            f"mcp_servers.pyrualean.tool_timeout_sec={protocol['wall_budget_s'] + 60}",
            "-",
        ]

    with (
        patch.object(play, "get_adapter", adapter),
        patch.object(play, "knowledge_text", knowledge),
        patch.object(play, "_codex_argv", codex_argv),
    ):
        result = play.play_episode(
            Cell(
                seed=args.seed,
                max_episode_steps=protocol["max_physics_steps"],
                backend_factory=args.backend_factory,
                candidate_store=str(args.store.resolve()),
                candidate_id=args.candidate,
            ),
            args.out.resolve(),
            arm="cells",
            robot="airport_checkpoint",
            model="gpt-6-astra",
            reasoning_effort="high",
            primitives="no-vla",
            runtime="codex",
            max_turns=protocol["max_code_calls"],
            budget_s=protocol["wall_budget_s"],
            images="on-demand",
            feedback="rich",
            codex_home=args.codex_home,
        )
    print(json.dumps({key: result.get(key) for key in ("status", "environment_success", "error")}))


if __name__ == "__main__":
    main()
