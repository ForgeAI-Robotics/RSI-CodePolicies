"""Fail-closed execution boundary; no LLM, VLA or trajectory replay dependency."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import dist
from time import perf_counter
from typing import Literal

from .policy import CheckpointPolicy, PolicyConfig
from .types import Action, Backend, Observation


@dataclass(frozen=True)
class DeploymentEvidence:
    """References to reviewable calibration/control evidence, not mere flags.

    Empty references block real execution. This module implements no physical
    robot transport. A concrete adapter must enforce these guarantees itself.
    """

    robot_interface: str = ""
    calibration_manifest: str = ""
    fixed_base_interlock: str = ""
    collision_checker: str = ""
    bounded_stop: str = ""
    force_compensation: str = ""
    tool_surface_alignment: str = ""
    perception_validation: str = ""

    def blockers(self) -> tuple[str, ...]:
        return tuple(name for name, value in vars(self).items() if not value.strip())


@dataclass(frozen=True)
class RunReport:
    protocol_complete: bool
    failure_reason: str | None
    state: str
    completed_person_ids: tuple[str, ...]
    ticks: int
    deployment_mode: str
    synthetic_sensors: tuple[str, ...] = ()
    action_log: tuple[dict, ...] = field(default_factory=tuple)
    # Intentionally unset: only a separate native evaluator may judge success.
    native_success: None = None


def run_policy(
    backend: Backend,
    config: PolicyConfig | None = None,
    *,
    max_ticks: int = 10000,
    deployment_mode: Literal["simulation", "real"] = "simulation",
    evidence: DeploymentEvidence | None = None,
) -> RunReport:
    if deployment_mode not in {"simulation", "real"}:
        raise ValueError("Unsupported deployment mode")
    if isinstance(max_ticks, bool) or not isinstance(max_ticks, int) or max_ticks <= 0:
        raise ValueError("max_ticks must be a positive integer")
    policy = CheckpointPolicy(config)
    log: list[dict] = []
    synthetic: set[str] = set()
    ticks = 0
    approach_path_denials = 0
    recheck_observation = None

    def recheck_person_safe(obs: Observation) -> bool:
        if policy.config.control_profile == "reactive_v1" and policy._person_id is None:
            # Pre-extract before selecting a passenger. This is not a waiver
            # during sampling: only the existing unheld PICK_APPROACH retry
            # reaches this helper, and native person/contact checks stay active.
            return not policy._has_paper and policy.state == "PICK_APPROACH"
        people = [p for p in obs.people if p.track_id == policy._person_id]
        return bool(
            len(people) == 1
            and policy._person_valid(people[0], obs)
            and dist(people[0].velocity, (0.0, 0.0, 0.0)) <= policy.config.stationary_speed_m_s
            and policy._person_anchor is not None
            and dist(people[0].position, policy._person_anchor) <= 0.025
        )

    def recheck_paper_safe(obs: Observation) -> bool:
        return obs.paper_tip is not None and all(
            policy._patch_valid(patch, obs) for patch in (obs.paper_grasp, obs.supply_exit)
        )

    def finish(reason: str | None = None) -> RunReport:
        if reason:
            policy._stop(reason)
        try:
            backend.stop(reason or "protocol complete; hold robot")
        except Exception as exc:
            policy._stop(f"backend stop failed: {type(exc).__name__}: {exc}")
        return RunReport(
            policy.protocol_complete,
            policy.failure_reason,
            policy.state,
            tuple(policy.completed_person_ids),
            ticks,
            deployment_mode,
            tuple(sorted(synthetic)),
            tuple(log),
        )

    if deployment_mode == "real":
        missing = (evidence or DeploymentEvidence()).blockers()
        if missing:
            return finish("real deployment missing evidence: " + ", ".join(missing))
    for ticks in range(1, max_ticks + 1):
        try:
            tick_started = perf_counter()
            obs = backend.observe()
            observed_at = perf_counter()
            synthetic.update(
                name for name, sensor in obs.sensors.items() if sensor.kind == "synthetic"
            )
            if deployment_mode == "real" and synthetic:
                return finish("synthetic sensors prohibited on real robot")
            action = policy.step(obs)
            log.append(
                {
                    "tick": ticks,
                    "sequence": obs.sequence,
                    "stamp": obs.stamp,
                    "state": policy.state,
                    "kind": action.kind,
                    "target": action.target,
                    "reason": action.reason,
                    "action": asdict(action),
                    "wall_timing_s": {
                        "observe": observed_at - tick_started,
                        "policy": perf_counter() - observed_at,
                    },
                }
            )
            if action.kind == "stop":
                return finish(policy.failure_reason or action.reason)
            # An unheld-paper approach may wait for at most two new measured
            # frames per consecutive denial streak. Never retry grasp, contact,
            # retreat, held-paper transport, or real control.
            if recheck_observation is not None:
                if obs.stamp - recheck_observation.stamp < 0.05 - 1e-9 or any(
                    obs.sensors[name].stamp <= recheck_observation.sensors[name].stamp
                    for name in ("rgbd", "joints", "gripper", "force")
                ):
                    return finish("perception recheck requires new sensor timestamps after hold")
                if not recheck_paper_safe(obs):
                    return finish("paper/slot geometry lost during perception recheck")
                if not recheck_person_safe(obs):
                    return finish("passenger changed or moved during perception recheck")
                if (
                    action.kind != "move_eef"
                    or action.reason != "PICK_APPROACH"
                    or policy._has_paper
                    or policy._gripped(obs)
                ):
                    return finish("unheld approach changed during perception recheck")
            if action.kind in {"move_eef", "retreat"}:
                if action.target is None:
                    return finish("path unreachable or collision/fixed-base check rejected")
                check_started = perf_counter()
                reachable = backend.check_reachability(action)
                log[-1]["wall_timing_s"]["precheck"] = perf_counter() - check_started
                log[-1]["reachability"] = {
                    "accepted": bool(reachable),
                    "action": asdict(action),
                    "sequence": obs.sequence,
                }
                if not reachable:
                    eligible = (
                        deployment_mode == "simulation"
                        and not policy._has_paper
                        and not policy._gripped(obs)
                        and action.kind == "move_eef"
                        and action.reason == "PICK_APPROACH"
                    )
                    if not eligible:
                        return finish("path unreachable or collision/fixed-base check rejected")
                    approach_path_denials += 1
                    log[-1]["approach_path_denials"] = approach_path_denials
                    if approach_path_denials >= 3:
                        return finish(
                            "path unreachable or collision/fixed-base check rejected; "
                            "recheck exhausted"
                        )
                    if not recheck_paper_safe(obs):
                        return finish("paper/slot geometry lost before perception recheck")
                    if not recheck_person_safe(obs):
                        return finish("passenger changed or moved before perception recheck")
                    hold = Action(
                        "hold_pose", duration_s=0.05, reason="bounded approach perception recheck"
                    )
                    log[-1]["recheck_hold"] = asdict(hold)
                    # Explicit encoder-target freeze, not generic HOLD: the backend
                    # replaces old motion targets from this sensor epoch and
                    # invalidates the denied plan before advancing physics.
                    backend.execute(hold)
                    recheck_observation = obs
                    continue
                recheck_observation = None

            if policy.finished:
                return finish()
            execute_started = perf_counter()
            backend.execute(action)
            log[-1]["wall_timing_s"]["execute"] = perf_counter() - execute_started
            log[-1]["wall_timing_s"]["total"] = perf_counter() - tick_started
            if action.kind in {"move_eef", "retreat"}:
                approach_path_denials = 0
        except Exception as exc:
            return finish(f"backend unavailable: {type(exc).__name__}: {exc}")
    return finish("execution tick budget exhausted")


def reject_base_command(action: Action) -> None:
    """Defence for adapters accepting deserialized external actions."""
    if action.kind not in {"hold", "hold_pose", "stop", "move_eef", "grip", "retreat"}:
        raise ValueError("Only fixed-base checkpoint actions are permitted")
