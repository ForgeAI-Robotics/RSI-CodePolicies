---
name: airport-checkpoint-workflow
description: Develop, train and audit the PyRUA-Lean fixed-base Isaac checkpoint policy for queued passengers carrying bags, half-exposed paper extraction and three-person batch disposal. Use for this simulator workflow and its versioned shared policy; physical hardware integration is a later separate task.
---

# Fixed-base checkpoint

This is a proposed simulator workflow, not a record of successful training. Read
the candidate manifest and native run artifacts before claiming a learned skill.
Current scope is the existing simulation robot. Uncalibrated physical hardware
capabilities do not block simulator experiments and must not be described as proven.

Use PyRUA-Lean with `gpt-6-astra`, `high` reasoning and `no-vla` during training.
Project-local primitives are permitted. Do not launch a VLA service or download
VLA weights. The deployment entry point is `airport.checkpoint.deployment.run_policy`;
it runs deterministic code with no model, agent runtime or recorded action replay.

## Episode contract

The base stays fixed beside the lane throughout the episode. No base translation,
rotation, wheel command or teleport is allowed. Passengers queue carrying bags,
stop for a measured three-second stable dwell, receive bag and forearm sampling,
and enter the waiting area. Keep identities associated across these transitions.
The default batch is three passengers; dispose of the same paper only after the
batch. Extract paper from a supply fixture with approximately half its length
exposed; do not start with paper attached or substitute a pre-held state.

Compose feedback-aware code using project primitives. `robo.state()` reports
measurements or an explicit unavailable reason. `robo.execute_workflow()` runs
the frozen candidate once; it delegates to the existing deterministic controller
and guards. Inspect its failure reason and per-stage evidence. Do not retry the
same finished episode or change its initial state. New candidates get new episodes.

Use `robo.measure()` for current sensor features and provenance, and
`robo.check_reachability(xyz, rpy)` for a read-only whole-path feasibility query.
`robo.propose_parameters(parameters, rationale)` saves a bounded next-trial
proposal without changing this frozen episode's parameters or native identity.

Policy input comes from current RGB/depth, encoders, gripper measurements and
contact/force sensors. Mark simulation sensors as synthetic; that is permitted
in simulation. Do not expose USD poses, native success flags, hidden collision
actor identities or scenario target coordinates as measured perception. Privileged
state belongs only to the independent native auditor. Calibrate the simulator's
camera geometry, tool frame and force units rather than inventing sensor values.

Stop on stale/missing measurements, identity ambiguity, moving passengers,
unreachable paths, unexpected contact, force violations or base drift. Preserve
failure logs. Use bounded controller retries and the fixed run budget. Lack of
an implemented perception/control capability is a recorded failed attempt, not
permission to bypass the measurement boundary. Real hardware later requires its
own interface and calibration evidence; do not claim this simulator proves them.

## Training and shared maintenance

The fixed experiment protocol is
`configs/airport-checkpoint-protocol.json`; `airport.checkpoint.scenarios` generates
the exact scene specifications and hashes. Train seeds are 1001–1008. Heldout
seeds 9001–9004 are reserved for a frozen final evaluation. Vary stopping position,
body orientation, height, bag side and bag position through the scenario generator.
Do not inspect heldout outcomes while proposing or selecting policy changes.

Multiple Astra workers may review train failures and submit independent candidate
code/parameter/skill snapshots. Only the named `checkpoint-maintainer` merges a
candidate. Run `python -m airport.checkpoint.learning --store STORE propose
--author NAME --rationale TEXT` (optionally `--parameters FILE`, `--base REVISION`,
`--evidence TRAIN_RUN`). The content-addressed candidate captures source hashes,
skill text, parameters, protocol and parent revision. Agents edit independent
candidate workspaces; never overwrite one another's active shared policy.

Run a fresh native episode with:

```sh
python -m airport.checkpoint.agent --store STORE --candidate ID --seed 1001 \
  --out NEW_RUN --backend-factory module:function
```

The factory connects the current simulator robot and returns the typed backend;
it receives `out_dir`, `scenario`, `parameters` and `binding`. Training reuses
the existing Codex authentication and PyRUA cells runtime. No new API service or
credentials are required. Snapshot working source must match the candidate hashes.

The existing remote Isaac host is connected by
`airport.checkpoint.remote_backend:make_backend`. The convenience entry point is
`scripts/run_checkpoint_training.py --store STORE --candidate ID --seed 1001
--out NEW_RUN --gpu 0 --sync-code`; `--review-out NEW_CYCLE` also runs the review
below. Coordinate GPU ownership with other experiment workers. The launcher
refuses an already active checkpoint container; it does not stop another run.
`--sync-code` uploads the frozen project modules and existing worker/launcher,
then verifies source hashes. Package installation and model downloads are absent.

Only `overview` is exposed, from the current head RGBD sensor; a wrist camera is
not implemented. The remote worker saves continuous camera frames with timestamps.
The host encodes these to `native_observations.mp4` with a provenance file. This
is a diagnostic recording of that episode, never a fresh test or replay controller.

After one or more live train attempts, run the three-role evidence review:

```sh
python -m airport.checkpoint.training_cycle --store STORE --candidate ID \
  --run TRAIN_RUN --out NEW_CYCLE
```

This launches separate `gpt-6-astra` / `high` Codex processes for proposer,
independent reviewer and sole maintainer. Each has its own workspace and recorded
prompt/events/stderr/result. Only the proposer may change its isolated `policy.py` and `vision.py`;
the reviewer checks that exact change and the maintainer may stage a new immutable
code/parameter/skill candidate. It cannot promote `CURRENT`, execute the proposed
code, reset a simulator or treat review acceptance as a successful experiment.
Missing native live evidence blocks this cycle. Native failures are valid training
evidence; unit-test fixtures and smoke construction are not.

Audit with `python -m airport.checkpoint.learning --store STORE audit --candidate
ID --run RUN`. Archive failures as well as successes. Candidate promotion requires
fresh native evidence for every train case. The maintainer uses `promote` with
`--actor checkpoint-maintainer` and repeated `--run` arguments. A filesystem lock
and compare-and-swap parent check prevent concurrent lost updates. `CURRENT` is
the promoted snapshot pointer, not a command to overwrite the source tree. Review
and materialize the snapshot in an isolated checkout before subsequent experiments.

Use `seal-heldout --candidate ID --actor checkpoint-maintainer` once after promotion.
Run that frozen revision without online LLM against the reserved set; retain all
outcomes, including failures. Heldout results cannot enter proposals. A later
study needs a new preregistered heldout set, not tuning against this one.

## Evidence and interpretation

Only the native environment may report task success. Agent finish text, a
controller's `protocol_complete`, successful API calls, smoke construction,
unit tests and replay videos are not native task success. Record the live engine,
run identity, seed, scene/source/protocol hashes, budgets, native physics steps,
contact evidence, stage ordering, base-lock checks and violations. The worker
writes `checkpoint_native.json`; the host writes `checkpoint_binding.json`.
Native observations/actions/events and scene manifest must have matching hashes.

The audit rejects absent native predicates, replay, partial scope, changed sources,
missing provenance and repeated evidence. It checks artifact consistency; it does
not cryptographically prove an honest simulator. Keep native outputs outside
proposal agents' writable workspaces and retain worker logs. Report structural
audit results separately from native success and from physical safety validation.

## Current measurement contracts

Before every PICK_APPROACH/PICK_GRASP movement or closure, require a fresh valid
paper patch. In the supply stage, use current depth, the current measured rim and
the declared paper dimensions; reject missing, ambiguous or expired identity.
Stored centers are association history, never current measurements or attachment
proof. Record rejection reasons and the transition to free paper tracking. The
post-grasp brightness-based tracker remains unvalidated for the dark paper back.

In simulation, an unheld PICK_APPROACH path rejection may trigger at most two
consecutive `hold_pose` rechecks; a third rejection stops. This internal executor
primitive re-anchors the 11 planning-joint targets to same-epoch own encoders,
invalidates the rejected plan and advances 50 ms with unchanged PD/FF/interlocks.
It preserves physical velocity; ordinary `hold` keeps its prior semantics.
Require fresh paper/slot, the same stationary lateral passenger within 25 mm of
the saved anchor, and advancing RGBD/joint/gripper/force timestamps before any
new plan. A successful checked movement resets the consecutive counter; existing
phase, tick, physics and wall budgets remain binding. Grasp/contact/held-paper and
real deployment never use this retry. The unloaded encoder-freeze native probe
qualified this bounded simulation primitive; complete-task benefits remain unproven.

Use the pinned `zero_delay_v1` experience and explicit half-pixel sampling only
with its recorded native qualification. Keep old calibration records at their
original sampling convention. The own-depth primitive requires matching render,
encoder and calibration provenance plus a passing frame-health check. Include
rendered finger pads in its isolated own model. Ray misses and axial mismatches
may become surface candidates only with an exact opaque own-triangle witness
within the empirical 20 µm Euclidean band; this differs from axial-depth matching.
Retain unqualified points and protect measured paper and instance-supported people.
Real foreground inside either uncertainty band can remain ambiguous. Preserve the
25 mm human clearance and native robot-person contact failure. Weak RGB skin
alone is not confirmed human identity. This is a controlled simulation candidate,
not a hardware safety claim; see `docs/airport-checkpoint-experiment.md` for evidence.

Wrist force uses a 50 ms momentum-balance window from measured wrench, encoders,
CAD mass/gravity and calibrated zero offset. Preserve window start/end and midpoint
provenance; reset after calibration or payload changes and wait for a complete
window. Never replace an unavailable force with zero or re-zero against contact.
Free-space motion and known loads were tested natively; the nominal 0.6 g payload
model does not measure flexible-paper motion and still needs native validation.

The reviewed `control_profile="reactive_v1"` candidate pre-extracts paper before
selecting a passenger, then waits with measured grip/paper feedback. Passenger
dwell, lateral association and sampling guards still apply. Stable transport
stages send a persistent goal/trajectory ID with speed and acceleration limits;
the backend interpolates only an approved path and requires fresh perception and
whole-path approval before each bounded segment. Contact/scan/retreat retain fine
feedback. IDs keep their goal/limit binding for the backend session, including
after rejection or hold. Never turn a stale ID into permission to continue moving.
Transport uses acceleration-bounded trapezoidal/triangular timing, checked through
FK at each command sample; velocity is continuous but this is not jerk-limited.
The default `legacy` profile and frozen pre-change baseline remain available.
The separately reviewed `collision_profile="oriented_v1"` uses a world broad
phase and locally expanded OBB narrow phase, with unchanged 25/3 mm human/unknown
clearance and a 1 µm numeric enclosure pad. Do not erase tracked-paper/skin overlap
to make a path pass: current false-corner cases can be resolved geometrically
while retaining all human flags. Rejection logs now include the actual point,
geometry and local gap. See `docs/airport-checkpoint-collision-repair.md` and its
native evidence before claiming a grasp or successful workflow.
The optional `vision_profile="supply_bounds_v1"` changes only the fixed supply
identity anchor from visible-point median to current rim/plane-aligned visible
min/max boundary midpoint. Preserve the 12 mm identity gate, unique geometry,
dimensions, rim and current depth requirements. Grasp/tip/span still come from
the current measured points; never return an old patch when current evidence is
missing. Partial contour loss, outliers and normal changes remain limitations.
See `docs/airport-checkpoint-reactive-control.md` for scope and validation: current
host perception is synchronous, so interpolation is not asynchronous real-time
vision or evidence of faster end-to-end execution. Only native tests establish
movement or task outcomes; selecting the profile is not a learned skill claim.

Current train attempts have not grasped paper or completed a three-person batch.
Successful detection, earlier fail-closed termination, review acceptance and force
calibration are separate evidence, not a learned end-to-end skill. Preserve all
guards and native acceptance while validating grasp, extraction, sampling and
disposal. Historical proposal rationales remain in immutable candidate snapshots.
