"""Image-only transport and observation-derived primitives for PyRUA-Lean."""

from __future__ import annotations

import base64
import io
import json
import math
import os
import shlex
import subprocess
import sys
import threading
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from tianji.description import ROOT, Description, VendorIK
from tianji.motion import linear_trajectory
from tianji.task_spec import HEAD_CENTER_TASK, SINGLE_LEFT_TASKS, TASK_VARIANTS

from .contract import CAMERAS, command_angles, initial_action, validate_action

SSH = [
    "ssh",
    "-i",
    "/data1/liuming/forgeai.pem",
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=12",
    "ubuntu@10.60.71.213",
]
REMOTE = "/mnt/shared/moonten/RSI"


class RGBBackend:
    def __init__(
        self,
        out: Path,
        *,
        gpu=7,
        seed=0,
        start=True,
        scenario="nominal",
        worker_slot=0,
        diagnostic_views=False,
        collision_planning_enabled=True,
        task_variant="two_can",
        camera_profile="urdf",
    ):
        self._out = out
        self._native_local = os.environ.get("TIANJI_NATIVE_LOCAL") == "1"
        if self._native_local and ROOT.resolve() != Path(REMOTE).resolve():
            raise RuntimeError("Local native transport requires the RTX project checkout")
        self._run = "rgb_" + uuid.uuid4().hex[:12]
        self._serial = 0
        self._count = 0
        self._closed = False
        self._transport_lock = threading.RLock()
        self._success = False
        self._command = initial_action(task_variant)
        self._initial_command = self._command.copy()
        if task_variant not in TASK_VARIANTS:
            raise ValueError("Invalid task variant")
        self._task_variant = task_variant
        from tianji.wrist_profile import PROFILES

        if camera_profile not in PROFILES:
            raise ValueError("Unknown wrist camera profile")
        self._camera_profile = camera_profile
        self._single_left_can = task_variant in SINGLE_LEFT_TASKS
        self._head_center_task = task_variant == HEAD_CENTER_TASK
        self._idle_right_command = self._command[8:].copy()
        self._images = {}
        self._history = {}
        self._diagnostic_views = bool(diagnostic_views)
        self._diagnostic_image = None
        self._training_state = None
        self._policy_execution = False
        self._collision_required = False
        self._collision_planning_enabled = bool(collision_planning_enabled)
        self._fixed_arm_order = False
        self._collision_approval = []
        self._description = Description()
        self._inverse = VendorIK(self._description)
        self._last_ik_diagnostic = None
        self._last_ik_request = None
        self._launch_pid = None
        metadata_path = out / "native_run.json"
        if metadata_path.is_file():
            prior = json.loads(metadata_path.read_text())
            if prior.get("simulation_start_requested") is True:
                # Author retries can reuse this output directory. Preserve every
                # owned native ID so shutdown cannot strand the previous renderer.
                with (out / "native_run_history.jsonl").open("a") as history:
                    history.write(json.dumps(prior) + "\n")
        metadata_path.write_text(
            json.dumps(
                {
                    "run": self._run,
                    "simulation_start_requested": bool(start),
                    "remote_root": REMOTE,
                    "gpu": gpu,
                    "worker_slot": worker_slot,
                    "scenario": scenario,
                    "seed": seed,
                    "training_diagnostic_views": self._diagnostic_views,
                    "collision_planning_enabled": self._collision_planning_enabled,
                    "task_variant": self._task_variant,
                    "camera_profile": self._camera_profile,
                },
                indent=2,
            )
        )
        if start:
            no_video = os.environ.get("TIANJI_NO_VIDEO", "1" if self._native_local else "0")
            launch = subprocess.run(
                [
                    sys.executable
                    if self._native_local
                    else str(ROOT / "environment/.venv/bin/python"),
                    str(ROOT / "scripts/start_tianji_isaac.py"),
                    self._run,
                    "--mode",
                    "rgb_policy",
                    "--gpu",
                    str(gpu),
                    "--scene-seed",
                    str(seed),
                    "--scenario",
                    scenario,
                    "--worker-slot",
                    str(worker_slot),
                    "--task-variant",
                    self._task_variant,
                    "--camera-profile",
                    self._camera_profile,
                    *(["--no-video"] if no_video == "1" else []),
                ],
                check=True,
                capture_output=True,
            )
            self._launch_pid = json.loads(launch.stdout)["pid"]
            self._exchange(None)
            if self._diagnostic_views:
                self._exchange({"operation": "diagnostic"})

    def _ssh(self, code):
        try:
            command = (
                [sys.executable, "-"]
                if self._native_local
                else [*SSH, shlex.join(["sudo", "-n", "python3", "-"])]
            )
            result = subprocess.run(
                command,
                input=code,
                text=True,
                capture_output=True,
                check=True,
                timeout=420,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            # Host audit only: never expose SSH/native stderr through actor feedback.
            stderr = exc.stderr or ""
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            with (self._out / "transport_errors.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "run": self._run,
                            "serial": self._serial,
                            "error_type": type(exc).__name__,
                            "returncode": getattr(exc, "returncode", None),
                            "stderr": stderr[-3000:],
                        }
                    )
                    + "\n"
                )
            raise
        return result.stdout

    def _exchange(self, request):
        with self._transport_lock:
            if self._closed:
                raise RuntimeError("Episode already closed")
            return self._exchange_locked(request)

    def _exchange_locked(self, request):
        folder = REMOTE + "/runs/tianji_native/" + self._run + "/ipc"
        serial = self._serial
        code = f"""from pathlib import Path
import time,json,os
p=Path({folder!r}); serial={serial}
launch_pid={self._launch_pid!r}
request={request!r}
if request is not None:
 temp=p/f'request_{{serial:06d}}.tmp'
 temp.write_text(json.dumps(request));temp.replace(p/f'request_{{serial:06d}}.json')
f=p/f'response_{{serial:06d}}.json'
for _ in range(4000):
 if f.exists(): print(f.read_text());break
 if (p/'finished.json').exists():raise RuntimeError('Trial stopped')
 if serial == 0 and launch_pid is not None:
  try:os.kill(launch_pid,0)
  except ProcessLookupError:
   raise RuntimeError('Native launcher exited before initial RGB; inspect launch log')
 time.sleep(.1)
else:raise TimeoutError('Native RGB observation timeout')
"""
        response = json.loads(self._ssh(code))
        allowed = {"images", "action_count", "stopped"}
        if self._diagnostic_views:
            allowed.add("diagnostic_images")
            allowed.add("training_state")
        if set(response) - allowed:
            raise ValueError("Unexpected observation fields")
        if set(response["images"]) != set(CAMERAS):
            raise ValueError("Expected exactly three RGB cameras")
        self._images = {c: base64.b64decode(v) for c, v in response["images"].items()}
        self._count = int(response["action_count"])
        self._history[self._count] = self._images.copy()
        if "diagnostic_images" in response:
            if set(response["diagnostic_images"]) != {"external"}:
                raise ValueError("Expected exactly one external diagnostic RGB image")
            self._diagnostic_image = base64.b64decode(response["diagnostic_images"]["external"])
            folder = self._out / "diagnostics"
            folder.mkdir(exist_ok=True)
            latest = folder / "external_latest.png"
            temporary = latest.with_suffix(".tmp")
            temporary.write_bytes(self._diagnostic_image)
            temporary.replace(latest)
        self._serial += 1
        if "training_state" in response:
            self._training_state = response["training_state"]
            (self._out / "diagnostics").mkdir(exist_ok=True)
            (self._out / "diagnostics/state_latest.json").write_text(
                json.dumps(self._training_state) + "\n"
            )
        if response.get("stopped"):
            raise RuntimeError("Native safety stop; inspect RGB, no privileged details exposed")
        return {"frames": self._count}

    def _act(self, actions):
        actions = [validate_action(a).tolist() for a in actions]
        if self._single_left_can and not np.allclose(
            np.asarray(actions)[:, 8:], self._idle_right_command, atol=1e-10, rtol=0
        ):
            raise ValueError("Single-can task requires the right arm to hold its initial command")
        if self._collision_required:
            approved = self._collision_approval[: len(actions)]
            if len(approved) != len(actions) or not np.array_equal(approved, actions):
                raise ValueError("Motion lacks matching ordered collision preflight approval")
            self._collision_approval = self._collision_approval[len(actions) :]
        for i in range(0, len(actions), 600):
            self._exchange({"operation": "step", "actions": actions[i : i + 600]})
        self._command = np.asarray(actions[-1])
        return {"frames": self._count, "commanded_action": self._command.tolist()}

    def execute_tool(self, name, args):
        if name == "diagnostic_state":
            if not self._diagnostic_views or self._policy_execution:
                raise RuntimeError("Training state diagnostics are disabled in frozen execution")
            if args.get("refresh", False):
                self._exchange(
                    {
                        "operation": "diagnostic_state",
                        "include_scene": bool(args.get("include_scene", False)),
                        "include_depth": bool(args.get("include_depth", False)),
                    }
                )
            if self._training_state is None:
                raise RuntimeError("Training state is not available")
            with (self._out / "training_state_access.jsonl").open("a") as stream:
                stream.write(json.dumps({"frames": self._count, "arguments": args}) + "\n")
            return self._training_state
        if name == "diagnostic_view":
            if not self._diagnostic_views or self._policy_execution:
                raise RuntimeError(
                    "External diagnostic view is disabled for frozen policy execution"
                )
            if self._diagnostic_image is None:
                raise RuntimeError("External diagnostic RGB image is not available")
            return {
                "frames": self._count,
                "purpose": "training author diagnosis only; not a policy sensor",
            }
        if name == "observe":
            return {"frames": self._count}
        if name == "collision_check":
            if not self._collision_planning_enabled:
                raise RuntimeError("Collision planner is disabled in the fixed-arm experiment")
            from .collision import check_motion

            self._collision_required = True
            self._collision_approval = []
            result = check_motion(**args)
            if result["clear"]:
                rows = [validate_action(row).tolist() for row in args["actions"]]
                if rows and np.array_equal(rows[0], self._command):
                    rows = rows[1:]
                self._collision_approval = rows
            with (self._out / "collision_preflight.jsonl").open("a") as stream:
                stream.write(json.dumps({"frames": self._count, "result": result}) + "\n")
            return result
        if name == "action16":
            return self._act([args["action"]] * int(args.get("repeat", 1)))
        if name == "action_sequence":
            if not args["actions"]:
                raise ValueError("Action sequence must not be empty")
            return self._act(args["actions"])
        if name == "move_to":
            return self._move(**args)
        if name == "gripper":
            arm = args["arm"]
            idx = 7 if arm == "left" else 15
            target = self._command.copy()
            target[idx] = float(args["opening"])
            validate_action(target)
            actions = [
                ((1 - t) * self._command + t * target).tolist() for t in np.linspace(0, 1, 41)[1:]
            ]
            return self._act(actions)
        if name == "detect":
            return self._detect(**args)
        if name == "rim":
            from .perception import circle_rim

            camera = args["camera"]
            model = self._camera_model(camera)
            rgb = np.asarray(Image.open(io.BytesIO(self.artifact(camera + "_rgb.png"))))[:, :, :3]
            fits = circle_rim(
                rgb,
                model["intrinsic"],
                model["optical_pose_from_command"],
                args["expected_xyz"],
                args["plane_height"],
                args["radius"],
            )
            return {
                "fits": fits,
                "source": "RGB Canny edges + explicit top-plane/radius prior; NOT depth",
            }
        if name == "stereo_rim_pose":
            from .stereo_rim import estimate_rim_pose

            cameras = ("head", "left_wrist")
            images = {
                camera: np.asarray(
                    Image.open(
                        io.BytesIO(
                            self.artifact(camera + "_rgb.png"),
                        )
                    ).convert("RGB")
                )
                for camera in cameras
            }
            models = {camera: self._camera_model(camera) for camera in cameras}
            return estimate_rim_pose(
                images,
                models,
                args["expected_xyz"],
                args["radius"],
                args["height"],
                args["height_spread"],
            )
        if name == "gripper_geometry":
            from tianji.wrist_profile import driver_max

            from .geometry import geometry_summary

            return geometry_summary(
                **args,
                driver_scale=driver_max(
                    args["arm"],
                    self._camera_profile,
                )
                / 0.6,
            )
        if name == "grasp_candidates":
            from .geometry import grasp_candidates

            return {
                "candidates": grasp_candidates(
                    self._inverse,
                    command_angles(self._command, self._task_variant, self._camera_profile),
                    **args,
                ),
                "source": "caller RGB estimates + static URDF/SDK; no scene state",
                "limits": "sampled hand endpoint clearance; no swept-arm collision guarantee",
            }
        if name == "last_ik_diagnostic":
            if self._last_ik_diagnostic is not None:
                self._record_ik_diagnostic(name, self._last_ik_request, self._last_ik_diagnostic)
            return {"diagnostic": self._last_ik_diagnostic}
        if name in {"solve_pose", "ik_diagnose", "solve_pose_continuous"}:
            arm = args["arm"]
            if arm not in ("left", "right"):
                raise ValueError("Invalid arm")
            xyz = np.asarray(args["xyz"], dtype=float)
            rpy = np.asarray(args["rpy"], dtype=float)
            if (
                xyz.shape != (3,)
                or rpy.shape != (3,)
                or not np.isfinite(xyz).all()
                or not np.isfinite(rpy).all()
            ):
                raise ValueError("IK xyz/rpy must be finite three-element vectors")
            target = np.eye(4)
            target[:3, :3] = Rotation.from_euler("xyz", rpy, degrees=True).as_matrix()
            target[:3, 3] = xyz
            reference = args.get("reference_action")
            reference = self._command if reference is None else validate_action(reference)
            if name == "solve_pose_continuous":
                from .continuous import solve_continuous

                return solve_continuous(
                    self._inverse,
                    target,
                    command_angles(reference, self._task_variant, self._camera_profile),
                    arm,
                )
            if hasattr(self._inverse, "diagnose"):
                report = self._inverse.diagnose(
                    target,
                    command_angles(reference, self._task_variant, self._camera_profile),
                    arm,
                )
                report["trace_id"] = self._run + ":ik:" + uuid.uuid4().hex[:12]
                self._last_ik_diagnostic = report
                requested = {
                    "arm": arm,
                    "xyz": list(args["xyz"]),
                    "rpy": list(args["rpy"]),
                    "reference_action": np.asarray(reference).tolist(),
                }
                self._last_ik_request = requested
                if name == "ik_diagnose" or not report["ok"]:
                    self._record_ik_diagnostic(name, requested, report)
                if name == "ik_diagnose":
                    return report
                if not report["ok"]:
                    reasons = sorted(
                        {attempt.get("failure", "unknown") for attempt in report["attempts"]}
                    )
                    raise ValueError(
                        "Bounded model IK failed: "
                        + ", ".join(reasons)
                        + "; inspect robo.last_ik_diagnostic() or robo.ik_diagnose(). "
                        "This is not proof of unreachable geometry."
                    )
                q = report["joint_target_rad"]
            else:
                q = self._inverse.solve(
                    target,
                    command_angles(reference, self._task_variant, self._camera_profile),
                    arm,
                )
            return {"joint_target_rad": q, "source": "static model IK only; no action executed"}
        if name == "camera_model":
            return self._camera_model(args["camera"])
        if name == "project":
            model = self._camera_model(args["camera"])
            u, v = args["col"], args["row"]
            pose = np.asarray(model["optical_pose_from_command"])
            ray = pose[:3, :3] @ np.linalg.solve(np.asarray(model["intrinsic"]), [u, v, 1.0])
            if abs(ray[2]) < 1e-7:
                raise ValueError("Pixel ray is parallel to assumed plane")
            scale = (float(args["height"]) - pose[2, 3]) / ray[2]
            if scale <= 0:
                raise ValueError("Requested plane is behind camera")
            return {
                "estimated_xyz": (pose[:3, 3] + scale * ray).tolist(),
                "source": "RGB ray + explicit plane prior",
            }
        if name == "fk":
            reference = args.get("reference_action")
            reference = self._command if reference is None else validate_action(reference)
            pose = self._description.forward(
                command_angles(reference, self._task_variant, self._camera_profile),
            )[args["arm"] + "_tool0"]
            return {
                "command_fk": pose.tolist(),
                "source": "model of issued/planned commands; NOT measured robot state",
            }
        raise ValueError("Unknown RGB primitive: " + name)

    def _record_ik_diagnostic(self, tool, requested, report):
        """Host audit of static calculations; no scene state is available here."""
        with (self._out / "ik_diagnostics.jsonl").open("a") as handle:
            handle.write(
                json.dumps(
                    {
                        "tool": tool,
                        "trace_id": report["trace_id"],
                        "requested": requested,
                        "diagnostic": report,
                    }
                )
                + "\n"
            )

    def _move(self, arm, xyz, rpy=None):
        if arm not in ("left", "right"):
            raise ValueError("arm must be left/right")
        current = command_angles(self._command, self._task_variant, self._camera_profile)
        old = self._description.forward(current)[arm + "_tool0"]
        target = old.copy()
        target[:3, 3] = xyz
        if rpy is not None:
            target[:3, :3] = Rotation.from_euler("xyz", rpy, degrees=True).as_matrix()
        offset = 0 if arm == "left" else 8
        angle = Rotation.from_matrix(old[:3, :3] @ target[:3, :3].T).magnitude()
        if angle < 0.005:
            plan = linear_trajectory(self._inverse, target, current, arm)
            points = np.asarray(plan["joint_positions_rad"])
            times = np.arange(len(points)) * plan["sample_dt_s"]
            samples = np.arange(0.05, plan["duration_s"] + 0.05, 0.05)
            joints = np.column_stack([np.interp(samples, times, points[:, j]) for j in range(7)])
        else:
            answer = self._inverse.solve(target, current, arm)
            end = np.array(
                [answer[f"Joint{i}_{'L' if arm == 'left' else 'R'}"] for i in range(1, 8)]
            )
            start = self._command[offset : offset + 7]
            count = max(40, math.ceil(1.875 * max(abs(end - start)) / 0.4 * 20))
            t = np.linspace(0, 1, count + 1)[1:]
            s = 10 * t**3 - 15 * t**4 + 6 * t**5
            joints = start + s[:, None] * (end - start)
        actions = np.tile(self._command, (len(joints), 1))
        actions[:, offset : offset + 7] = joints
        return self._act(actions)

    def _camera_model(self, camera):
        if camera not in CAMERAS:
            raise ValueError("Unknown camera")
        if self._head_center_task:
            from tianji.sample_calibration import camera_model

            return camera_model(
                self._description, self._command, camera, camera_profile=self._camera_profile
            )
        name = {
            "head": "cam_high_color_optical_frame",
            "left_wrist": "cam_left_wrist_color_optical_frame",
            "right_wrist": "cam_right_wrist_color_optical_frame",
        }[camera]
        pose = self._description.forward(
            command_angles(self._command, self._task_variant, self._camera_profile),
        )[name]
        focal = 21 if camera == "head" else 28.75
        f = focal / 30 * 640
        return {
            "intrinsic": [[f, 0, 320], [0, f, 240], [0, 0, 1]],
            "optical_pose_from_command": pose.tolist(),
            "source": "fixed model calibration and own action history; NOT measured pose",
        }

    def _detect(self, camera, color):
        rgb = np.asarray(Image.open(io.BytesIO(self.artifact(camera + "_rgb.png"))).convert("RGB"))
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        if color == "red":
            mask = (
                ((hsv[:, :, 0] < 12) | (hsv[:, :, 0] > 170))
                & (hsv[:, :, 1] > 85)
                & (hsv[:, :, 2] > 40)
            )
        elif color == "green":
            mask = (
                (hsv[:, :, 0] > 35)
                & (hsv[:, :, 0] < 90)
                & (hsv[:, :, 1] > 70)
                & (hsv[:, :, 2] > 40)
            )
        else:
            raise ValueError("Color must be red or green")
        binary = cv2.morphologyEx(
            mask.astype("uint8") * 255, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
        )
        count, _, stats, centres = cv2.connectedComponentsWithStats(binary)
        components = [
            {
                "bbox_xywh": list(map(int, stats[i, :4])),
                "area_px": int(stats[i, 4]),
                "center_uv": list(map(float, centres[i])),
            }
            for i in range(1, count)
            if stats[i, 4] >= 60
        ]
        return {
            "camera": camera,
            "color": color,
            "components": sorted(components, key=lambda r: -r["area_px"]),
            "source": "HSV segmentation of RGB pixels only",
        }

    def artifact(self, name, step=-1):
        if name == "diagnostic_external_rgb.png":
            if not self._diagnostic_views:
                raise ValueError("External diagnostic artifacts are disabled")
            if step != -1 or self._diagnostic_image is None:
                raise ValueError("Only the latest diagnostic RGB image is available")
            return self._diagnostic_image
        if name not in {c + "_rgb.png" for c in CAMERAS}:
            raise ValueError("Only RGB artifacts are exposed")
        camera = name.removesuffix("_rgb.png")
        return (self._images if step == -1 else self._history[int(step)])[camera]

    def env_steps(self):
        return self._count

    def solved(self):
        return self._success if self._closed else False

    def cancel(self):
        if not self._closed:
            folder = REMOTE + "/runs/tianji_native/" + self._run + "/ipc"
            self._ssh(f"from pathlib import Path\nPath({folder!r}, 'cancel_requested').touch()\n")
        self.conclude()

    def close(self):
        self.conclude()

    def conclude(self):
        with self._transport_lock:
            self._conclude_locked()

    def _conclude_locked(self):
        if self._closed:
            return
        folder = REMOTE + "/runs/tianji_native/" + self._run
        code = f"""from pathlib import Path
import json,time
p=Path({folder!r});ipc=p/'ipc'
f=ipc/'request_{self._serial:06d}.json'
if not (ipc/'finished.json').exists():f.write_text(json.dumps({{'operation':'finish'}}))
for _ in range(400):
 if (p/'sealed_verdict.json').exists():print((p/'sealed_verdict.json').read_text());break
 time.sleep(.1)
else:raise TimeoutError('Missing private verdict')
"""
        verdict = json.loads(self._ssh(code))
        self._success = bool(verdict["task_success"])
        if self._single_left_can:
            from .fixed_order import single_left_evidence

            _, contract = single_left_evidence(verdict)
            self._success = bool(self._success and contract["complete"])
            (self._out / "single_left_verdict.json").write_text(
                json.dumps(contract, indent=2) + "\n"
            )
        if self._fixed_arm_order:
            from .fixed_order import fixed_order_evidence

            _, contract = fixed_order_evidence(verdict)
            self._success = bool(self._success and contract["complete"])
            (self._out / "fixed_order_verdict.json").write_text(
                json.dumps(contract, indent=2) + "\n"
            )
        self._closed = True
        # This is host-only, outside the agent's confined workspace.
        (self._out / "sealed_verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
