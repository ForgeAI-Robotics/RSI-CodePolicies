"""PyRUA-Lean robot with a deliberately image-only public observation surface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from pyrualean._robot import RobotBase
from pyrualean.robots import RobotAdapter

from .contract import ACTION_NAMES, CAMERAS, initial_action


@dataclass(frozen=True)
class ActionResult:
    """Own issued command and count only; no encoder, contact or object state."""

    frames: int
    commanded_action: tuple[float, ...]


class TianjiRGBRobot(RobotBase):
    """Two 7-DoF arms and two grippers. RGB only, no proprioceptive observations.

    Native action order: left J1..J7, left opening, right J1..J7, right opening.
    Joints are absolute radians; openings 0=closed, 1=open. The static reset
    configuration and calibration are public. FK/camera poses use issued
    commands, never encoders. Moving one arm sends full 16D vectors holding
    the other arm's last command. Images are the only environmental feedback.
    """

    NAME = "tianji_rgb"
    CAMERAS: ClassVar = CAMERAS
    IMAGE_ARTIFACTS: ClassVar = {(c, "high"): c + "_rgb.png" for c in CAMERAS}
    ON_MOTION_IMAGES: ClassVar = tuple(c + "_rgb.png" for c in CAMERAS)
    ABSENT_MEMBERS: ClassVar = frozenset({"state", "done", "world_map"})
    STATEFUL_TOOLS: ClassVar = frozenset({"action16", "action_sequence", "move_to", "gripper"})
    RESULT_TYPES: ClassVar = (ActionResult,)

    def _state_from_envelope(self, envelope):
        return None

    def _summary_dict(self):
        return {}

    def _show_artifacts(self, cameras):
        ordinary = [c for c in cameras if c != "diagnostic_external"]
        artifacts = super()._show_artifacts(ordinary)
        if "diagnostic_external" in cameras:
            artifacts.append("diagnostic_external_rgb.png")
        return artifacts

    def diagnostic_view(self) -> dict:
        """Attach an external RGB observer image for the training author's diagnosis.

        Available only when the host explicitly enables training diagnostics.
        This is not one of the robot's three sensors. Never call it from the
        reusable policy: frozen validation disables this method and artifact.
        No depth, object pose, contact identities or evaluator data are returned.
        """
        result = self._readonly("diagnostic_view", {})
        if "diagnostic_external" not in self._show_requests:
            self._show_requests.append("diagnostic_external")
        return result

    def diagnostic_state(
        self, refresh: bool = False, include_scene: bool = False, include_depth: bool = False
    ) -> dict:
        """Training author only: actual joints/TCP, objects, contacts and scene evidence.

        Default returns cached evidence from the last action, including safety stops.
        Request refresh=True to obtain optional USD/depth; this advances no physics.
        Never call this API from policy.py or its helpers. Frozen execution rejects it.
        """
        return self._readonly("diagnostic_state", {
            "refresh": refresh, "include_scene": include_scene, "include_depth": include_depth,
        })

    def state(self, *args, **kwargs):
        raise AttributeError("Proprioception is not exposed")

    @property
    def done(self):
        raise AttributeError("Verdict is private until episode completion")

    def world_map(self, *args, **kwargs):
        raise AttributeError("No depth or world map is exposed")

    @property
    def task(self) -> str:
        """Task instruction; no target coordinates or hidden scene data."""
        if getattr(self._backend, "_head_center_task", False):
            return (
                "Use the left arm to place the can on the table within the central 20% "
                "of the head RGB image, then release and retreat. No green target is present."
            )
        if getattr(self._backend, "_single_left_can", False):
            return "Use the left arm to place the single red can upright on the central green mark."
        return "Place the two red cola cans upright on the two green circular tabletop marks."

    def initial_action(self) -> tuple[float, ...]:
        """Static reset action, not a measurement: both arms straight forward, grippers open."""
        return tuple(getattr(self._backend, "_initial_command", initial_action()))

    def action_names(self) -> tuple[str, ...]:
        """Names and ordering of the only 16 actuator channels."""
        return ACTION_NAMES

    def action(self, action: list[float], repeat: int = 1) -> ActionResult:
        """Send a 16D absolute action for repeat 20-Hz control ticks. No other command channel."""
        return self._stateful(
            "action16",
            {"action": action, "repeat": repeat},
            lambda p: ActionResult(p["frames"], tuple(p["commanded_action"])),
        )

    def move_to(self, arm: str, xyz: list[float], rpy: list[float] | None = None) -> ActionResult:
        """Generate 16D commands from model IK; xyz metres, optional XYZ Euler degrees.

        Uses own command history, not measured joints. Near-constant orientation
        uses vendor MOVL; orientation changes use smooth joint interpolation.
        This primitive does not know where objects or placement targets are.
        """
        return self._stateful(
            "move_to",
            {"arm": arm, "xyz": xyz, "rpy": rpy},
            lambda p: ActionResult(p["frames"], tuple(p["commanded_action"])),
        )

    def action_sequence(self, actions: list[list[float]]) -> ActionResult:
        """Batch an ordered sequence of 16D commands at the same 20Hz control rate.

        Every row is the same action format as action(). Validation happens
        before sending. This preserves every target, order and physics tick;
        images become available after a batch, rather than after every row.
        Use batches only where no image feedback is needed between rows.
        """
        return self._stateful(
            "action_sequence",
            {"actions": actions},
            lambda p: ActionResult(p["frames"], tuple(p["commanded_action"])),
        )

    def gripper(self, arm: str, opening: float) -> ActionResult:
        """Issue an opening target, 0 closed to 1 open, through 16D actions; no grasp status."""
        if arm not in ("left", "right"):
            raise ValueError("arm must be left/right")
        return self._stateful(
            "gripper",
            {"arm": arm, "opening": opening},
            lambda p: ActionResult(p["frames"], tuple(p["commanded_action"])),
        )

    def detect(self, camera: str, color: str) -> dict:
        """HSV components (bbox, centre, area) from red/green RGB pixels; no simulator labels."""
        return self._readonly("detect", {"camera": camera, "color": color})

    def rim(
        self, camera: str, expected_xyz: list[float], plane_height: float, radius: float = 0.029
    ) -> dict:
        """Estimate top-rim XY from RGB edges using caller-supplied priors.

        Canny pixels near the expected object estimate are projected to plane_height and
        fitted as circles by RANSAC/robust least squares. expected_xyz is YOUR
        coarse object-centre estimate, not a scene query. Returns XY, radial
        residual/coverage/radius; Z is assumed, never observed. No object IDs.
        """
        return self._readonly(
            "rim",
            {
                "camera": camera,
                "expected_xyz": expected_xyz,
                "plane_height": plane_height,
                "radius": radius,
            },
        )

    def stereo_rim_pose(
        self, expected_xyz: list[float], radius: float = .029,
        height: float = .1454, height_spread: float = .06,
    ) -> dict:
        """Fit a tilted circular rim jointly in head/left-wrist RGB using size priors.

        Returns estimated centre/axis and pixel support, never scene truth/depth.
        Rejects unsupported or ambiguous solutions; calibration remains a prior.
        """
        return self._readonly("stereo_rim_pose", {
            "expected_xyz": expected_xyz, "radius": radius, "height": height,
            "height_spread": height_spread,
        })

    def camera_model(self, camera: str) -> dict:
        """Static intrinsic calibration and estimated optical pose from own command history."""
        return self._readonly("camera_model", {"camera": camera})

    def project(self, camera: str, row: float, col: float, height: float) -> dict:
        """Intersect a colour pixel ray with an explicitly assumed horizontal plane.

        height is a policy-provided prior, not observed depth. Occlusion and
        pixel choice can bias the estimate. Use the RGB images to verify it.
        """
        return self._readonly(
            "project", {"camera": camera, "row": row, "col": col, "height": height}
        )

    def fk(self, arm: str, reference_action: list[float] | None = None) -> dict:
        """Model FK of own commands or a supplied planned 16D vector; never measured."""
        return self._readonly("fk", {"arm": arm, "reference_action": reference_action})

    def gripper_geometry(self, arm: str) -> dict:
        """Static URDF collision-mesh bounds in tool0 at several openings, metres.

        Includes opening-dependent finger retraction and wrist/camera casing.
        This is model geometry, not sensor data or actual joint configuration.
        """
        return self._readonly("gripper_geometry", {"arm": arm})

    def collision_check(
        self, actions: list[list[float]], obstacles: list[list[float]], arm: str,
        selected: int | None = None, allow_grasp_contact: bool = False,
        carried: dict | None = None, margin: float = 0.012,
        keepout_boxes: list[dict] | None = None,
    ) -> dict:
        """Read-only preflight using YOUR RGB object centers and static URDF.

        Checks sampled joint paths, both arms/grippers, unselected cylinders and
        an optional carried cylinder. Only active follower pads may touch the
        selected object. No scene/encoder/contact/evaluator reads occur.
        Convex envelopes can reject valid paths; finite sampling is not a proof
        of collision freedom. Native guards remain authoritative.
        """
        return self._readonly("collision_check", {
            "actions": actions, "obstacles": obstacles, "arm": arm,
            "selected": selected, "allow_grasp_contact": allow_grasp_contact,
            "carried": carried, "margin": margin,
            "keepout_boxes": keepout_boxes,
        })

    def solve_pose(
        self,
        arm: str,
        xyz: list[float],
        rpy: list[float],
        reference_action: list[float] | None = None,
    ) -> dict:
        """Check model IK without motion; optional 16D planned reference selects branch.

        Defaults to own last issued command, never measured state. Pass the
        preceding planned vector when preflighting a future waypoint sequence.
        """
        return self._readonly(
            "solve_pose", {"arm": arm, "xyz": xyz, "rpy": rpy, "reference_action": reference_action}
        )

    def solve_pose_continuous(
        self, arm: str, xyz: list[float], rpy: list[float],
        reference_action: list[float] | None = None,
    ) -> dict:
        """Bounded local static-model IK with sampled torso clearance; no motion.

        Uses own commands and fixed robot geometry, never encoders/object truth.
        A rejection is not a proof of unreachability. Native guards still apply.
        """
        return self._readonly("solve_pose_continuous", {
            "arm": arm, "xyz": xyz, "rpy": rpy, "reference_action": reference_action,
        })

    def ik_diagnose(
        self,
        arm: str,
        xyz: list[float],
        rpy: list[float],
        reference_action: list[float] | None = None,
    ) -> dict:
        """Explain bounded static IK attempts, without motion or simulator feedback.

        Returns ok, solution if accepted, and per-attempt target/reference,
        SDK flags, adapter/source URDF joint limits and FK consistency errors.
        A failure does not prove unreachability; success does not certify a
        collision-free path. Inputs are your target and own command history.
        """
        return self._readonly(
            "ik_diagnose", {
                "arm": arm, "xyz": xyz, "rpy": rpy, "reference_action": reference_action,
            }
        )

    def last_ik_diagnostic(self) -> dict:
        """Return the most recent model IK report without rerunning search; initially None."""
        return self._readonly("last_ik_diagnostic", {})

    def grasp_candidates(
        self,
        arm: str,
        centres: list[list[float]],
        radius: float = 0.029,
        height: float = 0.1454,
        table_height: float = 0.75,
        clearance: float = 0.10,
        limit: int = 5,
    ) -> dict:
        """Read-only common-orientation grasp search from YOUR RGB XYZ centre estimates.

        centres are cylinder centres (not bottoms), e.g. estimated pick and
        placement centres. Samples static hand meshes against the assumed
        cylinder/table and checks SDK IK. Each candidate gives above/contact
        tool XYZ pairs in centres order and XYZ Euler degrees. No scene state,
        motion, attachment or grasp-status query. No swept-arm safety guarantee.
        """
        return self._readonly(
            "grasp_candidates",
            {
                "arm": arm,
                "centres": centres,
                "radius": radius,
                "height": height,
                "table_height": table_height,
                "clearance": clearance,
                "limit": limit,
            },
        )


ADAPTER = RobotAdapter(
    name="tianji_rgb",
    robot_cls=TianjiRGBRobot,
    host="tianji.rgb.host",
    knowledge="tianji_rgb",
    blurb="an RGB-only Tianji dual-arm robot at a meeting table",
    show_examples='`robo.show("head", "left_wrist", "right_wrist")`',
    services_novla="model-based kinematics and RGB colour segmentation; no VLA",
)
