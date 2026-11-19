"""Single-can Code as Policy: RGB rim, model trajectories, boundary feedback."""

import math

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

HEAD_CENTER_TASK = True
SINGLE_LEFT_CAN = True
WRIST_CAMERA_PROFILE = "sample_wrist_v1"
TABLE_HEIGHT = 0.75
CAN_HEIGHT = 0.1454
CAN_RADIUS = 0.029
# Static visual lid geometry differs from the collision cylinder's maximum height.
LID_PLANE_DROP = 0.004


class RGBPolicy:
    def __init__(self, robo):
        self.robo = robo
        self.command = np.asarray(robo.initial_action(), dtype=float)
        self.state = {}
        self.completed = []
        self.stage = "initialize_scene"
        self.ticks = 0

    def send(self, actions):
        if not actions:
            return
        if self.ticks + len(actions) > 4200:
            raise ValueError("Motion budget exhausted")
        self.command = np.asarray(
            self.robo.action_sequence([list(map(float, row)) for row in actions]).commanded_action
        )
        self.ticks += len(actions)

    def hold(self, ticks=12):
        self.send([self.command.copy() for _ in range(ticks)])

    def grip(self, opening):
        start = self.command.copy()
        actions = []
        for phase in np.linspace(0, 1, 31)[1:]:
            blend = phase**3 * (10 - 15 * phase + 6 * phase**2)
            target = start.copy()
            target[7] = start[7] + blend * (opening - start[7])
            actions.append(target)
        self.send(actions)
        self.hold(10)

    def solve(self, point, reference, continuous=False):
        solve = self.robo.solve_pose_continuous if continuous else self.robo.solve_pose
        solution = solve(
            "left", list(map(float, point)), self.state["rpy"], reference.tolist()
        )["joint_target_rad"]
        target = reference.copy()
        target[:7] = [solution[f"Joint{i}_L"] for i in range(1, 8)]
        if target[3] < -2.0:
            raise ValueError("Elbow fold exceeds conservative self-contact margin")
        achieved = np.asarray(self.robo.fk("left", target.tolist())["command_fk"])[:3, 3]
        if np.linalg.norm(achieved - point) > 0.003:
            raise ValueError("Model IK endpoint residual")
        return target

    def line_plan(self, point, reference):
        start = np.asarray(self.robo.fk("left", reference.tolist())["command_fk"])[:3, 3]
        pieces = max(1, int(math.ceil(np.linalg.norm(point - start) / 0.01)))
        actions = []
        previous = reference.copy()
        for fraction in np.linspace(0, 1, pieces + 1)[1:]:
            target = self.solve(start + fraction * (point - start), previous, continuous=True)
            ticks = max(4, int(math.ceil(np.max(np.abs(target[:7] - previous[:7])) / 0.012)))
            for blend in np.linspace(0, 1, ticks + 1)[1:]:
                actions.append(previous + blend * (target - previous))
            previous = target
        return actions

    def line(self, point):
        self.send(self.line_plan(np.asarray(point), self.command))
        self.hold(6)

    def transition(self, point):
        target = self.solve(point, self.command)
        start = self.command.copy()
        ticks = max(60, int(math.ceil(np.max(abs(target[:7] - start[:7])) * 1.875 / 0.018)))
        rows = []
        for fraction in np.linspace(0, 1, ticks + 1)[1:]:
            blend = fraction**3 * (10 - 15 * fraction + 6 * fraction**2)
            rows.append(start + blend * (target - start))
        self.send(rows)
        self.hold(8)

    def red(self):
        image = np.asarray(self.robo.image("head"))
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = (
            ((hsv[:, :, 0] < 12) | (hsv[:, :, 0] > 170)) & (hsv[:, :, 1] > 85) & (hsv[:, :, 2] > 40)
        ).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        count, _, stats, centers = cv2.connectedComponentsWithStats(mask)
        regions = []
        for index in range(1, count):
            x, y, width, height, area = stats[index]
            if area >= 150 and width >= 10 and height >= 20:
                regions.append(
                    dict(
                        center=centers[index],
                        area=int(area),
                        bounds=np.array([x, y, width, height]),
                    )
                )
        if not regions:
            raise ValueError("No supported red can in head RGB")
        return max(regions, key=lambda item: item["area"])

    def refine_can(self, expected):
        fits = self.robo.rim(
            "head", expected.tolist(), TABLE_HEIGHT + CAN_HEIGHT - LID_PLANE_DROP, CAN_RADIUS
        )["fits"]
        valid = [
            f
            for f in fits
            if f["angular_bins_of_24"] >= 10
            and f["radial_rms_m"] < 0.004
            and np.linalg.norm(np.asarray(f["estimated_xy"]) - expected[:2]) < 0.06
        ]
        if not valid:
            raise ValueError("Can top rim lacks RGB support")
        best = max(valid, key=lambda fit: fit["score"])
        return np.array([*best["estimated_xy"], TABLE_HEIGHT + CAN_HEIGHT / 2])

    def initialize_scene(self):
        self.state["initial_blob"] = self.red()
        self.state["geometry"] = self.robo.gripper_geometry("left")

    def perceive_select(self):
        blob = self.state["initial_blob"]
        u, v = blob["center"]
        coarse = np.asarray(
            self.robo.project("head", float(v), float(u), TABLE_HEIGHT + CAN_HEIGHT / 2)[
                "estimated_xyz"
            ]
        )
        self.state["pickup"] = self.refine_can(coarse)
        image = np.asarray(self.robo.image("head"))
        height, width = image.shape[:2]
        failures = []
        # Candidate poses are model priors; no scene pose or encoder is read.
        for rpy, reach, z_offset in (
            ([180, -25, -80], 0.195, 0.020),
            ([180, -25, -65], 0.195, 0.020),
            ([180, -25, -95], 0.195, 0.020),
            ([0, 120, -60], 0.175, 0.0103),
            ([0, 90, -45], 0.175, 0.0103),
        ):
            self.state["rpy"] = rpy
            axis = Rotation.from_euler("xyz", rpy, degrees=True).as_matrix()[:, 2]
            contact = self.state["pickup"] + [0, 0, z_offset] - reach * axis
            above = contact + [0, 0, 0.15]
            for col, row in ((0.5, 0.48), (0.43, 0.56), (0.5, 0.54), (0.43, 0.43), (0.43, 0.58)):
                goal = np.asarray(
                    self.robo.project(
                        "head", height * row, width * col, TABLE_HEIGHT + CAN_HEIGHT / 2
                    )["estimated_xyz"]
                )
                placed = goal + [0, 0, z_offset] - reach * axis
                above_drop = placed + [0, 0, 0.15]
                try:
                    reference = self.solve(above, self.command)
                    withdrawal = placed - 0.09 * axis
                    for point in (
                        contact, above, above_drop, placed + [0, 0, 0.008],
                        placed, withdrawal, withdrawal + [0, 0, 0.10],
                    ):
                        reference = self.line_plan(np.asarray(point), reference)[-1]
                except ValueError as error:
                    failures.append(str(error))
                    continue
                self.state.update(
                    axis=axis,
                    reach=reach,
                    z_offset=z_offset,
                    goal=goal,
                    contact=contact,
                    above=above,
                    placed=placed,
                    above_drop=above_drop,
                )
                return
        raise ValueError("No model-feasible grasp/target family: " + str(failures[-2:]))

    def approach(self):
        self.transition(self.state["above"])
        self.grip(1.0)

    def align_grasp(self):
        self.line(self.state["above"])
        planned = self.line_plan(self.state["contact"], self.command)
        actions = []
        previous = self.command.copy()
        for target in planned:
            for fraction in (1 / 3, 2 / 3, 1.0):
                actions.append(previous + fraction * (target - previous))
            previous = target
        self.send(actions)
        self.hold(12)

    def visible_near(self, position):
        blob = self.red()
        model = self.robo.camera_model("head")
        optical = np.asarray(model["optical_pose_from_command"])
        point = optical[:3, :3].T @ (np.asarray(position) - optical[:3, 3])
        pixel = np.asarray(model["intrinsic"]) @ point
        if point[2] <= 0:
            raise ValueError("Expected can is behind the camera")
        distance = np.linalg.norm(blob["center"] - pixel[:2] / pixel[2])
        if distance > max(18, 0.40 * blob["bounds"][3]):
            raise ValueError("RGB can is inconsistent with the motion endpoint")
        return blob

    def close_lift(self):
        before = self.red()
        self.grip(0.0)
        self.hold(12)
        self.line(self.state["above"])
        after = self.visible_near(self.state["pickup"] + [0, 0, 0.15])
        if before["center"][1] - after["center"][1] < 15:
            raise ValueError("RGB can did not visibly rise")

    def transport_align(self):
        self.line(self.state["above_drop"])
        self.visible_near(self.state["goal"] + [0, 0, 0.156])

    def place_release(self):
        s = self.state
        self.line(s["placed"] + [0, 0, 0.008])
        self.line(s["placed"])
        self.hold(20)
        # Map the commissioned 0.264 rad unload to our 0.28 rad driver profile.
        # First support/unload, then withdraw, rise, and only then fully open.
        self.grip(0.95)
        withdrawal = s["placed"] - 0.09 * s["axis"]
        self.line(withdrawal)
        self.line(withdrawal + [0, 0, 0.10])
        self.grip(1.0)
        self.hold(20)

    def verify_placement(self):
        blob = self.red()
        height, width = np.asarray(self.robo.image("head")).shape[:2]
        u, v = blob["center"]
        if abs(u - width / 2) > width * 0.1 or abs(v - height / 2) > height * 0.1:
            coarse = np.asarray(
                self.robo.project(
                    "head", float(v), float(u), TABLE_HEIGHT + CAN_HEIGHT / 2
                )["estimated_xyz"]
            )
            centre = self.refine_can(coarse)
            model = self.robo.camera_model("head")
            optical = np.asarray(model["optical_pose_from_command"])
            point = optical[:3, :3].T @ (centre - optical[:3, 3])
            if point[2] <= 0:
                raise ValueError("Released can is behind the camera")
            pixel = np.asarray(model["intrinsic"]) @ point
            column, row = pixel[:2] / pixel[2]
            if abs(column - width / 2) > width * 0.1 or abs(row - height / 2) > height * 0.1:
                raise ValueError("Released can is outside head-image centre")

    def finalize(self):
        self.hold(20)

    def execute(self):
        dispatch = {
            "initialize_scene": self.initialize_scene,
            "perceive_select": self.perceive_select,
            "approach": self.approach,
            "align_grasp": self.align_grasp,
            "close_lift": self.close_lift,
            "transport_align": self.transport_align,
            "place_release": self.place_release,
            "verify_placement": self.verify_placement,
            "finalize": self.finalize,
        }
        stages = (
            "initialize_scene",
            "perceive_select",
            "approach",
            "align_grasp",
            "close_lift",
            "transport_align",
            "place_release",
            "verify_placement",
            "finalize",
        )
        try:
            for stage in stages:
                self.stage = stage
                dispatch[stage]()
                self.completed.append(
                    dict(
                        subtask=stage,
                        can_index=None if stage in ("initialize_scene", "finalize") else 0,
                    )
                )
            return dict(
                status="visually_complete",
                completed_subtasks=self.completed,
                failed_subtask=None,
                can_index=None,
            )
        except (ValueError, RuntimeError) as error:
            return dict(
                status="stuck",
                completed_subtasks=self.completed,
                failed_subtask=self.stage,
                can_index=0,
                reason=str(error),
            )


def policy(robo):
    return RGBPolicy(robo).execute()
