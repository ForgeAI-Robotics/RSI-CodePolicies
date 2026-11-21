"""Cleanup learning scaffold: preserved motion helpers, no replay or scene reads."""
import math
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

TABLE_CLEANUP_TASK = True
HEAD_CENTER_TASK = True
SINGLE_LEFT_CAN = True
WRIST_CAMERA_PROFILE = "sample_wrist_v1"

CLOSURE_SAMPLES = [[0.9996479012345679, 1.329059505034147e-05], [0.9973254320987655, 8.968477876636349e-05], [0.99144, 0.0002854433257883926], [0.9807841975308642, 0.0006363824466645851], [0.9645061728395061, 0.0011740932991261], [0.94208, 0.001909110983832507], [0.913276049382716, 0.0028472642138088755], [0.8781313580246913, 0.003983017376429976], [0.83692, 0.005296518778478754], [0.7901234567901235, 0.006766843171727599], [0.738400987654321, 0.008362913855362919], [0.68256, 0.010050615597927248], [0.6235264197530863, 0.011796500462211701], [0.562315061728395, 0.01355697248083848], [0.5, 0.015297054407256039], [0.43768493827160493, 0.016982202134981972], [0.3764735802469137, 0.01858180850986363], [0.31744000000000017, 0.02007199408242133], [0.26159901234567917, 0.021428070328872728], [0.2098765432098768, 0.02263901912577687], [0.16308000000000034, 0.0236964580668374], [0.12186864197530911, 0.024596203474234635], [0.08672395061728444, 0.02534136435289014], [0.057919999999999416, 0.02593464891222691], [0.03549382716049343, 0.026385768746511162], [0.019215802469135568, 0.026708308648015266], [0.008559999999999679, 0.026915451399382784], [0.002674567901234548, 0.02702985394849472], [0.000352098765432296, 0.027074277958897164], [0.0, 0.027081993922208336]]

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
        initial = np.asarray(self.robo.initial_action())
        if any(not np.array_equal(np.asarray(row)[8:], initial[8:]) for row in actions):
            raise ValueError("Right actuator channels must stay at reset")
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

    def plane_pixels(self, pixels, camera, height):
        model = self.robo.camera_model(camera)
        intrinsic = np.asarray(model["intrinsic"])
        pose = np.asarray(model["optical_pose_from_command"])
        pixels = np.asarray(pixels).reshape(-1, 2)
        rays = np.column_stack([pixels, np.ones(len(pixels))]) @ np.linalg.inv(intrinsic).T @ pose[:3, :3].T
        scales = (height - pose[2, 3]) / rays[:, 2]
        if np.any(scales <= 0):
            raise ValueError("Plane lies behind camera")
        return pose[:3, 3] + rays * scales[:, None]

    def locate_cup(self, height=0.8564):
        image = self.robo.image("head")
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        green = self.robo.detect("head", "green")["components"]
        if not green:
            raise ValueError("Cup straw not visible")
        straw = max(green, key=lambda item: item["area_px"])
        left, top, width, length = straw["bbox_xywh"]
        anchor = np.array([left + width / 2, top + length])
        mask = cv2.inRange(hsv, np.array([16, 20, 160]), np.array([32, 70, 255]))
        contours, unused = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        candidates = []
        for contour in contours:
            if not 200 < cv2.contourArea(contour) < 15000:
                continue
            bounds = cv2.boundingRect(contour)
            center = np.array(bounds[:2]) + np.array(bounds[2:]) / 2
            if np.linalg.norm(center - anchor) > 65:
                continue
            points = self.plane_pixels(cv2.convexHull(contour), "head", height)[:, :2]
            fit = least_squares(lambda candidate: np.linalg.norm(points - candidate[:2], axis=1) - candidate[2],
                                np.r_[np.mean(points, axis=0), 0.0368], loss="soft_l1", f_scale=0.002)
            residual = float(np.sqrt(np.mean(fit.fun ** 2)))
            if residual < 0.005 and 0.028 < fit.x[2] < 0.060:
                candidates.append((residual, fit.x[:2]))
        if not candidates:
            raise ValueError("No supported cup lid contour")
        residual, center = min(candidates, key=lambda item: item[0])
        return np.array([center[0] - 0.024, center[1] - 0.0102, height]), residual

    def restore_initial(self):
        start = self.command.copy()
        target = np.asarray(self.robo.initial_action())
        ticks = max(80, int(math.ceil(np.max(abs(target[:7] - start[:7])) * 1.875 / 0.018)))
        rows = []
        for phase in np.linspace(0, 1, ticks + 1)[1:]:
            blend = phase ** 3 * (10 - 15 * phase + 6 * phase ** 2)
            rows.append(start + blend * (target - start))
        self.send(rows)
        self.hold(10)

    def scan_bin(self):
        self.state["rpy"] = [180, 15, 100]
        self.transition(np.array([0.20, 0.55, 1.00]))
        self.hold(20)
        self.line(np.array([0.08, 0.65, 1.00]))
        self.hold(12)
        self.state["bin"] = self.locate_bin()
        self.restore_initial()

    def locate_bin(self):
        image = self.robo.image("left_wrist")
        gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), (5, 5), 0)
        model = self.robo.camera_model("left_wrist")
        pose = np.asarray(model["optical_pose_from_command"])
        intrinsic = np.asarray(model["intrinsic"])
        angles = np.linspace(0, 2 * np.pi, 96, endpoint=False)
        def score(center, evidence=False):
            samples = []
            validity = []
            for radius in [0.07, 0.115, 0.128, 0.142, 0.155]:
                xyz = np.column_stack([center[0] + radius * np.cos(angles),
                                       center[1] + radius * np.sin(angles), np.full(len(angles), 0.276)])
                optical = (xyz - pose[:3, 3]) @ pose[:3, :3]
                pixels = optical @ intrinsic.T
                pixels = pixels[:, :2] / pixels[:, 2:]
                valid = (pixels[:, 0] > 1) & (pixels[:, 0] < gray.shape[1] - 2) & (pixels[:, 1] > 1) & (pixels[:, 1] < gray.shape[0] - 2)
                values = cv2.remap(gray, pixels[:, 0].astype(np.float32).reshape(1, -1),
                                   pixels[:, 1].astype(np.float32).reshape(1, -1), cv2.INTER_LINEAR).ravel().astype(float)
                samples.append(values)
                validity.append(valid)
            valid = np.all(validity, axis=0)
            if valid.sum() < 45:
                return -999.0
            contrast = np.minimum(samples[3], samples[4]) - np.minimum(samples[1], samples[2])
            if evidence:
                return {"visible_angles": int(valid.sum()),
                        "positive_fraction": float(np.mean(contrast[valid] > 5))}
            return float(np.mean(np.clip(contrast[valid], -20, 30)) - 0.25 * np.mean(samples[0][valid]))
        pixels = [[column, row] for row in range(30, image.shape[0] - 30, 30)
                  for column in range(30, image.shape[1] - 30, 30)]
        seeds = self.plane_pixels(pixels, "left_wrist", 0.276)[:, :2]
        center = max(seeds, key=score)
        for spacing in [0.01, 0.002]:
            seeds = [center + np.array([dx, dy]) for dx in np.arange(-5, 6) * spacing
                     for dy in np.arange(-5, 6) * spacing]
            center = max(seeds, key=score)
        confidence = score(center)
        if confidence < 5:
            raise ValueError("No dark circular bin with supported exterior contrast")
        fits = self.robo.rim("left_wrist", np.r_[center, 0.276].tolist(), 0.276, 0.13)["fits"]
        fits = [fit for fit in fits if fit["angular_bins_of_24"] >= 12 and fit["radial_rms_m"] < 0.003
                and np.linalg.norm(np.array(fit["estimated_xy"]) - center) < 0.025
                and score(fit["estimated_xy"]) > confidence - 2]
        if fits:
            center = np.asarray(min(fits, key=lambda item: item["radial_rms_m"])["estimated_xy"])
        evidence = score(center, evidence=True)
        self.state["bin_fit"] = {"contrast_score": confidence, "rim_supported": bool(fits),
                                 "annular_supported": confidence > 8 and evidence["visible_angles"] >= 60
                                 and evidence["positive_fraction"] >= 0.65, **evidence}
        self.state["bin_scan_image"] = image.copy()
        return np.r_[center, 0.276]

    def approach_table_clearance(self, actions):
        geometry = self.robo.gripper_geometry("left")["openings"]
        minimum = float("inf")
        for action in actions:
            opening = float(action[7])
            lower = max((sample for sample in geometry if sample["opening"] <= opening), key=lambda item: item["opening"])
            upper = min((sample for sample in geometry if sample["opening"] >= opening), key=lambda item: item["opening"])
            fraction = (opening - lower["opening"]) / max(upper["opening"] - lower["opening"], 1e-9)
            corners = []
            for name, bounds in lower["links"].items():
                low = np.asarray(bounds["min_xyz"]) * (1 - fraction) + np.asarray(upper["links"][name]["min_xyz"]) * fraction - 0.001
                high = np.asarray(bounds["max_xyz"]) * (1 - fraction) + np.asarray(upper["links"][name]["max_xyz"]) * fraction + 0.001
                corners.extend([[horizontal, lateral, vertical] for horizontal in [low[0], high[0]]
                                for lateral in [low[1], high[1]] for vertical in [low[2], high[2]]])
            transform = np.asarray(self.robo.fk("left", list(action))["command_fk"])
            height = float(np.min(np.asarray(corners) @ transform[2, :3] + transform[2, 3]))
            minimum = min(minimum, height)
            if height < 0.762:
                raise ValueError("Opening-conditioned wrist/hand violates table clearance")
        self.state["approach_minimum_hand_height"] = minimum

    def approach_plan(self):
        body = self.state["cup"].copy()
        body[2] = 0.795
        failures = []
        for above_shift in [0.0, 0.04, 0.08, 0.12]:
            self.state["rpy"] = [180, 60, 135]
            axis = Rotation.from_euler("xyz", self.state["rpy"], degrees=True).as_matrix()[:, 2]
            horizontal = axis.copy()
            horizontal[2] = 0
            horizontal /= np.linalg.norm(horizontal)
            contact = body - 0.132471491950665 * axis
            precontact = contact - 0.10 * horizontal
            above = precontact + np.array([above_shift, 0, 0.15])
            try:
                target = self.solve(above, self.command)
                ticks = max(60, int(math.ceil(np.max(abs(target[:7] - self.command[:7])) * 1.875 / 0.018)))
                transition = []
                for fraction in np.linspace(0, 1, ticks + 1)[1:]:
                    blend = fraction**3 * (10 - 15 * fraction + 6 * fraction**2)
                    transition.append(self.command + blend * (target - self.command))
                descent = self.line_plan(precontact, target)
                entry = self.line_plan(contact, descent[-1])
                lift_reference = entry[-1].copy()
                lift_reference[7] = 1.0
                lift = self.line_plan(contact + np.array([0, 0, 0.15]), lift_reference)
                self.state["grasp_geometry"] = "reference_teacher_body"
                self.state["approach_above_shift_m"] = above_shift
            except ValueError as error:
                failures.append(str(error))
                continue
            self.state.update(axis=axis, horizontal=horizontal, contact=contact,
                              body_offset=body - contact)
            return transition, descent, entry
        raise ValueError("No preflighted body approach: " + "; ".join(failures))

    def straw_tip(self):
        image = self.robo.image("head")
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.array([35, 60, 25]), np.array([90, 255, 255]))
        count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
        eligible = [index for index in range(1, count) if stats[index, cv2.CC_STAT_AREA] > 25]
        if not eligible:
            raise ValueError("No visible straw tip")
        index = max(eligible, key=lambda item: stats[item, cv2.CC_STAT_AREA])
        rows, columns = np.nonzero(labels == index)
        top = np.min(rows)
        selected = rows <= top + 5
        return np.array([np.mean(columns[selected]), np.mean(rows[selected])])

    def check_stationary_cup(self, tolerance=55.0):
        movement = float(np.linalg.norm(self.straw_tip() - self.state["straw_reference"]))
        self.state["straw_displacement_px"] = movement
        if movement > tolerance:
            raise ValueError("Cup straw moved; approach/closure stopped")

    def approach(self):
        transition, descent, entry = self.approach_plan()
        self.state["entry_verified"] = False
        self.state["lid_reference"] = self.approach_lid_evidence()
        self.state["straw_reference"] = self.straw_tip()
        for segment in [transition, descent]:
            self.send(segment)
            self.hold(4)
            self.check_stationary_cup(3.0)
        for offset in range(0, len(entry), 5):
            self.send(entry[offset:offset + 5])
            self.check_stationary_cup()
        self.state["entry_verified"] = True

    def approach_lid_evidence(self):
        image = self.robo.image("head")
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        components = self.robo.detect("head", "green")["components"]
        if not components:
            raise ValueError("No visible straw for upright lid verification")
        straw = max(components, key=lambda item: item["area_px"])
        left, top, width, length = straw["bbox_xywh"]
        anchor = np.array([left + width / 2, top + length])
        mask = cv2.inRange(hsv, np.array([16, 20, 160]), np.array([32, 70, 255]))
        contours, unused = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        candidates = []
        for contour in contours:
            if not 200 < cv2.contourArea(contour) < 25000:
                continue
            bounds = cv2.boundingRect(contour)
            center = np.array(bounds[:2]) + np.array(bounds[2:]) / 2
            if np.linalg.norm(center - anchor) > 80:
                continue
            points = self.plane_pixels(cv2.convexHull(contour), "head", 0.8564)[:, :2]
            fit = least_squares(
                lambda candidate: np.linalg.norm(points - candidate[:2], axis=1) - candidate[2],
                np.r_[np.mean(points, axis=0), 0.0368], loss="soft_l1", f_scale=0.002)
            residual = float(np.sqrt(np.mean(fit.fun ** 2)))
            angles = np.arctan2(points[:, 1] - fit.x[1], points[:, 0] - fit.x[0])
            coverage = len(np.unique(np.floor((angles + np.pi) * 12 / np.pi).astype(int)))
            if residual < 0.006 and 0.028 < fit.x[2] < 0.080 and coverage >= 12:
                candidates.append((residual, float(fit.x[2])))
        if not candidates:
            raise ValueError("No broad, circular upright lid evidence")
        return min(candidates, key=lambda item: item[0])[1]

    def compensated_close(self, opening):
        samples = np.asarray(CLOSURE_SAMPLES)[::-1]
        start = self.command.copy()
        start_position = np.asarray(self.robo.fk("left", start.tolist())["command_fk"])[:3, 3]
        initial_shift = np.interp(start[7], samples[:, 0], samples[:, 1])
        previous = start.copy()
        actions = []
        for phase in np.linspace(0, 1, 31)[1:]:
            blend = phase**3 * (10 - 15 * phase + 6 * phase**2)
            aperture = start[7] + blend * (opening - start[7])
            shift = np.interp(aperture, samples[:, 0], samples[:, 1]) - initial_shift
            target = self.solve(start_position - shift * self.state["axis"], previous, continuous=True)
            target[7] = aperture
            actions.append(target)
            previous = target
        self.send(actions)
        self.hold(10)

    def close_and_lift(self):
        if not self.state.get("entry_verified", False):
            raise ValueError("Verified nondisplacing body entry required")
        self.compensated_close(0.0)
        self.line(self.state["contact"] + np.array([0, 0, 0.15]))
        self.hold(12)

    def verify_lid_rise(self, minimum):
        camera_height = np.asarray(self.robo.camera_model("head")["optical_pose_from_command"])[2, 3]
        try:
            radius = self.approach_lid_evidence()
            rise = (camera_height - 0.8564) * (1 - self.state["lid_reference"] / radius)
        except ValueError:
            tcp = np.asarray(self.robo.fk("left")["command_fk"])[:3, 3]
            expected = tcp + np.array([0.11184, -0.11260, -0.00640])
            fits = self.robo.rim("head", expected.tolist(), float(expected[2]), 0.0368)["fits"]
            fits = [fit for fit in fits if fit["angular_bins_of_24"] >= 12
                    and fit["edge_support"] >= 50 and fit["radial_rms_m"] < 0.002
                    and abs(fit["fit_radius_m"] - 0.0368) < 0.005
                    and np.linalg.norm(np.asarray(fit["estimated_xy"]) - expected[:2]) < 0.045]
            if not fits:
                raise ValueError("No supported lifted lid arc; transport prohibited")
            fit = min(fits, key=lambda item: item["radial_rms_m"])
            height = camera_height - (camera_height - expected[2]) * 0.0368 / fit["fit_radius_m"]
            rise = height - 0.8564
            self.state["lift_rim_evidence"] = fit
        self.state["lid_scale_rise_m"] = float(rise)
        if not minimum <= rise <= 0.23:
            raise ValueError("Upright lid scale does not verify lift; transport prohibited")

    def verify_wrist_retention(self):
        image = self.robo.image("left_wrist")
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.array([5, 125, 15]), np.array([30, 255, 240]))
        fraction = float(np.mean(mask > 0))
        self.state["wrist_body_fraction"] = fraction
        if fraction < 0.20:
            raise ValueError("Wrist no longer supports retained cup body")

    def transport_release(self):
        if not (self.state.get("bin_fit", {}).get("rim_supported", False) or
                self.state.get("bin_fit", {}).get("annular_supported", False)):
            raise ValueError("Unsupported bin rim; transport prohibited")
        offset = np.array([0.11184, -0.11260, -0.11280])
        current = np.asarray(self.robo.fk("left")["command_fk"])[:3, 3]
        goal = current.copy()
        goal[:2] = self.state["bin"][:2] - offset[:2]
        outward = current.copy()
        outward[1] = goal[1]
        self.verify_wrist_retention()
        self.line(outward)
        self.verify_wrist_retention()
        self.transition(goal)
        self.verify_wrist_retention()
        goal[2] = 0.55 - offset[2]
        self.line(goal)
        # At release height, the bin rim can occlude the cup from wrist RGB.
        # Earlier retention checks guard transport; do not block a planned release here.
        self.grip(1.0)
        self.line(goal - 0.06 * self.state["horizontal"] + np.array([0, 0, 0.10]))
        self.hold(100)

    def body_observation(self):
        image = self.robo.image("head")
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.array([5, 125, 25]), np.array([30, 255, 240]))
        contours, unused = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [contour for contour in contours if cv2.contourArea(contour) > 500]
        if not contours:
            raise ValueError("Cup body not supported by RGB")
        contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(contour)
        pixel = np.array([moments["m10"], moments["m01"]]) / moments["m00"]
        return pixel, contour

    def verify_lift(self, before):
        self.verify_lid_rise(0.08)

    def recover_tipped(self):
        raise ValueError("Tipped recovery disabled: coupler contact in author trial")

    def execute(self):
        self.state["cup"], self.state["cup_residual"] = self.locate_cup()
        self.scan_bin()
        if not (self.state["bin_fit"].get("rim_supported", False) or
                self.state["bin_fit"].get("annular_supported", False)):
            return {"status": "bin_not_verified", "ticks": self.ticks,
                    "evidence": "No supported rim or broad annular contrast"}
        self.state["cup"], self.state["cup_residual"] = self.locate_cup()
        before, unused = self.body_observation()
        try:
            self.approach()
        except ValueError as error:
            return {"status": "approach_not_verified", "ticks": self.ticks,
                    "evidence": str(error)}
        try:
            self.close_and_lift()
            self.verify_lift(before)
        except ValueError as error:
            return {"status": "grasp_not_verified", "ticks": self.ticks,
                    "evidence": str(error)}
        if not (self.state.get("bin_fit", {}).get("rim_supported", False) or
                self.state.get("bin_fit", {}).get("annular_supported", False)):
            self.line(self.state["contact"])
            self.grip(1.0)
            self.line(self.state["contact"] + np.array([0, 0, 0.15]))
            return {"status": "bin_not_verified", "ticks": self.ticks,
                    "evidence": "Cup returned; unsupported bin release prohibited"}
        self.transport_release()
        return {"status": "released_and_retreated", "ticks": self.ticks,
                "evidence": "Program completion only; native validation required"}



def policy(robo):
    return RGBPolicy(robo).execute()


def run(robo):
    return policy(robo)
